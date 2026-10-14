"""Compare unchanged E5/Laya weights on the original and enriched real cards."""

import argparse
import gc
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args):
    import numpy as np
    import onnxruntime as ort
    from run_launcher_model import Runtime
    from tokenizers import Tokenizer

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new evaluation output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read(args.input / "manifest.json")
    for name, expected in manifest["files"].items():
        assert sha(args.input / name) == expected, name
    query_file = args.queries or args.input / "evaluation.jsonl"
    data = [json.loads(line) for line in query_file.read_text(encoding="utf-8").splitlines()]
    assert len(data) == len({row["id"] for row in data}) == (24 if args.queries else 150)
    role = read(args.input / "diagnostic-role.json")["role"] if args.queries else manifest["role"]
    catalogs = {variant: sorted(read(args.input / f"{variant}-catalog.json")["objects"],
                               key=lambda item: item["object_id"]) for variant in ("before", "after")}
    assert [item["object_id"] for item in catalogs["before"]] == [item["object_id"] for item in catalogs["after"]]
    assert len(catalogs["before"]) == 177
    by_id = {item["object_id"]: item for item in catalogs["before"]}
    config = read(ROOT / "models/launcher-runtime.json")
    config["index_directory"] = str(args.output / "laya-index")
    results, sources = {}, {}

    def evaluate(method, search, *, full_ranking):
        predictions, metrics = {}, {}
        for variant, objects in catalogs.items():
            search({"query": "открыть приложение", "objects": objects})  # Rebuild and warm this index.
            records = []
            for index, row in enumerate(data, 1):
                began = time.perf_counter()
                output = search({"query": row["query"], "objects": objects})
                milliseconds = (time.perf_counter() - began) * 1000
                ids = output["object_ids"]
                shortlist = output["candidate_ids"]
                assert len(ids) == len(set(ids)) == (177 if full_ranking else 20)
                assert len(shortlist) == 20 and set(shortlist) <= set(by_id)
                record = {**row, "variant": variant, "method": method,
                          "predicted": ids[0], "correct": ids[0] == row["target"],
                          "candidate_ids": shortlist, "ranked_ids": ids,
                          "scores": output["scores"], "warm_pipeline_ms": milliseconds,
                          "target_in_top20": row["target"] in shortlist,
                          "target_kind": by_id[row["target"]]["kind"]}
                assert len(record["scores"]) == len(ids)
                assert np.isfinite(record["scores"]).all()
                records.append(record)
                if index % 25 == 0 or index == len(data):
                    print(json.dumps({"method": method, "variant": variant, "completed": index}), flush=True)
            predictions[variant] = records
            grouped = defaultdict(list)
            for row in records:
                grouped[row["profile"]].append(row)
            profiles = {profile: {"rows": len(local), "correct": sum(row["correct"] for row in local),
                                  "recall20": sum(row["target_in_top20"] for row in local)}
                        for profile, local in grouped.items()}
            app_rows = [row for row in records if row["target_kind"] == "app"]
            values = np.array([row["warm_pipeline_ms"] for row in records])
            metrics[variant] = {"profiles": profiles, "rows": len(records),
                                "correct": sum(row["correct"] for row in records),
                                "app_rows": len(app_rows), "app_correct": sum(row["correct"] for row in app_rows),
                                "recall20": sum(row["target_in_top20"] for row in records),
                                "latency": {"p50_ms": float(np.quantile(values, .5)),
                                            "p95_ms": float(np.quantile(values, .95)),
                                            "maximum_ms": float(values.max())}}
            path = args.output / f"{method}-{variant}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        before, after = predictions["before"], predictions["after"]
        assert [(r["id"], r["query"], r["target"]) for r in before] == [(r["id"], r["query"], r["target"]) for r in after]
        metrics["paired"] = {"fixed": sum(not a["correct"] and b["correct"] for a, b in zip(before, after)),
                             "lost": sum(a["correct"] and not b["correct"] for a, b in zip(before, after)),
                             "shortlist_changed": sum(set(a["candidate_ids"]) != set(b["candidate_ids"]) for a, b in zip(before, after))}
        assert metrics["after"]["correct"] == metrics["before"]["correct"] + metrics["paired"]["fixed"] - metrics["paired"]["lost"]
        return metrics, predictions

    for method in ("e5-original", "e5-trained"):
        runtime = Runtime(config, "e5")
        native = runtime.e5
        native.cache_dir = args.output / "indexes" / method
        if method == "e5-trained":
            exported = ROOT / "artifacts/kaggle/e5-evaluation-v1/selected-export"
            export = read(exported / "export.json")
            tokenizer_path = ROOT / "artifacts/kaggle/outputs/e5-small-clean-first-v1/results/checkpoint/tokenizer.json"
            assert sha(exported / "model.onnx") == export["onnx_sha256"]
            assert sha(tokenizer_path) == export["tokenizer_sha256"]
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            native._session = ort.InferenceSession(str(exported / "model.onnx"), sess_options=options,
                                                  providers=["CPUExecutionProvider"])
            native._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            native._tokenizer.enable_truncation(max_length=512)
            native._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            sources["trained_e5_onnx"] = {"sha256": export["onnx_sha256"], "checkpoint_sha256": export["checkpoint_sha256"]}
        else:
            sources["original_e5"] = {"model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3"}

        def search(request, engine=native):
            output = engine.rank(request["query"], request["objects"], limit=177)
            assert not output["query_truncated"] and not output["truncated_object_ids"]
            records = output["results"]
            return {"object_ids": [row["object_id"] for row in records],
                    "candidate_ids": [row["object_id"] for row in records[:20]],
                    "scores": [row["score"] for row in records]}

        results[method], predictions = evaluate(method, search, full_ranking=True)
        if not args.queries:
            reference_dir = "original-onnx" if method == "e5-original" else "selected-onnx"
            reference_path = ROOT / "artifacts/kaggle/e5-evaluation-v1" / reference_dir / "predictions.jsonl"
            reference = {row["id"]: row for row in map(json.loads, reference_path.read_text(encoding="utf-8").splitlines())}
            assert all(row["predicted"] == reference[row["id"]]["predicted"] for row in predictions["before"])
            sources[method + "_baseline_reference"] = sha(reference_path)
        del search, runtime, native
        gc.collect()
    runtime = Runtime(config, "laya")
    assert next(runtime.model.parameters()).dtype == runtime.torch.float32
    sources["laya_checkpoint_sha256"] = runtime.weights_sha
    results["e5-original-laya"], _ = evaluate("e5-original-laya", runtime.search, full_ranking=False)
    sources["runtime_policy"] = runtime.policy
    report = {"role": role, "changed_cards": manifest["changed_cards"],
              "optimizer_updates": 0, "checkpoint_selection": False,
              "original_queries_targets_and_ids_unchanged": True,
              "e5_before_top1_matches_prior150": True if not args.queries else None,
              "query_file_sha256": sha(query_file), "query_rows": len(data),
              "results": results, "sources": sources,
              "timing_scope": f"One warm serial pass of the same{len(data)} queries before and after; includes ranking/card formatting, excludes loading, initial indexing, ASR and artifact writes",
              "not_measured": "Kev, fictional catalogs, new blind generalization, repeated latency distribution, E5-trained -> Laya",
              "input_manifest_sha256": sha(args.input / "manifest.json"),
              "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in (
                  Path(__file__).resolve(), ROOT / "scripts/run_launcher_model.py",
                  ROOT / "src/jev_launcher/app_metadata.py")}}
    report["prediction_sha256"] = {path.name: sha(path) for path in args.output.glob("*.jsonl")}
    save(args.output / "comparison.json", report)
    print(json.dumps({"completed": True, "results": results}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/benchmarks/app-metadata-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/app-metadata-v2/evaluation")
    parser.add_argument("--queries", type=Path)
    run(parser.parse_args())
