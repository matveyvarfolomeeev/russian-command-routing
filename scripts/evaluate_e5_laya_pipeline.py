"""Paired application-path comparison: original/trained E5 -> fixed Laya FP32."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "artifacts/kaggle/e5-evaluation-v1"
METADATA = ROOT / "artifacts/benchmarks/app-metadata-v2"
os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metrics(records):
    import numpy as np

    profiles = defaultdict(list)
    for row in records:
        profiles[row["profile"]].append(row)
    detail = {name: {"rows": len(local), "correct": sum(r["correct"] for r in local),
                     "e5_correct": sum(r["e5_correct"] for r in local),
                     "recall20": sum(r["target_in_top20"] for r in local)}
              for name, local in profiles.items()}
    scenarios = defaultdict(lambda: {"rows": 0, "correct": 0, "e5_correct": 0})
    for name, local in detail.items():
        key = "/".join(name.split("/")[:2]) if name.startswith("independent/") else name
        for field in ("rows", "correct", "e5_correct"):
            scenarios[key][field] += local[field]
    values = np.array([row["warm_pipeline_ms"] for row in records])
    return {"rows": len(records), "correct": sum(row["correct"] for row in records),
            "e5_correct": sum(row["e5_correct"] for row in records),
            "recall20": sum(row["target_in_top20"] for row in records),
            "profiles": detail, "scenarios": dict(scenarios),
            "equal_scenario_accuracy": sum(r["correct"] / r["rows"] for r in scenarios.values()) / len(scenarios),
            "e5_equal_scenario_accuracy": sum(r["e5_correct"] / r["rows"] for r in scenarios.values()) / len(scenarios),
            "latency": {"p50_ms": float(np.quantile(values, .5)),
                        "p95_ms": float(np.quantile(values, .95)),
                        "maximum_ms": float(values.max()), "over_500_ms": int((values > 500).sum())}}


def paired(first, second):
    assert [(r["scope"], r["id"]) for r in first] == [(r["scope"], r["id"]) for r in second]
    pairs = list(zip(first, second, strict=True))
    fixes = [(a, b) for a, b in pairs if not a["correct"] and b["correct"]]
    losses = [(a, b) for a, b in pairs if a["correct"] and not b["correct"]]
    same_sets = [(a, b) for a, b in pairs if set(a["candidate_ids"]) == set(b["candidate_ids"])]
    return {"fixed": len(fixes), "lost": len(losses),
            "retrieval_gained": sum(not a["target_in_top20"] and b["target_in_top20"] for a, b in pairs),
            "retrieval_lost": sum(a["target_in_top20"] and not b["target_in_top20"] for a, b in pairs),
            "shortlist_sets_changed": len(pairs) - len(same_sets),
            "same_set_different_input_order": sum(a["candidate_ids"] != b["candidate_ids"] for a, b in same_sets),
            "fixes_with_same_set": sum(not a["correct"] and b["correct"] for a, b in same_sets),
            "losses_with_same_set": sum(a["correct"] and not b["correct"] for a, b in same_sets),
            "fixed_ids": [a["id"] for a, _ in fixes], "lost_ids": [a["id"] for a, _ in losses]}


def run(args):
    from run_launcher_model import Runtime

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh evaluation directory")
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_files = {}

    def freeze(path):
        frozen_files[str(path.resolve())] = sha(path)

    known_manifest = read(E5 / "input/input-manifest.json")
    for name, expected in known_manifest["files"].items():
        assert sha(E5 / "input" / name) == expected, name
    for name, expected in known_manifest["source_files"].items():
        assert sha(ROOT / name) == expected, name
        freeze(ROOT / name)
    freeze(E5 / "input/input-manifest.json")
    freeze(E5 / "input/full-evaluation.jsonl")
    freeze(E5 / "input/catalogs.json")
    catalog_specs = read(E5 / "input/catalogs.json")
    catalogs = {key: sorted(read(ROOT / value["source_file"])["objects"], key=lambda item: item["object_id"])
                for key, value in catalog_specs.items()}
    data = [{**row, "scope": "known800"} for row in rows(E5 / "input/full-evaluation.jsonl")]
    assert len(data) == 800
    # First inspect the difficult old STT; no changes to query text, labels or candidates.
    data.sort(key=lambda row: not row["profile"].startswith("launcher-asr-v1/"))
    meta_manifest = read(METADATA / "manifest.json")
    for name, expected in meta_manifest["files"].items():
        assert sha(METADATA / name) == expected, name
    freeze(METADATA / "manifest.json")
    freeze(METADATA / "after-catalog.json")
    catalogs["metadata-real"] = sorted(read(METADATA / "after-catalog.json")["objects"], key=lambda item: item["object_id"])
    for scope, name, count in (("metadata-real150", "evaluation.jsonl", 150),
                               ("metadata-diagnostic24", "diagnostic-queries.jsonl", 24)):
        local = rows(METADATA / name)
        assert len(local) == count
        data.extend({**row, "scope": scope, "catalog_key": "metadata-real"} for row in local)
        freeze(METADATA / name)
    freeze(METADATA / "diagnostic-role.json")
    assert len(data) == len({(row["scope"], row["id"]) for row in data}) == 974
    for key, objects in catalogs.items():
        assert len(objects) == len({item["object_id"] for item in objects}) == 177
        if key in catalog_specs:
            assert [item["object_id"] for item in objects] == catalog_specs[key]["candidate_ids"]

    config_path = ROOT / "models/launcher-runtime.json"
    config = read(config_path)
    freeze(config_path)
    config["index_directory"] = str(args.output / "unused-original-index")
    worktree = Path(config["worktree"])
    sources = [Path(__file__).resolve(), ROOT / "scripts/run_launcher_model.py",
               ROOT / "scripts/benchmark_choice_quantization.py", ROOT / "scripts/train_laya_kaggle.py",
               ROOT / "scripts/train_nanojev_kaggle.py", ROOT / "scripts/laya_cpu_weights.py",
               worktree / "src/jev_launcher/semantic.py", worktree / "src/jev_launcher/local_weights.py"]
    assert sha(worktree / "src/jev_launcher/semantic.py") == known_manifest["card_renderer_sha256"]
    for path in sources:
        freeze(path)
    source_dir = args.output / "source"
    source_dir.mkdir()
    for path in sources:
        (source_dir / path.name).write_bytes(path.read_bytes())
    save(args.output / "catalogs.json", catalogs)
    (args.output / "queries.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")
    runtime = Runtime(config, "laya")
    assert next(runtime.model.parameters()).dtype == runtime.torch.float32
    assert runtime.weights_sha == "acc8ee1848ae7c2fcca1b8e640b76823d8990482c0be2af592f987cd234f6bcc"
    freeze(Path(config["profiles"]["laya"]["checkpoint"]) / "model.safetensors")

    import onnxruntime as ort
    from jev_launcher.semantic import E5Search
    from tokenizers import Tokenizer

    export = read(E5 / "selected-export/export.json")
    onnx_path = E5 / "selected-export/model.onnx"
    tokenizer_path = ROOT / "artifacts/kaggle/outputs/e5-small-clean-first-v1/results/checkpoint/tokenizer.json"
    assert sha(onnx_path) == export["onnx_sha256"]
    assert sha(tokenizer_path) == export["tokenizer_sha256"]
    for path in (onnx_path, tokenizer_path, E5 / "selected-export/export.json"):
        freeze(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    trained_session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    trained_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    trained_tokenizer.enable_truncation(max_length=512)
    trained_tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    sessions = {"original": (runtime.e5._session, runtime.e5._tokenizer),
                "trained": (trained_session, trained_tokenizer)}
    engines, captured, references = {}, {}, {}
    for variant, (session, tokenizer) in sessions.items():
        path = E5 / ("original-onnx" if variant == "original" else "selected-onnx") / "predictions.jsonl"
        references[variant] = {r["id"]: r for r in rows(path)}
        freeze(path)
        for scope in ("metadata-real150", "metadata-diagnostic24"):
            directory = "evaluation-v2" if scope.endswith("150") else "diagnostic-evaluation"
            method = "e5-original" if variant == "original" else "e5-trained"
            path = METADATA / directory / f"{method}-after.jsonl"
            references[(variant, scope)] = {r["id"]: r for r in rows(path)}
            freeze(path)
        for key, objects in catalogs.items():
            engine = E5Search(args.output / "indexes" / variant / key.replace("/", "-"), offline=True)
            engine._session, engine._tokenizer = session, tokenizer
            engine._index(objects)

            def capture(query, objects, *, limit=20, method=engine.rank):
                result = method(query, objects, limit=limit)
                captured["ranked"] = result
                return result

            engine.rank = capture
            engines[variant, key] = engine

    protocol = {"created_at": datetime.now(timezone.utc).isoformat(), "rows_per_variant": len(data),
                "variants": ["original", "trained"], "decision_model": "laya-fp32",
                "laya_checkpoint_sha256": runtime.weights_sha, "trained_e5": export,
                "runtime_policy": runtime.policy, "optimizer_updates": 0, "checkpoint_selection": False,
                "production_config_changed": False, "card_changes_during_run": False,
                "roles": {"known800": known_manifest["role"],
                          "metadata-real150": "Known real150 on frozen enriched cards; overlaps known800 queries, separate scope",
                          "metadata-diagnostic24": read(METADATA / "diagnostic-role.json")["role"]},
                "order": "Old STT first, other known800 in frozen order, real150, diagnostic24; alternating original/trained and trained/original per query",
                "timing_scope": "One warm paired serial pass; native E5 -> application stable shuffle -> Laya -> ID; excludes ASR, loading, initial indexing and artifact/parity checks",
                "frozen_files": frozen_files, "python": sys.version, "onnxruntime": ort.__version__,
                "torch": runtime.torch.__version__,
                "query_sha256": sha(args.output / "queries.jsonl"), "catalogs_sha256": sha(args.output / "catalogs.json")}
    save(args.output / "protocol.json", protocol)
    first = data[0]
    for variant in sessions:
        runtime.e5 = engines[variant, first["catalog_key"]]
        runtime.search({"query": "открыть приложение", "objects": catalogs[first["catalog_key"]]})
    results = {variant: [] for variant in sessions}
    with ExitStack() as stack:
        streams = {variant: stack.enter_context((args.output / f"{variant}-predictions.jsonl").open("w", encoding="utf-8"))
                   for variant in sessions}
        for index, row in enumerate(data, 1):
            objects = catalogs[row["catalog_key"]]
            by_id = {item["object_id"]: item for item in objects}
            for variant in (("original", "trained") if index % 2 else ("trained", "original")):
                runtime.e5 = engines[variant, row["catalog_key"]]
                began = time.perf_counter()
                output = runtime.search({"query": row["query"], "objects": objects})
                elapsed = (time.perf_counter() - began) * 1000
                ranked = captured["ranked"]
                assert ranked["index_cache_hit"] and not ranked["query_truncated"] and not ranked["truncated_object_ids"]
                retrieved = [r["object_id"] for r in ranked["results"]]
                reference = references[variant][row["id"]] if row["scope"] == "known800" else references[variant, row["scope"]][row["id"]]
                expected = reference["top20"] if row["scope"] == "known800" else reference["candidate_ids"]
                assert retrieved == expected and retrieved[0] == reference["predicted"], row["id"]
                expected_order = retrieved.copy()
                random.Random("launcher-choice-v1:" + row["query"].strip().casefold()).shuffle(expected_order)
                assert output["candidate_ids"] == expected_order
                record = {**row, "variant": variant, "predicted": output["object_ids"][0],
                          "correct": output["object_ids"][0] == row["target"],
                          "ranked_ids": output["object_ids"], "candidate_ids": output["candidate_ids"],
                          "scores": output["scores"], "retrieval_ids": retrieved,
                          "retrieval_scores": [r["score"] for r in ranked["results"]],
                          "e5_predicted": retrieved[0], "e5_correct": retrieved[0] == row["target"],
                          "target_in_top20": row["target"] in retrieved, "target_kind": by_id[row["target"]]["kind"],
                          "warm_pipeline_ms": elapsed, "e5_elapsed_ms": ranked["elapsed_ms"]}
                results[variant].append(record)
                streams[variant].write(json.dumps(record, ensure_ascii=False) + "\n")
                streams[variant].flush()
            if index % 25 == 0 or index == len(data):
                print(json.dumps({"completed_pairs": index, "scope": row["scope"],
                                  "correct_so_far": {v: sum(r["correct"] for r in local) for v, local in results.items()}}), flush=True)
    report = {"optimizer_updates": 0, "checkpoint_selection": False, "production_config_changed": False,
              "protocol_sha256": sha(args.output / "protocol.json"), "scopes": {}}
    for scope in protocol["roles"]:
        local = {variant: [r for r in records if r["scope"] == scope] for variant, records in results.items()}
        report["scopes"][scope] = {variant: metrics(records) for variant, records in local.items()}
        report["scopes"][scope]["paired"] = paired(local["original"], local["trained"])
    report["prediction_sha256"] = {f"{variant}-predictions.jsonl": sha(args.output / f"{variant}-predictions.jsonl") for variant in sessions}
    save(args.output / "comparison.json", report)
    print(json.dumps({"completed": True, "scopes": report["scopes"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/e5-laya-pipeline-v1")
    run(parser.parse_args())
