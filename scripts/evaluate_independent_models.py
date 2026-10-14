"""Freeze the new clean/ASR evaluation and audit all fixed decision checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from audit_kev_run import check_predictions, metrics
from compare_nanojev_runs import load, paired_interval, require
from independent_dataset import audit
from train_kev_kaggle import checkpoint_hashes, sha
from train_laya_kaggle import selection_loss
from train_nanojev_kaggle import rows, validate_inputs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/benchmarks/launcher-independent-v1"
OUTPUT = ROOT / "artifacts/kaggle/independent-v1"


def freeze_models(worktree):
    old = load(ROOT / "artifacts/kaggle/transfer-test-v1/fixed-checkpoints.json")
    models = {}
    for name, previous in old["models"].items():
        path = Path(previous["checkpoint"]).resolve()
        require(sha(path / "model.safetensors") == previous["sha256"], "Previously chosen weights changed")
        if name.startswith("original-"):
            staged = "laya-data/laya-input-manifest.json" if name.endswith("laya") else "training-data/experiment-input-manifest.json"
            files = load(ROOT / "artifacts/kaggle" / staged)["checkpoint_files"]
            # The Nano archive has the pinned native implementation at its root.
            files = {n: h for n, h in files.items() if n != "native_model.py"}
        else:
            files = {p.relative_to(path).as_posix(): sha(p) for p in path.rglob("*") if p.is_file()}
        for filename, expected in files.items():
            require(sha(path / filename) == expected, f"Changed checkpoint file: {name}/{filename}")
        models[name] = {**previous, "checkpoint": str(path), "files": files}
    original = worktree / "models/rerank/kev-08"
    original_files = load(ROOT / "artifacts/kaggle/kev-data/kev-input-manifest.json")["checkpoint_files"]
    selected = ROOT / "artifacts/kaggle/outputs/kev-clean-first-v1/results/checkpoint"
    kev_result = load(selected.parent / "result.json")
    require(kev_result["best_step"] == 710 and kev_result["stop_reason"] == "validation_patience", "Kev selection changed")
    for name, path in (("original-kev", original), ("trained-kev", selected)):
        files = original_files if name == "original-kev" else {
            p.relative_to(path).as_posix(): sha(p) for p in path.rglob("*") if p.is_file()}
        for filename, expected in files.items():
            require(sha(path / filename) == expected, f"Changed Kev file: {filename}")
        models[name] = {"checkpoint": str(path.resolve()), "family": "kev", "files": files,
                        "sha256": checkpoint_hashes(path)}
    require(models["trained-kev"]["sha256"] == kev_result["checkpoint_sha256"], "Wrong selected Kev")
    return {"models": models, "selected_steps": {"trained-laya": old["laya_best_step"], "trained-kev": 710},
            "selection_source": "Previously completed training and ordinary validation; no new-data selection"}


def prepare(worktree):
    require(not OUTPUT.exists(), "Use a new evaluation revision; do not overwrite frozen inputs")
    verified = audit(check_audio=True)
    selections = freeze_models(worktree)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "fixed-checkpoints.json").write_text(json.dumps(selections, indent=2) + "\n")
    (OUTPUT / "dataset-audit.json").write_text(json.dumps(verified, indent=2) + "\n")
    sys.path.insert(0, str(worktree / "src"))
    from jev_launcher.semantic import MODEL_REVISION, E5Search, passage_text

    e5 = E5Search(worktree / "models/e5", offline=True)
    e5._load()
    e5.cache_dir = OUTPUT / "e5-cache"
    objects = load(DATA / "clean/catalog.json")["objects"]
    prepared, reference, source_files = [], [], {}
    for representation in ("clean", "asr"):
        manifest = load(DATA / representation / "manifest.json")
        for filename, expected in manifest["files"].items():
            require(sha(DATA / representation / filename) == expected, "New source data changed")
        for split in ("validation", "test"):
            for row in rows(DATA / representation / f"{split}.jsonl"):
                ranked = e5.rank(row["query"], objects, limit=20)
                require(not ranked["query_truncated"] and not ranked["truncated_object_ids"], "Retrieval input cut")
                shortlist = list(ranked["results"])
                first = shortlist[0]["object_id"]
                random.Random("independent-choice-v1:" + row["id"].removesuffix("-asr")).shuffle(shortlist)
                ids = [obj["object_id"] for obj in shortlist]
                item = {"id": row["id"], "query": row["query"], "source": "evaluation",
                        "group_id": row["group_id"], "target": row["expected_object_id"],
                        "profile": f"independent/{representation}/{split}", "candidate_ids": ids,
                        "documents": [passage_text(obj).removeprefix("passage: ") for obj in shortlist],
                        "retrieval_covered": row["expected_object_id"] in ids}
                prepared.append(item)
                reference.append({**{k: item[k] for k in ("id", "profile", "group_id", "target", "candidate_ids", "retrieval_covered")},
                                  "predicted": first, "correct": first == item["target"]})
            print(json.dumps({"prepared_profile": f"{representation}/{split}", "rows_so_far": len(prepared)}), flush=True)
        for filename in ("catalog.json", "validation.jsonl", "test.jsonl", "manifest.json"):
            source_files[f"{representation}/{filename}"] = sha(DATA / representation / filename)
    require(len(prepared) == 400 and len({r["id"] for r in prepared}) == 400, "Wrong evaluation size")
    validate_inputs(rows(ROOT / "artifacts/kaggle/training-data/prepared-train.jsonl"), prepared)
    for name, values in (("prepared-evaluation.jsonl", prepared), ("e5-predictions.jsonl", reference)):
        (OUTPUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")
    manifest = {"role": "evaluation_only; no training, checkpoint selection or threshold tuning",
                "rows": 400, "independent_commands": 200, "profiles": dict(Counter(r["profile"] for r in prepared)),
                "e5_revision": MODEL_REVISION, "positive_injections": 0, "source_files": source_files,
                "files": {n: sha(OUTPUT / n) for n in ("prepared-evaluation.jsonl", "e5-predictions.jsonl", "fixed-checkpoints.json", "dataset-audit.json")}}
    (OUTPUT / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": True, "e5_metrics": metrics(reference)}), flush=True)


def compare(results, previous_results=None):
    manifest = load(OUTPUT / "input-manifest.json")
    for filename, expected in manifest["files"].items():
        require(sha(OUTPUT / filename) == expected, "Frozen evaluation changed")
    for filename, expected in manifest["source_files"].items():
        require(sha(DATA / filename) == expected, "Original evaluation changed")
    data, reference = rows(OUTPUT / "prepared-evaluation.jsonl"), rows(OUTPUT / "e5-predictions.jsonl")
    selections = load(OUTPUT / "fixed-checkpoints.json")
    completion = load(results / "completion.json")
    expected_models = ["original-kev", "trained-kev"] if previous_results else list(selections["models"])
    require(completion["completed_models"] == expected_models
            and completion["optimizer_updates"] == 0
            and completion["input_manifest_sha256"] == sha(ROOT / "artifacts/kaggle/independent-data/input-manifest.json"),
            "Remote evaluation incomplete or changed")
    report = {"audit": "passed", "role": manifest["role"], "independent_commands": 200,
              "e5": metrics(reference), "models": {}, "input_sha256": manifest["files"]["prepared-evaluation.jsonl"],
              "result_sources": {}}
    predictions = {}
    for name, selection in selections["models"].items():
        directory = results if name in expected_models else previous_results
        report["result_sources"][name] = str((directory / name).resolve())
        result = load(directory / name / "result.json")
        require(result["checkpoint_sha256"] == selection["sha256"], "Wrong evaluated checkpoint")
        require(result["input_sha256"] == report["input_sha256"] and result["optimizer_updates"] == 0, "Wrong evaluation input/run")
        require(result["precision"] == "fp32" and result["device"] == "cuda", "Different evaluation precision/device")
        values = rows(directory / name / "validation-predictions.jsonl")
        indexed = check_predictions(values, data)
        values = [indexed[r["id"]] for r in data]
        actual_metrics = metrics(values)
        require(actual_metrics == result["metrics"], "Reported metric differs from predictions")
        predictions[name] = values
        paired = {}
        for profile in report["e5"]:
            pairs = [(a, b) for a, b in zip(values, reference, strict=True) if a["profile"] == profile]
            paired[profile] = {**paired_interval(pairs),
                               "fixed_e5_errors": sum(a["correct"] and not b["correct"] for a, b in pairs),
                               "lost_e5_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
                               "scope": "Scenario groups in this profile; does not cover training-seed variability"}
        _, profile_nll = selection_loss(values)
        report["models"][name] = {"metrics": actual_metrics, "paired_vs_e5": paired,
                                  "conditional_nll": profile_nll, "encoding": result["encoding"]}
    report["adaptation"] = {}
    for original, trained in (("original-nano", "mixed-nano"), ("original-nano", "clean-nano"),
                              ("original-laya", "trained-laya"), ("original-kev", "trained-kev"),
                              ("mixed-nano", "clean-nano")):
        label = trained if original != "mixed-nano" else "clean-nano-vs-mixed"
        report["adaptation"][label] = {}
        for profile in report["e5"]:
            pairs = [(a, b) for a, b in zip(predictions[trained], predictions[original], strict=True) if a["profile"] == profile]
            report["adaptation"][label][profile] = {**paired_interval(pairs),
                "fixed_errors": sum(a["correct"] and not b["correct"] for a, b in pairs),
                "lost_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
                "scope": "Scenario groups in this profile; does not cover training-seed variability"}
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"audit": report["audit"], "e5": report["e5"],
                      "models": {name: v["metrics"] for name, v in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--previous-results", type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.worktree is not None, "--prepare requires --worktree")
        prepare(args.worktree)
    else:
        require(args.results is not None, "Comparison requires --results")
        compare(args.results, args.previous_results)
