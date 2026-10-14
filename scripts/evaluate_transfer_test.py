"""Freeze selected weights first, then evaluate the new catalog without tuning access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

from compare_nanojev_runs import load, paired_interval, require
from train_nanojev_kaggle import rows, validate_inputs

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data/benchmarks/launcher-transfer-v1/synthetic"
OUTPUT = ROOT / "artifacts/kaggle/transfer-test-v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args):
    result = load(args.laya_run / "result.json")
    require(result["stop_reason"] in ("validation_patience", "maximum_one_pass"), "Laya checkpoint not selected")
    require(sha(args.laya_run / "checkpoint/model.safetensors") == result["checkpoint_sha256"], "Selected Laya changed")
    manifest = load(TEST / "manifest.json")
    require(manifest["role"] == "test_only" and not manifest["checkpoint_selection_access"], "Wrong test role")
    for name, expected in manifest["files"].items():
        require(sha(TEST / name) == expected, f"Frozen test changed: {name}")
    paths = {"original-nano": args.worktree / "models/rerank/nanojev-v01",
             "mixed-nano": ROOT / "artifacts/kaggle/outputs/mixed-v1/results/checkpoint",
             "clean-nano": ROOT / "artifacts/kaggle/outputs/clean-first-v1/results/checkpoint",
             "original-laya": args.worktree / "models/rerank/laya-multi",
             "trained-laya": args.laya_run / "checkpoint"}
    selections = {name: {"checkpoint": str(path), "sha256": sha(path / "model.safetensors"),
                          "family": "laya" if name.endswith("laya") else "nano"}
                  for name, path in paths.items()}
    require(selections["trained-laya"]["sha256"] == result["checkpoint_sha256"], "Wrong Laya selection")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection_path = OUTPUT / "fixed-checkpoints.json"
    require(not selection_path.exists(), "Transfer test already prepared; do not retune and overwrite it")
    selection_path.write_text(json.dumps({"models": selections, "laya_best_step": result["best_step"],
        "selection_source": "Completed training and regular validation, before any transfer-model prediction",
        "test_manifest_sha256": sha(TEST / "manifest.json")}, indent=2) + "\n")
    sys.path.insert(0, str(args.worktree / "src"))
    from jev_launcher.semantic import MODEL_REVISION, E5Search, passage_text

    objects = load(TEST / "catalog.json")["objects"]
    e5 = E5Search(args.worktree / "models/e5", offline=True)
    e5._load()
    e5.cache_dir = OUTPUT / "e5-cache"
    prepared, reference = [], []
    for profile in ("basic", "difficult"):
        for row in rows(TEST / f"{profile}-test.jsonl"):
            ranked = e5.rank(row["query"], objects, limit=20)
            require(not ranked["query_truncated"] and not ranked["truncated_object_ids"], "Retrieval input cut")
            shortlist = list(ranked["results"])
            first = shortlist[0]["object_id"]
            random.Random("transfer-choice-v1:" + row["id"]).shuffle(shortlist)
            ids = [obj["object_id"] for obj in shortlist]
            item = {"id": row["id"], "query": row["query"], "source": "test",
                    "group_id": row["group_id"], "target": row["expected_object_id"],
                    "profile": "transfer/" + profile, "candidate_ids": ids,
                    "documents": [passage_text(obj).removeprefix("passage: ") for obj in shortlist],
                    "retrieval_covered": row["expected_object_id"] in ids}
            prepared.append(item)
            reference.append({**{k: item[k] for k in ("id", "group_id", "target", "profile", "candidate_ids", "retrieval_covered")},
                              "predicted": first, "correct": first == item["target"]})
    validate_inputs(rows(ROOT / "artifacts/kaggle/training-data/prepared-train.jsonl"), prepared)
    for name, data in (("prepared-test.jsonl", prepared), ("e5-predictions.jsonl", reference)):
        (OUTPUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    (OUTPUT / "input-manifest.json").write_text(json.dumps({"rows": 100, "role": "test_only",
        "e5_revision": MODEL_REVISION, "positive_injections": 0,
        "files": {n: sha(OUTPUT / n) for n in ("prepared-test.jsonl", "e5-predictions.jsonl")},
        "fixed_checkpoints_sha256": sha(selection_path)}, indent=2) + "\n")
    print(json.dumps({"prepared_test_rows": 100, "checkpoints_fixed_before_inference": True}))


def score(args):
    selection = load(OUTPUT / "fixed-checkpoints.json")["models"][args.model]
    checkpoint = Path(selection["checkpoint"])
    require(sha(checkpoint / "model.safetensors") == selection["sha256"], "Tested checkpoint changed")
    input_manifest = load(OUTPUT / "input-manifest.json")
    require(sha(OUTPUT / "prepared-test.jsonl") == input_manifest["files"]["prepared-test.jsonl"], "Test inputs changed")
    if selection["family"] == "nano":
        from evaluate_nanojev_locally import score as score_nano

        score_nano(checkpoint, OUTPUT / "prepared-test.jsonl", OUTPUT / args.model)
    else:
        from evaluate_laya_locally import score as score_laya

        score_laya(argparse.Namespace(worktree=args.worktree, checkpoint=checkpoint,
                                     input=OUTPUT / "prepared-test.jsonl", output=OUTPUT / args.model))


def compare():
    selections = load(OUTPUT / "fixed-checkpoints.json")
    reference = rows(OUTPUT / "e5-predictions.jsonl")
    def metrics(data):
        reported = {}
        for profile in sorted({r["profile"] for r in data}):
            subset = [r for r in data if r["profile"] == profile]
            correct = sum(r["correct"] for r in subset)
            covered = sum(r["retrieval_covered"] for r in subset)
            reported[profile] = {
                "rows": len(subset), "correct": correct, "accuracy": correct / len(subset),
                "candidate_recall": covered / len(subset),
                "conditional_accuracy": correct / covered if covered else None,
                "retrieval_failures": len(subset) - covered, "choice_failures": covered - correct,
            }
        return reported

    manifest = load(OUTPUT / "input-manifest.json")
    require(sha(OUTPUT / "fixed-checkpoints.json") == manifest["fixed_checkpoints_sha256"], "Selections changed")
    for filename, expected in manifest["files"].items():
        require(sha(OUTPUT / filename) == expected, "Transfer inputs or E5 reference changed")
    result = {"role": "test_only; no checkpoint selection", "e5": metrics(reference), "models": {}}
    for name, model in selections["models"].items():
        evaluated = load(OUTPUT / name / "result.json")
        require(evaluated["checkpoint_sha256"] == model["sha256"], "Wrong tested weights")
        require(evaluated["input_sha256"] == manifest["files"]["prepared-test.jsonl"], "Wrong evaluated input")
        predictions = rows(OUTPUT / name / "validation-predictions.jsonl")
        require(len(predictions) == len(reference) == 100, "Incomplete transfer test")
        for actual, expected in zip(predictions, reference, strict=True):
            require(all(actual[k] == expected[k] for k in ("id", "profile", "group_id", "target", "candidate_ids", "retrieval_covered")), "Unpaired transfer rows")
            require(actual["correct"] == (actual["predicted"] == actual["target"]), "Incorrect result flag")
            scores, candidates = actual["scores"], actual["candidate_ids"]
            require(len(scores) == len(candidates) == len(set(candidates)) == 20
                    and all(math.isfinite(v) and 0 <= v <= 1 for v in scores)
                    and math.isclose(sum(scores), 1, abs_tol=1e-4), "Invalid transfer scores")
            require(actual["predicted"] == candidates[max(range(20), key=scores.__getitem__)], "Prediction differs from scores")
            require(actual["retrieval_covered"] == (actual["target"] in candidates), "Wrong target coverage")
        result["models"][name] = {"metrics": metrics(predictions), "paired_vs_e5": {
            p: paired_interval([(a, b) for a, b in zip(predictions, reference, strict=True) if a["profile"] == p])
            for p in result["e5"]}}
    (OUTPUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--laya-run", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--model", choices=("original-nano", "mixed-nano", "clean-nano", "original-laya", "trained-laya"))
    action.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.laya_run is None:
            parser.error("--prepare requires a completed --laya-run")
        prepare(args)
    elif args.compare:
        compare()
    else:
        score(args)
