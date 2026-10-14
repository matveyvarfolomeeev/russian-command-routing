"""Check Laya's actual exposure, stopping, checkpoint hashes and paired local results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from compare_nanojev_runs import load, paired_interval, require
from train_laya_kaggle import ValidationStopper, selection_loss
from train_nanojev_kaggle import digest, plan_order, rows

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    result, data = load(args.run / "result.json"), rows(ROOT / "artifacts/kaggle/laya-data/prepared-train.jsonl")
    plan = result["plan"]
    require(plan == load(args.run / "plan.json"), "Result/plan mismatch")
    require(plan["schedule"] == "clean_first" and not plan["smoke"] and plan["device"] == "cuda", "Wrong run")
    order, prefix = plan_order(data, "clean_first", plan["seed"])
    require(plan["rows"] == len(data) == 22656 and plan["maximum_epochs"] == 1, "Wrong data budget")
    require(plan["ordered_ids_sha256"] == digest([data[i]["id"] for i in order]) and prefix == 2272, "Wrong order")
    manifest = load(ROOT / "artifacts/kaggle/laya-data/laya-input-manifest.json")
    require(plan["input_manifest"] == manifest, "Input manifest changed")
    for name, expected in manifest["files"].items():
        with (ROOT / "artifacts/kaggle/laya-data" / name).open("rb") as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == expected, "Local frozen input changed")
    ledger = rows(args.run / "training.jsonl")
    require(len(ledger) == result["steps"] <= 1416, "Incorrect completed-step count")
    for step, entry in enumerate(ledger, 1):
        require(entry["step"] == step, "Missing or duplicate update")
        expected = Counter(data[i]["source"] for i in order[(step - 1) * 16:step * 16])
        require(entry["sources"] == expected, "Source exposure differs from declared order")
        require(all(math.isfinite(entry[key]) for key in ("loss", "grad_norm", "amp_scale"))
                and all(math.isfinite(v) for v in entry["lr"]), "Non-finite training")
    checks = rows(args.run / "validation.jsonl")
    expected_steps = [0] + list(range(142, result["steps"] + 1, 142))
    if result["steps"] not in expected_steps:
        expected_steps.append(result["steps"])
    require([c["step"] for c in checks] == expected_steps, "Missing validation checks")
    validation = rows(ROOT / "artifacts/kaggle/laya-data/prepared-validation.jsonl")
    for check in checks:
        name = "baseline" if check["step"] == 0 else f"step-{check['step']}"
        predictions = rows(args.run / f"{name}-predictions.jsonl")
        require(len(predictions) == len(validation) == 150, "Incomplete checkpoint validation")
        for actual, expected in zip(predictions, validation, strict=True):
            require(all(actual[key] == expected[key] for key in
                        ("id", "profile", "group_id", "target", "candidate_ids", "retrieval_covered")), "Different checkpoint inputs")
            scores = actual["scores"]
            require(len(scores) == 20 and all(math.isfinite(v) and 0 <= v <= 1 for v in scores)
                    and math.isclose(sum(scores), 1, abs_tol=1e-4), "Invalid checkpoint scores")
            require(actual["predicted"] == actual["candidate_ids"][max(range(20), key=scores.__getitem__)]
                    and actual["correct"] == (actual["predicted"] == actual["target"]), "Wrong checkpoint prediction")
        recomputed, profile_losses = selection_loss(predictions)
        require(math.isclose(recomputed, check["selection_loss"], abs_tol=1e-10)
                and profile_losses == check["profile_losses"], "Stored validation loss differs from predictions")
    stopper = ValidationStopper(checks[0]["selection_loss"], plan["patience"], plan["min_delta"])
    for check in checks[1:]:
        improved, stop = stopper.update(check["selection_loss"], check["step"])
        require(improved == check["improved"] and stopper.wait == check["wait"], "Stopping ledger changed")
        require(not stop or check == checks[-1], "Training continued after stopping criterion")
    require(stopper.best_step == result["best_step"], "Wrong selected step")
    require(stopper.best_loss == result["best"]["selection_loss"], "Wrong selected loss")
    require((result["stop_reason"] == "validation_patience" and stopper.wait >= plan["patience"])
            or (result["stop_reason"] == "maximum_one_pass" and result["steps"] == 1416), "Wrong stopping reason")
    for folder, key in (("checkpoint", "checkpoint_sha256"), ("last-checkpoint", "last_checkpoint_sha256")):
        with (args.run / folder / "model.safetensors").open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        require(actual == result[key], f"Downloaded weights differ: {folder}")
    baseline = load(args.local / "laya-baseline-v1/result.json")
    trained = load(args.local / "laya-clean-first-v1/result.json")
    require(baseline["checkpoint_sha256"] == manifest["original_weight_sha256"], "Wrong local baseline")
    require(trained["checkpoint_sha256"] == result["checkpoint_sha256"], "Wrong locally evaluated checkpoint")
    require(baseline["input_sha256"] == trained["input_sha256"], "Different local inputs")
    local_manifest = load(ROOT / "artifacts/kaggle/local-validation.manifest.json")
    with (ROOT / "artifacts/kaggle/local-validation.jsonl").open("rb") as stream:
        local_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    require(baseline["input_sha256"] == local_manifest["sha256"] == local_sha, "Canonical local input changed")
    before, after = (rows(args.local / name / "validation-predictions.jsonl")
                     for name in ("laya-baseline-v1", "laya-clean-first-v1"))
    def identity(r):
        return [r[k] for k in ("id", "profile", "group_id", "target", "candidate_ids", "retrieval_covered")]

    require(len(before) == len(after) == 300 and list(map(identity, before)) == list(map(identity, after)), "Unpaired evaluation")
    for row in before + after:
        require(row["correct"] == (row["target"] == row["predicted"]), "Wrong correct flag")
        require(len(row["scores"]) == 20 and all(math.isfinite(v) and 0 <= v <= 1 for v in row["scores"])
                and math.isclose(sum(row["scores"]), 1, abs_tol=1e-4), "Invalid choice scores")
    selected_name = "baseline" if result["best_step"] == 0 else f"step-{result['best_step']}"
    remote = rows(args.run / f"{selected_name}-predictions.jsonl")
    local = {(r["id"], r["profile"]): r for r in after}
    require(len(remote) == 150 and len({(r["id"], r["profile"]) for r in remote}) == 150, "Incomplete GPU validation")
    for row in remote:
        counterpart = local[row["id"], row["profile"] + "/synthetic"]
        require(all(row[key] == counterpart[key] for key in
                    ("id", "group_id", "target", "candidate_ids", "retrieval_covered")), "Different GPU/CPU inputs")
    differences = sum(r["predicted"] != local[r["id"], r["profile"] + "/synthetic"]["predicted"] for r in remote)
    intervals = {}
    for profile in baseline["metrics"]:
        pairs = [(b, a) for b, a in zip(before, after, strict=True) if b["profile"] == profile]
        # paired_interval's first argument is the minuend; report trained minus baseline.
        intervals[profile] = paired_interval([(a, b) for b, a in pairs])
    audit = {"audit": "Exposure, stopping, downloaded hashes and 300 paired local predictions checked",
             "steps": result["steps"], "best_step": result["best_step"], "stop_reason": result["stop_reason"],
             "baseline": baseline["metrics"], "trained": trained["metrics"], "paired": intervals,
             "gpu_cpu_prediction_differences": differences, "latency": "Not qualified",
             "training_seeds": 1, "test_access": False}
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--local", type=Path, default=ROOT / "artifacts/kaggle/local-validation")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kaggle/laya-comparison.json")
    main(parser.parse_args())
