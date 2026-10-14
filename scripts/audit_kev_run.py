"""Audit actual Kev exposure, stopping, downloaded checkpoints and CPU parity."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from compare_nanojev_runs import load, paired_interval, require
from train_kev_kaggle import checkpoint_hashes, sha
from train_laya_kaggle import ValidationStopper, selection_loss
from train_nanojev_kaggle import digest, plan_order, rows

ROOT = Path(__file__).resolve().parents[1]


def check_predictions(predictions, data):
    require(len(predictions) == len(data), "Incomplete predictions")
    actual = {row["id"]: row for row in predictions}
    require(len(actual) == len(predictions), "Duplicate prediction ID")
    for expected in data:
        row = actual[expected["id"]]
        require(all(row[key] == expected[key] for key in
                    ("profile", "group_id", "target", "candidate_ids", "retrieval_covered")), "Changed inputs")
        scores = row["scores"]
        require(len(scores) == 20 and all(math.isfinite(v) and 0 <= v <= 1 for v in scores)
                and math.isclose(sum(scores), 1, abs_tol=1e-4), "Invalid choice probabilities")
        require(row["predicted"] == row["candidate_ids"][max(range(20), key=scores.__getitem__)]
                and row["correct"] == (row["predicted"] == row["target"]), "Incorrect prediction mapping")
    return actual


def metrics(predictions):
    result = {}
    for profile in {row["profile"] for row in predictions}:
        subset = [row for row in predictions if row["profile"] == profile]
        covered, correct = sum(r["retrieval_covered"] for r in subset), sum(r["correct"] for r in subset)
        result[profile] = {"rows": len(subset), "correct": correct, "accuracy": correct / len(subset),
                           "candidate_recall": covered / len(subset),
                           "conditional_accuracy": correct / covered if covered else None}
    return result


def main(args):
    staged = ROOT / "artifacts/kaggle/kev-data"
    result, plan = load(args.run / "result.json"), load(args.run / "plan.json")
    require(result["plan"] == plan and not plan["smoke"] and plan["device"] == "cuda", "Wrong run")
    manifest = load(staged / "kev-input-manifest.json")
    require(plan["input_manifest"] == manifest, "Manifest changed")
    for name, expected in manifest["files"].items():
        require(sha(staged / name) == expected, "Frozen input/trainer changed")
    data, validation = rows(staged / "prepared-train.jsonl"), rows(staged / "prepared-validation.jsonl")
    order, prefix = plan_order(data, "clean_first", plan["seed"])
    require(len(data) == plan["rows"] == 22656 and prefix == plan["clean_prefix_rows"] == 2272, "Wrong budget")
    require(plan["maximum_epochs"] == 1 and plan["maximum_steps"] == 1416, "Wrong exposure limit")
    require(plan["micro_batch"] == 1 and plan["gradient_accumulation"] == 16
            and plan["lora_lr"] == plan["head_lr"] == 2e-5 and not plan["adapter_merge"]
            and plan["precision"] == "FP32 training, frozen base, LoRA/head, optimizer and validation",
            "Different learning protocol")
    require(plan["ordered_ids_sha256"] == digest([data[i]["id"] for i in order])
            and plan["sample_multiset_sha256"] == digest(sorted(r["id"] for r in data)), "Changed order/examples")
    ledger = rows(args.run / "training.jsonl")
    require(len(ledger) == result["steps"] <= 1416, "Wrong completed-update count")
    for step, entry in enumerate(ledger, 1):
        require(entry["step"] == step, "Missing/duplicate update")
        require(entry["sources"] == Counter(data[i]["source"] for i in order[(step - 1) * 16:step * 16]),
                "Actual source exposure differs from plan")
        require(all(math.isfinite(entry[k]) for k in ("loss", "grad_norm", "amp_scale"))
                and entry["amp_scale"] == 1.0 and all(math.isfinite(v) for v in entry["lr"]),
                "Non-finite training or different numerical protocol")
    checks = rows(args.run / "validation.jsonl")
    expected_steps = [0] + list(range(142, result["steps"] + 1, 142))
    if result["steps"] not in expected_steps:
        expected_steps.append(result["steps"])
    require([c["step"] for c in checks] == expected_steps, "Missing validation checks")
    stopper = ValidationStopper(checks[0]["selection_loss"], plan["patience"], plan["min_delta"])
    for check in checks:
        name = "baseline" if check["step"] == 0 else f"step-{check['step']}"
        predictions = rows(args.run / f"{name}-predictions.jsonl")
        check_predictions(predictions, validation)
        require(metrics(predictions) == check["metrics"], "GPU counts differ from predictions")
        loss, losses = selection_loss(predictions)
        require(math.isclose(loss, check["selection_loss"], abs_tol=1e-10)
                and losses == check["profile_losses"], "NLL differs from saved predictions")
        if check["step"]:
            improved, stop = stopper.update(loss, check["step"])
            require(improved == check["improved"] and stopper.wait == check["wait"], "Wrong stopping ledger")
            require(not stop or check == checks[-1], "Continued after patience stop")
    require(stopper.best_step == result["best_step"] and stopper.best_loss == result["best"]["selection_loss"],
            "Wrong selected checkpoint")
    by_step = {check["step"]: check for check in checks}
    for key, step in (("baseline", 0), ("best", result["best_step"]), ("last", result["steps"])):
        require(result[key] == {field: by_step[step][field] for field in
                                ("metrics", "selection_loss", "profile_losses")},
                "Summary metrics differ from the checked predictions")
    require((result["stop_reason"] == "validation_patience" and stopper.wait >= plan["patience"])
            or (result["stop_reason"] == "maximum_one_pass" and result["steps"] == 1416), "Wrong stop reason")
    for folder, key in (("checkpoint", "checkpoint_sha256"), ("last-checkpoint", "last_checkpoint_sha256")):
        require(checkpoint_hashes(args.run / folder) == result[key], "Downloaded checkpoint hash mismatch")
    encoding = load(args.run / "encoding.json")
    require(encoding["trainable_parameters"] == 11347456 and encoding["parameters"] == 763740480
            and encoding["initialization"]["tensors"] == 372, "Wrong native model/warm start")
    require(encoding["initialization"]["weights_sha256"] == manifest["checkpoint_files"]["adapter_model.safetensors"]
            and encoding["initialization"]["head_sha256"] == manifest["checkpoint_files"]["head.pt"], "Wrong initialization")
    for split in ("train", "validation"):
        require(encoding[split]["query_truncations"] == encoding[split]["metadata_truncations"] == 0
                and encoding[split]["choices_per_query"] == 20, "Incomplete native input")
    def exposure(step):
        return dict(Counter(data[i]["source"] for i in order[:step * 16]))

    report = {"gpu_audit": "passed", "steps": result["steps"], "best_step": result["best_step"],
              "stop_reason": result["stop_reason"], "best_exposure": exposure(result["best_step"]),
              "last_exposure": exposure(result["steps"]), "unused_rows": len(data) - result["steps"] * 16,
              "gpu_metrics": result["best"]["metrics"], "local_audit": "pending"}
    if not args.gpu_only:
        local_data = rows(ROOT / "artifacts/kaggle/local-validation.jsonl")
        baseline_result = load(args.local / "kev-baseline-v1/result.json")
        trained_result = load(args.local / "kev-clean-first-v1/result.json")
        require(trained_result["checkpoint_sha256"] == result["checkpoint_sha256"], "Different selected local weights")
        require(baseline_result["checkpoint_sha256"] == {
            name: manifest["checkpoint_files"][name]
            for name in ("adapter_model.safetensors", "adapter_config.json", "head.pt")
        }, "Different local baseline weights")
        for local in (baseline_result, trained_result):
            require(local["input_sha256"] == sha(ROOT / "artifacts/kaggle/local-validation.jsonl")
                    and local["precision"] == "fp32" and not local["adapter_merge"] and local["temperature"] == 1,
                    "Different CPU protocol")
            require(not local["preflight"] and local["device"] == "cpu"
                    and local["base_sha256"] == manifest["checkpoint_files"][
                        "base/model.safetensors-00001-of-00001.safetensors"], "Wrong local evaluation/base")
        before = check_predictions(rows(args.local / "kev-baseline-v1/validation-predictions.jsonl"), local_data)
        after = check_predictions(rows(args.local / "kev-clean-first-v1/validation-predictions.jsonl"), local_data)
        require(metrics(list(before.values())) == baseline_result["metrics"]
                and metrics(list(after.values())) == trained_result["metrics"], "CPU counts differ from predictions")
        selected_name = "baseline" if result["best_step"] == 0 else f"step-{result['best_step']}"
        gpu = check_predictions(rows(args.run / f"{selected_name}-predictions.jsonl"), validation)
        require(all(row["profile"] + "/synthetic" == after[key]["profile"]
                    and all(row[field] == after[key][field] for field in
                            ("group_id", "target", "candidate_ids", "retrieval_covered"))
                    for key, row in gpu.items()), "GPU/CPU predictions are not paired")
        parity = sum(row["predicted"] != after[key]["predicted"] for key, row in gpu.items())
        max_difference = max(abs(a - b) for key, row in gpu.items() for a, b in
                             zip(row["scores"], after[key]["scores"], strict=True))
        profiles = {}
        for profile in sorted({row["profile"] for row in local_data}):
            selected = [row for row in local_data if row["profile"] == profile]
            paired = [(after[row["id"]], before[row["id"]]) for row in selected]
            profiles[profile] = {"baseline": baseline_result["metrics"][profile],
                                 "trained": trained_result["metrics"][profile],
                                 "fixed_errors": sum(a["correct"] and not b["correct"] for a, b in paired),
                                 "lost_correct": sum(b["correct"] and not a["correct"] for a, b in paired),
                                 "paired": paired_interval(paired)}
        report.update(local_audit="passed", local_rows=len(local_data), profiles=profiles,
                      gpu_cpu_synthetic_top1_differences=parity,
                      gpu_cpu_max_probability_difference=max_difference)
    output = ROOT / ("artifacts/kaggle/kev-gpu-audit.json" if args.gpu_only else "artifacts/kaggle/kev-comparison.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/kaggle/outputs/kev-clean-first-v1/results")
    parser.add_argument("--local", type=Path, default=ROOT / "artifacts/kaggle/local-validation")
    parser.add_argument("--gpu-only", action="store_true")
    main(parser.parse_args())
