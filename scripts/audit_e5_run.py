"""Rebuild E5 training selection and paired full-catalog quality/latency evidence."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from evaluate_e5_locally import ROOT, add_scenarios
from train_laya_kaggle import ValidationStopper, selection_loss
from train_modernbert_kaggle import sha
from train_nanojev_kaggle import digest, plan_order, rows


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_predictions(predictions, data, catalogs):
    import numpy as np

    assert len(predictions) == len(data) and len({row["id"] for row in predictions}) == len(data)
    by_id = {row["id"]: row for row in data}
    for row in predictions:
        original = by_id[row["id"]]
        for key in ("id", "profile", "group_id", "target", "catalog_key"):
            assert row[key] == original[key], key
        ids = catalogs[row["catalog_key"]]["candidate_ids"]
        assert row["candidate_ids"] == ids and row["retrieval_covered"]
        assert len(ids) == len(row["similarities"]) == len(row["scores"]) == 177
        scores = np.array(row["similarities"])
        assert np.isfinite(scores).all() and np.isfinite(row["scores"]).all()
        order = sorted(range(len(ids)), key=lambda index: (-scores[index], ids[index]))
        assert row["predicted"] == ids[order[0]]
        assert row["correct"] == (row["predicted"] == row["target"])
        assert row["top20"] == [ids[index] for index in order[:20]]
        assert row["target_rank"] == order.index(ids.index(row["target"])) + 1
        probabilities = np.exp((scores - scores.max()) / .01)
        probabilities /= probabilities.sum()
        assert np.max(np.abs(probabilities - row["scores"])) < 1e-5
    metrics = {}
    for profile in sorted({row["profile"] for row in predictions}):
        local = [row for row in predictions if row["profile"] == profile]
        metrics[profile] = {"rows": len(local), "correct": sum(row["correct"] for row in local),
                            "accuracy": sum(row["correct"] for row in local) / len(local),
                            "recall20": sum(row["target_rank"] <= 20 for row in local) / len(local),
                            "mrr": sum(1 / row["target_rank"] for row in local) / len(local)}
    return metrics


def audit(args):
    staged = ROOT / "artifacts/kaggle/e5-data"
    manifest = load(staged / "e5-input-manifest.json")
    run, plan = load(args.run / "result.json"), load(args.run / "plan.json")
    assert run["plan"] == plan and plan["input_manifest"] == manifest
    assert plan["device"] == "cuda" and not plan["smoke"] and plan["head_added"] is False
    assert plan["precision"].startswith("FP32") and plan["temperature"] == .01
    assert plan["maximum_epochs"] == 1 and plan["maximum_steps"] == 1416
    assert plan["effective_batch"] == 16 and plan["micro_batch"] == 2 and plan["learning_rate"] == 1e-5
    assert plan["validation_interval"] == 142 and plan["patience"] == 3 and plan["min_delta"] == .005
    for name, expected in manifest["files"].items():
        assert sha(staged / name) == expected, name
    data = rows(staged / "prepared-train.jsonl")
    validation = rows(staged / "full-validation.jsonl")
    catalogs = load(staged / "validation-catalogs.json")
    assert not {row["group_id"] for row in data} & {row["group_id"] for row in validation}
    order, prefix = plan_order(data, "clean_first", plan["seed"])
    assert plan["schedule"] == "clean_first" and prefix == plan["clean_prefix_rows"] == 2272
    assert len(data) == plan["rows"] == 22656 and plan["source_counts"] == Counter(row["source"] for row in data)
    assert digest([data[index]["id"] for index in order]) == plan["ordered_ids_sha256"]
    assert digest(sorted(row["id"] for row in data)) == plan["sample_multiset_sha256"]
    ledger = rows(args.run / "training.jsonl")
    assert len(ledger) == run["steps"] <= 1416
    for step, entry in enumerate(ledger, 1):
        assert entry["step"] == step
        assert entry["sources"] == Counter(data[index]["source"] for index in order[(step - 1) * 16:step * 16])
        assert math.isfinite(entry["loss"]) and math.isfinite(entry["grad_norm"])
        factor = step / 85 if step < 85 else (1416 - step) / (1416 - 85)
        assert len(entry["lr"]) == 1 and math.isclose(entry["lr"][0], 1e-5 * factor, abs_tol=1e-12)
    checks = rows(args.run / "validation.jsonl")
    expected = [0] + list(range(142, run["steps"] + 1, 142))
    if run["steps"] not in expected:
        expected.append(run["steps"])
    assert [row["step"] for row in checks] == expected
    stopper = ValidationStopper(checks[0]["selection_loss"], 3, .005)
    stop = False
    for check in checks:
        label = "baseline" if not check["step"] else f"step-{check['step']}"
        predictions = rows(args.run / f"{label}-predictions.jsonl")
        assert validate_predictions(predictions, validation, catalogs) == check["metrics"]
        loss, losses = selection_loss(predictions)
        assert math.isclose(loss, check["selection_loss"], abs_tol=1e-10) and losses == check["profile_losses"]
        if check["step"]:
            improved, stop = stopper.update(loss, check["step"])
            assert check["improved"] == improved and check["wait"] == stopper.wait
            assert not stop or check == checks[-1]
    assert run["best_step"] == stopper.best_step and run["best"]["selection_loss"] == stopper.best_loss
    assert run["last"]["selection_loss"] == checks[-1]["selection_loss"]
    assert (run["stop_reason"] == "validation_patience" and stop) or (
        run["stop_reason"] == "maximum_one_pass" and run["steps"] == 1416 and not stop)
    if not stopper.best_step:
        assert run["checkpoint_sha256"] == manifest["original_weight_sha256"]
    for folder, key in (("checkpoint", "checkpoint_sha256"), ("last-checkpoint", "last_checkpoint_sha256")):
        weights = args.run / folder / "model.safetensors"
        assert sha(weights) == run[key]
        with weights.open("rb") as stream:
            header = json.loads(stream.read(int.from_bytes(stream.read(8), "little")))
        assert all(value["dtype"] == "F32" for name, value in header.items() if name != "__metadata__")
    report = {"gpu_audit": "passed", "completed_updates": run["steps"], "selected_update": run["best_step"],
              "completed_exposure": dict(Counter(data[index]["source"] for index in order[:run["steps"] * 16])),
              "selected_exposure": dict(Counter(data[index]["source"] for index in order[:run["best_step"] * 16])),
              "validation_checks": len(checks), "stop_reason": run["stop_reason"],
              "checkpoint_sha256": run["checkpoint_sha256"]}
    if args.local:
        full = ROOT / "artifacts/kaggle/e5-evaluation-v1/input"
        full_data, full_catalogs = rows(full / "full-evaluation.jsonl"), load(full / "catalogs.json")
        predictions = rows(args.local / "validation-predictions.jsonl")
        local = load(args.local / "result.json")
        assert len(predictions) == 800 and local["checkpoint_sha256"] == run["checkpoint_sha256"]
        assert local["input_sha256"] == sha(full / "full-evaluation.jsonl")
        assert validate_predictions(predictions, full_data, full_catalogs) == local["metrics"]
        rebuilt = add_scenarios({"metrics": local["metrics"]})
        assert len(rebuilt["scenario_counts"]) == 10
        for key in ("scenario_counts", "equal_scenario_accuracy", "equal_row_accuracy", "noisy_stt_equal_catalog_accuracy"):
            assert rebuilt[key] == local[key]
        assert math.isclose(selection_loss(predictions)[0], local["selection_loss"], abs_tol=1e-10)
        label = "baseline" if not run["best_step"] else f"step-{run['best_step']}"
        gpu = rows(args.run / f"{label}-predictions.jsonl")
        cpu = {row["id"]: row for row in predictions}
        differences = []
        for row in gpu:
            actual = cpu[row["id"]]
            for key in ("candidate_ids", "target", "predicted", "top20", "target_rank"):
                assert actual[key] == row[key], key
            differences.append(max(abs(a - b) for a, b in zip(actual["scores"], row["scores"], strict=True)))
        assert len(differences) == 150 and max(differences) < 1e-4
        report.update(local800_audit="passed", gpu_cpu150={"rows": 150, "all_top1_match": True,
                      "all_top20_match": True, "maximum_probability_difference": max(differences)},
                      quality=rebuilt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--local", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())
