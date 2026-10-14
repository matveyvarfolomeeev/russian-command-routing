"""Audit actual reranker exposure, stopping, weights and local predictions."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from audit_kev_run import check_predictions, metrics
from compare_nanojev_runs import load
from train_laya_kaggle import ValidationStopper, selection_loss
from train_modernbert_kaggle import sha
from train_nanojev_kaggle import digest, plan_order, rows

ROOT = Path(__file__).resolve().parents[1]


def audit(args):
    nano = args.family == "nano"
    staged = ROOT/"artifacts/kaggle"/("nano-continuation-data" if nano else f"{args.family}-data")
    manifest = load(staged/("nano-continuation-input-manifest.json" if nano
                           else f"{args.family}-input-manifest.json"))
    result, plan = load(args.run/"result.json"), load(args.run/"plan.json")
    assert result["plan"] == plan and plan["input_manifest"] == manifest
    assert plan["device"] == "cuda" and not plan["smoke"]
    assert plan["maximum_epochs"] == 1 and plan["maximum_steps"] == 1416
    assert plan["effective_batch"] == 16 and plan["micro_batch"] == (4 if nano else 2)
    assert plan["learning_rate"] == (1e-5 if nano else 2e-5) and plan["precision"].startswith("FP32")
    if nano:
        parent = load(ROOT/"artifacts/kaggle/outputs/clean-first-v1/results/result.json")
        assert manifest["initial_weight_sha256"] == parent["checkpoint_sha256"] == plan["initial_weight_sha256"]
        assert manifest["parent_result_sha256"] == sha(ROOT/"artifacts/kaggle/outputs/clean-first-v1/results/result.json")
        assert plan["initial_checkpoint_eligible"] and not manifest["optimizer_state_present"]
        assert "fresh AdamW" in plan["optimizer"]
    for name, expected in manifest["files"].items():
        assert sha(staged/name) == expected, name
    data, validation = rows(staged/"prepared-train.jsonl"), rows(staged/"prepared-validation.jsonl")
    schedule = "mixed" if nano else "clean_first"
    order, prefix = plan_order(data,schedule,plan["seed"])
    assert plan["schedule"] == schedule
    assert plan["clean_prefix_rows"] == (0 if nano else prefix)
    assert prefix == 2272
    assert len(data) == plan["rows"] == 22656
    assert plan["ordered_ids_sha256"] == digest([data[i]["id"] for i in order])
    assert plan["sample_multiset_sha256"] == digest(sorted(r["id"] for r in data))
    ledger = rows(args.run/"training.jsonl")
    assert len(ledger) == result["steps"] <= 1416
    for step, entry in enumerate(ledger,1):
        assert entry["step"] == step
        assert entry["sources"] == Counter(data[i]["source"] for i in order[(step-1)*16:step*16])
        assert math.isfinite(entry["loss"]) and math.isfinite(entry["grad_norm"])
        factor = step/85 if step<85 else (1416-step)/(1416-85)
        assert len(entry["lr"]) == 1 and math.isclose(entry["lr"][0],plan["learning_rate"]*factor,abs_tol=1e-12)
    checks = rows(args.run/"validation.jsonl")
    expected_steps = [0]+list(range(142,result["steps"]+1,142))
    if result["steps"] not in expected_steps:
        expected_steps.append(result["steps"])
    assert [r["step"] for r in checks] == expected_steps
    stopper = ValidationStopper(checks[0]["selection_loss"],plan["patience"],plan["min_delta"])
    stop = False
    for check in checks:
        name = "baseline" if check["step"] == 0 else f"step-{check['step']}"
        predictions = rows(args.run/f"{name}-predictions.jsonl")
        check_predictions(predictions,validation)
        assert metrics(predictions) == check["metrics"]
        loss, losses = selection_loss(predictions)
        assert math.isclose(loss,check["selection_loss"],abs_tol=1e-10)
        assert losses == check["profile_losses"]
        if check["step"]:
            improved, stop = stopper.update(loss,check["step"])
            assert check["improved"] == improved and check["wait"] == stopper.wait
            assert not stop or check == checks[-1], "Continued after stopping rule"
    assert result["best_step"] == stopper.best_step
    if nano:
        parent_predictions = rows(ROOT/"artifacts/kaggle/outputs/clean-first-v1/results/trained-predictions.jsonl")
        baseline_predictions = rows(args.run/"baseline-predictions.jsonl")
        parent_by_id = {r["id"]:r for r in parent_predictions}
        assert checks[0]["metrics"] == parent["trained"]
        for row in baseline_predictions:
            original = parent_by_id[row["id"]]
            assert row["candidate_ids"] == original["candidate_ids"] and row["predicted"] == original["predicted"]
            assert max(abs(a-b) for a,b in zip(row["scores"],original["scores"],strict=True)) < 1e-4
    best = next(r for r in checks if r["step"]==stopper.best_step)
    assert result["best"]["metrics"] == best["metrics"]
    assert math.isclose(result["best"]["selection_loss"],stopper.best_loss,abs_tol=1e-10)
    assert result["last"]["selection_loss"] == checks[-1]["selection_loss"]
    assert (result["stop_reason"] == "validation_patience" and stop) or (
        result["stop_reason"] == "maximum_one_pass" and result["steps"]==1416 and not stop)
    if nano and stopper.best_step == 0:
        assert result["checkpoint_sha256"] == manifest["initial_weight_sha256"]
    for folder,key in (("checkpoint","checkpoint_sha256"),("last-checkpoint","last_checkpoint_sha256")):
        weights = args.run/folder/"model.safetensors"
        assert sha(weights) == result[key]
        with weights.open("rb") as stream:
            header_length = int.from_bytes(stream.read(8),"little")
            header = json.loads(stream.read(header_length))
        assert all(v["dtype"]=="F32" for k,v in header.items() if k!="__metadata__")
    report = {"model_family":args.family, "gpu_audit":"passed", "completed_updates":result["steps"],
              "selected_update":stopper.best_step, "completed_rows":min(result["steps"]*16,len(data)),
              "selected_rows":min(stopper.best_step*16,len(data)), "validation_checks":len(checks),
              "best_nll":stopper.best_loss, "stop_reason":result["stop_reason"],
              "checkpoint_sha256":result["checkpoint_sha256"]}
    if args.local:
        data = rows(ROOT/"artifacts/kaggle/modernbert-evaluation-v1/prepared-all-known.jsonl")
        reference = {r["id"]:r for r in rows(args.run/("baseline-predictions.jsonl" if stopper.best_step==0
                      else f"step-{stopper.best_step}-predictions.jsonl"))}
        predictions = rows(args.local/"validation-predictions.jsonl")
        check_predictions(predictions,data)
        local = load(args.local/"result.json")
        assert len(predictions) == 800 and local["checkpoint_sha256"] == result["checkpoint_sha256"]
        assert local["input_sha256"] == sha(ROOT/"artifacts/kaggle/modernbert-evaluation-v1/prepared-all-known.jsonl")
        assert metrics(predictions) == local["metrics"]
        if "selection_loss" in local:
            assert abs(selection_loss(predictions)[0]-local["selection_loss"])<1e-10
        cpu = {r["id"]:r for r in predictions}
        differences = []
        for identity,gpu in reference.items():
            assert cpu[identity]["candidate_ids"] == gpu["candidate_ids"]
            assert cpu[identity]["target"] == gpu["target"] and cpu[identity]["predicted"] == gpu["predicted"]
            difference = max(abs(a-b) for a,b in zip(cpu[identity]["scores"],gpu["scores"],strict=True))
            assert difference<1e-4, difference
            differences.append(difference)
        assert len(differences)==150
        if nano:
            counts = {}
            for row in predictions:
                profile = row["profile"]
                if profile.startswith("independent/"):
                    profile = "independent/"+("asr" if "asr" in profile else "clean")
                value = counts.setdefault(profile, {"rows":0,"correct":0})
                value["rows"] += 1
                value["correct"] += int(row["correct"])
            for value in counts.values():
                value["accuracy"] = value["correct"]/value["rows"]
        else:
            counts = local["scenario_counts"]
        assert len(counts)==10 and sum(r["rows"] for r in counts.values())==800
        mean = sum(r["correct"]/r["rows"] for r in counts.values())/10
        if "equal_scenario_accuracy" in local:
            assert abs(mean-local["equal_scenario_accuracy"])<1e-12
        report.update(local800_audit="passed",gpu_cpu_parity_rows=150,
                      maximum_probability_difference=max(differences),
                      scenario_counts=counts, equal_scenario_accuracy=mean,
                      equal_row_accuracy=sum(r["correct"] for r in counts.values())/800)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family",choices=("modernbert","nano","rubert"),default="modernbert")
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--local",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    audit(parser.parse_args())
