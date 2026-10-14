"""Audit the matched GPU runs and compare their paired local validation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from train_nanojev_kaggle import digest, plan_order, rows

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(run, data, schedule):
    result = load(run / "result.json")
    plan = result["plan"]
    require(plan == load(run / "plan.json"), "Result/plan disagreement")
    order, prefix = plan_order(data, schedule, plan["seed"])
    require(plan["schedule"] == schedule and not plan["smoke"], "Wrong run")
    require(plan["epochs"] == 1 and plan["rows"] == len(data) == 22656, "Wrong exposure")
    require(plan["device"] == "cuda" and plan["precision"] == "fp32", "Wrong runtime")
    require(plan["micro_batch"] * plan["gradient_accumulation"] == 16, "Wrong effective batch")
    require(plan["source_counts"] == {"clean": 5664, "stt": 16992}, "Wrong source counts")
    require(plan["sample_multiset_sha256"] == digest(sorted(r["id"] for r in data)), "Samples changed")
    require(plan["ordered_ids_sha256"] == digest([data[i]["id"] for i in order]), "Order changed")
    require(plan["clean_prefix_rows"] == (prefix if schedule == "clean_first" else 0), "Wrong clean stage")
    ledger = rows(run / "training.jsonl")
    require(len(ledger) == result["steps"] == math.ceil(len(data) / 16) == 1416, "Incomplete training")
    for step, entry in enumerate(ledger, 1):
        expected = Counter(data[i]["source"] for i in order[(step - 1) * 16:step * 16])
        require(entry["step"] == step and entry["epoch"] == 1, "Missing/duplicate update")
        require(entry["sources"] == expected, "Exposure ledger differs from plan")
        require(math.isfinite(entry["loss"]) and math.isfinite(entry["lr"]), "Non-finite ledger")
    sha = hashlib.sha256((run / "checkpoint/model.safetensors").read_bytes()).hexdigest()
    require(sha == result["checkpoint_sha256"] != plan["initial_weight_sha256"], "Checkpoint mismatch")
    return result


def paired_interval(pairs):
    # Resample whole scenario groups together; the 50 related rows are not independent.
    grouped = defaultdict(list)
    for mixed, clean in pairs:
        grouped[mixed["group_id"]].append(int(mixed["correct"]) - int(clean["correct"]))
    totals = [(sum(values), len(values)) for values in grouped.values()]
    rng = random.Random(20261005)
    draws = []
    for _ in range(10000):
        sampled = rng.choices(totals, k=len(totals))
        draws.append(sum(v for v, _ in sampled) / sum(n for _, n in sampled))
    draws.sort()
    return {"groups": len(totals), "delta_accuracy": sum(v for v, _ in totals) / sum(n for _, n in totals),
            "cluster_bootstrap_95_percent": [draws[249], draws[9749]], "resamples": 10000,
            "scope": "Validation group sampling; does not measure training-seed variability"}


def main(args):
    training_path = ROOT / "artifacts/kaggle/training-data/prepared-train.jsonl"
    data = rows(training_path)
    runs = {name: audit(directory, data, name) for name, directory in
            (("mixed", args.mixed), ("clean_first", args.clean_first))}
    excluded = {"schedule", "clean_prefix_rows", "ordered_ids_sha256"}
    controls = [{k: v for k, v in r["plan"].items() if k not in excluded} for r in runs.values()]
    require(controls[0] == controls[1], "Run settings differ beyond presentation order")
    require(hashlib.sha256(training_path.read_bytes()).hexdigest() == controls[0]["candidate_inputs_sha256"],
            "Frozen training file changed locally")
    require(runs["mixed"]["baseline"] == runs["clean_first"]["baseline"], "Different baseline metrics")
    baseline_a = rows(args.mixed / "baseline-predictions.jsonl")
    baseline_b = rows(args.clean_first / "baseline-predictions.jsonl")
    require([(r["id"], r["predicted"]) for r in baseline_a] == [(r["id"], r["predicted"]) for r in baseline_b],
            "Different baseline choices")
    evaluations = {name: load(args.local / name / "result.json") for name in ("baseline-v1", "mixed-v1", "clean-first-v1")}
    require(len({r["input_sha256"] for r in evaluations.values()}) == 1, "Local evaluation inputs changed")
    reference = load(args.local / "e5-reference-v1/result.json")
    require(reference["input_sha256"] == evaluations["baseline-v1"]["input_sha256"] and reference["shortlist_parity"],
            "E5 reference did not use the frozen shortlist inputs")
    for name, local_name in (("mixed", "mixed-v1"), ("clean_first", "clean-first-v1")):
        require(evaluations[local_name]["checkpoint_sha256"] == runs[name]["checkpoint_sha256"], "Wrong evaluated weights")
    require(evaluations["baseline-v1"]["checkpoint_sha256"] == runs["mixed"]["plan"]["initial_weight_sha256"], "Wrong baseline weights")
    predictions = {name: rows(args.local / name / "validation-predictions.jsonl") for name in evaluations}
    for values in predictions.values():
        require(len(values) == 300, "Incomplete local validation")
        for row in values:
            require(row["correct"] == (row["predicted"] == row["target"]), "Prediction label mismatch")
            require(len(row["scores"]) == 20 and all(math.isfinite(s) for s in row["scores"])
                    and math.isclose(sum(row["scores"]), 1.0, abs_tol=1e-4), "Invalid choice scores")
    def identity(r):
        return (r["id"], r["profile"], r["group_id"], r["target"], r["candidate_ids"], r["retrieval_covered"])

    require(all([identity(r) for r in p] == [identity(r) for r in predictions["baseline-v1"]]
                for p in predictions.values()), "Local predictions are not paired")
    paired = {}
    for profile in sorted(evaluations["baseline-v1"]["metrics"]):
        mixed = [r for r in predictions["mixed-v1"] if r["profile"] == profile]
        clean = [r for r in predictions["clean-first-v1"] if r["profile"] == profile]
        pairs = list(zip(mixed, clean, strict=True))
        paired[profile] = {"mixed_only_correct": sum(a["correct"] and not b["correct"] for a, b in pairs),
                           "clean_first_only_correct": sum(b["correct"] and not a["correct"] for a, b in pairs),
                           **paired_interval(pairs)}
    gpu_cpu_differences = {}
    for name, directory, local_name in (("mixed", args.mixed, "mixed-v1"),
                                        ("clean_first", args.clean_first, "clean-first-v1")):
        gpu = rows(directory / "trained-predictions.jsonl")
        cpu = {(r["id"], r["profile"]): r for r in predictions[local_name]}
        gpu_cpu_differences[name] = sum(r["predicted"] != cpu[r["id"], r["profile"] + "/synthetic"]["predicted"]
                                        for r in gpu)
    result = {"audits": "Both complete ledgers, frozen controls and downloaded/evaluated checkpoint hashes passed",
              "gpu": {name: {k: r[k] for k in ("steps", "elapsed_seconds", "checkpoint_sha256", "trained")}
                      for name, r in runs.items()},
              "local": {name: r["metrics"] for name, r in evaluations.items()}, "paired": paired,
              "e5_reference": reference["metrics"],
              "gpu_cpu_prediction_differences": gpu_cpu_differences,
              "latency": "Not qualified", "training_seeds": 1, "test_access": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", type=Path, required=True, help="Downloaded GPU results directory")
    parser.add_argument("--clean-first", type=Path, required=True)
    parser.add_argument("--local", type=Path, default=ROOT / "artifacts/kaggle/local-validation")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kaggle/comparison.json")
    main(parser.parse_args())
