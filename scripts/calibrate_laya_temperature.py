"""Fit one choice:20 temperature on validation; evaluate it on separate query groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/benchmarks/e5-laya-pipeline-v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def log_probabilities(logits, temperature):
    values = [value / temperature for value in logits]
    maximum = max(values)
    normalizer = maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))
    return [value - normalizer for value in values]


def nll(records, temperature):
    covered = [row for row in records if row["target_in_top20"]]
    return math.fsum(-log_probabilities(row["logits"], temperature)[row["target_index"]]
                     for row in covered) / len(covered)


def fit(records, lower=0.05, upper=20.0):
    # NLL is convex in inverse temperature; bisection finds its minimum without a grid bias.
    def derivative(beta):
        total = 0.0
        for row in records:
            logits = row["logits"]
            logp = log_probabilities(logits, 1.0 / beta)
            total += math.fsum(math.exp(p) * z for p, z in zip(logp, logits))
            total -= logits[row["target_index"]]
        return total / len(records)

    left, right = 1.0 / upper, 1.0 / lower
    if derivative(left) >= 0:
        return upper
    if derivative(right) <= 0:
        return lower
    for _ in range(80):
        middle = (left + right) / 2
        if derivative(middle) < 0:
            left = middle
        else:
            right = middle
    return 1.0 / ((left + right) / 2)


def metrics(records, temperature):
    bins = [[] for _ in range(10)]
    confidences, briers, multiclass = [], [], []
    for row in records:
        logp = log_probabilities(row["logits"], temperature)
        probabilities = [math.exp(value) for value in logp]
        confidence = max(probabilities)
        correct = int(row["correct"])
        confidences.append(confidence)
        briers.append((confidence - correct) ** 2)
        bins[min(int(confidence * 10), 9)].append((confidence, correct))
        if row["target_in_top20"]:
            multiclass.append(math.fsum((p - int(i == row["target_index"])) ** 2
                                       for i, p in enumerate(probabilities)))
    reliability = []
    ece = 0.0
    for index, values in enumerate(bins):
        if not values:
            continue
        accuracy = sum(y for _, y in values) / len(values)
        mean = math.fsum(p for p, _ in values) / len(values)
        ece += len(values) / len(records) * abs(mean - accuracy)
        reliability.append({"bin": index, "rows": len(values), "accuracy": accuracy,
                            "mean_confidence": mean})
    return {"rows": len(records), "groups": len({r["group_id"] for r in records}),
            "correct": sum(r["correct"] for r in records),
            "accuracy": sum(r["correct"] for r in records) / len(records),
            "target_in_top20": sum(r["target_in_top20"] for r in records),
            "conditional_nll": nll(records, temperature),
            "conditional_multiclass_brier": math.fsum(multiclass) / len(multiclass),
            "top_label_brier": math.fsum(briers) / len(briers),
            "mean_confidence": math.fsum(confidences) / len(confidences),
            "ece_10_equal_width": ece, "reliability_bins": reliability}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source_protocol = json.loads((SOURCE / "protocol.json").read_text("utf-8"))
    source_comparison = json.loads((SOURCE / "comparison.json").read_text("utf-8"))
    source = SOURCE / "original-predictions.jsonl"
    assert sha(source) == source_comparison["prediction_sha256"][source.name]
    records = [json.loads(line) for line in source.read_text("utf-8").splitlines()]
    assert len(records) == 974
    for row in records:
        probabilities = row["scores"]
        assert len(probabilities) == len(row["candidate_ids"]) == 20
        assert all(math.isfinite(p) and p > 0 for p in probabilities), row["id"]
        assert abs(math.fsum(probabilities) - 1) < 1e-6, row["id"]
        # With no underflow, log(p) recovers the original logits up to an irrelevant constant.
        row["logits"] = [math.log(p) for p in probabilities]
        row["target_index"] = (row["candidate_ids"].index(row["target"])
                               if row["target_in_top20"] else None)
        assert row["candidate_ids"][max(range(20), key=row["logits"].__getitem__)] == row["predicted"]
    calibration = [r for r in records if r["profile"].endswith("/validation")]
    test = [r for r in records if r["profile"].endswith("/test")]
    assert len(calibration) == 100 and len(test) == 300
    assert all(r["target_in_top20"] for r in calibration)
    assert not ({r["group_id"] for r in calibration} & {r["group_id"] for r in test})
    assert {r["id"] for r in calibration}.isdisjoint(r["id"] for r in test)
    protocol = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": source_protocol["laya_checkpoint_sha256"],
        "retriever": "original_e5", "question_type": "choice", "candidate_count": 20,
        "input": str(source.relative_to(ROOT)), "input_sha256": sha(source),
        "source_protocol_sha256": sha(SOURCE / "protocol.json"),
        "script_sha256": sha(__file__), "optimizer_updates": 0,
        "objective": "mean conditional multinomial NLL on calibration only",
        "temperature_bounds": [0.05, 20.0], "comparison_temperatures": [0.1, 0.5, 1, 2, 5],
        "method": "80 bisection iterations in inverse temperature",
        "logits_source": "log of frozen FP32 softmax probabilities; all positive, no flooring",
        "calibration": {"rows": 100, "groups": 10,
                        "profiles": dict(Counter(r["profile"] for r in calibration)),
                        "ids": [r["id"] for r in calibration]},
        "test": {"rows": 300, "groups": 30,
                 "profiles": dict(Counter(r["profile"] for r in test)),
                 "ids": [r["id"] for r in test]},
        "roles": {"calibration": "known independent validation; not used to train these weights",
                  "test": "separate groups; known descriptive test, not a new blind test",
                  "old_stt": "100 known difficult STT examples; evaluation only, no fitting"},
        "metric_policy": "NLL and multiclass Brier exclude absent targets; top-label Brier/ECE include every query",
        "limits": "One scalar calibrated for choice:20 and this checkpoint/retriever; smaller shortlists use T=1",
    }
    # Persist the split, search bounds and objective before fitting or evaluating a temperature.
    save(output / "protocol.json", protocol)
    temperature = fit(calibration)
    scopes = {"calibration100": calibration, "test300": test,
              "old_stt100": [r for r in records if r["scope"] == "known800"
                             and r["profile"].startswith("launcher-asr-v1/")]}
    values = [0.1, 0.5, 1.0, 2.0, 5.0, temperature]
    report = {"schema_version": 1, "temperature": temperature,
              "checkpoint_sha256": protocol["checkpoint_sha256"],
              "retriever": "original_e5", "question_type": "choice", "candidate_count": 20,
              "protocol_sha256": sha(output / "protocol.json"),
              "at_search_boundary": temperature in protocol["temperature_bounds"],
              "probability_recovery_minimum": min(p for r in records for p in r["scores"]),
              "scopes": {name: {"original": metrics(rows, 1), "fitted": metrics(rows, temperature),
                                "comparisons": [{"temperature": t, **metrics(rows, t)} for t in values]}
                         for name, rows in scopes.items()},
              "argmax_changes": 0, "optimizer_updates": 0,
              "supports_automatic_launch": False}
    assert report["scopes"]["calibration100"]["fitted"]["conditional_nll"] <= nll(calibration, 1)
    save(output / "temperature.json", report)
    for name, rows in scopes.items():
        with (output / f"{name}-predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                logp = log_probabilities(row["logits"], temperature)
                stable_order = sorted(range(20), key=lambda i: -row["logits"][i])
                assert [row["candidate_ids"][i] for i in stable_order] == row["ranked_ids"]
                stream.write(json.dumps({"id": row["id"], "profile": row["profile"],
                                         "group_id": row["group_id"], "target": row["target"],
                                         "candidate_ids": row["candidate_ids"],
                                         "predicted": row["predicted"], "correct": row["correct"],
                                         "target_in_top20": row["target_in_top20"],
                                         "scores": [math.exp(value) for value in logp],
                                         "temperature": temperature}, ensure_ascii=False) + "\n")
    print(json.dumps({"temperature": temperature, "scopes": {
        name: {key: {k: v for k, v in report["scopes"][name][key].items()
                     if k != "reliability_bins"} for key in ("original", "fitted")}
        for name in scopes}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/benchmarks/laya-temperature-v1")
    run(parser.parse_args().output)
