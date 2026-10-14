"""Recompute published routing metrics without models, API calls or private data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_prediction(row, expected, catalog_ids):
    require(row["target"] == expected["expected_object_id"], "Prediction target differs from label")
    require(row["group_id"] == expected["group_id"], "Prediction group differs from dataset")
    candidates = row["candidate_ids"]
    retrieved = row["retrieved_ids"]
    require(len(candidates) in (20, 177) and len(set(candidates)) == len(candidates),
            "Invalid or duplicate candidates")
    require(set(candidates) <= catalog_ids, "Unknown catalog object in candidates")
    require(len(retrieved) == 20 and len(set(retrieved)) == 20
            and set(retrieved) <= set(candidates), "Invalid retrieval shortlist")
    require(row["predicted"] in candidates, "Prediction is outside the candidate set")
    require(row["correct"] == (row["predicted"] == row["target"]), "Incorrect correctness flag")
    require(row["retrieval_covered"] == (row["target"] in retrieved), "Incorrect retrieval flag")
    if "scores" in row:
        scores = row["scores"]
        require(len(scores) == len(candidates) and all(math.isfinite(p) and p >= 0 for p in scores),
                "Invalid probabilities")
        require(abs(sum(scores) - 1) < 1e-5, "Probabilities do not sum to one")
        require(row["predicted"] == candidates[max(range(len(scores)), key=scores.__getitem__)],
                "Prediction differs from probability argmax")


def summarize(values):
    profiles = defaultdict(list)
    for row in values:
        profiles[row["profile"]].append(row)
    return {
        profile: {"rows": len(rows), "correct": sum(row["correct"] for row in rows),
                  "accuracy": sum(row["correct"] for row in rows) / len(rows),
                  "recall_at_20": sum(row["retrieval_covered"] for row in rows) / len(rows)}
        for profile, rows in sorted(profiles.items())
    }


def calibration_metrics(values, field):
    # The original report recovers logits from FP32 scores, then applies softmax even at T=1.
    normalized = [(row, [p / math.fsum(row[field]) for p in row[field]]) for row in values]
    covered = [(row, scores) for row, scores in normalized if row["target"] in row["candidate_ids"]]
    nll = math.fsum(-math.log(scores[row["candidate_ids"].index(row["target"])])
                    for row, scores in covered) / len(covered)
    bins = defaultdict(list)
    for row, scores in normalized:
        confidence = max(scores)
        bins[min(9, int(confidence * 10))].append((confidence, row["correct"]))
    ece = sum(abs(sum(p for p, _ in group) - sum(ok for _, ok in group))
              for group in bins.values()) / len(values)
    return nll, ece


def verify(root=ROOT):
    evidence = root / "results"
    manifest = load(evidence / "manifest.json")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "Manifest path escaped the repository")
        require(sha(path) == expected, f"SHA-256 mismatch: {name}")
    truth = {}
    data = root / "data/benchmarks/launcher-independent-v1"
    catalog_ids = {row["object_id"] for row in load(data / "clean/catalog.json")["objects"]}
    for representation in ("clean", "asr"):
        for split in ("validation", "test"):
            profile = f"independent/{representation}/{split}"
            for line in (data / representation / f"{split}.jsonl").read_text("utf-8").splitlines():
                row = json.loads(line)
                truth[(profile, row["id"])] = row
    require(len(truth) == 400, "Expected 200 commands with clean and ASR representations")
    grouped = defaultdict(list)
    with gzip.open(evidence / "independent-v1/predictions.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["profile"], row["id"]
            require(key in truth, "Unknown evaluation row")
            validate_prediction(row, truth[key], catalog_ids)
            grouped[row["model"]].append(row)
    report = load(evidence / "independent-v1/summary.json")
    require(set(grouped) == set(report["models"]), "Published model set differs")
    for model, values in grouped.items():
        keys = [(row["profile"], row["id"]) for row in values]
        require(len(keys) == len(set(keys)) and set(keys) == set(truth),
                f"Missing or duplicate evaluation rows: {model}")
        require(summarize(values) == report["models"][model]["metrics"],
                f"Recomputed metrics differ: {model}")

    calibration = load(evidence / "calibration/laya.json")
    with gzip.open(evidence / "calibration/predictions.jsonl.gz", "rt", encoding="utf-8") as stream:
        calibration_rows = [json.loads(line) for line in stream]
    scopes = defaultdict(list)
    for row in calibration_rows:
        key = row["profile"], row["id"]
        require(key in truth, "Unknown calibration row")
        validate_prediction({**row, "scores": row["original_scores"]}, truth[key], catalog_ids)
        logits = [math.log(p) / calibration["temperature"] for p in row["original_scores"]]
        exp = [math.exp(p - max(logits)) for p in logits]
        fitted = [p / sum(exp) for p in exp]
        require(max(abs(a - b) for a, b in zip(fitted, row["calibrated_scores"], strict=True)) < 1e-10,
                "Calibrated probabilities differ")
        require(row["predicted"] == row["candidate_ids"][max(range(20), key=fitted.__getitem__)],
                "Temperature changed argmax")
        scopes[row["scope"]].append(row)
    require(set(scopes) == {"calibration100", "test300"}, "Incorrect calibration scopes")
    for scope, values in scopes.items():
        expected = calibration["scopes"][scope]
        require(len(values) == expected["original"]["rows"], "Incorrect calibration row count")
        require(len({row["id"] for row in values}) == len(values), "Duplicate calibration rows")
        for label, field in (("original", "original_scores"), ("fitted", "calibrated_scores")):
            nll, ece = calibration_metrics(values, field)
            require(abs(nll - expected[label]["conditional_nll"]) < 1e-10, "NLL differs")
            require(abs(ece - expected[label]["ece_10_equal_width"]) < 1e-10, "ECE differs")
    require({row["group_id"] for row in scopes["calibration100"]}.isdisjoint(
        row["group_id"] for row in scopes["test300"]), "Calibration/test groups overlap")

    with ZipFile(root / "data/prepared/launcher-choice-v1.zip") as archive:
        frozen = json.loads(archive.read("experiment-input-manifest.json"))
        for name, expected in frozen["files"].items():
            with archive.open(name) as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == expected,
                        f"Frozen training file differs: {name}")
        with archive.open("prepared-train.jsonl") as stream:
            counts = Counter(json.loads(line)["source"] for line in stream)
        require(counts == {"clean": 5664, "stt": 16992}, "Incorrect frozen training counts")
    return {"verified": True, "models": len(grouped), "predictions": sum(map(len, grouped.values())),
            "commands": 200, "calibration_rows": len(calibration_rows), "training_rows": sum(counts.values()),
            "scope": "Recomputed saved evidence; no model inference or new training"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
