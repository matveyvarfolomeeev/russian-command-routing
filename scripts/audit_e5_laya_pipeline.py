"""Independently check every pipeline prediction, frozen input and reported count."""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(folder):
    protocol = read(folder / "protocol.json")
    report = read(folder / "comparison.json")
    assert sha(folder / "protocol.json") == report["protocol_sha256"]
    assert protocol["rows_per_variant"] == 974
    assert protocol["optimizer_updates"] == report["optimizer_updates"] == 0
    assert protocol["checkpoint_selection"] is report["checkpoint_selection"] is False
    assert protocol["production_config_changed"] is report["production_config_changed"] is False
    for name, expected in protocol["frozen_files"].items():
        assert sha(Path(name)) == expected, name
    for path in (folder / "source").iterdir():
        matching = [value for name, value in protocol["frozen_files"].items() if Path(name).name == path.name]
        assert matching == [sha(path)], path.name
    assert sha(folder / "queries.jsonl") == protocol["query_sha256"]
    assert sha(folder / "catalogs.json") == protocol["catalogs_sha256"]
    queries = rows(folder / "queries.jsonl")
    catalogs = read(folder / "catalogs.json")
    assert len(queries) == len({(r["scope"], r["id"]) for r in queries}) == 974
    expected_queries = {}
    for scope, path in (("known800", ROOT / "artifacts/kaggle/e5-evaluation-v1/input/full-evaluation.jsonl"),
                        ("metadata-real150", ROOT / "artifacts/benchmarks/app-metadata-v2/evaluation.jsonl"),
                        ("metadata-diagnostic24", ROOT / "artifacts/benchmarks/app-metadata-v2/diagnostic-queries.jsonl")):
        expected_queries[scope] = {r["id"]: r for r in rows(path)}
    for row in queries:
        expected = expected_queries[row["scope"]][row["id"]]
        assert all(row[k] == expected[k] for k in ("id", "query", "target", "profile"))
        assert row["target"] in {o["object_id"] for o in catalogs[row["catalog_key"]]}
    variants = {}
    reference_max_difference = 0.0
    retrieval_max_difference = 0.0
    for variant in ("original", "trained"):
        e5_folder = ROOT / "artifacts/kaggle/e5-evaluation-v1"
        directory = "original-onnx" if variant == "original" else "selected-onnx"
        retrieval_references = {"known800": {r["id"]: r for r in rows(e5_folder / directory / "predictions.jsonl")}}
        method = "e5-original" if variant == "original" else "e5-trained"
        for scope, directory in (("metadata-real150", "evaluation-v2"), ("metadata-diagnostic24", "diagnostic-evaluation")):
            reference_path = ROOT / "artifacts/benchmarks/app-metadata-v2" / directory / f"{method}-after.jsonl"
            retrieval_references[scope] = {r["id"]: r for r in rows(reference_path)}
        path = folder / f"{variant}-predictions.jsonl"
        assert sha(path) == report["prediction_sha256"][path.name]
        records = rows(path)
        assert len(records) == len(queries)
        variants[variant] = records
        for row, query in zip(records, queries, strict=True):
            assert all(row[k] == value for k, value in query.items())
            assert row["variant"] == variant
            objects = {o["object_id"]: o for o in catalogs[row["catalog_key"]]}
            assert row["target_kind"] == objects[row["target"]]["kind"]
            assert len(row["candidate_ids"]) == len(set(row["candidate_ids"])) == 20
            assert set(row["candidate_ids"]) == set(row["ranked_ids"]) == set(row["retrieval_ids"])
            assert set(row["candidate_ids"]) <= set(objects)
            expected_order = row["retrieval_ids"].copy()
            random.Random("launcher-choice-v1:" + row["query"].strip().casefold()).shuffle(expected_order)
            assert row["candidate_ids"] == expected_order
            scores = dict(zip(row["candidate_ids"], row["scores"], strict=True))
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in scores.values())
            assert abs(sum(scores.values()) - 1) < 1e-5
            assert row["ranked_ids"] == sorted(row["candidate_ids"], key=lambda key: -scores[key])
            assert row["predicted"] == row["ranked_ids"][0]
            assert row["correct"] == (row["predicted"] == row["target"])
            assert row["e5_predicted"] == row["retrieval_ids"][0]
            assert row["e5_correct"] == (row["e5_predicted"] == row["target"])
            assert row["target_in_top20"] == (row["target"] in row["candidate_ids"])
            assert len(row["retrieval_scores"]) == 20
            assert all(math.isfinite(s) and abs(s) < 1.00001 for s in row["retrieval_scores"])
            assert row["retrieval_scores"] == sorted(row["retrieval_scores"], reverse=True)
            ref = retrieval_references[row["scope"]][row["id"]]
            if row["scope"] == "known800":
                assert row["retrieval_ids"] == ref["top20"]
                reference_scores = dict(zip(objects, ref["similarities"], strict=True))
                difference = max(abs(score - reference_scores[oid]) for oid, score in zip(
                    row["retrieval_ids"], row["retrieval_scores"], strict=True))
            else:
                assert row["retrieval_ids"] == ref["candidate_ids"]
                difference = max(abs(a - b) for a, b in zip(row["retrieval_scores"], ref["scores"][:20], strict=True))
            assert row["e5_predicted"] == ref["predicted"] and difference < 1e-5
            retrieval_max_difference = max(retrieval_max_difference, difference)
            assert math.isfinite(row["warm_pipeline_ms"]) and row["warm_pipeline_ms"] > 0
            assert row["warm_pipeline_ms"] >= row["e5_elapsed_ms"] - .02
        for scope, value in report["scopes"].items():
            local = [r for r in records if r["scope"] == scope]
            summary = value[variant]
            assert len(local) == summary["rows"]
            for field, source in (("correct", "correct"), ("e5_correct", "e5_correct"), ("recall20", "target_in_top20")):
                assert sum(r[source] for r in local) == summary[field]
            groups = defaultdict(list)
            scenarios = defaultdict(list)
            for row in local:
                groups[row["profile"]].append(row)
                name = "/".join(row["profile"].split("/")[:2]) if row["profile"].startswith("independent/") else row["profile"]
                scenarios[name].append(row)
            assert set(groups) == set(summary["profiles"])
            assert set(scenarios) == set(summary["scenarios"])
            for name, group in groups.items():
                assert summary["profiles"][name] == {
                    "rows": len(group), "correct": sum(r["correct"] for r in group),
                    "e5_correct": sum(r["e5_correct"] for r in group),
                    "recall20": sum(r["target_in_top20"] for r in group)}
            for name, group in scenarios.items():
                assert summary["scenarios"][name] == {
                    "rows": len(group), "correct": sum(r["correct"] for r in group),
                    "e5_correct": sum(r["e5_correct"] for r in group)}
            for label, field in (("equal_scenario_accuracy", "correct"), ("e5_equal_scenario_accuracy", "e5_correct")):
                mean = sum(sum(r[field] for r in group) / len(group) for group in scenarios.values()) / len(scenarios)
                assert abs(mean - summary[label]) < 1e-12
            times = sorted(r["warm_pipeline_ms"] for r in local)
            for label, quantile in (("p50_ms", .5), ("p95_ms", .95)):
                index = (len(times) - 1) * quantile
                lower = int(index)
                observed = times[lower] + (times[min(lower + 1, len(times) - 1)] - times[lower]) * (index - lower)
                assert abs(observed - summary["latency"][label]) < 1e-6
            assert times[-1] == summary["latency"]["maximum_ms"]
            assert sum(t > 500 for t in times) == summary["latency"]["over_500_ms"]
    changes = []
    for scope, value in report["scopes"].items():
        first = [r for r in variants["original"] if r["scope"] == scope]
        second = [r for r in variants["trained"] if r["scope"] == scope]
        counts = defaultdict(int)
        ids = {"fixed_ids": [], "lost_ids": []}
        for a, b in zip(first, second, strict=True):
            assert a["id"] == b["id"]
            same_set = set(a["candidate_ids"]) == set(b["candidate_ids"])
            counts["shortlist_sets_changed"] += not same_set
            counts["same_set_different_input_order"] += same_set and a["candidate_ids"] != b["candidate_ids"]
            counts["retrieval_gained"] += not a["target_in_top20"] and b["target_in_top20"]
            counts["retrieval_lost"] += a["target_in_top20"] and not b["target_in_top20"]
            fixed, lost = not a["correct"] and b["correct"], a["correct"] and not b["correct"]
            counts["fixed"] += fixed
            counts["lost"] += lost
            counts["fixes_with_same_set"] += fixed and same_set
            counts["losses_with_same_set"] += lost and same_set
            if fixed or lost:
                ids["fixed_ids" if fixed else "lost_ids"].append(a["id"])
                changes.append({"scope": scope, "id": a["id"], "query": a["query"], "target": a["target"],
                                "original": a["predicted"], "trained": b["predicted"],
                                "change": "fixed" if fixed else "lost", "same_shortlist_set": same_set,
                                "original_target_in_top20": a["target_in_top20"],
                                "trained_target_in_top20": b["target_in_top20"]})
            if a["candidate_ids"] == b["candidate_ids"]:
                assert a["predicted"] == b["predicted"]
        assert {**counts, **ids} == value["paired"]
        assert value["trained"]["correct"] - value["original"]["correct"] == counts["fixed"] - counts["lost"]
    # Previous application-order results on enriched cards must be reproduced exactly.
    for scope, directory in (("metadata-real150", "evaluation-v2"), ("metadata-diagnostic24", "diagnostic-evaluation")):
        reference_path = ROOT / "artifacts/benchmarks/app-metadata-v2" / directory / "e5-original-laya-after.jsonl"
        reference_report = read(reference_path.parent / "comparison.json")
        assert sha(reference_path) == reference_report["prediction_sha256"][reference_path.name]
        reference = {r["id"]: r for r in rows(reference_path)}
        for row in [r for r in variants["original"] if r["scope"] == scope]:
            ref = reference[row["id"]]
            assert row["predicted"] == ref["predicted"] and row["candidate_ids"] == ref["candidate_ids"]
            difference = max(abs(a - b) for a, b in zip(row["scores"], ref["scores"], strict=True))
            assert difference < 1e-5
            reference_max_difference = max(reference_max_difference, difference)
    (folder / "changed-answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in changes), encoding="utf-8")
    sensitivity = {"role": "Descriptive postprocessing of stored probabilities, no temperature fitting or selection",
                   "method": "q_i = p_i^(1/T) / sum_j p_j^(1/T)",
                   "limitation": "Stored FP32 softmax probabilities, not raw logits; underflowed zeros remain zero",
                   "temperature_fitted": False, "production_temperature": 1.0, "variants": {}}
    for variant, records in variants.items():
        local = [r for r in records if r["scope"] == "known800" and r["profile"].startswith("launcher-asr-v1/")]
        assert len(local) == 100
        values = []
        for temperature in (.5, 1., 2., 5.):
            probabilities, changed = [], 0
            for row in local:
                transformed = [p ** (1 / temperature) for p in row["scores"]]
                total = sum(transformed)
                best = max(range(20), key=lambda i: transformed[i])
                changed += row["candidate_ids"][best] != row["predicted"]
                probabilities.append(transformed[best] / total)
            assert changed == 0
            values.append({"temperature": temperature, "top1_changed": changed,
                           "mean_max_probability": sum(probabilities) / len(probabilities)})
        sensitivity["variants"][variant] = {"rows": len(local), "correct": sum(r["correct"] for r in local),
                                            "zero_probability_entries": sum(p == 0 for r in local for p in r["scores"]),
                                            "values": values}
    (folder / "temperature-sensitivity.json").write_text(json.dumps(sensitivity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {"status": "passed", "predictions_checked": 1948, "query_pairs": 974,
             "frozen_files_checked": len(protocol["frozen_files"]),
             "optimizer_updates": 0, "production_config_unchanged": True,
             "retrieval_references_reproduced": 1948,
             "retrieval_reference_maximum_cosine_difference": retrieval_max_difference,
             "enriched_original_laya_reference_maximum_probability_difference": reference_max_difference,
             "scope_counts": {scope: {v: value[v]["correct"] for v in ("original", "trained")}
                              for scope, value in report["scopes"].items()},
             "error_decomposition": {
                 scope: {variant: {
                     "retrieval_misses": sum(not r["target_in_top20"] for r in records if r["scope"] == scope),
                     "choice_errors_with_available_target": sum(r["target_in_top20"] and not r["correct"]
                                                               for r in records if r["scope"] == scope),
                     "laya_fixed_e5_errors": sum(not r["e5_correct"] and r["correct"] for r in records if r["scope"] == scope),
                     "laya_spoiled_e5_correct": sum(r["e5_correct"] and not r["correct"] for r in records if r["scope"] == scope)}
                         for variant, records in variants.items()}
                 for scope in report["scopes"]},
             "evidence_sha256": {p.name: sha(p) for p in (
                 folder / "protocol.json", folder / "comparison.json", folder / "queries.jsonl", folder / "catalogs.json",
                 folder / "original-predictions.jsonl", folder / "trained-predictions.jsonl", folder / "changed-answers.jsonl",
                 folder / "temperature-sensitivity.json")}}
    (folder / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/benchmarks/e5-laya-pipeline-v1")
    run(parser.parse_args().input)
