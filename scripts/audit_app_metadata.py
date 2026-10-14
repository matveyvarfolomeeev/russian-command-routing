"""Recompute card changes and every recorded quality result, without running models."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "artifacts/benchmarks/app-metadata-v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run():
    manifest = read(FOLDER / "manifest.json")
    for name, value in manifest["files"].items():
        assert sha(FOLDER / name) == value, name
    before, after = [read(FOLDER / f"{name}-catalog.json")["objects"] for name in ("before", "after")]
    assert len(before) == len(after) == 177
    assert [item["object_id"] for item in before] == [item["object_id"] for item in after]
    updates = read(FOLDER / "description-updates.json")
    changed = {}
    for first, second in zip(before, after, strict=True):
        if first != second:
            assert first["kind"] == "app" and first["description"] == ""
            assert {k: v for k, v in first.items() if k != "description"} == {k: v for k, v in second.items() if k != "description"}
            changed[first["object_id"]] = second["description"]
    assert changed == {key: item["description"] for key, item in updates.items()}
    assert len(changed) == manifest["changed_cards"] == 12
    cache = read(FOLDER / "package-cache.json")["packages"]
    installed = cache["__installed__"]["versions"]
    packages = [record["package"] for record in cache.values() if isinstance(record, dict) and record.get("package")]
    for update in updates.values():
        if update["description_source"].startswith("winget:"):
            package_id = update["description_source"].removeprefix("winget:")
            assert any(package["package_id"] == package_id and package["version"] == installed[package_id]
                       and package["description"] == update["description"] for package in packages)
    observed = {}
    predictions_checked = 0
    for scope, query_file in (("evaluation-v2", "evaluation.jsonl"), ("diagnostic-evaluation", "diagnostic-queries.jsonl")):
        report = read(FOLDER / scope / "comparison.json")
        assert report["optimizer_updates"] == 0 and report["checkpoint_selection"] is False
        query_rows = rows(FOLDER / query_file)
        if scope == "evaluation-v2":
            assert sha(FOLDER / scope / "evaluated-source.py") == report["source_sha256"]["scripts/evaluate_app_metadata.py"]
        else:
            assert report["query_file_sha256"] == sha(FOLDER / query_file)
        observed[scope] = {}
        for method, result in report["results"].items():
            records = {}
            for variant in ("before", "after"):
                path = FOLDER / scope / f"{method}-{variant}.jsonl"
                assert sha(path) == report["prediction_sha256"][path.name]
                local = rows(path)
                assert len(local) == len(query_rows)
                assert [(r["id"], r["query"], r["target"]) for r in local] == [(r["id"], r["query"], r["target"]) for r in query_rows]
                for row in local:
                    assert row["correct"] == (row["predicted"] == row["target"])
                    assert row["predicted"] == row["ranked_ids"][0]
                    assert row["target_in_top20"] == (row["target"] in row["candidate_ids"])
                    if method == "e5-original-laya":
                        # Runtime scores are in input-candidate order, not sorted-result order.
                        probabilities = dict(zip(row["candidate_ids"], row["scores"], strict=True))
                        assert abs(sum(probabilities.values()) - 1) < 1e-5
                        assert row["ranked_ids"] == sorted(row["candidate_ids"], key=lambda key: -probabilities[key])
                    else:
                        assert row["candidate_ids"] == row["ranked_ids"][:20]
                        assert row["scores"] == sorted(row["scores"], reverse=True)
                assert sum(r["correct"] for r in local) == result[variant]["correct"]
                assert sum(r["target_in_top20"] for r in local) == result[variant]["recall20"]
                values = sorted(r["warm_pipeline_ms"] for r in local)
                assert all(math.isfinite(value) and value > 0 for value in values)
                for label, quantile in (("p50_ms", .5), ("p95_ms", .95)):
                    position = (len(values) - 1) * quantile
                    lower = int(position)
                    measured = values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)
                    assert abs(measured - result[variant]["latency"][label]) < 1e-6
                assert values[-1] == result[variant]["latency"]["maximum_ms"]
                for profile, metric in result[variant]["profiles"].items():
                    selected = [r for r in local if r["profile"] == profile]
                    assert len(selected) == metric["rows"]
                    assert sum(r["correct"] for r in selected) == metric["correct"]
                records[variant] = local
                predictions_checked += len(local)
            first, second = records["before"], records["after"]
            assert sum(not a["correct"] and b["correct"] for a, b in zip(first, second)) == result["paired"]["fixed"]
            assert sum(a["correct"] and not b["correct"] for a, b in zip(first, second)) == result["paired"]["lost"]
            observed[scope][method] = {"before": result["before"]["correct"], "after": result["after"]["correct"]}
    assert predictions_checked == 1044
    applied = read(FOLDER / "application-readback.json")
    assert applied["applied"] == len(changed) == 12
    assert applied["only_missing_descriptions_and_provenance_changed"]
    assert sha(ROOT / applied["backup"]) == applied["backup_sha256"]
    evidence = list(FOLDER.glob("*.json")) + list(FOLDER.glob("*.jsonl"))
    evidence += [FOLDER / scope / "comparison.json" for scope in observed]
    report = {"status": "passed", "changed_cards": len(changed), "predictions_checked": predictions_checked,
              "packages_match_installed_ids_and_versions": True, "files_folders_names_aliases_and_ids_unchanged": True,
              "known150_targets_on_changed_cards": sum(r["target"] in changed for r in rows(FOLDER / "evaluation.jsonl")),
              "diagnostic24_targets_all_changed_cards": all(r["target"] in changed for r in rows(FOLDER / "diagnostic-queries.jsonl")),
              "observed": observed, "evidence_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in evidence}}
    (FOLDER / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "predictions_checked": predictions_checked, "observed": observed}))


if __name__ == "__main__":
    run()
