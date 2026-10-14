"""Verify paired E5 outcomes and both complete native CPU timing passes."""

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_e5_locally import ROOT
from train_modernbert_kaggle import sha
from train_nanojev_kaggle import rows


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(args):
    import numpy as np

    args.run = args.run.resolve()
    base = ROOT / "artifacts/kaggle/e5-evaluation-v1"
    audit = load(ROOT / "artifacts/kaggle/e5-final-audit.json")
    assert audit["gpu_audit"] == audit["local800_audit"] == "passed"
    input_rows = rows(base / "input/full-evaluation.jsonl")
    original = {row["id"]: row for row in rows(base / "original/validation-predictions.jsonl")}
    trained = {row["id"]: row for row in rows(base / "selected/validation-predictions.jsonl")}
    assert len(original) == len(trained) == len(input_rows) == 800
    assert set(original) == set(trained) == {row["id"] for row in input_rows}
    paired = {}
    for profile in sorted({row["profile"] for row in input_rows}):
        local = [row for row in input_rows if row["profile"] == profile]
        fixed, lost, gained20, lost20 = 0, 0, 0, 0
        for row in local:
            a, b = original[row["id"]], trained[row["id"]]
            assert all(a[key] == b[key] == row[key] for key in ("id", "target", "profile", "group_id", "catalog_key"))
            assert a["candidate_ids"] == b["candidate_ids"]
            fixed += not a["correct"] and b["correct"]
            lost += a["correct"] and not b["correct"]
            gained20 += a["target_rank"] > 20 >= b["target_rank"]
            lost20 += b["target_rank"] > 20 >= a["target_rank"]
        paired[profile] = {"rows": len(local), "fixed_top1": fixed, "lost_top1": lost,
                           "gained_top20_target": gained20, "lost_top20_target": lost20}
    quality = {label: load(base / label / "result.json") for label in ("original", "selected")}
    latency = {}
    sources = [base / "input/input-manifest.json", base / "input/full-evaluation.jsonl", base / "input/catalogs.json",
               ROOT / "artifacts/kaggle/e5-final-audit.json", args.run / "result.json", args.run / "plan.json",
               args.run / "training.jsonl", args.run / "validation.jsonl", args.run / "encoding.json",
               ROOT / "artifacts/kaggle/e5-source-v1/source.json", ROOT / "artifacts/kaggle/e5-data/e5-input-manifest.json"]
    for label in ("original", "selected"):
        folder = base / f"{label}-onnx"
        report = load(folder / "result.json")
        native_predictions = rows(folder / "predictions.jsonl")
        reference = original if label == "original" else trained
        assert len(native_predictions) == 800 and len({row["id"] for row in native_predictions}) == 800
        for row in native_predictions:
            assert row["predicted"] == reference[row["id"]]["predicted"]
            assert row["target_rank"] == reference[row["id"]]["target_rank"]
            assert set(row["top20"]) == set(reference[row["id"]]["top20"])
        assert report["all_reference_top1_reproduced"] and report["checkpoint_sha256"] == quality[label]["checkpoint_sha256"]
        assert report["metrics"] == quality[label]["metrics"]
        timed = rows(folder / "timed-rows.jsonl")
        ids = {row["id"] for row in input_rows if row["profile"].startswith("independent/")}
        assert len(timed) == 800 and len(ids) == 400
        assert Counter(row["id"] for row in timed) == Counter({identity: 2 for identity in ids})
        for repetition, summary in enumerate(report["repetitions"], 1):
            local = [row for row in timed if row["repetition"] == repetition]
            assert len(local) == 400 and {row["id"] for row in local} == ids
            values = np.array([row["pipeline_ms"] for row in local])
            assert np.isfinite(values).all() and (values >= 0).all()
            assert float(np.quantile(values, .5)) == summary["p50_ms"]
            assert float(np.quantile(values, .95)) == summary["p95_ms"]
            assert float(values.max()) == summary["maximum_ms"]
            assert int((values > 500).sum()) == summary["over_500_ms"]
            assert all(row["predicted"] == reference[row["id"]]["predicted"] for row in local)
        assert max(row["maximum_similarity_difference"] for row in native_predictions) == report["maximum_similarity_difference"]
        latency[label] = report
        sources.extend([folder / name for name in ("result.json", "predictions.jsonl", "timed-rows.jsonl")])
        sources.extend([base / label / name for name in ("result.json", "validation-predictions.jsonl")])
    compact = {label: {key: value for key, value in result.items() if key in
                        ("metrics", "scenario_counts", "equal_scenario_accuracy", "equal_row_accuracy",
                         "noisy_stt_equal_catalog_accuracy", "checkpoint_sha256")} for label, result in quality.items()}
    fixed = sum(row["fixed_top1"] for row in paired.values())
    lost = sum(row["lost_top1"] for row in paired.values())
    run_result = load(args.run / "result.json")
    report = {"recorded_on": "2026-10-08", "quality": compact, "paired_profiles": paired,
              "paired_total": {"fixed_top1": fixed, "lost_top1": lost}, "latency": latency,
              "training": {key: run_result[key] for key in ("steps", "best_step", "elapsed_seconds", "stop_reason",
                                                            "checkpoint_sha256", "last_checkpoint_sha256")},
              "audits": {"quality800": "passed", "latency400x2_both_models": "passed", "gpu_cpu150": audit["gpu_cpu150"]},
              "data_roles": "Same synthetic150 full-catalog selection; known800 descriptive afterward; oldtest150 unopened; no fresh blind test",
              "production": "Main Laya and installed original E5 unchanged; trained E5-to-Laya cascade not measured",
              "source_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paired_total": report["paired_total"], "macro10": {k: v["equal_scenario_accuracy"] for k, v in compact.items()},
                      "noisy_stt": {k: v["noisy_stt_equal_catalog_accuracy"] for k, v in compact.items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kaggle/e5-evaluation-v1/comparison.json")
    run(parser.parse_args())
