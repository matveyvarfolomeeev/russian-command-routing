"""Export a portable subset of completed local experiments for GitHub publication."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from verify_public_results import ROOT, load, require, sha, summarize

OUTPUT = ROOT / "results"
PROVENANCE = {}


def source(path):
    path = path.resolve()
    PROVENANCE[path.relative_to(ROOT).as_posix()] = sha(path)
    return path


def write(name, value):
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(path):
    with source(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_gzip(name, values):
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output, gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as stream:
        for value in values:
            stream.write((json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode())


def public_prediction(row):
    value = {key: row[key] for key in
             ("id", "profile", "group_id", "target", "candidate_ids", "predicted", "correct")}
    value["retrieved_ids"] = row.get("top20", row["candidate_ids"])
    value["retrieval_covered"] = row["target"] in value["retrieved_ids"]
    if "scores" in row:
        value["scores"] = row["scores"]
    return value


def export(registry_path):
    registry = load(source(registry_path))
    independent = ROOT / "artifacts/kaggle/independent-v1"
    comparison = load(source(independent / "comparison.json"))
    fixed = load(source(independent / "fixed-checkpoints.json"))
    models = {
        "e5-original": {"display_name": "E5 original", "stage": "initial_frozen_comparison",
                        "metrics": summarize(rows(independent / "e5-predictions.jsonl"))}
    }
    predictions = [{"model": "e5-original", **public_prediction(row)}
                   for row in rows(independent / "e5-predictions.jsonl")]
    directories = {name: (Path(directory), "initial_frozen_comparison")
                   for name, directory in comparison["result_sources"].items()}
    directories.update({
        "e5-trained": (ROOT / "artifacts/kaggle/e5-evaluation-v1/selected", "later_known_suite"),
        "modernbert-trained": (ROOT / "artifacts/kaggle/modernbert-evaluation-v1/trained", "later_known_suite"),
        "rubert-trained": (ROOT / "artifacts/kaggle/rubert-evaluation-v1/trained", "later_known_suite"),
    })
    for name, (directory, stage) in directories.items():
        result = load(source(directory / "result.json"))
        local = [row for row in rows(directory / "validation-predictions.jsonl")
                 if row["profile"].startswith("independent/")]
        require(len(local) == 400, f"Incomplete independent evidence: {name}")
        public = [public_prediction(row) for row in local]
        actual = summarize(public)
        for profile, metric in actual.items():
            original = result["metrics"][profile]
            require(metric["correct"] == original["correct"] and metric["rows"] == original["rows"],
                    f"Local result disagrees with predictions: {name}")
        models[name] = {"display_name": name, "stage": stage, "metrics": actual,
                        "checkpoint_sha256": result["checkpoint_sha256"]}
        if name in comparison["models"]:
            models[name]["paired_vs_e5"] = comparison["models"][name]["paired_vs_e5"]
            models[name]["selected_step"] = fixed["selected_steps"].get(name)
        predictions.extend({"model": name, **row} for row in public)
    write("independent-v1/summary.json", {
        "schema_version": 1, "commands": 200, "representations": ["clean", "asr"],
        "test_commands": 150, "catalog_objects": 177, "test_groups": 30,
        "role": "Initial frozen-model evaluation; later adaptations use a previously inspected suite",
        "annotation": "Author-reviewed; TTS audio; no independent human annotation or live human speech",
        "models": models, "adaptation": comparison["adaptation"],
    })
    write_gzip("independent-v1/predictions.jsonl.gz", predictions)

    training = []
    for experiment in registry["experiments"]:
        if experiment["category"] != "fine_tuning" or experiment["status"] != "complete":
            continue
        result = load(source(ROOT / experiment["result"]))
        entry = {key: experiment[key] for key in
                 ("id", "completed_updates", "selected_update", "elapsed_seconds", "stop_reason",
                  "selected_checkpoint_sha256", "last_checkpoint_sha256", "plan")}
        entry["selection_results"] = {key: result[key] for key in
                                      ("baseline", "trained", "best", "last") if key in result}
        training.append(entry)
        directory = ROOT / experiment["result"]
        for filename in ("training.jsonl", "validation.jsonl"):
            ledger = directory.parent / filename
            if ledger.exists():
                # Ledgers contain fictional training IDs and numeric diagnostics only.
                write_gzip(f"training/{experiment['id']}/{filename}.gz", rows(ledger))
    write("training-runs.json", {"schema_version": 1, "selection_data": "Original synthetic validation150",
                                  "runs": training, "single_seed_per_recipe": True})

    temperature_dir = ROOT / "artifacts/benchmarks/laya-temperature-v1"
    temperature = load(source(temperature_dir / "temperature.json"))
    protocol = load(source(temperature_dir / "protocol.json"))
    original = {row["id"]: row for row in rows(ROOT / protocol["input"]) if row["scope"] == "known800"}
    calibrated = []
    for scope in ("calibration100", "test300"):
        for row in rows(temperature_dir / f"{scope}-predictions.jsonl"):
            raw = original[row["id"]]
            value = public_prediction(row)
            value.pop("scores")
            calibrated.append({**value, "scope": scope, "original_scores": raw["scores"],
                               "calibrated_scores": row["scores"]})
    write("calibration/laya.json", {
        "schema_version": 1, "temperature": temperature["temperature"],
        "checkpoint_sha256": temperature["checkpoint_sha256"], "candidate_count": 20,
        "scope": "Known independent suite, 100 calibration / 300 test representations; disjoint groups",
        "scopes": {name: temperature["scopes"][name] for name in ("calibration100", "test300")},
        "optimizer_updates": 0, "argmax_changes": 0,
    })
    write_gzip("calibration/predictions.jsonl.gz", calibrated)

    model_sources = {}
    for family, manifest in {
        "nanojev": "training-data/experiment-input-manifest.json",
        "laya": "laya-data/laya-input-manifest.json",
        "kev": "kev-data/kev-input-manifest.json",
        "modernbert": "modernbert-data/modernbert-input-manifest.json",
    }.items():
        model_sources[family] = load(source(ROOT / "artifacts/kaggle" / manifest))["model"]
    for family in ("e5", "rubert"):
        model = load(source(ROOT / f"artifacts/kaggle/{family}-source-v1/source.json"))
        model_sources[family] = {key: model[key] for key in
                                 ("model_id", "revision", "source_url", "license", "parameters")}
    write("source-models.json", model_sources)

    archive_path = ROOT / "data/prepared/launcher-choice-v1.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ("prepared-train.jsonl", "prepared-validation.jsonl", "experiment-input-manifest.json"):
            archive.write(source(ROOT / "artifacts/kaggle/training-data" / name), name)
    files = [p for p in OUTPUT.rglob('*') if p.is_file() and p.name != "manifest.json"]
    files += [archive_path]
    for directory in (ROOT / "data/benchmarks", ROOT / "data/training"):
        files.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix in (".json", ".jsonl"))
    write("manifest.json", {
        "schema_version": 1, "published_on": "2026-10-14",
        "files": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)},
        "local_provenance_sha256": PROVENANCE,
        "scope": "Portable saved evidence, fictional data only; weights/audio/personal catalogs omitted",
    })
    print(json.dumps({"exported": True, "models": len(models), "predictions": len(predictions),
                      "training_runs": len(training), "prepared_archive_bytes": archive_path.stat().st_size}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path,
                        default=ROOT / "artifacts/research-notes/docs/EXPERIMENT_REGISTRY.json")
    export(parser.parse_args().registry)
