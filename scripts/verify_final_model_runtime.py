"""Verify the three GUI workers offline against frozen full-catalog predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from jev_launcher.model_search import MODELS, ModelRuntime, runtime_config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/model-integration-v2"
PIPELINE = ROOT / "artifacts/benchmarks/e5-laya-pipeline-v1"


def rows(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(packaged=False):
    OUTPUT.mkdir(exist_ok=True)
    config_path = runtime_config()
    config = json.loads(config_path.read_text("utf-8"))
    if packaged:
        worker = ROOT / "dist/JevLauncher/_internal/jev_launcher/runtime/run_launcher_model.py"
        for name in ("run_launcher_model.py", "benchmark_choice_quantization.py",
                     "train_laya_kaggle.py", "train_nanojev_kaggle.py", "train_modernbert_kaggle.py"):
            assert sha(worker.parent / name) == sha(ROOT / "scripts" / name), name
        config["worker"] = str(worker)
        config_path = OUTPUT / "bundled-runtime.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    assert set(config["profiles"]) == {"laya", "modernbert", "e5_trained"}
    assert {key for key, _ in MODELS if key != "filter"} == set(config["profiles"])
    catalogs = json.loads((PIPELINE / "catalogs.json").read_text("utf-8"))
    original = rows(PIPELINE / "original-predictions.jsonl")
    known = [row for row in original if row["scope"] == "known800"]
    trained = {row["id"]: row for row in rows(
        ROOT / "artifacts/kaggle/e5-evaluation-v1/selected-onnx/predictions.jsonl")}
    modern = {row["id"]: row for row in rows(
        ROOT / "artifacts/kaggle/modernbert-evaluation-v1/trained/validation-predictions.jsonl")}
    selected = []
    for profile in ("launcher-asr-v1/synthetic", "launcher-asr-v1/real",
                    "independent/clean/validation", "independent/asr/test"):
        selected.extend([row for row in known if row["profile"] == profile][:(2 if packaged else 12)])
    assert len(selected) == (8 if packaged else 48)
    temperature = json.loads(Path(config["profiles"]["laya"]["calibration"]).read_text("utf-8"))["temperature"]
    result = {"schema_version": 1, "models": {}, "rows_per_model": len(selected),
              "bundled_worker": packaged,
              "roles": "known-input offline integration parity, not a new benchmark",
              "config_sha256": sha(config_path),
              "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
                  Path(__file__), ROOT / "scripts/run_launcher_model.py",
                  ROOT / "scripts/configure_model_runtime.py", ROOT / "src/jev_launcher/model_search.py")}}
    for key in ("e5_trained", "laya", "modernbert"):
        service = ModelRuntime(config_path, timeout=180)
        records, prepared = [], set()
        maximum_probability_difference = 0.0
        try:
            for row in selected:
                objects = catalogs[row["catalog_key"]]
                assert len(objects) == 177
                if row["catalog_key"] not in prepared:
                    assert service.prepare(key, objects)["event"] == "ready"
                    assert service.prepare(key, objects)["cached"]
                    prepared.add(row["catalog_key"])
                response = service.search(key, row["query"], objects)
                if key == "e5_trained":
                    reference = trained[row["id"]]
                    assert response["candidate_ids"] == reference["top20"], row["id"]
                    assert response["object_ids"][0] == reference["predicted"], row["id"]
                    assert response["score_type"] == "cosine_similarity"
                    assert response["temperature"] is None
                elif key == "laya":
                    assert response["object_ids"] == row["ranked_ids"], row["id"]
                    assert response["candidate_ids"] == row["candidate_ids"]
                    assert response["temperature"] == temperature
                    transformed = [math.exp(math.log(p) / temperature) for p in row["scores"]]
                    total = math.fsum(transformed)
                    difference = max(abs(p - expected / total) for p, expected
                                     in zip(response["scores"], transformed))
                    assert difference < 2e-6, (row["id"], difference)
                    maximum_probability_difference = max(maximum_probability_difference, difference)
                else:
                    assert response["object_ids"][0] == modern[row["id"]]["predicted"], row["id"]
                    assert response["temperature"] == 1.0
                assert response["checkpoint_sha256"] == config["profiles"][key]["checkpoint_sha256"]
                records.append({"id": row["id"], "predicted": response["object_ids"][0],
                                "elapsed_ms": response["elapsed_ms"]})
            if key == "laya":
                small = catalogs["independent"][:5]
                response = service.search(key, "открой приложение", small)
                assert response["temperature"] == 1.0
                assert len(response["object_ids"]) == 5
                assert math.fsum(response["scores"]) == 1.0
            result["models"][key] = {"rows": len(records), "known_input_parity": True,
                                     "maximum_probability_difference": maximum_probability_difference,
                                     "predictions": records}
            print(json.dumps({"verified": key, "rows": len(records)}), flush=True)
        finally:
            process = service._process
            service.close(wait=True)
            if process is not None:
                assert process.poll() is not None
    result["passed"] = True
    name = "bundled-worker-check.json" if packaged else "runtime-check.json"
    (OUTPUT / name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packaged", action="store_true")
    run(parser.parse_args().packaged)
