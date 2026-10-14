"""Package fictional training data and synthetic validation for private Kaggle use."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stt_training_dataset as stt  # noqa: E402
import training_dataset as clean  # noqa: E402


def prepare(output: Path, owner: str) -> dict:
    # Reuse the source validators, including the frozen evaluation guards.
    reports = [clean.validate(clean.DEFAULT_DATASET), stt.validate(stt.DEFAULT_DATASET)]
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "catalogs.jsonl": clean.DEFAULT_DATASET / "catalogs.jsonl",
        "clean.jsonl": clean.DEFAULT_DATASET / "train.jsonl",
        "stt.jsonl": stt.DEFAULT_DATASET / "train.jsonl",
        "clean.source-manifest.json": clean.DEFAULT_DATASET / "manifest.json",
        "stt.source-manifest.json": stt.DEFAULT_DATASET / "manifest.json",
    }
    for profile in ("launcher-basic-v1", "launcher-v1", "launcher-asr-v1"):
        directory = ROOT / "data" / "benchmarks" / profile / "synthetic"
        for name in ("catalog.json", "validation.jsonl"):
            sources[f"validation/{profile}/{name}"] = directory / name
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    manifest = {
        "schema_version": 1,
        "privacy": "fictional_data_only; private_kaggle_dataset",
        "files": hashes,
        "training": {r["dataset_id"]: r["rows"] for r in reports},
        "validation": "Synthetic basic/difficult/ASR, reported separately. No test rows uploaded.",
        "real_evaluation": "Remains local; this package does not include personal catalogs.",
        "model_input": "Only query and catalog metadata. expected_object_id is a label.",
        "grouping": "Keep related clean/STT instances and substitutions in the same group.",
    }
    # A single archive avoids duplicating full catalogs in every training example.
    archive = output / "launcher-data.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name, path in sources.items():
            bundle.writestr(name, path.read_bytes())
        bundle.writestr("bundle-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    (output / "bundle-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "id": f"{owner}/launcher-decision-training-v1",
        "title": "Launcher decision training v1",
        "licenses": [{"name": "other"}],
        "description": "Fictional Russian launcher training data; synthetic validation only. "
        "Clean: 5664 rows; simulated STT: 16992 rows. Test and personal catalogs stay local. "
        "Private experiment: training data remains copyright of its authors. "
        "Model files retain their original license notices. No additional public license is granted.",
    }
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"dataset": metadata["id"], "archive": str(archive.resolve()),
            "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "training": manifest["training"], "files": len(sources)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kaggle/training-data")
    parser.add_argument("--owner", default="matveyvarfolomeev")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.owner), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
