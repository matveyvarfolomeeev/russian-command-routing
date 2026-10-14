"""Freeze the existing E5 top-20 inputs and Nano-Jev v0.1 weights for Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_kaggle_training import prepare  # noqa: E402
from training_dataset import read_rows  # noqa: E402


def freeze(worktree: Path, output: Path) -> dict:
    prepare(output, "matveyvarfolomeev")
    sys.path.insert(0, str(worktree / "src"))
    import numpy as np
    from jev_launcher.semantic import MODEL_REVISION, E5Search, passage_text

    checkpoint = worktree / "models/rerank/nanojev-v01"
    spec = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    model_hash = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
    if model_hash != spec["weight_lfs_sha256"]:
        raise ValueError("Pinned Nano-Jev weights changed")
    e5 = E5Search(worktree / "models/e5", offline=True)
    e5._load()
    e5.cache_dir = output.parent / "e5-preparation-cache"
    train_dir = ROOT / "data/training"
    catalogs = {c["catalog_id"]: c["objects"] for c in read_rows(
        train_dir / "launcher-train-v1/catalogs.jsonl"
    )}
    prepared, injected = [], Counter()

    def make(row, objects, shortlist, source):
        target = row["expected_object_id"]
        candidate_ids = [obj["object_id"] for obj in shortlist]
        covered = target in candidate_ids
        if source != "validation" and not covered:
            # Training labels may construct positives; evaluation never repairs retrieval.
            shortlist[-1] = next(obj for obj in objects if obj["object_id"] == target)
            injected[source] += 1
        random.Random("nano-choice-v1:" + row["id"]).shuffle(shortlist)
        return {
            "id": row["id"], "query": row["query"], "source": source,
            "group_id": row["group_id"], "target": target,
            "candidate_ids": [obj["object_id"] for obj in shortlist],
            "documents": [passage_text(obj).removeprefix("passage: ") for obj in shortlist],
            "retrieval_covered": covered,
        }

    for source, dataset in (("clean", "launcher-train-v1"), ("stt", "launcher-stt-train-v1")):
        rows = read_rows(train_dir / dataset / "train.jsonl")
        for catalog_id, objects in catalogs.items():
            local = [row for row in rows if row["catalog_id"] == catalog_id]
            e5._index(objects)
            vectors, truncated = e5._encode(["query: " + row["query"].strip() for row in local])
            if any(truncated) or e5._truncated_ids:
                raise ValueError("Training retrieval truncated an input")
            similarities = vectors @ e5._embeddings.T
            ids = [obj["object_id"] for obj in objects]
            for row, scores in zip(local, similarities, strict=True):
                order = sorted(range(len(objects)), key=lambda i: (-float(scores[i]), ids[i]))[:20]
                prepared.append(make(row, objects, [objects[i] for i in order], source))
            print(f"Prepared {source} {catalog_id}: {len(local)}", flush=True)
    validation = []
    for profile in ("launcher-basic-v1", "launcher-v1", "launcher-asr-v1"):
        directory = ROOT / "data/benchmarks" / profile / "synthetic"
        objects = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))["objects"]
        for row in read_rows(directory / "validation.jsonl"):
            result = e5.rank(row["query"], objects, limit=20)
            if result["query_truncated"] or result["truncated_object_ids"]:
                raise ValueError("Validation retrieval truncated an input")
            item = make(row, objects, list(result["results"]), "validation")
            validation.append({**item, "profile": profile})
    paths = []
    for name, rows in (("prepared-train.jsonl", prepared), ("prepared-validation.jsonl", validation)):
        path = output / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        paths.append(path)
    manifest = {
        "model": spec["spec"], "original_weight_sha256": model_hash,
        "retrieval_revision": MODEL_REVISION, "shortlist_size": 20,
        "train_rows": len(prepared), "validation_rows": len(validation),
        "train_positive_injections": dict(injected), "evaluation_positive_injections": 0,
        "candidate_order": "Fixed per-row shuffle, identical across both training schedules",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "validation_recall": {
            profile: float(np.mean([r["retrieval_covered"] for r in validation if r["profile"] == profile]))
            for profile in sorted({r["profile"] for r in validation})
        },
    }
    (output / "experiment-input-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with ZipFile(output / "nanojev-v01.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "nanojev_config.json", "README.md", "calibration.json"):
            bundle.write(checkpoint / name, name)
        bundle.write(checkpoint / "code/nanojev/model.py", "native_model.py")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kaggle/training-data")
    args = parser.parse_args()
    print(json.dumps(freeze(args.worktree, args.output), indent=2))
