"""Prepare local real/synthetic validation, then evaluate one Nano-Jev checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_nanojev_kaggle import encode_all, evaluate, rows  # noqa: E402


def prepare(worktree, path):
    sys.path.insert(0, str(worktree / "src"))
    from jev_launcher.semantic import E5Search, passage_text

    e5 = E5Search(worktree / "models/e5", offline=True)
    e5._load()
    e5.cache_dir = path.parent / "local-e5-cache"
    prepared = []
    # Reuse the exact synthetic inputs given to both GPU experiments.
    for row in rows(ROOT / "artifacts/kaggle/training-data/prepared-validation.jsonl"):
        prepared.append({**row, "profile": row["profile"] + "/synthetic"})
    hashes = {}
    for profile in ("launcher-basic-v1", "launcher-v1", "launcher-asr-v1"):
        directory = ROOT / "artifacts/benchmarks" / profile / "real"
        catalog_path, validation_path = directory / "catalog.json", directory / "validation.jsonl"
        hashes[profile] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (catalog_path, validation_path)}
        objects = json.loads(catalog_path.read_text(encoding="utf-8"))["objects"]
        for row in rows(validation_path):
            ranked = e5.rank(row["query"], objects, limit=20)
            if ranked["query_truncated"] or ranked["truncated_object_ids"]:
                raise ValueError("Local retrieval truncated an input")
            shortlist = ranked["results"]
            random.Random("nano-choice-v1:" + row["id"]).shuffle(shortlist)
            ids = [obj["object_id"] for obj in shortlist]
            prepared.append({"id": row["id"], "query": row["query"], "source": "validation",
                             "group_id": row["group_id"], "target": row["expected_object_id"],
                             "profile": profile + "/real", "candidate_ids": ids,
                             "documents": [passage_text(obj).removeprefix("passage: ") for obj in shortlist],
                             "retrieval_covered": row["expected_object_id"] in ids})
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prepared), encoding="utf-8")
    path.with_suffix(".manifest.json").write_text(json.dumps({"rows": len(prepared),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sources": hashes,
        "positive_injections": 0, "test_access": False}, indent=2) + "\n")
    print(json.dumps({"prepared_validation_rows": len(prepared), "path": str(path)}), flush=True)


def score(checkpoint, input_path, output):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new evaluation output directory")
    output.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    data = rows(input_path)
    metrics = evaluate(model, tokenizer, encode_all(tokenizer, data), data, "cpu", output, "validation")
    result = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(
        (checkpoint / "model.safetensors").read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(), "metrics": metrics,
        "device": "cpu", "precision": "fp32", "timing": "Batched accuracy evaluation; not a latency qualification"}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/kaggle/local-validation.jsonl")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if not args.worktree:
            parser.error("--prepare requires --worktree")
        prepare(args.worktree, args.input)
    else:
        if not args.checkpoint or not args.output:
            parser.error("Evaluation requires --checkpoint and --output")
        score(args.checkpoint, args.input, args.output)
