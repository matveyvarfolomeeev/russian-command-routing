"""Evaluate Laya on the frozen 300-row real/synthetic validation set, offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from train_laya_kaggle import encode_all, evaluate, load_model
from train_nanojev_kaggle import rows


def score(args):
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    sys.path.insert(0, str(args.worktree / "src"))
    from laya_cpu_weights import cpu_weights

    with (args.checkpoint / "model.safetensors").open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    # Validate/widen before heavy imports and avoid a private copy of all parameters.
    weights, storage = cpu_weights(args.checkpoint / "model.safetensors", checkpoint_sha,
                                  Path(__file__).resolve().parents[1] / "artifacts/kaggle/float32-cache")
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    # Parsing the tokenizer's large JSON also needs temporary memory: do it before weights.
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer", local_files_only=True)
    data = rows(args.input)
    encoded, stats = encode_all(tokenizer, data)
    print(json.dumps({"stage": "encoded", "rows": len(data), "checkpoint_sha256": checkpoint_sha}), flush=True)
    with patch("jev_launcher.local_weights.CpuSafeFile", return_value=weights), \
            patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer):
        model, _ = load_model(args.checkpoint, "cpu")
    model.requires_grad_(False)
    print(json.dumps({"stage": "model_loaded", "precision": "fp32"}), flush=True)
    result = evaluate(model, tokenizer, encoded, data, "cpu", args.output, "validation", micro_batch=1)
    result.update(checkpoint_sha256=checkpoint_sha,
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  encoding=stats, device="cpu", precision="fp32", weight_storage=storage,
                  timing="accuracy evaluation; no latency qualification")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/kaggle/local-validation.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    score(parser.parse_args())
