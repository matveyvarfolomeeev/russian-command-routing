"""Evaluate unmerged FP32 Kev on frozen local inputs without copying its base."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from train_kev_kaggle import checkpoint_hashes, encode_all, evaluate, load_model, native_code, sha
from train_nanojev_kaggle import plan_order, rows

ROOT = Path(__file__).resolve().parents[1]


def score(args):
    os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    sys.path.insert(0, str(args.worktree / "src"))
    from laya_cpu_weights import cpu_weights

    original = args.worktree / "models/rerank/kev-08"
    checkpoint = args.checkpoint or original
    pinned = json.loads((original / "manifest.json").read_text())
    for name, record in pinned["files"].items():
        if sha(original / name.removeprefix("kev-08/")) != record["sha256"]:
            raise ValueError(f"Original pinned Kev file changed: {name}")
    native_code(original)
    base_file = original / "base/model.safetensors-00001-of-00001.safetensors"
    base_sha = pinned["lfs_sha256"]["kev-08/base/model.safetensors-00001-of-00001.safetensors"]
    weights, storage = cpu_weights(base_file, base_sha, ROOT / "artifacts/kaggle/float32-cache")
    import torch
    from jev_launcher.local_weights import CpuSafeFile
    from kev.model import load_tokenizer
    from transformers import core_model_loading

    torch.set_num_threads(4)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(str(original / "base"))
    data = rows(args.input)
    if args.gradient_preflight:
        if any(row.get("source") not in ("clean", "stt") for row in data):
            raise ValueError("Gradient preflight requires training rows; never use validation/test labels")
        order, _ = plan_order(data, "clean_first", 20261005)
        data = [data[order[0]]]
    elif args.preflight:
        data = [data[0]]
    encoded, stats = encode_all(tokenizer, data)
    print(json.dumps({"stage": "encoded", "rows": len(data), "encoding": stats}), flush=True)

    def safe_open(filename, framework="pt", device="cpu"):
        if Path(filename).resolve() == base_file.resolve():
            if framework != "pt" or device not in (None, "cpu"):
                raise ValueError("Mapped weights are CPU-only")
            return weights
        return CpuSafeFile(filename, framework=framework, device=device)

    with patch("transformers.modeling_utils.safe_open", safe_open), \
            patch("safetensors.torch.safe_open", safe_open), \
            patch.object(core_model_loading, "GLOBAL_WORKERS", 1):
        model, _, meta, model_stats = load_model(original, "cpu", tokenizer, checkpoint)
    if meta.base != pinned["spec"]["base_repo"] or meta.base_revision != pinned["spec"]["base_revision"]:
        raise ValueError("Checkpoint base revision differs from the frozen original")
    print(json.dumps({"stage": "model_loaded", **model_stats}), flush=True)
    if args.gradient_preflight:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        parameters = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in parameters]
        source_hashes = checkpoint_hashes(checkpoint)
        model.train()
        target = data[0]["candidate_ids"].index(data[0]["target"])
        loss = torch.nn.functional.cross_entropy(model(encoded[0])[0][None], torch.tensor([target]))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("Non-finite FP32 numerical preflight")
        torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.01).step()
        changed = sum(not torch.equal(old, p) for old, p in zip(before, parameters, strict=True))
        if not changed or source_hashes != checkpoint_hashes(checkpoint):
            raise ValueError("Preflight failed to change private trainables or modified source files")
        result = {"numerical_preflight_only": True, "training_rows": 1, "optimizer_updates": 1,
                  "loss": float(loss.detach()), "grad_norm": float(norm), "changed_trainable_tensors": changed,
                  "source_checkpoint_unchanged": True}
        print(json.dumps(result), flush=True)
    else:
        model.requires_grad_(False)
        result = evaluate(model, encoded, data, args.output, "validation")
    result.update(checkpoint_sha256=checkpoint_hashes(checkpoint), base_sha256=base_sha,
                  input_sha256=sha(args.input), encoding=stats, model=model_stats,
                  device="cpu", precision="fp32", adapter_merge=False, temperature=1.0,
                  weight_storage=storage, preflight=args.preflight or args.gradient_preflight,
                  timing="accuracy evaluation; no latency qualification")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/kaggle/local-validation.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--gradient-preflight", action="store_true")
    score(parser.parse_args())
