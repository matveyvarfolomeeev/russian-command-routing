"""Inference-only GPU evaluation using previously pinned models and E5 shortlists."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

from train_kev_kaggle import checkpoint_hashes, sha
from train_nanojev_kaggle import rows


def run(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires GPU")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.input / "input-manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        if sha(args.input / filename) != expected:
            raise ValueError(f"Evaluation input changed: {filename}")
    specs = json.loads((args.input / "fixed-checkpoints.json").read_text())["models"]
    selected_names = args.models or list(specs)
    data = rows(args.input / "prepared-evaluation.jsonl")
    if len(data) != manifest["rows"] or manifest["positive_injections"] != 0:
        raise ValueError("Wrong frozen evaluation")
    hashes = {}

    def find(filename, expected):
        matches = []
        for candidate in args.sources.rglob(filename):
            if candidate not in hashes:
                hashes[candidate] = sha(candidate)
            if hashes[candidate] == expected:
                matches.append(candidate.parent)
        if len(matches) != 1:
            raise ValueError(f"Expected one pinned source for {filename}, got {len(matches)}")
        return matches[0]

    kev_original = None
    for name in selected_names:
        spec = specs[name]
        started = time.monotonic()
        primary = "adapter_model.safetensors" if spec["family"] == "kev" else "model.safetensors"
        expected = spec["sha256"][primary] if spec["family"] == "kev" else spec["sha256"]
        checkpoint = find(primary, expected)
        for filename, expected_hash in spec["files"].items():
            if sha(checkpoint / filename) != expected_hash:
                raise ValueError(f"Pinned checkpoint changed: {name}/{filename}")
        if name == "original-kev":
            kev_original = checkpoint
        output = args.output / name
        output.mkdir()
        print(json.dumps({"stage": "loading", "model": name, "checkpoint": str(checkpoint)}), flush=True)
        if spec["family"] == "nano":
            from train_nanojev_kaggle import encode_all, evaluate

            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            encoded = encode_all(tokenizer, data)
            encoding = {"max_sequence_tokens": max(len(pair["input_ids"]) for row in encoded for pair in row),
                        "query_truncations": 0, "metadata_truncations": 0, "choices_per_query": 20}
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).float().to("cuda")
            model.requires_grad_(False)
            result = {"metrics": evaluate(model, tokenizer, encoded, data, "cuda", output, "validation")}
        elif spec["family"] == "laya":
            from train_laya_kaggle import encode_all, evaluate, load_model

            model, tokenizer = load_model(checkpoint, "cuda")
            model.requires_grad_(False)
            encoded, encoding = encode_all(tokenizer, data)
            result = evaluate(model, tokenizer, encoded, data, "cuda", output, "validation", micro_batch=1)
        else:
            from train_kev_kaggle import encode_all, evaluate, load_model, native_code

            if kev_original is None:
                raise ValueError("Kev original source must be loaded before its adaptation")
            native_code(kev_original)
            from kev.model import load_tokenizer

            tokenizer = load_tokenizer(str(kev_original / "base"))
            encoded, encoding = encode_all(tokenizer, data)
            model, _, _, stats = load_model(kev_original, "cuda", tokenizer, checkpoint)
            model.requires_grad_(False)
            result = evaluate(model, encoded, data, output, "validation")
            base_files = [n for n in specs["original-kev"]["files"] if n.startswith("base/") and n.endswith(".safetensors")]
            if len(base_files) != 1:
                raise ValueError("Expected one pinned Kev base shard")
            result.update(model=stats, base_sha256=specs["original-kev"]["files"][base_files[0]])
            if checkpoint_hashes(checkpoint) != spec["sha256"]:
                raise ValueError("Kev source weights changed during inference")
        if any(parameter.dtype != torch.float32 or parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Inference must use frozen FP32 parameters")
        result.update(checkpoint_sha256=spec["sha256"], input_sha256=manifest["files"]["prepared-evaluation.jsonl"],
                      encoding=encoding, device="cuda", precision="fp32", optimizer_updates=0,
                      elapsed_seconds=time.monotonic() - started, gpu=torch.cuda.get_device_name(),
                      torch=torch.__version__, timing="Accuracy evaluation; no CPU latency qualification")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"stage": "completed", "model": name, "elapsed_seconds": result["elapsed_seconds"],
                          "metrics": result["metrics"]}), flush=True)
        del model, tokenizer, encoded
        gc.collect()
        torch.cuda.empty_cache()
    (args.output / "completion.json").write_text(json.dumps({"completed_models": selected_names, "optimizer_updates": 0,
        "input_manifest_sha256": sha(args.input / "input-manifest.json")}, indent=2) + "\n")


if __name__ == "__main__":
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("original-nano", "mixed-nano", "clean-nano",
                         "original-laya", "trained-laya", "original-kev", "trained-kev"))
    run(parser.parse_args())
