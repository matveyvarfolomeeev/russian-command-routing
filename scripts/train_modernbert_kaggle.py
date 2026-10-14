"""Train pinned Russian scalar pair scorers for launcher top-20 choice on Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

if __package__:
    from .train_laya_kaggle import ValidationStopper, selection_loss
    from .train_nanojev_kaggle import batch, digest, plan_order, rows, validate_inputs
else:
    from train_laya_kaggle import ValidationStopper, selection_loss
    from train_nanojev_kaggle import batch, digest, plan_order, rows, validate_inputs

MAX_PAIR_TOKENS = 2048


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encode_all(tokenizer, data):
    """Native query/card pairs; IDs, labels and provenance never enter the input."""
    encoded, lengths = [], []
    for row in data:
        if len(row["documents"]) != 20:
            raise ValueError("Expected exactly twenty candidate cards")
        values = tokenizer([row["query"]] * 20, row["documents"],
                           padding=False, truncation=False)
        sizes = list(map(len, values["input_ids"]))
        if len(sizes) != 20 or max(sizes) > MAX_PAIR_TOKENS:
            raise ValueError(f"Input would lose candidates or truncate: {row['id']}")
        lengths.extend(sizes)
        encoded.append([{key: value[i] for key, value in values.items()} for i in range(20)])
    return encoded, {"pairs": len(lengths), "max_pair_tokens": max(lengths),
                     "mean_pair_tokens": sum(lengths)/len(lengths), "truncated_pairs": 0}


def load_model(checkpoint, device):
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config.reference_compile = False
    config.compile_model = False
    if config.model_type not in {"modernbert", "bert"} or config.num_labels != 1:
        raise ValueError("Expected a supported encoder with a scalar relevance head")
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, config=config, local_files_only=True, attn_implementation="sdpa",
        output_loading_info=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise ValueError(f"Checkpoint is not loaded exactly: {key}: {loading[key]}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    return model.float().to(device).eval(), tokenizer


def evaluate(model, tokenizer, encoded, data, device, output, name, micro_batch=2):
    import torch

    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data), micro_batch):
            indices = list(range(start, min(start+micro_batch, len(data))))
            logits = model(**batch(tokenizer, encoded, indices, device)).logits.reshape(len(indices), 20)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite evaluation logits")
            for index, scores in zip(indices, logits.float().softmax(-1).cpu().tolist(), strict=True):
                row = data[index]
                prediction = row["candidate_ids"][max(range(20), key=scores.__getitem__)]
                predictions.append({key: row[key] for key in
                                    ("id", "profile", "group_id", "target", "candidate_ids", "retrieval_covered")}
                                   | {"predicted": prediction, "correct": prediction == row["target"],
                                      "scores": scores})
    metrics = {}
    for profile in sorted({r["profile"] for r in predictions}):
        local = [r for r in predictions if r["profile"] == profile]
        correct, covered = sum(r["correct"] for r in local), sum(r["retrieval_covered"] for r in local)
        metrics[profile] = {"rows": len(local), "correct": correct, "accuracy": correct/len(local),
                            "candidate_recall": covered/len(local),
                            "conditional_accuracy": correct/covered if covered else None}
    loss, profile_losses = selection_loss(predictions)
    result = {"metrics": metrics, "selection_loss": loss, "profile_losses": profile_losses}
    (output/f"{name}-predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in predictions), encoding="utf8")
    print(json.dumps({"evaluation": name, **result}), flush=True)
    return result


def save_checkpoint(model, tokenizer, original, destination):
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    shutil.copy2(original/"README.md", destination/"ORIGINAL_MODEL_CARD.md")
    (destination/"finetuning.json").write_text(json.dumps({
        "task": "Russian launcher top-20 choice", "input": "query/card pairs",
        "score_type": "uncalibrated_choice_softmax", "source": str(original),
    }, indent=2)+"\n", encoding="utf8")


def run(args):
    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    import torch
    import transformers
    from transformers import get_linear_schedule_with_warmup

    started, output = time.monotonic(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.smoke:
        raise RuntimeError("Full training requires a Kaggle GPU")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    family = getattr(args, "family", "modernbert")
    if family not in {"modernbert", "rubert"}:
        raise ValueError("Unsupported model family")
    manifest = json.loads((args.input/f"{family}-input-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(args.input/name) != expected:
            raise ValueError(f"Frozen input/code changed: {name}")
    if sha(Path(__file__)) != manifest["files"]["train_modernbert_kaggle.py"]:
        raise ValueError("Executing a different training script")
    original = args.input/f"{family}-original"
    if not original.is_dir():
        archive_path = args.input/f"{family}-original.zip"
        if sha(archive_path) != manifest["initial_model_archive_sha256"]:
            raise ValueError("Initial checkpoint archive changed")
        original = output/"original"
        with ZipFile(archive_path) as archive:
            if set(archive.namelist()) != set(manifest["checkpoint_files"]):
                raise ValueError("Unexpected checkpoint archive files")
            for name in archive.namelist():
                if not (original/name).resolve().is_relative_to(original.resolve()):
                    raise ValueError("Checkpoint archive path escapes output")
            archive.extractall(original)
    for name, expected in manifest["checkpoint_files"].items():
        if sha(original/name) != expected:
            raise ValueError(f"Initial model file changed: {name}")
    data, validation = rows(args.input/"prepared-train.jsonl"), rows(args.input/"prepared-validation.jsonl")
    validate_inputs(data, validation)
    if args.smoke:
        data = [r for r in data if r["source"] == "clean"][:16] + [r for r in data if r["source"] == "stt"][:16]
        validation = [next(r for r in validation if r["profile"] == p)
                      for p in sorted({r["profile"] for r in validation})]
    order, prefix = plan_order(data, "clean_first", args.seed)
    maximum_steps = math.ceil(len(data)/16)
    plan = {"model_family": family, "schedule": "clean_first", "seed": args.seed,
            "rows": len(data), "maximum_epochs": 1,
            "maximum_steps": maximum_steps, "clean_prefix_rows": prefix,
            "sample_multiset_sha256": digest(sorted(r["id"] for r in data)),
            "ordered_ids_sha256": digest([data[i]["id"] for i in order]),
            "source_counts": dict(Counter(r["source"] for r in data)), "input_manifest": manifest,
            "learning_rate": 2e-5, "weight_decay": .01, "warmup_fraction": .06,
            "micro_batch": 2 if device == "cuda" else 1, "effective_batch": 16, "gradient_clip": 1.,
            "precision": "FP32 weights, activations, optimizer and validation", "maximum_pair_tokens": MAX_PAIR_TOKENS,
            "validation_interval": 142, "patience": 3, "min_delta": .005,
            "selection": "per-profile per-group reachable-label NLL; baseline eligible; synthetic validation only",
            "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "torch": torch.__version__, "transformers": transformers.__version__, "smoke": args.smoke}
    (output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"stage": "plan", **plan}), flush=True)
    model, tokenizer = load_model(original, device)
    if model.config.model_type != ("bert" if family == "rubert" else "modernbert"):
        raise ValueError("Model architecture differs from the experiment family")
    encoded, train_stats = encode_all(tokenizer, data)
    validation_encoded, validation_stats = encode_all(tokenizer, validation)
    stats = {"train": train_stats, "validation": validation_stats,
             "parameters": sum(p.numel() for p in model.parameters())}
    (output/"encoding.json").write_text(json.dumps(stats, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"stage": "encoding", **stats}), flush=True)
    baseline = evaluate(model, tokenizer, validation_encoded, validation, device, output, "baseline")
    best, last, step, reason = baseline, baseline, 0, "maximum_one_pass"
    stopper = ValidationStopper(baseline["selection_loss"], plan["patience"], plan["min_delta"])
    save_checkpoint(model, tokenizer, original, output/"checkpoint")
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, round(maximum_steps*.06), maximum_steps)
    labels = [r["candidate_ids"].index(r["target"]) for r in data]
    with (output/"training.jsonl").open("w", encoding="utf8") as ledger, \
            (output/"validation.jsonl").open("w", encoding="utf8") as checks:
        checks.write(json.dumps({"step": 0, **baseline})+"\n")
        checks.flush()
        for start in range(0, len(order), 16):
            model.train()
            indices = order[start:start+16]
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.
            for micro_start in range(0, len(indices), plan["micro_batch"]):
                micro = indices[micro_start:micro_start+plan["micro_batch"]]
                logits = model(**batch(tokenizer, encoded, micro, device)).logits.reshape(len(micro), 20)
                loss = torch.nn.functional.cross_entropy(logits.float(),
                                                        torch.tensor([labels[i] for i in micro], device=device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                (loss*len(micro)/len(indices)).backward()
                weighted_loss += float(loss.detach())*len(micro)/len(indices)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite gradients; no optimizer update claimed")
            optimizer.step()
            scheduler.step()
            step += 1
            entry = {"step": step, "loss": weighted_loss, "grad_norm": float(norm),
                     "lr": scheduler.get_last_lr(), "elapsed_seconds": time.monotonic()-started,
                     "sources": dict(Counter(data[i]["source"] for i in indices))}
            ledger.write(json.dumps(entry)+"\n")
            ledger.flush()
            if step == 1 or step % 100 == 0:
                print(json.dumps(entry), flush=True)
            if step % plan["validation_interval"] == 0 or step == maximum_steps:
                last = evaluate(model, tokenizer, validation_encoded, validation, device, output, f"step-{step}")
                improved, stop = stopper.update(last["selection_loss"], step)
                checks.write(json.dumps({"step": step, "improved": improved, "wait": stopper.wait, **last})+"\n")
                checks.flush()
                if improved:
                    best = last
                    save_checkpoint(model, tokenizer, original, output/"checkpoint")
                if stop:
                    reason = "validation_patience"
                    break
    save_checkpoint(model, tokenizer, original, output/"last-checkpoint")
    result = {"plan": plan, "steps": step, "best_step": stopper.best_step, "baseline": baseline,
              "best": best, "last": last, "stop_reason": reason, "elapsed_seconds": time.monotonic()-started,
              "checkpoint_sha256": sha(output/"checkpoint/model.safetensors"),
              "last_checkpoint_sha256": sha(output/"last-checkpoint/model.safetensors")}
    (output/"result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"completed": True, "steps": step, "best_step": stopper.best_step,
                      "stop_reason": reason, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--family", choices=("modernbert", "rubert"), default="modernbert")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
