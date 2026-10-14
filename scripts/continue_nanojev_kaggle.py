"""One extra Nano-Jev pass from trained weights, with validation selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

if __package__:
    from .train_laya_kaggle import ValidationStopper, selection_loss
    from .train_nanojev_kaggle import (
        INSTRUCTION,
        batch,
        digest,
        encode_all,
        evaluate,
        plan_order,
        rows,
        validate_inputs,
    )
else:
    from train_laya_kaggle import ValidationStopper, selection_loss
    from train_nanojev_kaggle import (
        INSTRUCTION,
        batch,
        digest,
        encode_all,
        evaluate,
        plan_order,
        rows,
        validate_inputs,
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(checkpoint, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model, info = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, output_loading_info=True,
    )
    if any(info.get(key) for key in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint did not load exactly: {info}")
    if model.config.num_labels != 1:
        raise ValueError("Nano requires its native single-logit head")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    return model.float().to(device).eval(), tokenizer


def score(model, tokenizer, encoded, data, device, output, name):
    metrics = evaluate(model, tokenizer, encoded, data, device, output, name)
    predictions = rows(output / f"{name}-predictions.jsonl")
    for row in predictions:
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in row["scores"]):
            raise RuntimeError("Non-finite validation probabilities")
    loss, losses = selection_loss(predictions)
    result = {"metrics": metrics, "selection_loss": loss, "profile_losses": losses}
    print(json.dumps({"selection": name, **result}), flush=True)
    return result


def save_checkpoint(model, tokenizer, original, destination):
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    for name in ("nanojev_config.json", "native_model.py", "ORIGINAL_MODEL_CARD.md"):
        shutil.copy2(original / name, destination / name)
    (destination / "calibration.json").write_text(json.dumps({
        "temperature": 1.0, "status": "uncalibrated_after_finetuning",
    }) + "\n", encoding="utf-8")


def run(args):
    import torch
    import transformers
    from transformers import get_linear_schedule_with_warmup

    started, output = time.monotonic(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.smoke:
        raise RuntimeError("The full continuation requires a Kaggle GPU")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.input / "nano-continuation-input-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(args.input / name) != expected:
            raise ValueError(f"Frozen input or trainer changed: {name}")
    original = args.input / "nanojev-trained"
    extracted = not original.is_dir()
    if extracted:
        original = output / "initial-checkpoint"
        archive_path = args.input / "nanojev-trained.zip"
        if sha(archive_path) != manifest["initial_model_archive_sha256"]:
            raise ValueError("Initial archive changed")
        with ZipFile(archive_path) as archive:
            if set(archive.namelist()) != set(manifest["checkpoint_files"]):
                raise ValueError("Unexpected archive entries")
            for name in archive.namelist():
                if Path(name).name != name:
                    raise ValueError("Checkpoint archive must contain only flat files")
            archive.extractall(original)
    for name, expected in manifest["checkpoint_files"].items():
        if sha(original / name) != expected:
            raise ValueError(f"Initial checkpoint changed: {name}")
    if sha(original / "model.safetensors") != manifest["initial_weight_sha256"]:
        raise ValueError("Starting weights differ from the selected trained Nano")
    data, validation = rows(args.input / "prepared-train.jsonl"), rows(args.input / "prepared-validation.jsonl")
    validate_inputs(data, validation)
    if args.smoke:
        data = [r for r in data if r["source"] == "clean"][:16] + [r for r in data if r["source"] == "stt"][:16]
        validation = [next(r for r in validation if r["profile"] == p)
                      for p in sorted({r["profile"] for r in validation})]
    order, _ = plan_order(data, "mixed", args.seed)
    maximum_steps = math.ceil(len(data) / 16)
    plan = {
        "schedule": "mixed", "seed": args.seed, "rows": len(data), "maximum_epochs": 1,
        "maximum_steps": maximum_steps, "clean_prefix_rows": 0,
        "source_counts": dict(Counter(r["source"] for r in data)),
        "sample_multiset_sha256": digest(sorted(r["id"] for r in data)),
        "ordered_ids_sha256": digest([data[i]["id"] for i in order]),
        "instruction_sha256": digest(INSTRUCTION), "input_manifest": manifest,
        "initial_weight_sha256": manifest["initial_weight_sha256"],
        "micro_batch": 1 if device == "cpu" else 4, "effective_batch": 16,
        "learning_rate": 1e-5, "weight_decay": 0.01, "warmup_fraction": 0.06,
        "optimizer": "fresh AdamW; previous run did not save optimizer state",
        "gradient_clip": 1.0, "precision": "FP32 weights, training, optimizer and validation",
        "validation_interval": 1 if args.smoke else 142, "patience": 3, "min_delta": 0.005,
        "selection": "mean per-profile per-group reachable-label NLL; original synthetic150 only",
        "initial_checkpoint_eligible": True, "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch": torch.__version__, "transformers": transformers.__version__, "smoke": args.smoke,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan), flush=True)
    model, tokenizer = load_model(original, device)
    encoded, encoded_validation = encode_all(tokenizer, data), encode_all(tokenizer, validation)
    stats = {"parameters": sum(p.numel() for p in model.parameters()),
             "train_pairs": len(data) * 20, "validation_pairs": len(validation) * 20,
             "max_train_tokens": max(len(p["input_ids"]) for item in encoded for p in item),
             "max_validation_tokens": max(len(p["input_ids"]) for item in encoded_validation for p in item),
             "truncated_pairs": 0}
    (output / "encoding.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"encoding": stats}), flush=True)
    baseline = score(model, tokenizer, encoded_validation, validation, device, output, "baseline")
    best, last = baseline, baseline
    stopper = ValidationStopper(baseline["selection_loss"], plan["patience"], plan["min_delta"])
    # Copy original bytes, so best_step=0 retains the exact initial artifact.
    shutil.copytree(original, output / "checkpoint")
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, round(maximum_steps * 0.06), maximum_steps)
    labels = [r["candidate_ids"].index(r["target"]) for r in data]
    step, reason = 0, "maximum_one_pass"
    with (output / "training.jsonl").open("w", encoding="utf-8") as ledger, \
            (output / "validation.jsonl").open("w", encoding="utf-8") as checks:
        checks.write(json.dumps({"step": 0, **baseline}) + "\n")
        checks.flush()
        for start in range(0, len(order), 16):
            model.train()
            indices = order[start:start + 16]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro_start in range(0, len(indices), plan["micro_batch"]):
                local = indices[micro_start:micro_start + plan["micro_batch"]]
                logits = model(**batch(tokenizer, encoded, local, device)).logits.reshape(len(local), 20)
                loss = torch.nn.functional.cross_entropy(
                    logits.float(), torch.tensor([labels[i] for i in local], device=device),
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                (loss * len(local) / len(indices)).backward()
                losses.append((float(loss.detach()), len(local)))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan["gradient_clip"])
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite gradients; no update claimed")
            optimizer.step()
            scheduler.step()
            step += 1
            entry = {"step": step, "loss": sum(v * n for v, n in losses) / len(indices),
                     "lr": scheduler.get_last_lr(), "grad_norm": float(norm),
                     "elapsed_seconds": time.monotonic() - started,
                     "sources": dict(Counter(data[i]["source"] for i in indices))}
            ledger.write(json.dumps(entry) + "\n")
            if step == 1 or step % 50 == 0 or step == maximum_steps:
                ledger.flush()
                print(json.dumps(entry), flush=True)
            if step % plan["validation_interval"] == 0 or step == maximum_steps:
                last = score(model, tokenizer, encoded_validation, validation, device, output, f"step-{step}")
                improved, stop = stopper.update(last["selection_loss"], step)
                checks.write(json.dumps({"step": step, "improved": improved, "wait": stopper.wait, **last}) + "\n")
                checks.flush()
                if improved:
                    best = last
                    save_checkpoint(model, tokenizer, original, output / "checkpoint")
                if stop:
                    reason = "validation_patience"
                    break
    save_checkpoint(model, tokenizer, original, output / "last-checkpoint")
    result = {"plan": plan, "steps": step, "best_step": stopper.best_step, "stop_reason": reason,
              "baseline": baseline, "best": best, "last": last,
              "elapsed_seconds": time.monotonic() - started,
              "checkpoint_sha256": sha(output / "checkpoint/model.safetensors"),
              "last_checkpoint_sha256": sha(output / "last-checkpoint/model.safetensors")}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if extracted:
        if original.resolve().parent != output or original.is_symlink():
            raise ValueError("Initial checkpoint escaped the run output")
        shutil.rmtree(original)
    print(json.dumps({"completed": True, "steps": step, "best_step": stopper.best_step,
                      "stop_reason": reason, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
