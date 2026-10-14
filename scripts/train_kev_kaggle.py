"""Warm-start native Kev LoRA/pointer training on frozen E5 choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from zipfile import ZipFile

if __package__:
    from .train_laya_kaggle import ValidationStopper, selection_loss
    from .train_nanojev_kaggle import INSTRUCTION, digest, plan_order, rows, validate_inputs
else:
    from train_laya_kaggle import ValidationStopper, selection_loss
    from train_nanojev_kaggle import INSTRUCTION, digest, plan_order, rows, validate_inputs


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def native_code(original):
    sys.path.insert(0, str(original / "code"))


def encode_all(tokenizer, data):
    from kev.api import SystemOneRequest, to_record
    from kev.model import encode

    encoded, stats = [], Counter()
    for row in data:
        request = SystemOneRequest.model_validate({
            "state": row["query"], "questions": {"open": {
                "type": "choice", "instructions": INSTRUCTION,
                "criteria": {f"o{i}": doc for i, doc in enumerate(row["documents"])},
            }},
        })
        record, metadata = to_record(request)
        # The native dummy label is deliberately independent of target/object IDs.
        item = encode(tokenizer, record, max_state=8192, max_branch=8192, strict=True)
        if item["state_truncated"] or len(item["opt_idx"]) != 1 or len(set(item["opt_idx"][0])) != 20:
            raise ValueError(f"Kev input loses query or options: {row['id']}")
        if metadata[0]["keys"] != [f"o{i}" for i in range(20)]:
            raise ValueError("Native option order changed")
        item.pop("labels")
        stats["max_sequence_tokens"] = max(stats["max_sequence_tokens"], len(item["ids"]))
        stats["max_query_tokens"] = max(stats["max_query_tokens"], item["state_tokens"])
        encoded.append(item)
    return encoded, {**dict(stats), "query_truncations": 0, "metadata_truncations": 0,
                     "choices_per_query": 20, "input_label": "removed; supervision is separate"}


def load_model(original, device, tokenizer=None, checkpoint_dir=None):
    import torch
    from kev.checkpoint import Checkpoint, Meta
    from kev.model import DecisionModel, load_tokenizer
    from transformers import AutoConfig, AutoModelForCausalLM

    checkpoint = Checkpoint(checkpoint_dir or original)
    meta = Meta.from_dict(checkpoint.meta.to_dict())
    if meta.weights != "lora" or meta.lora != 16 or meta.special_embeddings or meta.option_isolation:
        raise ValueError("Unexpected original Kev architecture")
    tokenizer = tokenizer or load_tokenizer(str(original / "base"))
    # This base has a nested text_config recorded as BF16. Transformers can
    # retain that dtype even with dtype=float32 on the outer multimodal config.
    config = AutoConfig.from_pretrained(original / "base", local_files_only=True)
    config.dtype = torch.float32
    config.get_text_config().dtype = torch.float32
    from_pretrained = AutoModelForCausalLM.from_pretrained

    def fp32_base(*values, **options):
        return from_pretrained(*values, **options, config=config, local_files_only=True)

    with patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=fp32_base):
        model = DecisionModel(str(original / "base"), tokenizer, device, lora=meta.lora,
                              head_dim=meta.head_dim, dtype=torch.float32, attn="sdpa" if device == "cuda" else "eager")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Kev base/LoRA/head must remain FP32")
    # warm_start only needs suite.digest. The unrelated suite builder imports
    # Unix fcntl and external data generators; keep its file-hash contract here.
    hashing = ModuleType("kev.suite")
    hashing.digest = sha
    with patch.dict(sys.modules, {"kev.suite": hashing}):
        provenance = checkpoint.warm_start(model, meta)
    model.head.temperature = 1.0
    model.lm.config.use_cache = False
    model.eval()
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(not (name.startswith("head.") or ".lora_" in name) for name in trainable):
        raise ValueError("Training would change weights outside LoRA and pointer head")
    return model, tokenizer, meta, {"initialization": provenance,
                                   "trainable_parameters": sum(trainable.values()),
                                   "trainable_tensors": len(trainable),
                                   "parameters": sum(p.numel() for p in model.parameters()),
                                   "temperature": 1.0, "adapter_merge": False}


def evaluate(model, encoded, data, output, name):
    import torch

    model.eval()
    predictions = []
    with torch.inference_mode():
        for position, (item, row) in enumerate(zip(encoded, data, strict=True), 1):
            logits = model(item)[0].float()
            if logits.shape != (20,) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid Kev validation logits")
            scores = logits.softmax(-1).cpu().tolist()
            predicted = row["candidate_ids"][max(range(20), key=scores.__getitem__)]
            predictions.append({key: row[key] for key in
                                ("id", "profile", "group_id", "target", "candidate_ids", "retrieval_covered")}
                               | {"predicted": predicted, "correct": predicted == row["target"], "scores": scores})
            if position % 50 == 0:
                print(json.dumps({"evaluation": name, "evaluated": position, "total": len(data)}), flush=True)
    metrics = {}
    for profile in sorted({row["profile"] for row in predictions}):
        local = [row for row in predictions if row["profile"] == profile]
        correct, covered = sum(r["correct"] for r in local), sum(r["retrieval_covered"] for r in local)
        metrics[profile] = {"rows": len(local), "correct": correct, "accuracy": correct / len(local),
                            "candidate_recall": covered / len(local),
                            "conditional_accuracy": correct / covered if covered else None}
    loss, profile_losses = selection_loss(predictions)
    result = {"metrics": metrics, "selection_loss": loss, "profile_losses": profile_losses}
    (output / f"{name}-predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    print(json.dumps({"evaluation": name, **result}), flush=True)
    return result


def save_checkpoint(model, tokenizer, meta, original, destination, step):
    from kev.checkpoint import Meta, write_meta

    destination.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(destination, safe_serialization=True)
    # Keep portable original base pins, rather than the Kaggle extraction directory.
    config_path = destination / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(base_model_name_or_path=meta.base, revision=meta.base_revision)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    saved = Meta.from_dict(meta.to_dict())
    saved.head = {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()}
    saved.temperature = 1.0
    saved.extra = {**saved.extra, "launcher_finetuning": {
        "task": "launcher choice", "step": step, "adapter_merge": False,
        "calibration": "uncalibrated_after_finetuning", "other_tasks": "not trained or validated",
    }}
    write_meta(destination, saved)
    tokenizer.save_pretrained(destination)
    shutil.copy2(original / "README.md", destination / "ORIGINAL_MODEL_CARD.md")


def checkpoint_hashes(path):
    return {name: sha(path / name) for name in ("adapter_model.safetensors", "adapter_config.json", "head.pt")}


def run(args):
    import torch
    from transformers import get_linear_schedule_with_warmup

    started, output = time.monotonic(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.smoke:
        raise RuntimeError("The full Kev experiment requires Kaggle GPU")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = json.loads((args.input / "kev-input-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(args.input / name) != expected:
            raise ValueError(f"Input hash mismatch: {name}")
    original = args.input / "kev-original"
    if not original.is_dir():
        original = output / "original"
        with ZipFile(args.input / "kev-original.zip") as archive:
            for entry in archive.infolist():
                if not (original / entry.filename).resolve().is_relative_to(original.resolve()):
                    raise ValueError("Archive path escaped original directory")
            archive.extractall(original)
    for name, expected in manifest["checkpoint_files"].items():
        if sha(original / name) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    native_code(original)
    data, validation = rows(args.input / "prepared-train.jsonl"), rows(args.input / "prepared-validation.jsonl")
    validate_inputs(data, validation)
    if args.smoke:
        data = [r for r in data if r["source"] == "clean"][:16] + [r for r in data if r["source"] == "stt"][:16]
        validation = [next(r for r in validation if r["profile"] == profile)
                      for profile in sorted({r["profile"] for r in validation})]
    order, prefix = plan_order(data, "clean_first", args.seed)
    total_steps = math.ceil(len(data) / 16)
    plan = {"schedule": "clean_first", "seed": args.seed, "rows": len(data), "maximum_epochs": 1,
            "maximum_steps": total_steps, "clean_prefix_rows": prefix,
            "sample_multiset_sha256": digest(sorted(r["id"] for r in data)),
            "ordered_ids_sha256": digest([data[i]["id"] for i in order]),
            "source_counts": dict(Counter(r["source"] for r in data)), "input_manifest": manifest,
            "micro_batch": 1, "gradient_accumulation": 16, "lora_lr": 2e-5, "head_lr": 2e-5,
            "weight_decay": 0.01, "warmup_fraction": 0.06, "gradient_clip": 1.0,
            "precision": "FP32 training, frozen base, LoRA/head, optimizer and validation",
            "validation_interval": 142, "patience": 3, "min_delta": 0.005,
            "selection": "mean per-profile per-group conditional NLL; synthetic validation only",
            "max_state": 8192, "max_branch": 8192, "adapter_merge": False,
            "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "torch": torch.__version__, "smoke": args.smoke}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan), flush=True)
    model, tokenizer, meta, model_stats = load_model(original, device)
    encoded, train_stats = encode_all(tokenizer, data)
    validation_encoded, validation_stats = encode_all(tokenizer, validation)
    stats = {"train": train_stats, "validation": validation_stats, **model_stats}
    (output / "encoding.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({"encoding": stats}), flush=True)
    baseline = evaluate(model, validation_encoded, validation, output, "baseline")
    stopper, best = ValidationStopper(baseline["selection_loss"]), baseline
    save_checkpoint(model, tokenizer, meta, original, output / "checkpoint", 0)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, round(total_steps * 0.06), total_steps)
    labels = [row["candidate_ids"].index(row["target"]) for row in data]
    reason, step, last = "maximum_one_pass", 0, baseline
    with (output / "training.jsonl").open("w", encoding="utf-8") as ledger, \
            (output / "validation.jsonl").open("w", encoding="utf-8") as checks:
        checks.write(json.dumps({"step": 0, **baseline}) + "\n")
        for start in range(0, len(order), 16):
            model.train()
            indices, losses = order[start:start + 16], []
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                logits = model(encoded[index])[0].float()
                loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([labels[index]], device=device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                (loss / len(indices)).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite gradient norm; no optimizer update claimed")
            optimizer.step()
            scheduler.step()
            step += 1
            entry = {"step": step, "loss": sum(losses) / len(losses), "lr": scheduler.get_last_lr(),
                     "grad_norm": float(norm), "amp_scale": 1.0,
                     "elapsed_seconds": time.monotonic() - started,
                     "sources": dict(Counter(data[i]["source"] for i in indices))}
            ledger.write(json.dumps(entry) + "\n")
            if step == 1 or step % 50 == 0 or step == total_steps:
                ledger.flush()
                print(json.dumps(entry), flush=True)
            if step % plan["validation_interval"] == 0 or step == total_steps:
                last = evaluate(model, validation_encoded, validation, output, f"step-{step}")
                improved, stop = stopper.update(last["selection_loss"], step)
                checks.write(json.dumps({"step": step, "improved": improved, "wait": stopper.wait, **last}) + "\n")
                checks.flush()
                if improved:
                    best = last
                    save_checkpoint(model, tokenizer, meta, original, output / "checkpoint", step)
                if stop:
                    reason = "validation_patience"
                    break
    save_checkpoint(model, tokenizer, meta, original, output / "last-checkpoint", step)
    result = {"plan": plan, "steps": step, "stop_reason": reason, "best_step": stopper.best_step,
              "baseline": baseline, "best": best, "last": last,
              "elapsed_seconds": time.monotonic() - started,
              "checkpoint_sha256": checkpoint_hashes(output / "checkpoint"),
              "last_checkpoint_sha256": checkpoint_hashes(output / "last-checkpoint")}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if original.parent == output:
        if original.is_symlink() or not original.resolve().is_relative_to(output):
            raise ValueError("Extracted original escaped output directory")
        shutil.rmtree(original)
    print(json.dumps({"completed": True, "steps": step, "best_step": stopper.best_step,
                      "stop_reason": reason, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
