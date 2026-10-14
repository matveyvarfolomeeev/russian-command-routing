"""Native Laya choice fine-tuning with a clean prefix and validation stopping."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

if __package__:
    from .train_nanojev_kaggle import INSTRUCTION, digest, plan_order, rows, validate_inputs
else:
    from train_nanojev_kaggle import INSTRUCTION, digest, plan_order, rows, validate_inputs


def selection_loss(predictions):
    """Group-balanced NLL of reachable labels, then equal weight per profile."""
    profiles = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        if row["retrieval_covered"]:
            p = row["scores"][row["candidate_ids"].index(row["target"])]
            profiles[row["profile"]][row["group_id"]].append(-math.log(max(p, 1e-12)))
    expected = {row["profile"] for row in predictions}
    if set(profiles) != expected:
        raise ValueError("A validation profile has no reachable labels")
    losses = {profile: sum(sum(v) / len(v) for v in groups.values()) / len(groups)
              for profile, groups in profiles.items()}
    return sum(losses.values()) / len(losses), losses


class ValidationStopper:
    def __init__(self, initial_loss, patience=3, min_delta=0.005):
        if not math.isfinite(initial_loss):
            raise ValueError("Non-finite validation loss")
        self.best_loss, self.best_step = initial_loss, 0
        self.wait, self.patience, self.min_delta = 0, patience, min_delta

    def update(self, value, step):
        if not math.isfinite(value):
            raise ValueError("Non-finite validation loss")
        improved = value < self.best_loss - self.min_delta
        if improved:
            self.best_loss, self.best_step, self.wait = value, step, 0
        else:
            self.wait += 1
        return improved, self.wait >= self.patience


def load_model(checkpoint, device):
    import torch
    from accelerate import init_empty_weights
    from laya import common
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    torch.backends.mha.set_fastpath_enabled(False)
    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text(encoding="utf-8"))
    with init_empty_weights(include_buffers=False):
        model = common.build_model(cfg, encoder_dir=str(checkpoint / "encoder"), pretrained=False)
    if os.name == "nt" and device == "cpu":
        # Reuse the project's validated Windows loader, supplied by the local evaluator.
        from jev_launcher.local_weights import CpuSafeFile

        with CpuSafeFile(checkpoint / "model.safetensors") as weights:
            values = {key: weights.get_tensor(key) for key in weights.keys()}
    else:
        values = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    model.load_state_dict(values, strict=True, assign=True)
    del values
    model = model.float().to(device).eval()
    model.encoder.config.reference_compile = False
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    return model, tokenizer


def encode_all(tokenizer, data):
    from laya import common

    @common._reuse_question_tokens
    def encode():
        encoded, stats = [], Counter()
        lengths = {}
        for row in data:
            question = {"t": "choice", "ins": INSTRUCTION,
                        "crit": {f"o{i}": doc for i, doc in enumerate(row["documents"])}}
            ids, markers, options, state = common.build_sequence(
                tokenizer, row["query"], question, max_len=8192, head_max_len=8192,
                return_stats=True, return_truncation_stats=True,
            )
            if state["truncated"] or len(markers) != 20 or options["options_distinct"] != 20:
                raise ValueError(f"Laya input loses query or choices: {row['id']}")
            if options["tokens_per_option"] is not None:
                raise ValueError("An additional head budget cut changed the native input")
            for text in common.render_options(question):
                if text not in lengths:
                    lengths[text] = len(common.encode_text(
                        tokenizer, " " + text.replace(tokenizer.mask_token, " "),
                        add_special_tokens=False)["input_ids"])
                stats["metadata_capped_options"] += lengths[text] > 48
                stats["metadata_tokens_dropped"] += max(0, lengths[text] - 48)
            stats["max_sequence_tokens"] = max(stats["max_sequence_tokens"], len(ids))
            encoded.append({"ids": ids, "markers": markers, "qtype": common.QTYPES["choice"]})
        return encoded, dict(stats)

    return encode()


def batch(tokenizer, encoded, indices, device):
    from laya.common import collate_items

    values = collate_items([[encoded[i]] for i in indices], tokenizer.pad_token_id)
    return {key: values[key].to(device) for key in
            ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}


def evaluate(model, tokenizer, encoded, data, device, output, name, micro_batch=2):
    import torch

    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data), micro_batch):
            indices = list(range(start, min(start + micro_batch, len(data))))
            logits, _ = model(**batch(tokenizer, encoded, indices, device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite validation logits")
            for index, scores in zip(indices, logits.float().softmax(-1).cpu().tolist(), strict=True):
                row = data[index]
                predicted = row["candidate_ids"][max(range(20), key=scores.__getitem__)]
                predictions.append({key: row[key] for key in
                                    ("id", "profile", "group_id", "target", "candidate_ids", "retrieval_covered")}
                                   | {"predicted": predicted, "correct": predicted == row["target"],
                                      "scores": scores})
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


def save_checkpoint(model, original, destination):
    from safetensors.torch import save_file

    destination.mkdir(parents=True, exist_ok=True)
    # Serialize CPU tensors; parameters/optimizer stay FP32 during AMP training.
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
              str(destination / "model.safetensors"))
    for folder in ("encoder", "tokenizer"):
        shutil.copytree(original / folder, destination / folder, dirs_exist_ok=True)
    cfg = json.loads((original / "rl_agent_config.json").read_text(encoding="utf-8"))
    cfg.update(temperature=[1.0, 1.0, 1.0], temperature_by_options={})
    (destination / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copy2(original / "README.md", destination / "ORIGINAL_MODEL_CARD.md")
    (destination / "finetuning.json").write_text(json.dumps({
        "task": "launcher choice", "calibration": "uncalibrated_after_finetuning",
        "act_head": "unchanged weights; not trained or validated for escalation",
    }, indent=2) + "\n")


def run(args):
    import torch
    from transformers import get_linear_schedule_with_warmup

    started, output = time.monotonic(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.smoke:
        raise RuntimeError("The full Laya experiment requires Kaggle GPU")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.input / "laya-input-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((args.input / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input hash mismatch: {name}")
    original = args.input / "laya-multilingual"
    extracted = not original.is_dir()
    if extracted:
        original = output / "original"
        with ZipFile(args.input / "laya-multilingual.zip") as archive:
            archive.extractall(original)
    for name, expected in manifest["checkpoint_files"].items():
        if hashlib.sha256((original / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
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
            "source_counts": dict(Counter(r["source"] for r in data)),
            "sample_multiset_sha256": digest(sorted(r["id"] for r in data)),
            "ordered_ids_sha256": digest([data[i]["id"] for i in order]),
            "input_manifest": manifest, "micro_batch": 1, "gradient_accumulation": 16,
            "encoder_lr": 1e-5, "head_lr": 3e-5, "weight_decay": 0.01,
            "warmup_fraction": 0.06, "gradient_clip": 1.0,
            "precision": "AMP FP16 training; FP32 weights/optimizer/validation",
            "validation_interval": 142, "patience": 3, "min_delta": 0.005,
            "selection": "mean per-profile per-group conditional NLL; synthetic validation only",
            "native_option_token_cap": 48, "max_len": 8192, "head_max_len": 8192,
            "act_head": "frozen; no escalation training or confidence calibration",
            "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "torch": torch.__version__, "smoke": args.smoke}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan), flush=True)
    model, tokenizer = load_model(original, device)
    encoded, train_stats = encode_all(tokenizer, data)
    validation_encoded, validation_stats = encode_all(tokenizer, validation)
    stats = {"train": train_stats, "validation": validation_stats,
             "parameters": sum(p.numel() for p in model.parameters())}
    (output / "encoding.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({"encoding": stats}), flush=True)
    baseline = evaluate(model, tokenizer, validation_encoded, validation, device, output, "baseline")
    stopper = ValidationStopper(baseline["selection_loss"])
    best = baseline
    save_checkpoint(model, original, output / "checkpoint")
    for param in model.act_head.parameters():
        param.requires_grad_(False)
    model.encoder.gradient_checkpointing_enable()
    model.head_checkpointing = True
    encoder = list(model.encoder.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": 1e-5}, {"params": head, "lr": 3e-5}], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, round(total_steps * 0.06), total_steps)
    scaler = torch.amp.GradScaler("cuda", init_scale=256, enabled=device == "cuda")
    labels = [r["candidate_ids"].index(r["target"]) for r in data]
    reason, step, last = "maximum_one_pass", 0, baseline
    with (output / "training.jsonl").open("w", encoding="utf-8") as ledger, \
            (output / "validation.jsonl").open("w", encoding="utf-8") as checks:
        checks.write(json.dumps({"step": 0, **baseline}) + "\n")
        for start in range(0, len(order), 16):
            model.train()
            indices = order[start:start + 16]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in indices:
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                    logits, _ = model(**batch(tokenizer, encoded, [index], device))
                    loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([labels[index]], device=device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss / len(indices)).backward()
                losses.append(float(loss.detach()))
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite gradient norm; training stopped without claiming an update")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            entry = {"step": step, "loss": sum(losses) / len(losses), "lr": scheduler.get_last_lr(),
                     "grad_norm": float(norm), "amp_scale": scaler.get_scale(),
                     "elapsed_seconds": time.monotonic() - started,
                     "sources": dict(Counter(data[i]["source"] for i in indices))}
            ledger.write(json.dumps(entry) + "\n")
            if step == 1 or step % 50 == 0 or step == total_steps:
                ledger.flush()
                print(json.dumps(entry), flush=True)
            if step % plan["validation_interval"] == 0 or step == total_steps:
                last = evaluate(model, tokenizer, validation_encoded, validation, device, output, f"step-{step}")
                improved, stop = stopper.update(last["selection_loss"], step)
                checks.write(json.dumps({"step": step, "improved": improved, "wait": stopper.wait, **last}) + "\n")
                checks.flush()
                if improved:
                    best = last
                    save_checkpoint(model, original, output / "checkpoint")
                if stop:
                    reason = "validation_patience"
                    break
    save_checkpoint(model, original, output / "last-checkpoint")
    result = {"plan": plan, "steps": step, "stop_reason": reason,
              "best_step": stopper.best_step, "baseline": baseline, "best": best, "last": last,
              "elapsed_seconds": time.monotonic() - started,
              "checkpoint_sha256": hashlib.sha256((output / "checkpoint/model.safetensors").read_bytes()).hexdigest(),
              "last_checkpoint_sha256": hashlib.sha256((output / "last-checkpoint/model.safetensors").read_bytes()).hexdigest()}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if extracted:
        if original.parent != output or original.is_symlink():
            raise ValueError("Original checkpoint escaped output")
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
