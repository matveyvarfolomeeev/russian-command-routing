"""Adapt the pinned E5 encoder with hard negatives and full-catalog validation."""

from __future__ import annotations

import argparse
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
    from .train_modernbert_kaggle import sha
    from .train_nanojev_kaggle import digest, plan_order, rows
else:
    from train_laya_kaggle import ValidationStopper, selection_loss
    from train_modernbert_kaggle import sha
    from train_nanojev_kaggle import digest, plan_order, rows

TEMPERATURE = .01
MAX_TOKENS = 512


def load_model(checkpoint, device, attention="sdpa"):
    from transformers import AutoModel, AutoTokenizer

    model, loading = AutoModel.from_pretrained(checkpoint, local_files_only=True,
                                              attn_implementation=attention, output_loading_info=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise ValueError(f"E5 was not loaded exactly: {key}: {loading[key]}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    return model.float().to(device).eval(), tokenizer


def mean_pool_normalize(hidden, mask):
    import torch.nn.functional as functional

    values = hidden.masked_fill(~mask[..., None].bool(), 0.)
    pooled = values.sum(1) / mask.sum(1)[..., None]
    return functional.normalize(pooled.float(), p=2, dim=1)


def text_bank(tokenizer, texts):
    """Tokenize each distinct prefixed text once; fail rather than truncate."""
    unique = list(dict.fromkeys(texts))
    encoded = tokenizer(unique, padding=False, truncation=False)
    lengths = list(map(len, encoded["input_ids"]))
    if not unique or min(lengths) < 1 or max(lengths) > MAX_TOKENS:
        raise ValueError("E5 input is empty or exceeds the native 512-token budget")
    records = [{key: values[index] for key, values in encoded.items()} for index in range(len(unique))]
    return {"texts": unique, "encoded": records, "index": {text: i for i, text in enumerate(unique)},
            "stats": {"unique_texts": len(unique), "maximum_tokens": max(lengths),
                      "mean_tokens": sum(lengths) / len(lengths), "truncated_texts": 0}}


def prepare_training(tokenizer, data):
    texts = []
    for row in data:
        if len(row["documents"]) != 20 or len(set(row["candidate_ids"])) != 20:
            raise ValueError("Expected the frozen twenty distinct training candidates")
        if row["target"] not in row["candidate_ids"]:
            raise ValueError("Training target is missing")
        texts.append("query: " + row["query"].strip())
        texts.extend("passage: " + text for text in row["documents"])
    bank = text_bank(tokenizer, texts)
    prepared = [{"query": bank["index"]["query: " + row["query"].strip()],
                 "documents": [bank["index"]["passage: " + text] for text in row["documents"]],
                 "label": row["candidate_ids"].index(row["target"])} for row in data]
    return bank, prepared


def encode_indices(model, tokenizer, bank, indices, device):
    values = tokenizer.pad([bank["encoded"][index] for index in indices], padding=True, return_tensors="pt")
    values = {key: value.to(device) for key, value in values.items()}
    return mean_pool_normalize(model(**values).last_hidden_state, values["attention_mask"])


def training_logits(model, tokenizer, bank, prepared, indices, device):
    import torch

    local = [prepared[index] for index in indices]
    requested = [row["query"] for row in local] + [doc for row in local for doc in row["documents"]]
    unique = list(dict.fromkeys(requested))
    position = {index: i for i, index in enumerate(unique)}
    vectors = encode_indices(model, tokenizer, bank, unique, device)
    queries = vectors[[position[row["query"]] for row in local]]
    documents = vectors[[position[doc] for row in local for doc in row["documents"]]].reshape(len(local), 20, -1)
    # Per-query negatives avoid calling another valid query's target a negative.
    return torch.einsum("bd,bkd->bk", queries, documents) / TEMPERATURE


def prepare_validation(tokenizer, data, catalogs):
    for row in data:
        catalog = catalogs[row["catalog_key"]]
        if len(catalog["candidate_ids"]) != len(catalog["documents"]) or len(set(catalog["candidate_ids"])) != len(catalog["candidate_ids"]):
            raise ValueError("Incomplete or duplicate full-catalog metadata")
        if row["target"] not in catalog["candidate_ids"]:
            raise ValueError("Full validation catalog lacks the target")
    return text_bank(tokenizer, ["query: " + row["query"].strip() for row in data] +
                     ["passage: " + text for catalog in catalogs.values() for text in catalog["documents"]])


def evaluate(model, tokenizer, bank, data, catalogs, device, output, name, batch_size=32):
    import torch

    model.eval()
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(bank["texts"]), batch_size):
            indices = list(range(start, min(start + batch_size, len(bank["texts"]))))
            encoded.append(encode_indices(model, tokenizer, bank, indices, device).cpu())
    vectors = torch.cat(encoded)
    predictions = []
    for row in data:
        catalog = catalogs[row["catalog_key"]]
        ids = catalog["candidate_ids"]
        q = vectors[bank["index"]["query: " + row["query"].strip()]]
        docs = vectors[[bank["index"]["passage: " + text] for text in catalog["documents"]]]
        similarity = docs @ q
        if not torch.isfinite(similarity).all():
            raise RuntimeError("Non-finite retrieval scores")
        scores = similarity.tolist()
        order = sorted(range(len(ids)), key=lambda index: (-scores[index], ids[index]))
        probabilities = (similarity / TEMPERATURE).softmax(-1).tolist()
        pred = {key: row[key] for key in ("id", "profile", "group_id", "target", "catalog_key")}
        pred.update({"candidate_ids": ids, "retrieval_covered": True,
                     "predicted": ids[order[0]], "correct": ids[order[0]] == row["target"],
                     "scores": probabilities, "similarities": scores,
                     "top20": [ids[index] for index in order[:20]],
                     "target_rank": order.index(ids.index(row["target"])) + 1})
        predictions.append(pred)
    metrics = {}
    for profile in sorted({row["profile"] for row in predictions}):
        local = [row for row in predictions if row["profile"] == profile]
        count = len(local)
        metrics[profile] = {"rows": count, "correct": sum(row["correct"] for row in local),
                            "accuracy": sum(row["correct"] for row in local) / count,
                            "recall20": sum(row["target_rank"] <= 20 for row in local) / count,
                            "mrr": sum(1 / row["target_rank"] for row in local) / count}
    loss, profile_losses = selection_loss(predictions)
    result = {"metrics": metrics, "selection_loss": loss, "profile_losses": profile_losses,
              "scope": "full-catalog retrieval; every target present; temperature .01 for NLL"}
    (output / f"{name}-predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    print(json.dumps({"evaluation": name, **result}), flush=True)
    return result


def save_checkpoint(model, tokenizer, original, destination):
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    shutil.copy2(original / "README.md", destination / "ORIGINAL_MODEL_CARD.md")
    (destination / "finetuning.json").write_text(json.dumps({
        "task": "Russian launcher full-catalog retrieval", "pooling": "masked mean + L2 normalization",
        "prefixes": ["query: ", "passage: "], "temperature": TEMPERATURE, "head_added": False,
    }, indent=2) + "\n", encoding="utf-8")


def run(args):
    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    import torch
    import transformers
    from transformers import get_linear_schedule_with_warmup

    started = time.monotonic()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.smoke:
        raise RuntimeError("Full fine-tuning requires the Kaggle GPU")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.input / "e5-input-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha(args.input / name) != expected:
            raise ValueError(f"Changed input/code: {name}")
    if sha(Path(__file__)) != manifest["files"]["train_e5_kaggle.py"]:
        raise ValueError("Executing different E5 training code")
    original = args.input / "e5-original"
    if not original.is_dir():
        original = output / "original"
        source_zip = args.input / "e5-original.zip"
        if sha(source_zip) != manifest["initial_model_archive_sha256"]:
            raise ValueError("Changed original weights archive")
        with ZipFile(source_zip) as archive:
            if set(archive.namelist()) != set(manifest["checkpoint_files"]):
                raise ValueError("Unexpected original checkpoint files")
            for name in archive.namelist():
                if not (original / name).resolve().is_relative_to(original.resolve()):
                    raise ValueError("Archive path escapes output")
            archive.extractall(original)
    for name, expected in manifest["checkpoint_files"].items():
        if sha(original / name) != expected:
            raise ValueError(f"Changed original file: {name}")
    data = rows(args.input / "prepared-train.jsonl")
    validation = rows(args.input / "full-validation.jsonl")
    catalogs = json.loads((args.input / "validation-catalogs.json").read_text(encoding="utf-8"))
    if set(row["id"] for row in data) & set(row["id"] for row in validation) or \
            set(row["group_id"] for row in data) & set(row["group_id"] for row in validation):
        raise ValueError("Training/validation group overlap")
    if args.smoke:
        data = [row for row in data if row["source"] == "clean"][:16] + [row for row in data if row["source"] == "stt"][:16]
        validation = [next(row for row in validation if row["profile"] == profile)
                      for profile in sorted({row["profile"] for row in validation})]
    if len({row["id"] for row in data}) != len(data):
        raise ValueError("Duplicate training IDs")
    order, prefix = plan_order(data, "clean_first", args.seed)
    maximum_steps = math.ceil(len(data) / 16)
    plan = {"model_family": "e5", "schedule": "clean_first", "seed": args.seed, "rows": len(data),
            "maximum_epochs": 1, "maximum_steps": maximum_steps, "clean_prefix_rows": prefix,
            "sample_multiset_sha256": digest(sorted(row["id"] for row in data)),
            "ordered_ids_sha256": digest([data[index]["id"] for index in order]),
            "source_counts": dict(Counter(row["source"] for row in data)), "input_manifest": manifest,
            "learning_rate": 1e-5, "weight_decay": .01, "warmup_fraction": .06,
            "micro_batch": 2 if device == "cuda" else 1, "effective_batch": 16, "gradient_clip": 1.,
            "temperature": TEMPERATURE, "head_added": False, "maximum_tokens": MAX_TOKENS,
            "training_negatives": "frozen original-E5 top20; per-query CE; no in-batch false negatives",
            "precision": "FP32 weights, activations, optimizer and validation",
            "validation_interval": 142, "patience": 3, "min_delta": .005,
            "selection": "full-catalog synthetic150 group/profile NLL; baseline eligible",
            "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "torch": torch.__version__, "transformers": transformers.__version__, "smoke": args.smoke}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "plan", **{key: value for key, value in plan.items() if key != "input_manifest"}}), flush=True)
    model, tokenizer = load_model(original, device)
    train_bank, prepared = prepare_training(tokenizer, data)
    validation_bank = prepare_validation(tokenizer, validation, catalogs)
    (output / "encoding.json").write_text(json.dumps({"train": train_bank["stats"],
        "validation": validation_bank["stats"], "parameters": sum(value.numel() for value in model.parameters())}, indent=2) + "\n", encoding="utf-8")
    baseline = evaluate(model, tokenizer, validation_bank, validation, catalogs, device, output, "baseline")
    best = last = baseline
    stopper = ValidationStopper(baseline["selection_loss"], plan["patience"], plan["min_delta"])
    (output / "checkpoint").mkdir()
    for name in manifest["checkpoint_files"]:
        shutil.copy2(original / name, output / "checkpoint" / name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, round(maximum_steps * .06), maximum_steps)
    step, reason = 0, "maximum_one_pass"
    with (output / "training.jsonl").open("w", encoding="utf-8") as ledger, \
            (output / "validation.jsonl").open("w", encoding="utf-8") as checks:
        checks.write(json.dumps({"step": 0, **baseline}) + "\n")
        checks.flush()
        for start in range(0, len(order), 16):
            model.train()
            indices = order[start:start + 16]
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.
            for micro_start in range(0, len(indices), plan["micro_batch"]):
                micro = indices[micro_start:micro_start + plan["micro_batch"]]
                logits = training_logits(model, tokenizer, train_bank, prepared, micro, device)
                labels = torch.tensor([prepared[index]["label"] for index in micro], device=device)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite contrastive loss")
                (loss * len(micro) / len(indices)).backward()
                weighted_loss += float(loss.detach()) * len(micro) / len(indices)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite gradients; no update claimed")
            optimizer.step()
            scheduler.step()
            step += 1
            entry = {"step": step, "loss": weighted_loss, "grad_norm": float(norm), "lr": scheduler.get_last_lr(),
                     "elapsed_seconds": time.monotonic() - started,
                     "sources": dict(Counter(data[index]["source"] for index in indices))}
            ledger.write(json.dumps(entry) + "\n")
            ledger.flush()
            if step == 1 or step % 100 == 0:
                print(json.dumps(entry), flush=True)
            if step % plan["validation_interval"] == 0 or step == maximum_steps:
                last = evaluate(model, tokenizer, validation_bank, validation, catalogs, device, output, f"step-{step}")
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
    result = {"plan": plan, "steps": step, "best_step": stopper.best_step, "baseline": baseline,
              "best": best, "last": last, "stop_reason": reason, "elapsed_seconds": time.monotonic() - started,
              "checkpoint_sha256": sha(output / "checkpoint/model.safetensors"),
              "last_checkpoint_sha256": sha(output / "last-checkpoint/model.safetensors")}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
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
