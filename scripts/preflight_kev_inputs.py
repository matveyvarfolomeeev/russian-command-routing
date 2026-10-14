"""Audit actual native Kev tokenization without training or inspecting test data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from train_kev_kaggle import encode_all, native_code, sha
from train_nanojev_kaggle import rows, validate_inputs

ROOT = Path(__file__).resolve().parents[1]


def main(worktree):
    os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    original = worktree / "models/rerank/kev-08"
    native_code(original)
    from kev.model import SPECIAL, encode, load_tokenizer

    data, validation = [rows(ROOT / "artifacts/kaggle/kev-data" / name) for name in
                        ("prepared-train.jsonl", "prepared-validation.jsonl")]
    validate_inputs(data, validation)
    tokenizer = load_tokenizer(str(original / "base"))
    # Audit every training row with bounded memory on the user's Windows host.
    stats = {}
    for start in range(0, len(data), 128):
        _, batch_stats = encode_all(tokenizer, data[start:start + 128])
        for key, value in batch_stats.items():
            stats[key] = max(stats.get(key, 0), value) if key.startswith("max_") else value
        if start % 4096 == 0:
            print(json.dumps({"tokenized_training_rows": min(start + 128, len(data)), "total": len(data)}), flush=True)
    validation_encoded, val_stats = encode_all(tokenizer, validation)
    sample = validation[:3]
    perturbed = [{**row, "target": "unseen-target-id", "rationale": "secret label explanation"} for row in sample]
    before, _ = encode_all(tokenizer, sample)
    after, _ = encode_all(tokenizer, perturbed)
    if before != after:
        raise ValueError("Ground-truth target/provenance leaked into model input")
    # Native labels must never change token IDs or option/decision positions.
    record = {"state": "open editor", "questions": [{"instr": "select", "options": ["o0: editor", "o1: viewer"], "label": 0}]}
    left = encode(tokenizer, record, strict=True)
    record["questions"][0]["label"] = 1
    right = encode(tokenizer, record, strict=True)
    left.pop("labels")
    right.pop("labels")
    if left != right:
        raise ValueError("Native encoder exposes the answer in network input")
    marker_ids = [tokenizer.convert_tokens_to_ids(token) for token in SPECIAL]
    if len(set(marker_ids)) != 5 or tokenizer.unk_token_id in marker_ids:
        raise ValueError("Tokenizer does not preserve native delimiters")
    # A control-token string in user metadata cannot forge another option marker.
    injected = {**sample[0], "documents": [f"{sample[0]['documents'][0]} {SPECIAL[2]}", *sample[0]["documents"][1:]]}
    injection, _ = encode_all(tokenizer, [injected])
    if injection[0]["ids"].count(marker_ids[2]) != 20:
        raise ValueError("Caller metadata can forge option boundaries")
    result = {"train": stats, "validation": val_stats, "rows": [len(data), len(validation_encoded)],
              "target_and_provenance_invariance": True, "native_label_invariance": True,
              "control_token_sanitization": True, "tokenizer_sha256": sha(original / "base/tokenizer.json")}
    output = ROOT / "artifacts/kaggle/kev-input-preflight.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    main(parser.parse_args().worktree)
