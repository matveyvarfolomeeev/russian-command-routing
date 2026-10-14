"""Evaluate a Russian reranker checkpoint on frozen local E5 shortlists."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from train_modernbert_kaggle import encode_all, evaluate, load_model, sha
from train_nanojev_kaggle import rows


def score(args):
    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    import torch
    import transformers

    torch.set_num_threads(4)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh evaluation directory")
    args.output.mkdir(parents=True, exist_ok=True)
    data = rows(args.input)
    if len({r["id"] for r in data}) != len(data):
        raise ValueError("Duplicate evaluation IDs")
    for row in data:
        if len(row["candidate_ids"]) != 20 or len(set(row["candidate_ids"])) != 20:
            raise ValueError("Changed candidate list")
        if row["retrieval_covered"] != (row["target"] in row["candidate_ids"]):
            raise ValueError("Repaired evaluation shortlist")
    model, tokenizer = load_model(args.checkpoint, "cpu")
    model.requires_grad_(False)
    encoded, stats = encode_all(tokenizer, data)
    result = evaluate(model, tokenizer, encoded, data, "cpu", args.output, "validation", micro_batch=2)
    scenarios = defaultdict(lambda: {"rows": 0, "correct": 0})
    for row in rows(args.output/"validation-predictions.jsonl"):
        profile = row["profile"]
        if profile.startswith("independent/"):
            profile = "independent/" + ("asr" if "asr" in profile else "clean")
        scenarios[profile]["rows"] += 1
        scenarios[profile]["correct"] += int(row["correct"])
    for values in scenarios.values():
        values["accuracy"] = values["correct"]/values["rows"]
    result.update(checkpoint_sha256=sha(args.checkpoint/"model.safetensors"),
                  input_sha256=sha(args.input), encoding=stats, device="cpu", precision="fp32",
                  transformers=transformers.__version__, scenario_counts=dict(scenarios),
                  equal_scenario_accuracy=sum(c["accuracy"] for c in scenarios.values())/len(scenarios),
                  equal_row_accuracy=sum(c["correct"] for c in scenarios.values())/len(data),
                  checkpoint_selection=False, timing="frozen-shortlist quality; excludes E5, not a latency measurement")
    (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"rows": len(data), "equal_scenario_accuracy": result["equal_scenario_accuracy"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    score(parser.parse_args())
