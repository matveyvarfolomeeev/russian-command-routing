"""Check dynamic INT8 against saved FP32 predictions on a frozen local suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import evaluate_laya_locally as api
from audit_kev_run import check_predictions, metrics
from benchmark_choice_quantization import configure_cpu, quantize_choice_linear, read, rows, sha

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    args.threads, args.performance_cores, args.no_e5_spin = 6, True, True
    policy = configure_cpu(args)
    args.checkpoint = ROOT / "artifacts/kaggle/outputs/laya-clean-first-v1/results/checkpoint"
    if args.suite == "transfer":
        args.input = ROOT / "artifacts/kaggle/transfer-test-v1/prepared-test.jsonl"
        baseline = ROOT / "artifacts/kaggle/transfer-test-v1/trained-laya"
    else:
        args.input = ROOT / "artifacts/kaggle/local-validation.jsonl"
        baseline = ROOT / "artifacts/kaggle/local-validation/laya-clean-first-v1"
    previous = read(baseline / "result.json")
    if sha(args.input) != previous["input_sha256"]:
        raise ValueError("Frozen evaluation input changed")
    expected_rows = 100 if args.suite == "transfer" else 300
    if len(rows(args.input)) != expected_rows:
        raise ValueError("Unexpected local suite size")
    original_evaluate = api.evaluate
    conversion = {}

    def evaluate(model, *values, **kwargs):
        import torch
        torch.set_num_threads(6)
        model.eval().requires_grad_(False)
        model, information = quantize_choice_linear(model, torch, "laya", args.backbone_only, args.per_channel)
        conversion.update(information)
        print(json.dumps({"stage": "int8_model_ready", "suite": args.suite, "rows": expected_rows,
                          "scope": conversion["scope"], "weight_granularity": conversion["weight_granularity"],
                          "quantized_layers": len(conversion["quantized_linear_layers"])}), flush=True)
        return original_evaluate(model, *values, **kwargs)

    with patch.object(api, "evaluate", side_effect=evaluate):
        api.score(args)
    result_path = args.output / "result.json"
    result = read(result_path)
    predictions = rows(args.output / "validation-predictions.jsonl")
    check_predictions(predictions, rows(args.input))
    old = {r["id"]: r for r in rows(baseline / "validation-predictions.jsonl")}
    result.update(precision="dynamic_int8_linear_with_fp32_attention_embeddings", runtime_policy=policy,
                  conversion=conversion, suite=args.suite,
                  optimizer_updates=0, reference_sha256=sha(baseline / "validation-predictions.jsonl"),
                  top1_changes=sum(r["predicted"] != old[r["id"]]["predicted"] for r in predictions),
                  fixes=sum(r["correct"] and not old[r["id"]]["correct"] for r in predictions),
                  regressions=sum(not r["correct"] and old[r["id"]]["correct"] for r in predictions),
                  fp32_metrics=metrics(list(old.values())))
    if result["checkpoint_sha256"] != previous["checkpoint_sha256"] \
            or sha(args.checkpoint / "model.safetensors") != previous["checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "local_int8_complete", "suite": args.suite,
                      "metrics": result["metrics"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path,
                        default=Path("C:/Users/matva/.codex/worktrees/c58b/jev_pet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone-only", action="store_true")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--suite", choices=("canonical", "transfer"), default="canonical")
    main(parser.parse_args())
