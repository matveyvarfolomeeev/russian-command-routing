"""Compare selected FP32/dynamic-INT8 choice models on frozen CPU inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "artifacts/kaggle/independent-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quantization_exclusions(model, torch, family, backbone_only=False):
    if backbone_only:
        if family != "laya":
            raise ValueError("Encoder-only quantization is implemented for Laya")
        # Exact module names override the Linear type rule in quantize_dynamic.
        return tuple(name for name, module in model.named_modules()
                     if type(module) is torch.nn.Linear and not name.startswith("encoder."))
    return ("act_head.0",) if family == "laya" else ("head.q", "head.k")


def quantize_choice_linear(model, torch, family, backbone_only=False, per_channel=False):
    from jev_launcher.local_weights import quantize_cpu_linear

    excluded = quantization_exclusions(model, torch, family, backbone_only)
    configuration = (torch.ao.quantization.per_channel_dynamic_qconfig if per_channel
                     else torch.ao.quantization.default_dynamic_qconfig)
    with patch.object(torch.ao.quantization, "default_dynamic_qconfig", configuration):
        model = quantize_cpu_linear(model, torch, exclude=excluded)
    quantized = {name: module for name, module in model.named_modules()
                 if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)}
    if not quantized or (backbone_only and any(not name.startswith("encoder.") for name in quantized)):
        raise ValueError("Unexpected quantized layer set")
    weight_schemes = sorted({str(module.weight().qscheme()) for module in quantized.values()})
    expected = "torch.per_channel_affine" if per_channel else "torch.per_tensor_affine"
    if weight_schemes != [expected]:
        raise ValueError(f"Unexpected weight quantization: {weight_schemes}")
    return model, {"engine": torch.backends.quantized.engine,
                   "scope": "encoder_linear_only" if backbone_only else "compatible_linear",
                   "weight_granularity": "per_channel" if per_channel else "per_tensor",
                   "weight_qschemes": weight_schemes, "excluded": list(excluded),
                   "quantized_linear_layers": list(quantized),
                   "int8_weight_elements": sum(module.weight().numel() for module in quantized.values())}


def configure_cpu(args):
    import psutil

    process = psutil.Process()
    runtime_policy = {"threads": args.threads, "performance_cores": args.performance_cores,
                      "e5_spinning": not args.no_e5_spin, "initial_affinity": process.cpu_affinity()}
    if args.performance_cores:
        import ctypes
        import struct
        from ctypes import wintypes

        api_sets = ctypes.WinDLL("kernel32", use_last_error=True).GetSystemCpuSetInformation
        api_sets.argtypes = [ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
                            wintypes.HANDLE, wintypes.ULONG]
        api_sets.restype = wintypes.BOOL
        size = wintypes.ULONG()
        api_sets(None, 0, ctypes.byref(size), None, 0)
        buffer = ctypes.create_string_buffer(size.value)
        if not api_sets(buffer, size.value, ctypes.byref(size), None, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        cpu_sets, offset = [], 0
        while offset < size.value:
            record_size, kind = struct.unpack_from("<II", buffer.raw, offset)
            if kind == 0:
                _, group, logical, core, _, _, efficiency, _ = struct.unpack_from("<IHBBBBBB", buffer.raw, offset+8)
                cpu_sets.append((group, logical, core, efficiency))
            offset += record_size
        if not cpu_sets or any(group != 0 for group, *_ in cpu_sets):
            raise ValueError("Performance-core experiment requires one Windows processor group")
        fastest = max(record[3] for record in cpu_sets)
        cores = {}
        for _, logical, core, efficiency in cpu_sets:
            if efficiency == fastest:
                cores.setdefault(core, logical)
        process.cpu_affinity(sorted(cores.values()))
        runtime_policy["cpu_sets"] = cpu_sets
    runtime_policy["effective_affinity"] = process.cpu_affinity()
    print(json.dumps({"stage": "runtime_policy", **runtime_policy}), flush=True)
    return runtime_policy


def run(args):
    if args.backbone_only and (args.model != "laya" or args.precision != "int8"):
        raise ValueError("--backbone-only requires --model laya --precision int8")
    if args.per_channel and (args.model != "laya" or args.precision != "int8"):
        raise ValueError("--per-channel requires --model laya --precision int8")
    os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1")
    runtime_policy = configure_cpu(args)
    sys.path.insert(0, str(args.worktree / "src"))
    api = importlib.import_module(f"evaluate_{args.model}_locally")
    native = importlib.import_module(f"train_{args.model}_kaggle")
    from audit_kev_run import check_predictions, metrics
    from train_laya_kaggle import selection_loss

    manifest = read(EVALUATION / "input-manifest.json")
    for name, expected in manifest["files"].items():
        if sha(EVALUATION / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    parity = rows(EVALUATION / "cpu-parity-input.jsonl")
    if len(parity) != 12 or len({r["id"] for r in parity}) != 12:
        raise ValueError("Expected the existing twelve-query parity sample")
    args.input = EVALUATION / ("prepared-evaluation.jsonl" if args.quality else "cpu-parity-input.jsonl")
    args.checkpoint = ROOT / f"artifacts/kaggle/outputs/{args.model}-clean-first-v1/results/checkpoint"
    args.preflight = args.gradient_preflight = False
    reference_path = (ROOT / "artifacts/kaggle/outputs" /
                      ("independent-v2" if args.model == "laya" else "independent-v5") /
                      f"results/trained-{args.model}/validation-predictions.jsonl")
    reference = {r["id"]: r for r in rows(reference_path)}
    original_encode = api.encode_all
    tokenizer_holder = {}

    def capture_tokenizer(tokenizer, data):
        tokenizer_holder["value"] = tokenizer
        return original_encode(tokenizer, data)

    def evaluate(model, *values, **_kwargs):
        import numpy as np
        import torch
        from jev_launcher.semantic import E5Search, passage_text

        data = values[2] if args.model == "laya" else values[1]
        tokenizer = tokenizer_holder["value"]
        torch.set_num_threads(args.threads)
        torch.backends.mha.set_fastpath_enabled(False)
        model.eval().requires_grad_(False)
        original_parameters = sum(p.numel() for p in model.parameters())
        conversion = {"precision": args.precision, "adapter_merge": False}
        if args.onnx:
            graph = read(args.onnx.parent / "export.json")
            original_parameters = graph["parameters"]
            conversion.update(graph=str(args.onnx.resolve()), graph_sha256=sha(args.onnx),
                              export=graph, providers=model.session.get_providers(),
                              onnxruntime=importlib.import_module("onnxruntime").__version__)
        if args.precision == "int8":
            began = time.perf_counter()
            if args.model == "kev":
                # Base tensors are read-only file views. Merge only private copies.
                from peft.tuners.lora.layer import LoraLayer

                copied = 0
                for module in model.lm.modules():
                    if isinstance(module, LoraLayer):
                        weight = module.get_base_layer().weight
                        module.get_base_layer().weight = torch.nn.Parameter(weight.detach().clone(),
                                                                           requires_grad=False)
                        copied += 1
                model.lm = model.lm.merge_and_unload(safe_merge=True)
                conversion.update(adapter_merge=True, privately_copied_lora_base_layers=copied)
            model, information = quantize_choice_linear(model, torch, args.model,
                                                       args.backbone_only, args.per_channel)
            conversion.update(information, seconds=time.perf_counter() - began)
        print(json.dumps({"stage": "conversion_complete", "model": args.model,
                          "precision": args.precision,
                          "quantized_layers": len(conversion.get("quantized_linear_layers", []))}), flush=True)

        catalog_file = ROOT / "data/benchmarks/launcher-independent-v1/clean/catalog.json"
        if sha(catalog_file) != manifest["source_files"]["clean/catalog.json"]:
            raise ValueError("Catalog changed")
        objects = read(catalog_file)["objects"]
        e5 = E5Search(args.worktree / "models/e5", offline=True)
        if args.no_e5_spin:
            import onnxruntime as ort

            original_options = ort.SessionOptions

            def no_spin_options():
                options = original_options()
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                options.add_session_config_entry("session.inter_op.allow_spinning", "0")
                return options

            with patch.object(ort, "SessionOptions", side_effect=no_spin_options):
                e5._load()
        else:
            e5._load()
        e5.cache_dir = EVALUATION / "e5-cache"

        def infer(row):
            began = time.perf_counter()
            ranked = e5.rank(row["query"], objects, limit=20)
            retrieval_end = time.perf_counter()
            shortlist = list(ranked["results"])
            random.Random("independent-choice-v1:" + row["id"].removesuffix("-asr")).shuffle(shortlist)
            ids = [obj["object_id"] for obj in shortlist]
            documents = [passage_text(obj).removeprefix("passage: ") for obj in shortlist]
            encoded, _ = original_encode(tokenizer, [{"query": row["query"], "documents": documents,
                                                      "id": row["id"]}])
            encoding_end = time.perf_counter()
            if args.model == "laya":
                logits, _ = model(**native.batch(tokenizer, encoded, [0], "cpu"))
                logits = logits[0].float()
            else:
                logits = model(encoded[0])[0].float()
            scores = logits.softmax(-1).cpu().tolist()
            predicted = ids[max(range(len(ids)), key=scores.__getitem__)]
            ended = time.perf_counter()
            if ranked["query_truncated"] or ranked["truncated_object_ids"]:
                raise ValueError("Unexpected retrieval truncation")
            if ids != row["candidate_ids"] or documents != row["documents"]:
                raise ValueError("Fresh inputs differ from the frozen benchmark")
            if len(scores) != 20 or not all(math.isfinite(x) and 0 <= x <= 1 for x in scores) \
                    or not math.isclose(sum(scores), 1, abs_tol=1e-5):
                raise ValueError("Invalid choice probabilities")
            expected = reference[row["id"]]
            difference = max(abs(a-b) for a, b in zip(scores, expected["scores"], strict=True))
            if args.precision in ("fp32", "onnx_fp32") \
                    and (predicted != expected["predicted"] or difference > 1e-4):
                raise ValueError("FP32 does not reproduce the audited reference")
            prediction = {key: row[key] for key in
                          ("id", "profile", "group_id", "target", "candidate_ids", "retrieval_covered")}
            prediction.update(predicted=predicted, correct=predicted == row["target"], scores=scores,
                              fp32_predicted=expected["predicted"], max_fp32_probability_difference=difference)
            timing = {"id": row["id"], "profile": row["profile"],
                      "sequence_tokens": len(encoded[0]["ids"]),
                      "retrieval_ms": (retrieval_end-began)*1000,
                      "input_build_ms": (encoding_end-retrieval_end)*1000,
                      "choice_and_mapping_ms": (ended-encoding_end)*1000,
                      "pipeline_ms": (ended-began)*1000}
            return prediction, timing

        def summarize(records):
            return {field: {"p50": float(np.quantile(numbers, .5)),
                            "p95": float(np.quantile(numbers, .95)),
                            "max": float(numbers.max()), "mean": float(numbers.mean())}
                    for field in ("retrieval_ms", "input_build_ms", "choice_and_mapping_ms", "pipeline_ms")
                    for numbers in [np.asarray([r[field] for r in records])]}

        data_by_id = {row["id"]: row for row in data}
        if any(data_by_id[row["id"]] != row for row in parity):
            raise ValueError("Parity sample differs from full benchmark inputs")
        probe, probe_predictions, predictions, timings = [], [], [], []
        with torch.inference_mode():
            for _ in range(2):
                infer(parity[0])
            for repetition, ordered in enumerate((parity, list(reversed(parity))), 1):
                for row in ordered:
                    prediction, timing = infer(row)
                    probe.append({"repetition": repetition, **timing})
                    probe_predictions.append(prediction)
            probe_summary = summarize(probe)
            (args.output / "probe-rows.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in probe), encoding="utf-8")
            print(json.dumps({"stage": "probe_complete", "model": args.model,
                              "precision": args.precision, "summary_ms": probe_summary}), flush=True)
            if args.quality:
                with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
                    for index, row in enumerate(data, 1):
                        prediction, timing = infer(row)
                        predictions.append(prediction)
                        timings.append(timing)
                        stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                        if index % 20 == 0 or index == len(data):
                            stream.flush()
                            print(json.dumps({"stage": "quality", "model": args.model,
                                              "precision": args.precision, "completed": index,
                                              "total": len(data)}), flush=True)
                check_predictions(predictions, data)

            if args.save:
                if args.precision != "int8":
                    raise ValueError("Only save the separate quantized state")
                saved = args.output / "quantized-state.pt"
                torch.save(model.state_dict(), saved)
                before = [infer(row)[0] for row in parity]
                state = torch.load(saved, map_location="cpu", weights_only=True, mmap=True)
                model.load_state_dict(state, strict=True, assign=True)
                after = [infer(row)[0] for row in parity]
                if any(a["scores"] != b["scores"] for a, b in zip(before, after, strict=True)):
                    raise ValueError("Serialized INT8 state changes predictions")
                conversion.update(state_file=saved.name, state_sha256=sha(saved),
                                  state_bytes=saved.stat().st_size, reload_scores_exact=True,
                                  reload_checked_queries=len(parity))

        report = {"model": args.model, "precision": args.precision, "device": "cpu",
                  "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                  "runtime_policy": runtime_policy,
                  "parameters_before_quantization": original_parameters, "conversion": conversion,
                  "optimizer_updates": 0, "reference_sha256": sha(reference_path),
                  "parity_input_sha256": sha(EVALUATION / "cpu-parity-input.jsonl"),
                  "timing": "E5 top20 -> same shuffled metadata -> native encoding -> choice -> ID",
                  "excluded_from_timing": "ASR, cold loading/indexing, output writes and integrity checks",
                  "warmup_queries": 2, "probe_unique_queries": 12, "probe_timed_queries": 24,
                  "probe_summary_ms": probe_summary,
                  "probe_top1_changes": sum(r["predicted"] != r["fp32_predicted"] for r in probe_predictions)}
        if args.quality:
            loss, losses = selection_loss(predictions)
            report.update(quality_rows=len(predictions), metrics=metrics(predictions),
                          conditional_group_nll=loss, profile_nll=losses,
                          quality_summary_ms=summarize(timings),
                          top1_changes=sum(r["predicted"] != r["fp32_predicted"] for r in predictions),
                          fixes=sum(r["correct"] and not reference[r["id"]]["correct"] for r in predictions),
                          regressions=sum(not r["correct"] and reference[r["id"]]["correct"] for r in predictions),
                          max_fp32_probability_difference=max(r["max_fp32_probability_difference"]
                                                              for r in predictions))
            (args.output / "quality-timings.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in timings), encoding="utf-8")
        (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": "benchmark_complete", "model": args.model,
                          "precision": args.precision, "quality_rows": len(predictions)}), flush=True)
        return {"benchmark": report}

    api.encode_all, api.evaluate = capture_tokenizer, evaluate
    if args.onnx:
        if args.model != "laya" or args.precision not in ("onnx_fp32", "onnx_int8"):
            raise ValueError("ONNX graph is only supported for the exported Laya")
        import onnxruntime as ort
        import torch
        from transformers import AutoTokenizer

        graph = read(args.onnx.parent / "export.json")
        if sha(args.checkpoint / "model.safetensors") != graph["source_checkpoint_sha256"]:
            raise ValueError("Graph belongs to another checkpoint")
        expected = graph["fp32"].get(args.onnx.name) if args.precision == "onnx_fp32" else graph["int8"]
        if sha(args.onnx) != expected["sha256"]:
            raise ValueError("Exported graph changed")
        for filename, record in graph["fp32"].items():
            if sha(args.onnx.parent / filename) != record["sha256"]:
                raise ValueError("FP32 export changed")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Use a new output directory")
        args.output.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer", local_files_only=True)
        data = rows(args.input)
        encoded, stats = capture_tokenizer(tokenizer, data)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = args.threads, 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        class ONNXModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                                    providers=["CPUExecutionProvider"])

            def forward(self, **inputs):
                output = self.session.run(["logits", "act_logits"],
                                          {key: value.cpu().numpy() for key, value in inputs.items()})
                return tuple(torch.from_numpy(value) for value in output)

        result = evaluate(ONNXModel(), tokenizer, encoded, data, "cpu", args.output, "validation")
        result.update(checkpoint_sha256=graph["source_checkpoint_sha256"], input_sha256=sha(args.input),
                      encoding=stats, device="cpu", weight_storage="validated separate ONNX graph")
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        api.score(args)
    result_path = args.output / "result.json"
    result = read(result_path)
    result.update(precision=args.precision, adapter_merge=args.model == "kev" and args.precision == "int8",
                  timing="small paired warm latency probe; optional full400 quantization regression check")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    selected = read(EVALUATION / "fixed-checkpoints.json")["models"][f"trained-{args.model}"]["sha256"]
    if result["checkpoint_sha256"] != selected:
        raise ValueError("Source checkpoint differs from the previously selected weights")
    if args.model == "laya" and sha(args.checkpoint / "model.safetensors") != selected:
        raise ValueError("Source Laya was modified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("laya", "kev"), required=True)
    parser.add_argument("--precision", choices=("fp32", "int8", "onnx_fp32", "onnx_int8"), required=True)
    parser.add_argument("--worktree", type=Path,
                        default=Path("C:/Users/matva/.codex/worktrees/c58b/jev_pet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 6, 8), default=4)
    parser.add_argument("--performance-cores", action="store_true")
    parser.add_argument("--no-e5-spin", action="store_true")
    parser.add_argument("--backbone-only", action="store_true")
    parser.add_argument("--per-channel", action="store_true")
    run(parser.parse_args())
