"""Measure E5 with the application's native pooling, cached catalogs and ID mapping."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_choice_quantization import configure_cpu
from evaluate_e5_locally import ROOT, WORKTREE, add_scenarios
from train_modernbert_kaggle import sha
from train_nanojev_kaggle import rows


def run(args):
    policy = configure_cpu(SimpleNamespace(threads=6, performance_cores=True, no_e5_spin=True))
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh ONNX evaluation directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(WORKTREE / "src"))
    from jev_launcher.semantic import E5Search

    options_factory = ort.SessionOptions

    def options():
        value = options_factory()
        value.intra_op_num_threads = 4
        value.inter_op_num_threads = 1
        value.add_session_config_entry("session.intra_op.allow_spinning", "0")
        value.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return value

    if args.onnx:
        exported = json.loads((args.onnx.parent / "export.json").read_text(encoding="utf-8"))
        if sha(args.onnx) != exported["onnx_sha256"] or sha(args.tokenizer) != exported["tokenizer_sha256"]:
            raise ValueError("Export weights/tokenizer changed")
        session = ort.InferenceSession(str(args.onnx), sess_options=options(), providers=["CPUExecutionProvider"])
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        tokenizer.enable_truncation(max_length=512)
        tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        weight_sha = exported["checkpoint_sha256"]
    else:
        native = E5Search(WORKTREE / "models/e5", offline=True)
        with patch.object(ort, "SessionOptions", side_effect=options):
            native._load()
        session, tokenizer = native._session, native._tokenizer
        weight_sha = "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"

    manifest = json.loads((args.input / "input-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha(args.input / name) != expected:
            raise ValueError(f"Changed full evaluation input: {name}")
    data = rows(args.input / "full-evaluation.jsonl")
    catalogs = json.loads((args.input / "catalogs.json").read_text(encoding="utf-8"))
    reference = {row["id"]: row for row in rows(args.reference)}
    reference_result = json.loads((args.reference.parent / "result.json").read_text(encoding="utf-8"))
    if reference_result["checkpoint_sha256"] != weight_sha:
        raise ValueError("Quality reference uses different weights")
    if set(reference) != {row["id"] for row in data}:
        raise ValueError("Reference does not cover all 800 queries")
    searches, objects = {}, {}
    for key, catalog in catalogs.items():
        catalog_file = ROOT / catalog["source_file"]
        if sha(catalog_file) != manifest["source_files"][catalog["source_file"]]:
            raise ValueError("Changed catalog")
        objects[key] = sorted(json.loads(catalog_file.read_text(encoding="utf-8"))["objects"],
                              key=lambda item: item["object_id"])
        if [item["object_id"] for item in objects[key]] != catalog["candidate_ids"]:
            raise ValueError("Catalog order differs")
        search = E5Search(args.output / "indexes" / weight_sha / key, offline=True)
        search._session, search._tokenizer = session, tokenizer
        search._index(objects[key])  # New weights always receive their own new index.
        searches[key] = search

    def infer(row):
        start = time.perf_counter()
        ranked = searches[row["catalog_key"]].rank(row["query"], objects[row["catalog_key"]], limit=177)
        ended = time.perf_counter()
        return ranked, (ended - start) * 1000

    def verify(row, ranked):
        if not ranked["index_cache_hit"] or ranked["query_truncated"] or ranked["truncated_object_ids"]:
            raise ValueError("Index cache or token budget differs")
        predictions = ranked["results"]
        scores = {item["object_id"]: item["score"] for item in predictions}
        ref = reference[row["id"]]
        difference = max(abs(scores[oid] - value) for oid, value in zip(ref["candidate_ids"], ref["similarities"], strict=True))
        ids = [item["object_id"] for item in predictions]
        if ids[0] != ref["predicted"] or difference > 1e-5 or not all(np.isfinite(list(scores.values()))):
            raise ValueError(f"Native ONNX differs from quality reference: {row['id']} / {difference}")
        return ids, scores, difference

    predictions = []
    for index, row in enumerate(data, 1):
        ranked, _ = infer(row)
        ids, scores, difference = verify(row, ranked)
        predictions.append({**row, "predicted": ids[0], "correct": ids[0] == row["target"],
                            "target_rank": ids.index(row["target"]) + 1, "top20": ids[:20],
                            "similarities": [scores[oid] for oid in catalogs[row["catalog_key"]]["candidate_ids"]],
                            "maximum_similarity_difference": difference})
        if index % 200 == 0:
            print(json.dumps({"stage": "quality", "completed": index}), flush=True)
    metrics = {}
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["profile"]].append(row)
    for profile, local in grouped.items():
        metrics[profile] = {"rows": len(local), "correct": sum(row["correct"] for row in local),
                            "accuracy": sum(row["correct"] for row in local) / len(local),
                            "recall20": sum(row["target_rank"] <= 20 for row in local) / len(local),
                            "mrr": sum(1 / row["target_rank"] for row in local) / len(local)}

    timed, repeats = [], []
    independent = [row for row in data if row["profile"].startswith("independent/")]
    assert len(independent) == len({row["id"] for row in independent}) == 400
    for _ in range(2):
        infer(independent[0])
    for repetition in range(2):
        local = []
        for index, row in enumerate(independent if repetition == 0 else reversed(independent), 1):
            ranked, latency = infer(row)
            ids, _, difference = verify(row, ranked)
            record = {"repetition": repetition + 1, "id": row["id"], "profile": row["profile"],
                      "predicted": ids[0], "pipeline_ms": latency, "maximum_similarity_difference": difference}
            local.append(record)
            timed.append(record)
            if index % 100 == 0:
                print(json.dumps({"stage": "timing", "repetition": repetition + 1, "completed": index}), flush=True)
        values = np.array([row["pipeline_ms"] for row in local])
        repeats.append({"repetition": repetition + 1, "p50_ms": float(np.quantile(values, .5)),
                        "p95_ms": float(np.quantile(values, .95)), "maximum_ms": float(values.max()),
                        "over_500_ms": int((values > 500).sum())})
    report = add_scenarios({"metrics": metrics})
    report.update({"runtime": "onnxruntime-cpu-fp32", "runtime_policy": policy,
                   "checkpoint_sha256": weight_sha, "onnx_sha256": sha(args.onnx) if args.onnx else None,
                   "input_sha256": sha(args.input / "full-evaluation.jsonl"), "reference_sha256": sha(args.reference),
                   "all_reference_top1_reproduced": True,
                   "maximum_similarity_difference": max(row["maximum_similarity_difference"] for row in predictions),
                   "top20_order_matches": sum(row["top20"] == reference[row["id"]]["top20"] for row in predictions),
                   "target_rank_matches": sum(row["target_rank"] == reference[row["id"]]["target_rank"] for row in predictions),
                   "unique_timed_queries": 400, "timed_queries": 800, "warmup_queries": 2,
                   "repetitions": repeats,
                   "timing_scope": "native E5 query tokenization/encoding -> cached177 cosine ranking -> ID; full177 returned",
                   "excluded": "ASR, loading, initial index construction, artifact writes and parity checks"})
    for filename, records in (("predictions.jsonl", predictions), ("timed-rows.jsonl", timed)):
        (args.output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"equal_scenario_accuracy": report["equal_scenario_accuracy"], "repetitions": repeats}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    if bool(args.onnx) != bool(args.tokenizer):
        parser.error("Custom export requires both --onnx and --tokenizer")
    run(args)
