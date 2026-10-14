"""Warm E5-to-ID latency of a frozen native reranker on real shortlists."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_choice_quantization import configure_cpu
from train_modernbert_kaggle import sha
from train_nanojev_kaggle import batch, rows


def run(args):
    if args.family == "nano":
        from continue_nanojev_kaggle import load_model
        from train_nanojev_kaggle import encode_all
    else:
        from train_modernbert_kaggle import encode_all, load_model

    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    policy = configure_cpu(SimpleNamespace(threads=6, performance_cores=True, no_e5_spin=True))
    import numpy as np
    import onnxruntime as ort
    import torch

    sys.path.insert(0, str(args.worktree/"src"))
    from jev_launcher.semantic import E5Search, passage_text

    torch.set_num_threads(6)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh benchmark output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    catalog_file = root/"data/benchmarks/launcher-independent-v1/clean/catalog.json"
    objects = json.loads(catalog_file.read_text(encoding="utf8"))["objects"]
    data = rows(args.input)
    reference = {r["id"]: r for r in rows(args.reference)}
    reference_result = json.loads((args.reference.parent/"result.json").read_text())
    if sha(args.checkpoint/"model.safetensors") != reference_result["checkpoint_sha256"]:
        raise ValueError("Quality reference uses different weights")
    model, tokenizer = load_model(args.checkpoint, "cpu")
    model.requires_grad_(False)
    e5 = E5Search(args.worktree/"models/e5", offline=True)
    original_options = ort.SessionOptions

    def no_spin_options():
        options = original_options()
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return options

    with patch.object(ort, "SessionOptions", side_effect=no_spin_options):
        e5._load()
    e5.cache_dir = root/"artifacts/kaggle/independent-v1/e5-cache"

    def infer(row):
        began = time.perf_counter()
        ranked = e5.rank(row["query"], objects, limit=20)
        retrieved = time.perf_counter()
        shortlist = list(ranked["results"])
        random.Random("independent-choice-v1:" + row["id"].removesuffix("-asr")).shuffle(shortlist)
        ids = [o["object_id"] for o in shortlist]
        documents = [passage_text(o).removeprefix("passage: ") for o in shortlist]
        encoded = encode_all(tokenizer, [{"id":row["id"], "query":row["query"], "documents":documents}])
        if args.family != "nano":
            encoded, _ = encoded
        values = batch(tokenizer, encoded, [0], "cpu")
        prepared = time.perf_counter()
        scores = model(**values).logits.reshape(20).float().softmax(-1).cpu().tolist()
        predicted = ids[max(range(20), key=scores.__getitem__)]
        ended = time.perf_counter()
        if ids != row["candidate_ids"] or documents != row["documents"]:
            raise ValueError("Fresh E5 differs from frozen quality inputs")
        if ranked["query_truncated"] or ranked["truncated_object_ids"]:
            raise ValueError("Fresh retrieval truncates inputs")
        if not np.isfinite(scores).all() or abs(sum(scores)-1)>1e-5:
            raise ValueError("Invalid probability vector")
        difference = max(abs(a-b) for a,b in zip(scores, reference[row["id"]]["scores"], strict=True))
        if predicted != reference[row["id"]]["predicted"] or difference>1e-4:
            raise ValueError(f"Benchmark does not reproduce quality reference: {row['id']} / {difference}")
        return {"id":row["id"], "profile":row["profile"], "predicted":predicted,
                "retrieval_ms":(retrieved-began)*1000, "input_build_ms":(prepared-retrieved)*1000,
                "choice_and_mapping_ms":(ended-prepared)*1000, "pipeline_ms":(ended-began)*1000,
                "maximum_probability_difference":difference}

    records, repetitions = [], []
    with torch.inference_mode():
        for _ in range(2):
            infer(data[0])
        for repetition in range(args.repetitions):
            local = []
            for i, row in enumerate(data if repetition%2==0 else list(reversed(data)), 1):
                record = {"repetition":repetition+1, **infer(row)}
                local.append(record)
                records.append(record)
                if i%100==0:
                    print(json.dumps({"stage":"latency", "repetition":repetition+1, "completed":i}),flush=True)
            summary = {}
            for field in ("retrieval_ms", "input_build_ms", "choice_and_mapping_ms", "pipeline_ms"):
                values = np.array([r[field] for r in local])
                summary[field] = {"p50":float(np.quantile(values,.5)), "p95":float(np.quantile(values,.95)),
                                  "maximum":float(values.max())}
            repetitions.append({"repetition":repetition+1, "summary_ms":summary,
                                "over_500_ms":sum(r["pipeline_ms"]>500 for r in local)})
    report = {"model_family":args.family, "device":"cpu", "precision":"fp32", "runtime_policy":policy,
              "unique_queries":len(data), "timed_queries":len(records), "warmup_queries":2,
              "complete_independent400":len(data)==400 and len({r['id'] for r in data})==400,
              "scope":f"warm cached catalog; fresh E5 -> shuffled top20 -> native {args.family} pairs -> ID",
              "excluded":"ASR, loading, initial indexing, artifact writes and post-timing checks",
              "checkpoint_sha256":sha(args.checkpoint/"model.safetensors"), "input_sha256":sha(args.input),
              "reference_sha256":sha(args.reference), "catalog_sha256":sha(catalog_file),
              "all_reference_ids_reproduced":True,
              "maximum_probability_difference":max(r["maximum_probability_difference"] for r in records),
              "repetitions":repetitions}
    (args.output/"timed-rows.jsonl").write_text(
        "".join(json.dumps(r)+"\n" for r in records), encoding="utf8")
    (args.output/"result.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family",type=str,choices=("modernbert","nano","rubert"),default="modernbert")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worktree",type=Path,default=Path("C:/Users/matva/.codex/worktrees/c58b/jev_pet"))
    parser.add_argument("--repetitions",type=int,choices=(1,2),default=2)
    run(parser.parse_args())
