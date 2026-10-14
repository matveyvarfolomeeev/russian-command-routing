"""Evaluate E5 over full known catalogs; never reopen the old sealed tests."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from train_e5_kaggle import evaluate, load_model, prepare_validation
from train_modernbert_kaggle import sha
from train_nanojev_kaggle import rows

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = Path("C:/Users/matva/.codex/worktrees/c58b/jev_pet")


def prepare(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new evaluation input directory")
    output.mkdir(parents=True, exist_ok=True)
    original = ROOT / "artifacts/kaggle/modernbert-evaluation-v1/prepared-all-known.jsonl"
    reference = rows(original)
    assert len(reference) == 800 and len({row["id"] for row in reference}) == 800
    sys.path.insert(0, str(WORKTREE / "src"))
    from jev_launcher.semantic import passage_text

    catalogs, data, sources = {}, [], {}
    for row in reference:
        profile = row["profile"]
        if profile.startswith("independent/"):
            key = "independent"
            path = ROOT / "data/benchmarks/launcher-independent-v1/clean/catalog.json"
        elif profile.startswith("transfer/"):
            key = "transfer"
            path = ROOT / "data/benchmarks/launcher-transfer-v1/synthetic/catalog.json"
        else:
            key = profile
            family, population = profile.split("/")
            base = ROOT / ("data/benchmarks" if population == "synthetic" else "artifacts/benchmarks")
            path = base / family / population / "catalog.json"
        if key not in catalogs:
            objects = sorted(json.loads(path.read_text(encoding="utf-8"))["objects"], key=lambda item: item["object_id"])
            ids = [item["object_id"] for item in objects]
            assert len(ids) == len(set(ids)) == 177
            catalogs[key] = {"candidate_ids": ids,
                             "documents": [passage_text(item).removeprefix("passage: ") for item in objects],
                             "source_file": path.relative_to(ROOT).as_posix()}
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
        catalog = catalogs[key]
        by_id = dict(zip(catalog["candidate_ids"], catalog["documents"], strict=True))
        assert row["target"] in by_id
        assert all(by_id[oid] == document for oid, document in zip(row["candidate_ids"], row["documents"], strict=True))
        data.append({key: row[key] for key in ("id", "query", "profile", "group_id", "target")} | {"catalog_key": key})
    (output / "full-evaluation.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")
    (output / "catalogs.json").write_text(json.dumps(catalogs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"rows": len(data), "catalogs": len(catalogs), "candidates_per_catalog": 177,
                "source_prepared_sha256": sha(original), "source_files": sources,
                "card_renderer_sha256": sha(WORKTREE / "src/jev_launcher/semantic.py"),
                "files": {name: sha(output / name) for name in ("full-evaluation.jsonl", "catalogs.json")},
                "role": "Known800 descriptive evaluation after checkpoint selection; old test not opened"}
    (output / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": len(data), "catalogs": len(catalogs), "scope": manifest["role"]}), flush=True)


def add_scenarios(result):
    counts = defaultdict(lambda: {"rows": 0, "correct": 0})
    for profile, metrics in result["metrics"].items():
        name = "/".join(profile.split("/")[:2]) if profile.startswith("independent/") else profile
        counts[name]["rows"] += metrics["rows"]
        counts[name]["correct"] += metrics["correct"]
    for value in counts.values():
        value["accuracy"] = value["correct"] / value["rows"]
    result.update({"scenario_counts": counts,
                   "equal_scenario_accuracy": sum(value["accuracy"] for value in counts.values()) / len(counts),
                   "equal_row_accuracy": sum(value["correct"] for value in counts.values()) / sum(value["rows"] for value in counts.values()),
                   "noisy_stt_equal_catalog_accuracy": sum(result["metrics"][f"launcher-asr-v1/{name}"]["accuracy"] for name in ("real", "synthetic")) / 2})
    return result


def run(args):
    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    import torch

    torch.set_num_threads(4)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new result directory")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.input / "input-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha(args.input / name) != expected:
            raise ValueError(f"Changed evaluation input: {name}")
    for name, expected in manifest["source_files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Changed catalog: {name}")
    data = rows(args.input / "full-evaluation.jsonl")
    catalogs = json.loads((args.input / "catalogs.json").read_text(encoding="utf-8"))
    model, tokenizer = load_model(args.checkpoint, "cpu")
    model.requires_grad_(False)
    bank = prepare_validation(tokenizer, data, catalogs)
    result = add_scenarios(evaluate(model, tokenizer, bank, data, catalogs, "cpu", args.output, "validation", batch_size=16))
    result.update({"device": "cpu", "precision": "fp32", "optimizer_updates": 0,
                   "checkpoint_sha256": sha(args.checkpoint / "model.safetensors"),
                   "input_sha256": sha(args.input / "full-evaluation.jsonl"),
                   "encoding": bank["stats"], "checkpoint_selection": False,
                   "timing": "batched full-catalog quality; not a latency measurement"})
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"equal_scenario_accuracy": result["equal_scenario_accuracy"],
                      "equal_row_accuracy": result["equal_row_accuracy"],
                      "noisy_stt_equal_catalog_accuracy": result["noisy_stt_equal_catalog_accuracy"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        if not all((args.checkpoint, args.input, args.output)):
            parser.error("Evaluation requires --checkpoint, --input and --output")
        run(args)
