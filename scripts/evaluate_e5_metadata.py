"""Remove application metadata and evaluate the fixed trained E5 on known800."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from evaluate_e5_locally import ROOT, WORKTREE, add_scenarios
from train_e5_kaggle import evaluate, load_model, prepare_validation
from train_modernbert_kaggle import sha
from train_nanojev_kaggle import rows


def run(args):
    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    import torch

    torch.set_num_threads(4)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new metadata evaluation directory")
    args.output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "artifacts/kaggle/e5-evaluation-v1"
    source = base / "input"
    manifest = json.loads((source / "input-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        assert sha(source / name) == expected, name
    renderer = WORKTREE / "src/jev_launcher/semantic.py"
    assert sha(renderer) == manifest["card_renderer_sha256"]
    sys.path.insert(0, str(WORKTREE / "src"))
    from jev_launcher.semantic import passage_text

    data = rows(source / "full-evaluation.jsonl")
    original_catalogs = json.loads((source / "catalogs.json").read_text(encoding="utf-8"))
    original_result = json.loads((base / "selected/result.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / "artifacts/kaggle/outputs/e5-small-clean-first-v1/results/checkpoint"
    weight_sha = sha(checkpoint / "model.safetensors")
    assert original_result["checkpoint_sha256"] == weight_sha
    assert original_result["input_sha256"] == sha(source / "full-evaluation.jsonl")
    original_predictions = rows(base / "selected/validation-predictions.jsonl")
    original_by_id = {row["id"]: row for row in original_predictions}
    raw, app_ids, catalog_stats = {}, {}, {}
    for key, catalog in original_catalogs.items():
        path = ROOT / catalog["source_file"]
        assert sha(path) == manifest["source_files"][catalog["source_file"]]
        items = sorted(json.loads(path.read_text(encoding="utf-8"))["objects"], key=lambda item: item["object_id"])
        assert [item["object_id"] for item in items] == catalog["candidate_ids"]
        assert [passage_text(item).removeprefix("passage: ") for item in items] == catalog["documents"]
        raw[key] = items
        apps = [item for item in items if item["kind"] == "app"]
        app_ids[key] = {item["object_id"] for item in apps}
        catalog_stats[key] = {"objects": len(items), "apps": len(apps),
                              "apps_with_description": sum(bool(item.get("description", "").strip()) for item in apps),
                              "apps_with_aliases": sum(bool(item.get("aliases")) for item in apps)}

    def app_metrics(predictions):
        metrics = {}
        for profile in sorted({row["profile"] for row in predictions}):
            local = [row for row in predictions if row["profile"] == profile and row["target"] in app_ids[row["catalog_key"]]]
            assert local
            metrics[profile] = {"rows": len(local), "correct": sum(row["correct"] for row in local),
                                "accuracy": sum(row["correct"] for row in local) / len(local)}
        return add_scenarios({"metrics": metrics})

    results = {"full": {"all_objects": original_result, "app_targets": app_metrics(original_predictions)}}
    model, tokenizer = load_model(checkpoint, "cpu")
    model.requires_grad_(False)
    for variant in ("no_app_descriptions", "app_name_and_kind"):
        folder = args.output / variant
        folder.mkdir()
        catalogs = copy.deepcopy(original_catalogs)
        for key, items in raw.items():
            changed = []
            for item in items:
                item = dict(item)
                if item["kind"] == "app":
                    item["description"] = ""
                    if variant == "app_name_and_kind":
                        item["aliases"] = []
                changed.append(item)
            catalogs[key]["documents"] = [passage_text(item).removeprefix("passage: ") for item in changed]
            assert catalogs[key]["candidate_ids"] == original_catalogs[key]["candidate_ids"]
        (folder / "catalogs.json").write_text(json.dumps(catalogs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        bank = prepare_validation(tokenizer, data, catalogs)
        result = add_scenarios(evaluate(model, tokenizer, bank, data, catalogs, "cpu", folder, "evaluation", batch_size=16))
        predictions = rows(folder / "evaluation-predictions.jsonl")
        assert len(predictions) == len(original_by_id) == len(data) == 800
        assert {row["id"] for row in predictions} == set(original_by_id)
        fixed = sum(not original_by_id[row["id"]]["correct"] and row["correct"] for row in predictions)
        lost = sum(original_by_id[row["id"]]["correct"] and not row["correct"] for row in predictions)
        assert sum(row["correct"] for row in predictions) == sum(row["correct"] for row in original_predictions) + fixed - lost
        results[variant] = {"all_objects": result, "app_targets": app_metrics(predictions),
                            "paired_with_full": {"fixed": fixed, "lost": lost}, "encoding": bank["stats"]}
        print(json.dumps({"variant": variant, "macro10": result["equal_scenario_accuracy"],
                          "old_stt": result["noisy_stt_equal_catalog_accuracy"], "paired": {"fixed": fixed, "lost": lost}}), flush=True)
    sources = [source / "input-manifest.json", source / "full-evaluation.jsonl", source / "catalogs.json",
               base / "selected/result.json", base / "selected/validation-predictions.jsonl", Path(__file__).resolve()]
    for variant in ("no_app_descriptions", "app_name_and_kind"):
        sources.extend([args.output / variant / name for name in ("catalogs.json", "evaluation-predictions.jsonl")])
    report = {"recorded_on": "2026-10-08", "checkpoint_sha256": weight_sha, "head_added": False,
              "optimizer_updates": 0, "checkpoint_selection": False, "catalog_stats": catalog_stats,
              "scope": "Same800 queries/targets/catalog IDs; only app descriptions/aliases removed; files/folders unchanged",
              "role": "Known-suite metadata ablation after fixed checkpoint selection; sealed oldtest unopened",
              "results": results,
              "source_sha256": {path.resolve().relative_to(ROOT).as_posix(): sha(path) for path in sources}}
    (args.output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kaggle/e5-metadata-ablation-v1")
    run(parser.parse_args())
