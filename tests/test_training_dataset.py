import hashlib
import itertools
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from scripts.training_dataset import (
    decision_examples,
    evaluation_guard,
    read_json,
    read_rows,
    screen_augmentation,
    validate,
)
from scripts.validate_benchmark import BenchmarkError

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/training/launcher-train-v1"
HELDOUT = ROOT / "data/benchmarks/launcher-v1/synthetic"


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "training"
    shutil.copytree(DATASET, directory)
    return directory


def rewrite_rows(directory, filename, rows):
    path = directory / filename
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = read_json(directory / "manifest.json")
    manifest["files"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_training_is_positive_only_and_protected_from_both_evaluation_profiles():
    report = validate(DATASET)
    assert report["rows"] == 5664
    assert report["actions"] == {"open": 5664}
    assert report["seed_queries"] == 708
    assert report["target_objects"] == 1416
    assert {"launcher-v1-synthetic", "launcher-basic-v1-synthetic"} <= set(report["evaluation_checked"])
    assert "author_only" in report["semantic_review"]


def test_rehashed_but_wrong_known_target_is_rejected(dataset):
    rows = read_rows(dataset / "train.jsonl")
    catalog = next(c for c in read_rows(dataset / "catalogs.jsonl") if c["catalog_id"] == rows[0]["catalog_id"])
    other = next(o for o in catalog["objects"] if o["object_id"] != rows[0]["expected_object_id"] and o["kind"] == rows[0]["tags"][0])
    rows[0]["expected_object_id"] = other["object_id"]
    rewrite_rows(dataset, "train.jsonl", rows)
    with pytest.raises(BenchmarkError, match="authored seed labels/provenance"):
        validate(dataset)


def test_removed_variant_qualifier_is_rejected(dataset):
    rows = read_rows(dataset / "train.jsonl")
    row = next(r for r in rows if r["seed_id"].startswith("variant-03-") and r["seed_id"].endswith("q3"))
    row["query"] = row["query"].replace("64-битную", "").replace("32-битную", "")
    rewrite_rows(dataset, "train.jsonl", rows)
    with pytest.raises(BenchmarkError, match="authored seed labels/provenance"):
        validate(dataset)


def test_evaluation_profile_cannot_be_silently_removed_from_the_guard(dataset):
    path = dataset / "manifest.json"
    manifest = read_json(path)
    manifest["evaluation_guard"] = [
        guard for guard in manifest["evaluation_guard"]
        if guard["dataset_id"] != "launcher-basic-v1-synthetic"
    ]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="both public evaluation profiles"):
        validate(dataset)


def test_execution_fields_cannot_enter_a_training_catalog(dataset):
    catalogs = read_rows(dataset / "catalogs.jsonl")
    catalogs[0]["objects"][0]["target"] = "C:/example/program.exe"
    # Rehash both layers to test the schema boundary rather than file integrity.
    catalogs[0]["catalog_hash"] = hashlib.sha256(
        json.dumps(catalogs[0]["objects"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rewrite_rows(dataset, "catalogs.jsonl", catalogs)
    with pytest.raises(BenchmarkError, match="unsafe catalog fields"):
        validate(dataset)


@pytest.mark.parametrize("leak", ["target_id", "target_name", "query"])
def test_guard_rejects_eval_target_and_query_leakage(leak):
    evaluation_row = read_rows(HELDOUT / "test.jsonl")[0]
    evaluation_object = next(o for o in read_json(HELDOUT / "catalog.json")["objects"] if o["object_id"] == evaluation_row["expected_object_id"])
    catalog = {"objects": [{"object_id": "new-id", "name": "New invented object"}]}
    row = {"query": "Новый независимый запрос для обучения"}
    if leak == "target_id":
        catalog["objects"][0]["object_id"] = evaluation_object["object_id"]
    elif leak == "target_name":
        catalog["objects"][0]["name"] = evaluation_object["name"]
    else:
        row["query"] = "  " + evaluation_row["query"].upper() + "!"
    with pytest.raises(BenchmarkError, match="leakage"):
        evaluation_guard([catalog], [row], [HELDOUT])


def test_typo_is_removed_when_it_matches_a_different_object():
    row = {
        "query": "Запусти Лавро", "expected_object_id": "one", "tags": ["app", "exact_name", "typo"],
        "source": "controlled_augmentation",
        "augmentation": {"original_query": "Запусти Лаверо"},
    }
    screen_augmentation(row, [
        {"object_id": "one", "name": "Лаверо", "aliases": []},
        {"object_id": "two", "name": "Лавро", "aliases": []},
    ])
    assert row["query"] == "Запусти Лаверо"
    assert row["augmentation"] is None
    assert "typo" not in row["tags"]


def test_export_keeps_every_candidate_and_only_label_contains_the_answer():
    rows = read_rows(DATASET / "train.jsonl")
    catalogs = {c["catalog_id"]: c["objects"] for c in read_rows(DATASET / "catalogs.jsonl")}
    positions = Counter()
    for row, example in zip(rows, decision_examples(DATASET), strict=True):
        assert set(example) == {"state", "questions"}
        assert example["state"] == {"query": row["query"]}
        question = example["questions"]["open"]
        assert set(question) == {"type", "instructions", "criteria", "label"}
        assert question["type"] == "choice"
        assert question["label"] == row["expected_object_id"]
        expected = {o["object_id"] for o in catalogs[row["catalog_id"]]}
        assert set(question["criteria"]) == expected
        assert len(expected) == 177
        assert row["rationale"] not in json.dumps(example, ensure_ascii=False)
        positions[list(question["criteria"]).index(question["label"])] += 1
    assert set(positions) == set(range(177))
    assert max(positions.values()) < 2 * len(rows) / 177


def test_choice_order_is_reproducible_and_varies_between_same_catalog_rows():
    first = list(itertools.islice(decision_examples(DATASET), 10))
    second = list(itertools.islice(decision_examples(DATASET), 10))
    assert first == second
    orders = [list(e["questions"]["open"]["criteria"]) for e in first]
    assert len({tuple(order) for order in orders}) == len(first)
