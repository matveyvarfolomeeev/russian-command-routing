"""Training provenance, recoverable corruption and model-input boundaries."""

import itertools
import shutil
from collections import Counter

import pytest

from scripts import stt_training_dataset as stt
from scripts import training_dataset as base
from scripts.validate_benchmark import BenchmarkError


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "stt-training"
    shutil.copytree(stt.DEFAULT_DATASET, directory)
    return directory


def rewrite(directory, name, value):
    if name.endswith("jsonl"):
        base.write_rows(directory / name, value)
    else:
        base.write_json(directory / name, value)
    manifest = base.read_json(directory / "manifest.json")
    manifest["files"][name] = base.digest((directory / name).read_bytes())
    base.write_json(directory / "manifest.json", manifest)


def test_corpus_is_train_only_diverse_and_separated_from_all_three_profiles():
    report = stt.validate(stt.DEFAULT_DATASET)
    assert report["rows"] == 16992 and report["parent_rows"] == 5664
    assert report["actions"] == {"open": 16992} and report["actual_stt"] is False
    assert report["target_objects"] == 1416 and report["seed_queries"] == 708
    assert {"phonetic_name", "phonetic_word", "word_split", "word_merge", "spoken_number",
            "spoken_acronym", "script_transcription", "word_repeat", "command_dropout",
            "command_mishearing", "filler", "punctuation_and_case"} == set(report["operator_occurrences"])
    assert all(report["operator_occurrences"].values())
    assert {"launcher-v1-synthetic", "launcher-basic-v1-synthetic", "launcher-asr-v1-synthetic"} <= set(report["evaluation_checked"])


def test_every_edit_replays_from_its_clean_parent_and_keeps_the_parent_label():
    parents = {r["id"]: r for r in base.read_rows(base.DEFAULT_DATASET / "train.jsonl")}
    counts, queries = Counter(), set()
    for row in base.read_rows(stt.DEFAULT_DATASET / "train.jsonl"):
        aug = row["augmentation"]
        parent = parents[aug["parent_id"]]
        assert aug["original_query"] == stt.clean_query(parent)
        assert aug["parent_query_sha256"] == base.digest(parent["query"].encode())
        assert aug["restored_parent_keyboard_typo"] == bool(parent["augmentation"])
        assert stt.replay(aug["original_query"], aug["operations"]) == row["query"]
        assert row["expected_object_id"] == parent["expected_object_id"]
        assert row["group_id"] == parent["group_id"]
        key = (row["catalog_id"], base.normalize_query(row["query"]))
        assert key not in queries
        queries.add(key)
        counts[aug["parent_id"]] += 1
    assert set(counts.values()) == {3}


def test_phonetic_name_collision_is_rejected_instead_of_inheriting_a_wrong_label():
    objects = [
        {"kind": "app", "name": "Лаверо", "aliases": ["Lavero"]},
        {"kind": "app", "name": "Лаворо", "aliases": ["Lavoro"]},
    ]
    assert not stt.unambiguous_brand("Лаверо", "Лаворо", objects)
    assert stt.unambiguous_brand("Лаверо", "Лаверо", [objects[0]])


@pytest.mark.parametrize("damage", ["wrong_target", "removed_qualifier"])
def test_rehashed_target_or_qualifier_changes_are_rejected(dataset, damage):
    rows = base.read_rows(dataset / "train.jsonl")
    row = next(r for r in rows if r["seed_id"] == "variant-03-2/q3" and r["augmentation"]["view"] == 1)
    if damage == "wrong_target":
        catalog = next(c for c in base.read_rows(dataset / "catalogs.jsonl") if c["catalog_id"] == row["catalog_id"])
        row["expected_object_id"] = next(o["object_id"] for o in catalog["objects"] if o["object_id"] != row["expected_object_id"])
    else:
        assert "64-битную" in row["query"]
        row["query"] = row["query"].replace("64-битную", "")
    rewrite(dataset, "train.jsonl", rows)
    with pytest.raises(BenchmarkError, match="parent labels/qualifiers/edit provenance"):
        stt.validate(dataset)


def test_asr_holdout_and_clean_annotation_cannot_be_removed_from_the_boundary(dataset):
    manifest = base.read_json(dataset / "manifest.json")
    manifest["evaluation_guard"] = [g for g in manifest["evaluation_guard"] if g["dataset_id"] != "launcher-asr-v1-synthetic"]
    base.write_json(dataset / "manifest.json", manifest)
    with pytest.raises(BenchmarkError, match="Evaluation profiles missing"):
        stt.validate(dataset)
    manifest["protocol"]["model_input"].append("augmentation.original_query")
    base.write_json(dataset / "manifest.json", manifest)
    with pytest.raises(BenchmarkError, match="Invalid STT model input"):
        stt.validate(dataset)


def test_decision_export_contains_only_noisy_query_and_all_safe_choices():
    rows = base.read_rows(stt.DEFAULT_DATASET / "train.jsonl")
    examples = list(itertools.islice(base.decision_examples(stt.DEFAULT_DATASET), 8))
    positions = []
    for row, example in zip(rows, examples, strict=False):
        assert example["state"] == {"query": row["query"]}
        question = example["questions"]["open"]
        assert len(question["criteria"]) == 177
        assert question["label"] == row["expected_object_id"] and question["label"] in question["criteria"]
        assert "augmentation" not in str(example) and "parent_id" not in str(example)
        positions.append(list(question["criteria"]).index(question["label"]))
    assert len(set(positions)) > 1
