import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.validate_benchmark import BenchmarkError, main, validate_bundle, validate_dataset

DATASET = Path(__file__).resolve().parents[1] / "data/benchmarks/launcher-v1/synthetic"
BASIC_DATASET = DATASET.parents[1] / "launcher-basic-v1/synthetic"


@pytest.fixture
def dataset(tmp_path):
    target = tmp_path / "benchmark"
    shutil.copytree(DATASET, target)
    return target


def read_rows(directory, split):
    return [
        json.loads(line)
        for line in (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def write_rows(directory, split, rows):
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (directory / f"{split}.jsonl").write_text(text, encoding="utf-8")
    refresh_file_hashes(directory)


def refresh_file_hashes(directory):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in manifest["files"]
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("directory", [DATASET, BASIC_DATASET], ids=["difficult", "basic"])
def test_shipped_fictional_benchmark_is_valid(directory):
    report = validate_dataset(directory)
    assert report["catalog_objects"] == 177
    assert report["distinct_open_targets"] == 50
    assert report["splits"]["validation"]["rows"] == 50
    assert report["splits"]["test"]["rows"] == 150
    assert report["splits"]["validation"]["actions"] == {"open": 50}
    assert report["splits"]["test"]["actions"] == {"open": 150}
    assert report["splits"]["validation"]["groups"] == 13
    assert report["splits"]["test"]["groups"] == 37
    assert report["annotation_review"]["independent"] is False


def test_basic_and_difficult_keep_the_same_targets_and_splits():
    reports = validate_bundle([DATASET, BASIC_DATASET])
    assert reports[0]["catalog_hash"] == reports[1]["catalog_hash"]
    families = []
    for directory in (DATASET, BASIC_DATASET):
        families.append({
            row["group_id"]: (split, row["expected_object_id"])
            for split in ("validation", "test") for row in read_rows(directory, split)
        })
    assert families[0] == families[1]


def test_file_tampering_is_rejected(dataset):
    with (dataset / "test.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(BenchmarkError, match="file hash mismatch"):
        validate_dataset(dataset)


@pytest.mark.parametrize(
    ("split", "change", "message"),
    [
        ("validation", lambda rows: rows[0].update(expected_object_id="absent"), "unknown target"),
        ("validation", lambda rows: rows[0].update(ambiguity_object_ids=[]), "invalid row fields"),
        ("validation", lambda rows: rows[0].update(expected_object_id=None), "unknown target"),
        ("validation", lambda rows: rows[0].update(query="  "), "empty actionable query"),
        ("test", lambda rows: rows[1].update(id=rows[0]["id"]), "duplicate example ID"),
        ("test", lambda rows: rows[1].update(
            query="  " + rows[0]["query"].upper() + "!!  "
        ), "duplicate normalized query"),
        ("test", lambda rows: rows[0].update(input_type="asr"), "only contains text"),
        ("test", lambda rows: rows.pop(), "unexpected action counts"),
    ],
)
def test_invalid_annotations_are_rejected(dataset, split, change, message):
    rows = read_rows(dataset, split)
    change(rows)
    write_rows(dataset, split, rows)
    with pytest.raises(BenchmarkError, match=message):
        validate_dataset(dataset)


@pytest.mark.parametrize("action", ["clarify", "not_found", "no_action"])
def test_non_selection_actions_are_rejected(dataset, action):
    rows = read_rows(dataset, "test")
    rows[0].update(expected_action=action, expected_object_id=None)
    write_rows(dataset, "test", rows)
    with pytest.raises(BenchmarkError, match="only contains open actions"):
        validate_dataset(dataset)


def test_family_cannot_cross_validation_and_test(dataset):
    validation = read_rows(dataset, "validation")
    test = read_rows(dataset, "test")
    test[0]["group_id"] = validation[0]["group_id"]
    write_rows(dataset, "test", test)
    with pytest.raises(BenchmarkError, match="group crosses splits"):
        validate_dataset(dataset)


def test_private_launch_fields_are_rejected(dataset):
    path = dataset / "catalog.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    catalog["objects"][0]["launch_target"] = "private-path"
    path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    refresh_file_hashes(dataset)
    with pytest.raises(BenchmarkError, match="invalid fields"):
        validate_dataset(dataset)


def test_catalog_canonical_hash_is_verified(dataset):
    path = dataset / "catalog.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    catalog["objects"][0]["description"] += " changed"
    path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    refresh_file_hashes(dataset)
    with pytest.raises(BenchmarkError, match="catalog hash mismatch"):
        validate_dataset(dataset)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("splits", [], "invalid split metadata"),
        ("annotation_review", {"status": "pending"}, "annotation review"),
        ("protocol", {"allowed_actions": ["open", "clarify"]}, "top-1 selection only"),
    ],
)
def test_invalid_manifest_is_rejected(dataset, field, value, message):
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(BenchmarkError, match=message):
        validate_dataset(dataset)


def make_second_catalog(dataset):
    other = dataset.parent / "second"
    shutil.copytree(dataset, other)
    manifest_path = other / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_id"] = "second-catalog"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for split in ("validation", "test"):
        rows = read_rows(other, split)
        for row in rows:
            row["id"] = "second-" + row["id"]
        write_rows(other, split, rows)
    return other


def test_shared_families_are_allowed_with_the_same_split(dataset):
    other = make_second_catalog(dataset)
    assert len(validate_bundle([dataset, other])) == 2


def test_family_cannot_cross_splits_between_catalogs(dataset):
    other = make_second_catalog(dataset)
    # Rename both complete families; each dataset remains valid on its own.
    for split in ("validation", "test"):
        rows = read_rows(other, split)
        for row in rows:
            if row["group_id"] == "p01":
                row["group_id"] = "renamed-validation-family"
            elif row["group_id"] == "p02":
                row["group_id"] = "p01"
        write_rows(other, split, rows)
    assert validate_dataset(other)["catalog_objects"] == 177
    with pytest.raises(BenchmarkError, match="across catalogs"):
        validate_bundle([dataset, other])


def test_cli_reports_errors_without_a_traceback(dataset, capsys):
    (dataset / "catalog.json").unlink()
    assert main([str(dataset)]) == 1
    assert "Invalid benchmark:" in capsys.readouterr().err
