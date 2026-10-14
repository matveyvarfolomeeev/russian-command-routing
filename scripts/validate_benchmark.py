"""Validate frozen launcher datasets; never imports the launcher or calls a model.

Usage: python scripts/validate_benchmark.py DATASET_DIR [DATASET_DIR ...]
Supplying several datasets also checks families across catalogs and difficulty sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

FILES = {"catalog.json", "validation.jsonl", "test.jsonl"}
OBJECT_FIELDS = {"object_id", "name", "kind", "description", "aliases"}
ROW_FIELDS = {
    "id", "query", "expected_action", "expected_object_id", "input_type",
    "group_id", "tags", "source", "rationale",
}
COUNTS = {
    "validation": {"open": 50},
    "test": {"open": 150},
}
KINDS = {"app": 139, "folder": 20, "file": 18}


class BenchmarkError(ValueError):
    """An invalid benchmark, with a message suitable for the CLI."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkError(message)


def nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def strings(value) -> bool:
    return (
        isinstance(value, list)
        and all(nonempty(item) for item in value)
        and len(set(value)) == len(value)
    )


def normalize_query(query: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", query).casefold().replace("ё", "е").split()
    ).strip(" .!?…")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise BenchmarkError(f"{path}: {exc}") from exc


def read_rows(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BenchmarkError(f"{path}: {exc}") from exc
    rows = []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise BenchmarkError(f"{path}:{number}: invalid JSON: {exc}") from exc
        require(isinstance(row, dict), f"{path}:{number}: expected a JSON object")
        rows.append(row)
    return rows


def _validate_dataset(directory: Path) -> tuple[dict, list[tuple[str, dict]]]:
    manifest = read_json(directory / "manifest.json")
    require(isinstance(manifest, dict), f"{directory}: invalid manifest")
    require(manifest.get("schema_version") == 1, f"{directory}: unsupported manifest schema")
    require(nonempty(manifest.get("dataset_id")), f"{directory}: missing dataset_id")
    require(manifest.get("query_source") == "authored", f"{directory}: invalid query source")
    protocol = manifest.get("protocol")
    require(
        isinstance(protocol, dict) and protocol.get("allowed_actions") == ["open"]
        and protocol.get("metric") == "top1_accuracy"
        and protocol.get("require_unique_target") is True,
        f"{directory}: launcher-v1 requires a unique target and top-1 selection only",
    )
    split_metadata = manifest.get("splits")
    require(
        isinstance(split_metadata, dict) and set(split_metadata) == set(COUNTS),
        f"{directory}: invalid split metadata",
    )
    review = manifest.get("annotation_review")
    require(
        isinstance(review, dict) and review.get("status") == "author_reviewed"
        and review.get("reviewed_rows") == 200 and isinstance(review.get("independent"), bool),
        f"{directory}: missing completed annotation review declaration",
    )
    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == FILES, f"{directory}: invalid file list")
    for name, expected in files.items():
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            f"{name}: invalid SHA-256",
        )
        try:
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        except OSError as exc:
            raise BenchmarkError(f"{directory / name}: {exc}") from exc
        require(actual == expected, f"{directory / name}: file hash mismatch")

    catalog = read_json(directory / "catalog.json")
    require(
        isinstance(catalog, dict)
        and set(catalog) == {"schema_version", "catalog_hash", "objects"}
        and catalog["schema_version"] == 1
        and isinstance(catalog["objects"], list),
        f"{directory}: invalid catalog schema",
    )
    objects = catalog["objects"]
    object_ids = set()
    for number, obj in enumerate(objects, 1):
        where = f"catalog object {number}"
        require(isinstance(obj, dict) and set(obj) == OBJECT_FIELDS, f"{where}: invalid fields")
        require(nonempty(obj["object_id"]), f"{where}: missing object ID")
        require(obj["object_id"] not in object_ids, f"{where}: duplicate object ID")
        object_ids.add(obj["object_id"])
        require(nonempty(obj["name"]), f"{where}: missing name")
        require(isinstance(obj["kind"], str) and obj["kind"] in KINDS, f"{where}: invalid kind")
        require(isinstance(obj["description"], str), f"{where}: invalid description")
        require(strings(obj["aliases"]), f"{where}: invalid aliases")
    kinds = Counter(obj["kind"] for obj in objects)
    require(kinds == KINDS, f"{directory}: expected catalog kinds {KINDS}, got {dict(kinds)}")
    canonical = json.dumps(objects, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    catalog_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    require(catalog["catalog_hash"] == catalog_hash, f"{directory}: catalog hash mismatch")
    require(manifest.get("catalog_hash") == catalog_hash, f"{directory}: manifest catalog mismatch")
    require(manifest.get("catalog_counts") == dict(kinds), f"{directory}: manifest kinds mismatch")

    all_rows = []
    seen_ids, seen_queries = set(), set()
    groups = {}
    split_reports = {}
    by_id = {obj["object_id"]: obj for obj in objects}
    targets = set()
    for split, expected_counts in COUNTS.items():
        rows = read_rows(directory / f"{split}.jsonl")
        for number, row in enumerate(rows, 1):
            where = f"{directory.name}/{split}:{number}"
            require(set(row) == ROW_FIELDS, f"{where}: invalid row fields")
            for key in ("id", "group_id", "rationale"):
                require(nonempty(row[key]), f"{where}: invalid {key}")
            require(row["id"] not in seen_ids, f"{where}: duplicate example ID")
            seen_ids.add(row["id"])
            require(isinstance(row["query"], str), f"{where}: query must be text")
            query_key = normalize_query(row["query"])
            require(query_key not in seen_queries, f"{where}: duplicate normalized query")
            seen_queries.add(query_key)
            action = row["expected_action"]
            require(action == "open", f"{where}: launcher-v1 only contains open actions")
            require(bool(query_key), f"{where}: empty actionable query")
            require(row["input_type"] == "text", f"{where}: launcher-v1 only contains text")
            require(row["source"] == "authored", f"{where}: invalid source")
            require(strings(row["tags"]) and bool(row["tags"]), f"{where}: invalid tags")
            target = row["expected_object_id"]
            require(isinstance(target, str) and target in object_ids, f"{where}: unknown target ID")
            targets.add(target)
            group = row["group_id"]
            require(group not in groups or groups[group] == split, f"{where}: group crosses splits")
            groups[group] = split
            all_rows.append((split, row))
        counts = Counter(row["expected_action"] for row in rows)
        require(counts == expected_counts, f"{directory}/{split}: unexpected action counts {dict(counts)}")
        require(
            split_metadata[split] == {"rows": len(rows), "actions": dict(counts)},
            f"{directory}/{split}: manifest counts mismatch",
        )
        target_kinds = Counter(
            by_id[row["expected_object_id"]]["kind"]
            for row in rows
        )
        require(set(target_kinds) == set(KINDS), f"{directory}/{split}: missing target kind")
        split_reports[split] = {
            "rows": len(rows), "actions": dict(counts), "target_kinds": dict(target_kinds),
            "groups": len({row["group_id"] for row in rows}),
        }
    require(len(targets) >= 40, f"{directory}: fewer than 40 distinct open targets")
    require(manifest.get("group_count") == len(groups), f"{directory}: manifest group count mismatch")
    report = {
        "dataset_id": manifest["dataset_id"], "catalog_hash": catalog_hash,
        "catalog_objects": len(objects), "catalog_counts": dict(kinds),
        "distinct_open_targets": len(targets), "splits": split_reports,
        "annotation_review": review,
        "tags": dict(sorted(Counter(tag for _, row in all_rows for tag in row["tags"]).items())),
    }
    return report, all_rows


def validate_dataset(directory: Path | str) -> dict:
    return _validate_dataset(Path(directory))[0]


def validate_bundle(directories: list[Path | str]) -> list[dict]:
    reports, datasets, examples, groups, queries = [], set(), set(), {}, {}
    for directory in directories:
        report, rows = _validate_dataset(Path(directory))
        dataset_id = report["dataset_id"]
        require(dataset_id not in datasets, f"duplicate dataset ID: {dataset_id}")
        datasets.add(dataset_id)
        for split, row in rows:
            require(row["id"] not in examples, f"duplicate example ID across catalogs: {row['id']}")
            examples.add(row["id"])
            group = row["group_id"]
            require(
                group not in groups or groups[group] == split,
                f"group crosses splits across catalogs: {group}",
            )
            groups[group] = split
            query = normalize_query(row["query"])
            require(
                query not in queries or queries[query] == (split, group),
                "shared query crosses splits or families across catalogs",
            )
            queries[query] = split, group
        reports.append(report)
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        reports = validate_bundle(args.datasets)
    except BenchmarkError as exc:
        print(f"Invalid benchmark: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"valid": True, "datasets": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
