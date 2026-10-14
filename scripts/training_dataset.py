"""Build, verify and export positive launcher training data using only the stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

try:
    from .validate_benchmark import (
        OBJECT_FIELDS,
        normalize_query,
        read_json,
        read_rows,
        require,
    )
except ImportError:  # Also support: python -I scripts/training_dataset.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_benchmark import (
        OBJECT_FIELDS,
        normalize_query,
        read_json,
        read_rows,
        require,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/training/launcher-train-v1"
DEFAULT_EXPORT = ROOT / "artifacts/training/launcher-train-v1/kev.train.jsonl"
HELDOUT = [
    ROOT / f"{parent}/{profile}/{kind}"
    for profile in ("launcher-v1", "launcher-basic-v1")
    for parent, kind in (("data/benchmarks", "synthetic"), ("artifacts/benchmarks", "real"))
]
TRAIN_FIELDS = {
    "id", "query", "catalog_id", "expected_action", "expected_object_id", "input_type",
    "group_id", "seed_id", "tags", "source", "rationale", "augmentation",
}
FILES = {"profiles.json", "catalogs.jsonl", "train.jsonl"}
KINDS = {"app": 139, "folder": 20, "file": 18}
INSTRUCTIONS = (
    "Выбери ровно один объект каталога, который пользователь просит открыть. "
    "Учитывай название, тип, описание, альтернативные названия и уточнение варианта. "
    "Запрос однозначен; правильный ответ — ключ соответствующего варианта."
)
MONTHS = [
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
]
PROJECT_ADJECTIVES = [
    "Липовая", "Облачная", "Медная", "Янтарная", "Ивовая", "Сосновая", "Песчаная", "Ледяная",
]
PROJECT_NOUNS = ["терраса", "долина", "бухта", "аллея", "гавань"]
BRAND_START = ["La", "Ve", "Ro", "Mi", "Sa", "Ta", "Ne", "Ba", "Fi", "Do", "Ka", "Pu"]
BRAND_MIDDLE = ["li", "ra", "vo", "na", "me", "si", "to", "la", "vi", "re", "mo", "nu"]
BRAND_END = ["ron", "lis", "van", "mera", "nora", "vika", "taro", "lino", "sona", "rena"]
TRANSLIT = str.maketrans({
    "a": "а", "b": "б", "d": "д", "e": "е", "f": "ф", "i": "и", "k": "к", "l": "л",
    "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в",
})


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def rng(key) -> random.Random:
    return random.Random(int(digest(str(key).encode()), 16))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def within_one_edit(first: str, second: str) -> bool:
    """Conservative screen: a noisy name must still identify one catalog object."""
    first, second = normalize_query(first), normalize_query(second)
    if abs(len(first) - len(second)) > 1:
        return False
    if len(first) == len(second):
        return sum(a != b for a, b in zip(first, second)) <= 1
    if len(first) > len(second):
        first, second = second, first
    position = next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), len(first))
    return first[position:] == second[position + 1:]


def screen_augmentation(row: dict, objects: list[dict]) -> None:
    if row["augmentation"] is None:
        return
    noisy_name = row["query"].removeprefix("Запусти ")
    matches = {
        obj["object_id"] for obj in objects
        if any(within_one_edit(noisy_name, name) for name in [obj["name"], *obj["aliases"]])
    }
    if matches != {row["expected_object_id"]}:
        row["query"] = row["augmentation"]["original_query"]
        row["augmentation"] = None
        row["tags"].remove("typo")
        row["source"] = "authored_pattern_instantiation"


def instances(source: dict) -> tuple[list[dict], list[dict]]:
    """Regenerate labels from authored profiles, never from predictions or eval labels."""
    catalogs, rows = [], []
    used_brands = set()
    for world in range(source["catalog_count"]):
        catalog_id = f"train-world-{world + 1:02}"
        randomizer = rng(f"{source['seed']}:{catalog_id}")
        brands = {}
        projects = [f"{PROJECT_ADJECTIVES[world]} {noun}" for noun in PROJECT_NOUNS]
        month1, month2 = MONTHS[world:world + 2]
        objects = []
        first_row = len(rows)
        for profile in source["profiles"]:
            profile_id = profile["profile_id"]
            brand_group = profile.get("brand_group")
            if brand_group and brand_group not in brands:
                while True:
                    brand = (
                        randomizer.choice(BRAND_START) + randomizer.choice(BRAND_MIDDLE)
                        + randomizer.choice(BRAND_END)
                    )
                    if brand not in used_brands:
                        used_brands.add(brand)
                        break
                translated = brand.lower().translate(TRANSLIT).capitalize()
                brands[brand_group] = (brand, translated) if world % 2 else (translated, brand)
            brand, brand_alias = brands.get(brand_group, ("", ""))
            context = {
                "brand": brand, "brand_alias": brand_alias,
                "project": projects[profile.get("project", 0)],
                "month1": month1, "month2": month2,
            }
            name = profile["name"].format(**context)
            alias = profile["alias"].format(**context)
            context.update(name=name, alias=alias)
            object_id = "obj_" + digest(
                f"{source['dataset_id']}:{source['seed']}:{catalog_id}:{profile_id}".encode()
            )[:20]
            objects.append({
                "object_id": object_id, "name": name, "kind": profile["kind"],
                "description": profile["description"].format(**context), "aliases": [alias],
            })
            for query_index, seed in enumerate(profile["queries"]):
                row_id = f"train-v1-w{world + 1:02}-{profile_id}-q{query_index + 1}"
                query = seed["text"].format(**context)
                augmentation = None
                tags = [profile["kind"], seed["tag"]]
                if profile["kind"] == "app" and query_index == 0:
                    noise = rng(row_id)
                    if noise.random() < 0.4:
                        position = noise.randrange(1, len(brand) - 1)
                        typo = brand[:position] + brand[position + 1:]
                        augmentation = {
                            "type": "single_character_deletion", "original_query": query,
                            "original_fragment": brand, "changed_fragment": typo,
                        }
                        query = query.replace(brand, typo, 1)
                        tags.append("typo")
                family = brand_group or f"project-{profile['project']}"
                rows.append({
                    "id": row_id, "query": query, "catalog_id": catalog_id,
                    "expected_action": "open", "expected_object_id": object_id,
                    "input_type": "text", "group_id": f"train-v1/{family}",
                    "seed_id": f"{profile_id}/q{query_index + 1}", "tags": tags,
                    "source": "controlled_augmentation" if augmentation else "authored_pattern_instantiation",
                    "rationale": profile["rationale"] + f" Цель: {name}.",
                    "augmentation": augmentation,
                })
        for row in rows[first_row:]:
            screen_augmentation(row, objects)
        randomizer.shuffle(objects)
        catalogs.append({
            "schema_version": 1, "catalog_id": catalog_id,
            "catalog_hash": digest(encoded(objects)), "objects": objects,
        })
    rng(f"{source['seed']}:row_order").shuffle(rows)
    return catalogs, rows


def evaluation_guard(catalogs: list[dict], rows: list[dict], directories) -> list[dict]:
    train_ids = {obj["object_id"] for catalog in catalogs for obj in catalog["objects"]}
    train_names = {normalize_query(obj["name"]) for c in catalogs for obj in c["objects"]}
    train_queries = {normalize_query(row["query"]) for row in rows}
    result = []
    for directory in directories:
        directory = Path(directory)
        catalog = read_json(directory / "catalog.json")
        evaluation = [row for split in ("validation", "test") for row in read_rows(directory / f"{split}.jsonl")]
        targets = {row["expected_object_id"] for row in evaluation}
        target_names = {normalize_query(o["name"]) for o in catalog["objects"] if o["object_id"] in targets}
        queries = {normalize_query(row["query"]) for row in evaluation}
        require(not train_ids & targets, f"{directory}: held-out target ID leakage")
        require(not train_names & target_names, f"{directory}: held-out target name leakage")
        require(not train_queries & queries, f"{directory}: held-out query leakage")
        result.append({
            "dataset_id": read_json(directory / "manifest.json")["dataset_id"],
            "path": directory.relative_to(ROOT).as_posix(),
            "files": {
                name: digest((directory / name).read_bytes())
                for name in ("catalog.json", "validation.jsonl", "test.jsonl", "manifest.json")
            },
            "query_rows": len(evaluation), "held_out_targets": len(targets),
            "overlap": {"target_ids": 0, "target_names": 0, "normalized_queries": 0},
        })
    return result


def stats(catalogs: list[dict], rows: list[dict]) -> dict:
    return {
        "catalogs": len(catalogs), "objects_per_catalog": dict(Counter(len(c["objects"]) for c in catalogs)),
        "rows": len(rows), "seed_queries": len({row["seed_id"] for row in rows}),
        "target_families": len({row["group_id"] for row in rows}),
        "target_objects": len({row["expected_object_id"] for row in rows}),
        "unique_normalized_queries": len({normalize_query(row["query"]) for row in rows}),
        "actions": dict(Counter(row["expected_action"] for row in rows)),
        "kinds": dict(Counter(row["tags"][0] for row in rows)),
        "sources": dict(Counter(row["source"] for row in rows)),
        "tags": dict(Counter(tag for row in rows for tag in row["tags"])),
    }


def build(directory: Path) -> dict:
    source = read_json(directory / "profiles.json")
    catalogs, rows = instances(source)
    # Read evaluation only to exclude overlap and freeze hashes, never to author train.
    guard = evaluation_guard(catalogs, rows, [path for path in HELDOUT if path.is_dir()])
    require(len(guard) >= 2, "The two portable held-out datasets are required")
    write_rows(directory / "catalogs.jsonl", catalogs)
    write_rows(directory / "train.jsonl", rows)
    manifest = {
        "schema_version": 1, "dataset_id": source["dataset_id"],
        "created_on": source["created_on"], "privacy": "shareable_fictional",
        "split": "train_only", "files": {name: digest((directory / name).read_bytes()) for name in sorted(FILES)},
        "statistics": stats(catalogs, rows), "seed": source["seed"],
        "query_source": "authored_patterns_with_controlled_instantiation_and_typos",
        "annotation_review": {
            "status": "author_reviewed", "reviewer": "Codex (dataset author)",
            "reviewed_on": source["created_on"], "independent": False,
            "seed_queries": sum(len(p["queries"]) for p in source["profiles"]),
            "method": "Author review of profile purposes, explicit variant qualifiers and file/folder distinctions; deterministic instantiation and structural checks of all rows.",
        },
        "protocol": {
            "task": "Select one uniquely specified catalog object to open",
            "allowed_actions": ["open"], "require_unique_target": True,
            "model_input": ["query", "full_catalog.objects"],
            "private_annotation_fields": sorted(TRAIN_FIELDS - {"query"}),
            "candidate_order": "Deterministic independent shuffle per row ID at export",
            "evaluation": "Frozen launcher-basic-v1 and launcher-v1, reported separately; no random row split of this training corpus",
        },
        "evaluation_guard": guard,
        "limitations": [
            "Synthetic author-written data; no production logs, teacher responses, independent human annotation or real ASR.",
            "5664 rows are instantiations of 708 seed queries, not 5664 independent linguistic scenarios.",
            "Object/name and exact normalized query separation does not imply unseen intents or surface command templates.",
            "Only forced unique open choices; no evidence about uncertainty, clarification or refusal.",
            "No checkpoint trained or model runtime checked; tokenizer counts do not establish model quality or GPU compatibility.",
            "Kev's default training context cannot hold 177 options; the checked export requires --max_state 16384 with the documented tokenizer.",
        ],
        "kev_format_reference": "https://github.com/jaredpalmer/kev#by-hand",
    }
    write_json(directory / "manifest.json", manifest)
    return validate(directory)


def validate(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("schema_version") == 1 and manifest.get("split") == "train_only", "invalid training manifest")
    require(set(manifest.get("files", {})) == FILES, "invalid training file list")
    for name, expected in manifest["files"].items():
        require(digest((directory / name).read_bytes()) == expected, f"{name}: file hash mismatch")
    source = read_json(directory / "profiles.json")
    require(len(source["profiles"]) == 177 and source["catalog_count"] == 8, "invalid source size")
    require(len({p["profile_id"] for p in source["profiles"]}) == 177, "duplicate profile ID")
    require(all(len(p["queries"]) == 4 for p in source["profiles"]), "invalid seed query count")
    catalogs, rows = read_rows(directory / "catalogs.jsonl"), read_rows(directory / "train.jsonl")
    require(len(catalogs) == 8 and len(rows) == 5664, "invalid training row count")
    by_catalog = {}
    all_ids = set()
    for catalog in catalogs:
        require(set(catalog) == {"schema_version", "catalog_id", "catalog_hash", "objects"}, "invalid catalog fields")
        catalog_id, objects = catalog["catalog_id"], catalog["objects"]
        require(catalog_id not in by_catalog, "duplicate catalog ID")
        require(digest(encoded(objects)) == catalog["catalog_hash"], "catalog hash mismatch")
        require(Counter(o["kind"] for o in objects) == KINDS, "invalid catalog kind counts")
        require(len({normalize_query(o["name"]) for o in objects}) == 177, "duplicate object name")
        require(len({normalize_query(a) for o in objects for a in o["aliases"]}) == 177, "duplicate alias")
        for obj in objects:
            require(set(obj) == OBJECT_FIELDS, "unsafe catalog fields")
            require(obj["object_id"] not in all_ids, "duplicate training object ID")
            all_ids.add(obj["object_id"])
            require(all(isinstance(obj[key], str) and obj[key].strip() for key in ("object_id", "name", "kind", "description")), "empty object metadata")
            require(isinstance(obj["aliases"], list) and obj["aliases"] and all(isinstance(a, str) and a.strip() for a in obj["aliases"]), "invalid aliases")
        by_catalog[catalog_id] = {o["object_id"]: o for o in objects}
    example_ids, queries = set(), set()
    for row in rows:
        require(set(row) == TRAIN_FIELDS, "invalid training row fields")
        require(row["id"] not in example_ids, "duplicate training example ID")
        example_ids.add(row["id"])
        require(row["catalog_id"] in by_catalog, "unknown catalog")
        obj = by_catalog[row["catalog_id"]].get(row["expected_object_id"])
        require(obj is not None, "unknown target")
        require(row["expected_action"] == "open" and row["input_type"] == "text", "invalid action/input")
        require(isinstance(row["query"], str) and row["query"].strip(), "empty query")
        require(row["tags"][0] == obj["kind"], "target kind mismatch")
        query_key = (row["catalog_id"], normalize_query(row["query"]))
        require(query_key not in queries, "duplicate query in one catalog")
        queries.add(query_key)
    # Detect changed labels, missing qualifiers and augmentation provenance, even after rehashing.
    expected_catalogs, expected_rows = instances(source)
    require(catalogs == expected_catalogs, "catalog does not match authored profiles")
    require(rows == expected_rows, "rows do not match authored seed labels/provenance")
    require(encoded(stats(catalogs, rows)) == encoded(manifest["statistics"]), "statistics mismatch")
    require(manifest["protocol"]["allowed_actions"] == ["open"] and manifest["protocol"]["require_unique_target"] is True, "invalid training protocol")
    require(manifest["annotation_review"]["independent"] is False, "unsupported independent annotation claim")
    checked, unavailable = [], []
    frozen_ids = [item["dataset_id"] for item in manifest["evaluation_guard"]]
    require(len(frozen_ids) == len(set(frozen_ids)), "duplicate evaluation guard")
    require(
        {"launcher-v1-synthetic", "launcher-basic-v1-synthetic"} <= set(frozen_ids),
        "both public evaluation profiles must remain protected",
    )
    for frozen in manifest["evaluation_guard"]:
        path = ROOT / frozen["path"]
        if not path.is_dir():
            require(frozen["path"].startswith("artifacts/"), "missing public evaluation dataset")
            unavailable.append(frozen["dataset_id"])
            continue
        for name, expected in frozen["files"].items():
            require(digest((path / name).read_bytes()) == expected, f"{path / name}: held-out data changed")
        current = evaluation_guard(catalogs, rows, [path])[0]
        require(current == frozen, "held-out guard mismatch")
        checked.append(frozen["dataset_id"])
    require(len(checked) >= 2, "held-out evaluation guard missing")
    return {
        "valid": True, "dataset_id": manifest["dataset_id"], **manifest["statistics"],
        "evaluation_checked": checked, "evaluation_unavailable": unavailable,
        "semantic_review": "author_only; structural validity is not independent label verification",
    }


def decision_examples(directory: Path):
    """No annotations/rationale in state or criteria; the answer is only label."""
    catalogs = {c["catalog_id"]: c["objects"] for c in read_rows(directory / "catalogs.jsonl")}
    for row in read_rows(directory / "train.jsonl"):
        objects = list(catalogs[row["catalog_id"]])
        rng(f"choice_order:{row['id']}").shuffle(objects)
        criteria = {}
        for obj in objects:
            criteria[obj["object_id"]] = (
                f"[{obj['kind']}] {obj['name']}. {obj['description']} "
                f"Альтернативное название: {', '.join(obj['aliases'])}."
            )
        yield {
            "state": {"query": row["query"]},
            "questions": {"open": {
                "type": "choice", "instructions": INSTRUCTIONS,
                "criteria": criteria, "label": row["expected_object_id"],
            }},
        }


def export(directory: Path, output: Path) -> dict:
    report = validate(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    positions, criteria_chars = Counter(), []
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for example in decision_examples(directory):
            question = example["questions"]["open"]
            positions[list(question["criteria"]).index(question["label"])] += 1
            criteria_chars.append(sum(len(k) + len(v) for k, v in question["criteria"].items()))
            stream.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "dataset_id": report["dataset_id"], "format": "kev_labeled_choice",
        "rows": report["rows"], "options_per_row": 177,
        "sha256": digest(output.read_bytes()), "bytes": output.stat().st_size,
        "source_manifest_sha256": digest((directory / "manifest.json").read_bytes()),
        "label_position_counts": dict(sorted(positions.items())),
        "criteria_characters": {"min": min(criteria_chars), "max": max(criteria_chars)},
        "tokenizer_checked": False, "training_run": False,
    }
    tokenization = output.with_suffix(".tokenization.json")
    if tokenization.is_file():
        budget = read_json(tokenization)
        expected_sources = {name: digest((directory / name).read_bytes()) for name in sorted(FILES)}
        if (
            budget.get("export_sha256") == summary["sha256"]
            and budget.get("source_files") == expected_sources
            and budget.get("rows_fit_recommended_context") == summary["rows"]
        ):
            summary.update(
                tokenizer_checked=True,
                tokenization_report_sha256=digest(tokenization.read_bytes()),
                required_training_flag=budget["recommended_training_flag"],
                full_row_tokens=budget["full_row_tokens"],
                tokenizer_model=budget["tokenizer_model"],
                tokenizer_revision=budget["tokenizer_revision"],
            )
    write_json(output.with_suffix(".manifest.json"), summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "export"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORT)
    args = parser.parse_args(argv)
    try:
        result = build(args.dataset) if args.command == "build" else (
            validate(args.dataset) if args.command == "validate" else export(args.dataset, args.output)
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    if "label_position_counts" in result:
        counts = result["label_position_counts"]
        result = {key: value for key, value in result.items() if key != "label_position_counts"}
        result["label_positions"] = {"covered": len(counts), "min": min(counts.values()), "max": max(counts.values())}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
