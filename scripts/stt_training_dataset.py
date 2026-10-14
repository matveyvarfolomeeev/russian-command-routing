"""Reproducible, training-only text simulations of STT errors; no API calls."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    from . import training_dataset as base
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import training_dataset as base

ROOT = base.ROOT
DEFAULT_DATASET = ROOT / "data/training/launcher-stt-train-v1"
SOURCE = base.DEFAULT_DATASET
DEFAULT_EXPORT = ROOT / "artifacts/training/launcher-stt-train-v1/kev.train.jsonl"
FILES = {"operators.json", "catalogs.jsonl", "train.jsonl"}
WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+")
PROTECTED = re.compile(
    r"^(обыч|расшир|приват|стабил|тестов|установ|перенос|настоль|веб|стандарт|"
    r"профес|личн|рабоч|админ|гост|финал|чернов|архив|актив|исход|отчет|отчёт|"
    r"январ|феврал|март|апрел|май|июн|июл|август|сентябр|октябр|ноябр|декабр)",
    re.IGNORECASE,
)
CONFIG = {
    "schema_version": 1, "dataset_id": "launcher-stt-train-v1", "created_on": "2026-10-04",
    "seed": "stt-text-v1-20261004", "views_per_parent": 3,
    "phonetic_pairs": {
        "а": "о", "о": "а", "е": "и", "и": "е", "б": "п", "п": "б",
        "д": "т", "т": "д", "в": "ф", "ф": "в", "с": "з", "з": "с", "к": "г", "г": "к",
        "a": "o", "o": "a", "e": "i", "i": "e", "b": "p", "p": "b", "d": "t",
        "t": "d", "v": "f", "f": "v", "s": "z", "z": "s", "k": "g", "g": "k",
    },
    "spoken_numbers": {"32": "тридцать два", "64": "шестьдесят четыре", "2026": "две тысячи двадцать шесть"},
    "spoken_acronyms": {
        "PDF": "пи ди эф", "DOCX": "док икс", "XLSX": "икс эл эс икс", "CSV": "си эс ви",
        "TXT": "ти икс ти", "SQL": "эс кью эл", "DNS": "ди эн эс", "SMART": "смарт",
        "COM": "ком", "3D": "три дэ", "ping": "пинг",
    },
    "fillers": ["ну ", "так ", "эм ", "слушай ", "пожалуйста "],
    "simulation": "Plausible text operators, not measured vendor error probabilities or actual ASR responses",
}


def clean_query(row: dict) -> str:
    """Do not stack keyboard typos from the parent dataset with speech errors."""
    return (row["augmentation"] or {}).get("original_query", row["query"])


def operation(text: str, start: int, end: int, replacement: str, kind: str) -> dict:
    return {"type": kind, "start": start, "end": end, "original": text[start:end], "replacement": replacement}


def replay(text: str, operations: list[dict]) -> str:
    for op in operations:
        base.require(0 <= op["start"] <= op["end"] <= len(text), "Invalid STT edit span")
        base.require(text[op["start"]:op["end"]] == op["original"], "STT edit provenance mismatch")
        text = text[:op["start"]] + op["replacement"] + text[op["end"]:]
    return text


def brand_names(obj: dict) -> list[str]:
    if obj["kind"] != "app":
        return []
    return [obj["name"].split(" — ")[0], *(a.split()[0] for a in obj["aliases"])]


def unambiguous_brand(original: str, changed: str, objects: list[dict]) -> bool:
    """Variants share a brand; their unchanged qualifier still chooses the variant."""
    original = base.normalize_query(original)
    rivals = {
        base.normalize_query(name) for obj in objects for name in brand_names(obj)
        if base.normalize_query(name) != original
        and bool(re.search(r"[а-я]", name.lower())) == bool(re.search(r"[а-я]", original))
    }
    return not any(base.within_one_edit(changed, name) for name in rivals)


def content_edits(text: str, obj: dict, objects: list[dict], key: str, rejected: Counter | None = None) -> list[dict]:
    result, randomizer = [], base.rng(key + ":content")
    names = brand_names(obj)
    pairs = CONFIG["phonetic_pairs"]
    for name in names:
        match = re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE)
        if not match:
            continue
        positions = list(range(1, len(name) - 1))
        randomizer.shuffle(positions)
        actual = match.group()
        for position in positions:
            changed_letter = pairs.get(actual[position].lower())
            if not changed_letter:
                continue
            changed = actual[:position] + changed_letter + actual[position + 1:]
            if unambiguous_brand(actual, changed, objects):
                result.append(operation(text, match.start(), match.end(), changed, "phonetic_name"))
                break
            if rejected is not None:
                rejected["ambiguous_phonetic_name"] += 1
    # One vowel substitution in a long word; preserve product names, project names,
    # variant qualifiers, months, file formats and negation exactly.
    protected_words = {
        word.group().lower() for value in [obj["name"], *obj["aliases"]]
        for word in WORD.finditer(value)
    }
    words = [
        word for word in WORD.finditer(text)
        if len(word.group()) >= 7 and word.group().lower() not in protected_words
        and not PROTECTED.match(word.group())
    ]
    randomizer.shuffle(words)
    for word in words:
        value = word.group()
        positions = [i for i in range(1, len(value) - 1) if value[i].lower() in "аоеи"]
        if positions:
            position = randomizer.choice(positions)
            changed = value[:position] + pairs[value[position].lower()] + value[position + 1:]
            result.append(operation(text, word.start(), word.end(), changed, "phonetic_word"))
            break
    # An inflection of the command is common in transcripts and cannot change the target.
    for original, changed in (
        ("Открой", "Открою"), ("Запусти", "Запустить"), ("Хочу", "Хочет"),
        ("Нужно", "Нужна"), ("Нужен", "Нужна"), ("Нужна", "Нужно"), ("Мне", "Меня"),
    ):
        if text.startswith(original + " "):
            result.append(operation(text, 0, len(original), changed, "command_mishearing"))
            break
    return result


def representation_edits(text: str, obj: dict, key: str) -> list[dict]:
    result, randomizer = [], base.rng(key + ":representation")
    for original, replacement in CONFIG["spoken_numbers"].items():
        match = re.search(r"(?<!\d)" + original + r"(?!\d)", text)
        if match:
            result.append(operation(text, match.start(), match.end(), replacement, "spoken_number"))
    for original, replacement in CONFIG["spoken_acronyms"].items():
        match = re.search(r"(?<!\w)" + original + r"(?!\w)", text, re.IGNORECASE)
        if match:
            result.append(operation(text, match.start(), match.end(), replacement, "spoken_acronym"))
    names = brand_names(obj)
    if len(names) == 2:
        for index, name in enumerate(names):
            match = re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE)
            if match:
                result.append(operation(text, match.start(), match.end(), names[1 - index], "script_transcription"))
                break
    words = list(WORD.finditer(text))
    long_words = [w for w in words if len(w.group()) >= 6 and w.start() != 0]
    randomizer.shuffle(long_words)
    if long_words:
        word = long_words[0]
        position = randomizer.randrange(2, len(word.group()) - 1)
        result.append(operation(text, word.start() + position, word.start() + position, " ", "word_split"))
    spaces = [
        (left.end(), right.start()) for left, right in zip(words, words[1:])
        if text[left.end():right.start()] == " " and min(len(left.group()), len(right.group())) >= 3
        and left.group().lower() not in ("нет", "без") and right.group().lower() not in ("нет", "без")
    ]
    if spaces:
        start, end = randomizer.choice(spaces)
        result.append(operation(text, start, end, "", "word_merge"))
    return result


def surface_edits(text: str, key: str) -> list[dict]:
    randomizer = base.rng(key + ":surface")
    filler = randomizer.choice(CONFIG["fillers"])
    result = [operation(text, 0, 0, filler, "filler")]
    words = list(WORD.finditer(text))
    if words:
        word = randomizer.choice(words)
        result.append(operation(text, word.start(), word.start(), word.group() + " ", "word_repeat"))
    match = re.match(r"(?:Открой|Запусти|Нужен|Нужна|Нужно|Хочу открыть) ", text)
    if match:
        result.append(operation(text, 0, match.end(), "", "command_dropout"))
    transcript_style = re.sub(r"[«»\".,:;!?—–-]", " ", text.lower())
    transcript_style = re.sub(r"\s+", " ", transcript_style).strip()
    if transcript_style != text:
        result.append(operation(text, 0, len(text), transcript_style, "punctuation_and_case"))
    randomizer.shuffle(result)
    return result


def instances(catalogs: list[dict], parents: list[dict]) -> tuple[list[dict], dict]:
    by_catalog = {c["catalog_id"]: c["objects"] for c in catalogs}
    used = {(r["catalog_id"], base.normalize_query(clean_query(r))) for r in parents}
    used |= {(r["catalog_id"], base.normalize_query(r["query"])) for r in parents}
    rows, rejected, operator_counts = [], Counter(), Counter()
    for parent in sorted(parents, key=lambda r: r["id"]):
        text, key = clean_query(parent), parent["id"]
        objects = by_catalog[parent["catalog_id"]]
        obj = next(o for o in objects if o["object_id"] == parent["expected_object_id"])
        content = content_edits(text, obj, objects, key, rejected)
        representation = representation_edits(text, obj, key)
        # Prefer changes in the target's name over a command inflection.
        content.sort(key=lambda op: (op["type"] != "phonetic_name", op["type"] == "command_mishearing"))
        # Include less frequent numbers and acronyms whenever applicable.
        representation.sort(key=lambda op: (op["type"] not in ("spoken_number", "spoken_acronym"), operator_counts[op["type"]]))
        composed = []
        for first in content + representation:
            changed = replay(text, [first])
            for second in surface_edits(changed, key + ":" + first["type"]):
                composed.append([first, second])
        pools = [[[op] for op in content], [[op] for op in representation], composed]
        for view, pool in enumerate(pools, 1):
            accepted = None
            for operations in pool:
                query = replay(text, operations)
                query_key = (parent["catalog_id"], base.normalize_query(query))
                if query_key in used:
                    rejected["duplicate_or_clean_query"] += 1
                    continue
                accepted = (operations, query, query_key)
                break
            base.require(accepted is not None, f"No safe STT view {view} for {key}")
            operations, query, query_key = accepted
            used.add(query_key)
            for op in operations:
                operator_counts[op["type"]] += 1
            rows.append({
                **parent, "id": parent["id"].replace("train-v1-", "stt-train-v1-") + f"-v{view}",
                "query": query, "input_type": "simulated_asr_text", "source": "simulated_stt_text",
                "tags": [t for t in parent["tags"] if t != "typo"] + ["stt_simulation", *dict.fromkeys(op["type"] for op in operations)],
                "augmentation": {
                    "type": "simulated_stt_text", "parent_id": parent["id"], "original_query": text,
                    "parent_query_sha256": base.digest(parent["query"].encode()),
                    "restored_parent_keyboard_typo": parent["augmentation"] is not None,
                    "operations": operations, "view": view,
                    "severity": "composed" if len(operations) > 1 else "single",
                },
            })
    base.rng(CONFIG["seed"] + ":order").shuffle(rows)
    return rows, {"rejected_candidates": dict(rejected), "operator_occurrences": dict(operator_counts)}


def heldout_paths() -> list[Path]:
    paths = [p for p in base.HELDOUT if p.is_dir()]
    paths += [p for p in (
        ROOT / "data/benchmarks/launcher-asr-v1/synthetic",
        ROOT / "artifacts/benchmarks/launcher-asr-v1/real",
    ) if (p / "manifest.json").is_file()]
    return paths


def build(directory: Path) -> dict:
    base.validate(SOURCE)
    catalogs, parents = base.read_rows(SOURCE / "catalogs.jsonl"), base.read_rows(SOURCE / "train.jsonl")
    rows, details = instances(catalogs, parents)
    guard = base.evaluation_guard(catalogs, rows, heldout_paths())
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE / "catalogs.jsonl", directory / "catalogs.jsonl")
    base.write_json(directory / "operators.json", CONFIG)
    base.write_rows(directory / "train.jsonl", rows)
    manifest = {
        "schema_version": 1, "dataset_id": CONFIG["dataset_id"], "created_on": CONFIG["created_on"],
        "privacy": "shareable_fictional", "split": "train_only", "query_source": "artificial_stt_text_errors",
        "files": {name: base.digest((directory / name).read_bytes()) for name in sorted(FILES)},
        "parent_dataset": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "files": {name: base.digest((SOURCE / name).read_bytes()) for name in sorted(base.FILES | {"manifest.json"})},
            "rows": len(parents), "clean_queries": len({(r["catalog_id"], clean_query(r)) for r in parents}),
        },
        "statistics": {**base.stats(catalogs, rows), **details,
            "parent_rows": len({r["augmentation"]["parent_id"] for r in rows}),
            "severity": dict(Counter(r["augmentation"]["severity"] for r in rows)),
        },
        "evaluation_guard": guard,
        "protocol": {
            "allowed_actions": ["open"], "require_unique_target": True,
            "model_input": ["query", "full_catalog.objects"],
            "private_annotation_fields": sorted(base.TRAIN_FIELDS - {"query"}),
            "candidate_order": "Deterministic independent shuffle per row ID at export",
            "split_rule": "Train only; related views retain parent group_id. Never randomly split augmented rows.",
            "label_preservation": "Retain the original target; keep all distinguishing qualifiers. Phonetic name edits are screened against rival brands; no deletion of identity words or arbitrary hallucinations.",
        },
        "annotation_review": {
            "status": "author_reviewed_rules_and_parent_labels", "independent": False,
            "reviewer": "Codex (dataset author)",
            "method": "Inherited authored parent labels; reversible boundary/script/number edits, protected variant words and conservative one-edit brand screen. No independent human annotation of each generated row.",
        },
        "limitations": [
            "No speech/audio or STT API used to generate these training rows; simulated errors are not a measured provider confusion distribution.",
            "Augmented views and eight catalog instantiations are correlated; row count is not the number of independent linguistic scenarios.",
            "Exact query/target separation is checked against frozen evaluation data; shared task purposes and command templates remain.",
            "Only recoverable, uniquely intended open requests; unrecoverable ASR, clarification and refusal are outside this training version.",
            "No model training or GPU runtime verification performed.",
        ],
    }
    base.write_json(directory / "manifest.json", manifest)
    return validate(directory)


def validate(directory: Path) -> dict:
    manifest = base.read_json(directory / "manifest.json")
    base.require(manifest["dataset_id"] == CONFIG["dataset_id"] and manifest["split"] == "train_only", "Invalid STT training manifest")
    base.require(set(manifest["files"]) == FILES, "Invalid STT file list")
    for name, expected in manifest["files"].items():
        base.require(base.digest((directory / name).read_bytes()) == expected, f"{name}: file hash mismatch")
    base.require(base.read_json(directory / "operators.json") == CONFIG, "STT operator rules changed")
    source = ROOT / manifest["parent_dataset"]["path"]
    base.require(source == SOURCE, "STT parent must be the frozen launcher-train-v1")
    for name, expected in manifest["parent_dataset"]["files"].items():
        base.require(base.digest((source / name).read_bytes()) == expected, f"{name}: frozen parent training dataset changed")
    base.require((directory / "catalogs.jsonl").read_bytes() == (source / "catalogs.jsonl").read_bytes(), "STT catalogs differ from parent")
    catalogs, parents = base.read_rows(source / "catalogs.jsonl"), base.read_rows(source / "train.jsonl")
    rows = base.read_rows(directory / "train.jsonl")
    base.require(manifest["protocol"]["model_input"] == ["query", "full_catalog.objects"] and manifest["protocol"]["allowed_actions"] == ["open"], "Invalid STT model input")
    frozen_ids = {g["dataset_id"] for g in manifest["evaluation_guard"]}
    required = {"launcher-v1-synthetic", "launcher-basic-v1-synthetic"}
    if (ROOT / "data/benchmarks/launcher-asr-v1/synthetic/manifest.json").is_file():
        required.add("launcher-asr-v1-synthetic")
    base.require(required <= frozen_ids, "Evaluation profiles missing from STT guard")
    expected_rows, details = instances(catalogs, parents)
    base.require(rows == expected_rows, "STT rows differ from deterministic parent labels/qualifiers/edit provenance")
    base.require(all(set(r) == base.TRAIN_FIELDS and r["expected_action"] == "open" and r["input_type"] == "simulated_asr_text" for r in rows), "Invalid STT row schema/action/input")
    statistics = {**base.stats(catalogs, rows), **details,
        "parent_rows": len({r["augmentation"]["parent_id"] for r in rows}),
        "severity": dict(Counter(r["augmentation"]["severity"] for r in rows)),
    }
    base.require(base.encoded(statistics) == base.encoded(manifest["statistics"]), "STT statistics mismatch")
    checked, unavailable = [], []
    for guard in manifest["evaluation_guard"]:
        path = ROOT / guard["path"]
        if not path.is_dir():
            base.require(guard["path"].startswith("artifacts/"), "Public held-out dataset missing")
            unavailable.append(guard["dataset_id"])
            continue
        base.require(base.evaluation_guard(catalogs, rows, [path])[0] == guard, "Frozen STT held-out guard mismatch")
        checked.append(guard["dataset_id"])
    return {"valid": True, "dataset_id": manifest["dataset_id"], **statistics,
        "evaluation_checked": checked, "evaluation_unavailable": unavailable,
        "actual_stt": False, "semantic_review": "author_rules_and_parent_labels; no independent annotation"}


def export(directory: Path, output: Path) -> dict:
    report = validate(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    base.write_rows(output, base.decision_examples(directory))
    summary = {"dataset_id": report["dataset_id"], "format": "kev_labeled_choice",
        "rows": report["rows"], "options_per_row": 177, "sha256": base.digest(output.read_bytes()),
        "bytes": output.stat().st_size, "source_manifest_sha256": base.digest((directory / "manifest.json").read_bytes()),
        "tokenizer_checked": False, "training_run": False}
    tokenization = output.with_suffix(".tokenization.json")
    if tokenization.is_file():
        budget = base.read_json(tokenization)
        source_files = {name: base.digest((directory / name).read_bytes()) for name in sorted(FILES)}
        if budget.get("export_sha256") == summary["sha256"] and budget.get("source_files") == source_files and budget.get("rows_fit_recommended_context") == summary["rows"]:
            summary.update(tokenizer_checked=True, tokenization_report_sha256=base.digest(tokenization.read_bytes()),
                required_training_flag=budget["recommended_training_flag"], full_row_tokens=budget["full_row_tokens"],
                tokenizer_model=budget["tokenizer_model"], tokenizer_revision=budget["tokenizer_revision"])
    base.write_json(output.with_suffix(".manifest.json"), summary)
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
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
