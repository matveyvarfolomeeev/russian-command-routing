"""Fresh authored catalog and queries, neural TTS and actual Large V3 ASR; no training."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

try:
    from . import asr_dataset as asr
    from .validate_benchmark import KINDS, validate_dataset
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import asr_dataset as asr
    from validate_benchmark import KINDS, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/benchmarks/launcher-independent-v1"
CLEAN = DATA / "clean"
SPEECH = DATA / "asr"
ARTIFACTS = ROOT / "artifacts/benchmarks/launcher-independent-v1"
PARAMETERS = {
    "provider": "groq", "model": "whisper-large-v3", "language": "ru",
    "response_format": "json", "prompt": None,
}
PROFILES = {
    "quiet_background": {"snr_db": 30, "gain": 0.8, "hum": 0.02, "lowpass": 1.0},
    "soft_room": {"snr_db": 26, "gain": 0.75, "hum": 0.05, "lowpass": 0.98},
    "soft_fan": {"snr_db": 22, "gain": 0.65, "hum": 0.15, "lowpass": 0.95},
    "soft_distance": {"snr_db": 18, "gain": 0.5, "hum": 0.04, "lowpass": 0.85},
}
VOICES = {"validation": "ru-RU-SvetlanaNeural", "test": "ru-RU-DmitryNeural"}
VALIDATION_GROUPS = {
    "apiary", "malt", "soap", "pasture", "grapes", "slope", "snow", "dye",
    "fair-folder", "tasting-file",
}
PINS = {"edge-tts": "7.2.8", "imageio-ffmpeg": "0.6.0"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return " ".join(re.findall(r"[^\W_]+", text, re.UNICODE))


def prior_sources(*, public_only=False):
    paths = set()
    bases = [ROOT / "data/benchmarks", ROOT / "data/training"]
    if not public_only:
        bases.append(ROOT / "artifacts/benchmarks")
    for base in bases:
        for manifest_path in base.rglob("manifest.json"):
            if manifest_path.is_relative_to(DATA) or manifest_path.is_relative_to(ARTIFACTS):
                continue
            manifest = asr.read_json(manifest_path)
            paths.add(manifest_path)
            for name in manifest.get("files", {}):
                path = manifest_path.parent / name
                asr.require(path.is_file(), f"Missing prior source: {path}")
                paths.add(path)
    for base in ([] if public_only else (ROOT / "artifacts/benchmarks").glob("asr-comparison-*")):
        for name in ("responses.jsonl", "comparison.json", "live-client.json"):
            path = base / name
            if path.exists():
                paths.add(path)
    return sorted(paths)


def inventory(paths):
    result = {key: set() for key in ("queries", "ids", "names", "descriptions", "audio", "voices")}
    query_rows = 0

    def visit(value):
        nonlocal query_rows
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(item, str):
                    continue
                if key in ("query", "clean_query", "transcript", "text") or key.endswith(("_full", "_turbo")):
                    if normalize(item):
                        result["queries"].add(normalize(item))
                        query_rows += 1
                if key in ("id", "object_id", "expected_object_id", "recording_id", "group_id"):
                    result["ids"].add(item)
                if key in ("audio_sha256", "clean_audio_sha256"):
                    result["audio"].add(item)
                if key == "voice":
                    result["voices"].add(item)
            if "name" in value and "kind" in value:
                result["names"].add(normalize(value["name"]))
                result["names"].update(normalize(x) for x in value.get("aliases", []))
                result["descriptions"].add(normalize(value.get("description", "")))
            for item in value.values():
                if isinstance(item, (dict, list)):
                    visit(item)

    for path in paths:
        visit(asr.read_rows(path) if path.suffix == ".jsonl" else asr.read_json(path))
    result["names"].discard("")
    result["descriptions"].discard("")
    result["query_rows"] = query_rows
    return result


def check_overlap(objects, rows, previous, *, recordings=(), transcripts=()):
    checks = {
        "object_ids": {o["object_id"] for o in objects} & previous["ids"],
        "object_names_or_aliases": {
            normalize(name) for o in objects for name in (o["name"], *o["aliases"])
        } & previous["names"],
        "object_descriptions": {normalize(o["description"]) for o in objects} & previous["descriptions"],
        "example_or_group_ids": {r[key] for r in rows for key in ("id", "group_id")} & previous["ids"],
        "clean_queries": {normalize(r["query"]) for r in rows} & previous["queries"],
        "asr_queries": {normalize(r["transcript"]) for r in transcripts if r["transcript"].strip()} & previous["queries"],
        "audio_hashes": {r[key] for r in recordings for key in ("audio_sha256", "clean_audio_sha256")} & previous["audio"],
        "voices": {r["voice"] for r in recordings} & previous["voices"],
    }
    for key, overlaps in checks.items():
        asr.require(not overlaps, f"Overlap with previous data ({key}): {sorted(overlaps)[:3]}")
    return {key: len(value) for key, value in checks.items()}


def catalog_hash(objects):
    return asr.sha(json.dumps(objects, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def require_frozen_parameters(plan):
    for field, expected in {
        "asr": PARAMETERS, "noise_profiles": PROFILES, "tts_packages": PINS,
        "speaker_policy": VOICES, "tts_rate_unit": "percent",
        "audio_origin": "new_neural_tts_with_deterministic_mild_noise",
    }.items():
        asr.require(plan.get(field) == expected, f"Frozen {field} changed; use a new revision")


def prepare():
    source = asr.read_json(DATA / "authoring.json")
    paths = prior_sources()
    previous = inventory(paths)
    snapshot = {p.relative_to(ROOT).as_posix(): asr.file_sha(p) for p in paths}
    plan_path = SPEECH / "plan_manifest.json"
    if plan_path.exists():
        frozen = asr.read_json(plan_path)
        require_frozen_parameters(frozen)
        asr.require(frozen["authoring_sha256"] == asr.file_sha(DATA / "authoring.json"), "Frozen authoring changed")
        asr.require(frozen["guard_source_files"] == snapshot, "Prior data changed; use a new revision")
        asr.require(frozen["asr_client_sha256"] == asr.file_sha(ROOT / "src/jev_launcher/speech.py"), "ASR client changed")
    objects, splits, plan = [], {"validation": [], "test": []}, []

    def obj(kind, name, description):
        value = {"object_id": "independent-v1-" + asr.sha((kind + ":" + name).encode())[:24],
                 "name": name, "kind": kind, "description": description, "aliases": []}
        objects.append(value)
        return value

    for target in source["targets"]:
        value = obj(target["kind"], target["name"], target["description"])
        split = "validation" if target["key"] in VALIDATION_GROUPS else "test"
        asr.require(len(target["queries"]) == 5, "Need five authored queries per target")
        for index, text in enumerate(target["queries"]):
            row_id = f"launcher-independent-v1-{target['key']}-{index + 1}"
            row = {
                "id": row_id, "query": text, "expected_action": "open", "expected_object_id": value["object_id"],
                "input_type": "text", "group_id": "independent-v1/" + target["key"],
                "tags": ["natural_request", target["kind"], "new_catalog"], "source": "authored",
                "rationale": f"Единственный объект типа {target['kind']} с указанной функцией или материалами: {target['name']}. {target['description']}",
            }
            splits[split].append(row)
            plan.append({
                "recording_id": row_id + "-asr", "source_example_id": row_id,
                "split": split, "clean_query": text, "expected_action": "open",
                "expected_object_id": value["object_id"], "group_id": row["group_id"],
                "tags": row["tags"], "rationale": row["rationale"], "speaker_id": VOICES[split],
                "voice": VOICES[split], "tts_rate": (-4, 0, 4)[index % 3],
                "condition": list(PROFILES)[len(plan) % 4], "noise_seed": asr.seed("independent-noise:" + row_id),
            })
        for neighbor in target.get("neighbors", []):
            name, description = neighbor.split("|", 1)
            obj("app", name, description)
    for key, kind in (("extra_apps", "app"), ("extra_folders", "folder"), ("extra_files", "file")):
        for item in source[key]:
            name, description = item.split("|", 1)
            obj(kind, name, description)
    asr.require(Counter(o["kind"] for o in objects) == KINDS, "Catalog sizes differ")
    asr.require(len({normalize(o['name']) for o in objects}) == len(objects), "Duplicate catalog names")
    rows = splits["validation"] + splits["test"]
    overlap = check_overlap(objects, rows, previous)
    # Catalog order contains no target/authoring order cue.
    objects.sort(key=lambda o: asr.sha(("catalog-order:" + o["object_id"]).encode()))
    value = {"schema_version": 1, "catalog_hash": catalog_hash(objects), "objects": objects}
    for directory in (CLEAN, SPEECH, ARTIFACTS):
        directory.mkdir(parents=True, exist_ok=True)
    asr.write_json(CLEAN / "catalog.json", value)
    asr.write_json(SPEECH / "catalog.json", value)
    for split, split_rows in splits.items():
        asr.write_rows(CLEAN / f"{split}.jsonl", split_rows)
    asr.write_json(CLEAN / "manifest.json", {
        "schema_version": 1, "dataset_id": "launcher-independent-v1-clean", "created_on": source["created_on"],
        "query_source": "authored", "catalog_hash": value["catalog_hash"], "catalog_counts": KINDS,
        "group_count": 40, "splits": {s: {"rows": len(r), "actions": {"open": len(r)}} for s, r in splits.items()},
        "annotation_review": {"status": "author_reviewed", "reviewed_rows": 200, "independent": False},
        "protocol": {"allowed_actions": ["open"], "metric": "top1_accuracy", "require_unique_target": True,
                     "independent_of_previous_datasets": True, "training_access": False,
                     "model_input": ["query", "full_catalog.objects"]},
        "files": {name: asr.file_sha(CLEAN / name) for name in ("catalog.json", "validation.jsonl", "test.jsonl")},
    })
    asr.write_rows(SPEECH / "clean_queries.jsonl", plan)
    asr.write_json(plan_path, {
        "schema_version": 1, "dataset_id": "launcher-independent-v1-asr", "created_on": source["created_on"],
        "source_dataset": "launcher-independent-v1-clean", "source_files": asr.source_hashes(CLEAN),
        "source_path": CLEAN.relative_to(ROOT).as_posix(), "clean_queries_sha256": asr.file_sha(SPEECH / "clean_queries.jsonl"),
        "asr": PARAMETERS, "asr_client_sha256": asr.file_sha(ROOT / "src/jev_launcher/speech.py"),
        "authoring_sha256": asr.file_sha(DATA / "authoring.json"), "guard_source_files": snapshot,
        "audio_origin": "new_neural_tts_with_deterministic_mild_noise", "noise_profiles": PROFILES,
        "tts_packages": PINS, "tts_rate_unit": "percent", "speaker_policy": VOICES,
        "independent_of_previous_datasets": True, "training_access": False,
        "guarded_query_occurrences": previous["query_rows"],
    })
    asr.write_json(DATA / "overlap_sources.json", snapshot)
    asr.write_json(DATA / "overlap_report.json", {"stage": "authored_text", "previous_files": len(paths),
                   "previous_query_occurrences": previous["query_rows"], "previous_unique_queries": len(previous["queries"]),
                   "previous_unique_object_names_or_aliases": len(previous["names"]), "overlaps": overlap})
    return validate_dataset(CLEAN)


@contextmanager
def speech_context():
    changes = {"DIRS": {"synthetic": SPEECH}, "BASE": {"synthetic": CLEAN},
               "ARTIFACTS": ARTIFACTS, "ASR": PARAMETERS, "CONDITIONS": PROFILES,
               "AUDIO_ORIGIN": "new_neural_tts_with_deterministic_mild_noise"}
    before = {key: getattr(asr, key) for key in changes}
    try:
        for key, value in changes.items():
            setattr(asr, key, value)
        yield
    finally:
        for key, value in before.items():
            setattr(asr, key, value)


async def synthesize():
    import edge_tts
    import imageio_ffmpeg

    for package, version in PINS.items():
        asr.require(importlib.metadata.version(package) == version, f"Use pinned {package}=={version}")
    plan = asr.read_rows(SPEECH / "clean_queries.jsonl")
    semaphore = asyncio.Semaphore(4)
    completed = 0

    async def one(row):
        nonlocal completed
        async with semaphore:
            clean_path = ARTIFACTS / "audio/clean" / (row["recording_id"] + ".wav")
            mp3_path = ARTIFACTS / "audio/tts" / (row["recording_id"] + ".mp3")
            mp3_path.parent.mkdir(parents=True, exist_ok=True)
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            if not clean_path.exists():
                for attempt in range(3):
                    try:
                        await edge_tts.Communicate(row["clean_query"], row["voice"], rate=f"{row['tts_rate']:+d}%").save(str(mp3_path))
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2 ** (attempt + 1))
                result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(mp3_path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(clean_path)],
                    capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                asr.require(result.returncode == 0, "Neural TTS WAV conversion failed")
            data, details = asr.degrade_audio(clean_path.read_bytes(), row["condition"], row["noise_seed"], profile=PROFILES[row["condition"]])
            noisy_path = ARTIFACTS / "audio/noisy" / clean_path.name
            noisy_path.parent.mkdir(parents=True, exist_ok=True)
            noisy_path.write_bytes(data)
            completed += 1
            if completed % 20 == 0:
                print(json.dumps({"neural_tts_completed": completed, "total": len(plan)}), flush=True)
            return {
                "recording_id": row["recording_id"], "audio_origin": "tts_noisy",
                "speaker_id": row["speaker_id"], "voice": row["voice"], "tts_rate": row["tts_rate"],
                "tts_backend": "edge-tts", "tts_mp3_sha256": asr.file_sha(mp3_path),
                "condition": row["condition"], "noise_seed": row["noise_seed"],
                "clean_audio_path": clean_path.relative_to(ROOT).as_posix(), "clean_audio_sha256": asr.file_sha(clean_path),
                "audio_path": noisy_path.relative_to(ROOT).as_posix(), "audio_sha256": asr.sha(data), "audio": details,
            }

    recordings = await asyncio.gather(*(one(row) for row in plan))
    asr.write_rows(SPEECH / "recordings.jsonl", recordings)
    return {"audio_completed": len(recordings), "voices": VOICES}


def audit(*, check_audio=True, public_only=False):
    plan = asr.read_json(SPEECH / "plan_manifest.json")
    require_frozen_parameters(plan)
    frozen = asr.read_json(DATA / "overlap_sources.json")
    asr.require(plan["guard_source_files"] == frozen, "Frozen prior-source manifests differ")
    public_snapshot = {p.relative_to(ROOT).as_posix(): asr.file_sha(p)
                       for p in prior_sources(public_only=True)}
    asr.require(public_snapshot == {name: expected for name, expected in frozen.items()
                                   if name.startswith("data/")}, "Prior corpus changed: public sources")
    # Audit the original frozen corpus; subsequent experiments are not part of this holdout.
    selected = {name: expected for name, expected in frozen.items()
                if not public_only or name.startswith("data/")}
    paths = []
    for name, expected in selected.items():
        path = (ROOT / name).resolve()
        asr.require(path.is_relative_to(ROOT.resolve()), "Prior source escaped the workspace")
        asr.require(path.is_file(), f"Missing frozen prior source: {name}")
        asr.require(asr.file_sha(path) == expected, f"Prior corpus changed: {name}")
        paths.append(path)
    asr.require(asr.file_sha(DATA / "authoring.json") == plan["authoring_sha256"], "Authoring hash mismatch")
    objects = asr.read_json(CLEAN / "catalog.json")["objects"]
    rows = asr.read_rows(CLEAN / "validation.jsonl") + asr.read_rows(CLEAN / "test.jsonl")
    authored = {
        f"launcher-independent-v1-{target['key']}-{index + 1}": (target, text)
        for target in asr.read_json(DATA / "authoring.json")["targets"]
        for index, text in enumerate(target["queries"])
    }
    asr.require(len(authored) == 200 and {r["id"] for r in rows} == set(authored), "Authored example IDs differ")
    for row in rows:
        target, text = authored[row["id"]]
        object_id = "independent-v1-" + asr.sha((target["kind"] + ":" + target["name"]).encode())[:24]
        asr.require(row["query"] == text and row["expected_object_id"] == object_id, "Authored query or intended object changed")
    previous = inventory(paths)
    recordings = asr.read_rows(SPEECH / "recordings.jsonl") if (SPEECH / "recordings.jsonl").exists() else []
    transcripts = asr.read_rows(SPEECH / "transcriptions.jsonl") if (SPEECH / "transcriptions.jsonl").exists() else []
    overlaps = check_overlap(objects, rows, previous, recordings=recordings, transcripts=transcripts)
    result = {"valid": True, "clean": validate_dataset(CLEAN), "overlaps": overlaps,
              "previous_files": len(paths), "previous_query_occurrences": previous["query_rows"],
              "previous_unique_queries": len(previous["queries"]), "audio_records": len(recordings), "asr_responses": len(transcripts)}
    result["audit_scope"] = "public_frozen_sources" if public_only else "all_frozen_sources"
    result["excluded_private_prior_files"] = len(frozen) - len(selected)
    if (SPEECH / "manifest.json").exists():
        with speech_context():
            result["asr"] = asr.validate(SPEECH, check_audio=check_audio)
        asr.require(result["asr"]["asr_statistics"]["recoverability"].get("not_reviewed", 0) == 0,
                    "Changed ASR transcripts still need semantic review")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "synthesize", "transcribe", "finalize", "validate"))
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--public-only", action="store_true",
                        help="Validate frozen public sources; report excluded private provenance")
    args = parser.parse_args()
    asr.require(not args.public_only or args.command == "validate", "--public-only is for validation")
    if args.command == "prepare":
        result = prepare()
    elif args.command == "synthesize":
        audit(check_audio=False)
        result = asyncio.run(synthesize())
    elif args.command == "transcribe":
        audit()
        with speech_context():
            result = asr.transcribe_audio(max_new=args.max_new, interval=3.2)
    elif args.command == "finalize":
        with speech_context():
            result = asr.finalize()
        result = audit()
        asr.write_json(DATA / "overlap_report.json", result)
    else:
        result = audit(check_audio=not args.text_only, public_only=args.public_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
