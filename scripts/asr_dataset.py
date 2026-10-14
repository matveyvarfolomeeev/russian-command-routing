"""Noisy TTS -> the launcher's actual ASR -> a frozen, open-only benchmark.

Network requests occur only in the explicit transcribe command. Validation is offline.
"""

from __future__ import annotations

import argparse
import array
import base64
import hashlib
import io
import json
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from .validate_benchmark import (
        OBJECT_FIELDS,
        ROW_FIELDS,
        normalize_query,
        read_json,
        read_rows,
        require,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_benchmark import (
        OBJECT_FIELDS,
        ROW_FIELDS,
        normalize_query,
        read_json,
        read_rows,
        require,
    )

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/benchmarks/launcher-asr-v1"
DIRS = {
    "synthetic": ROOT / "data/benchmarks/launcher-asr-v1/synthetic",
    "real": ARTIFACTS / "real",
}
BASE = {
    "synthetic": ROOT / "data/benchmarks/launcher-basic-v1/synthetic",
    "real": ROOT / "artifacts/benchmarks/launcher-basic-v1/real",
}
ASR = {"provider": "groq", "model": "whisper-large-v3-turbo", "language": "ru", "response_format": "json", "prompt": None}
AUDIO_ORIGIN = "windows_tts_with_deterministic_noise"
SPEAKERS = {"validation": "Microsoft Irina Desktop", "test": "Microsoft Pavel"}
CONDITIONS = {
    "room": {"snr_db": 18, "gain": 0.8, "hum": 0.1, "lowpass": 1.0},
    "fan": {"snr_db": 12, "gain": 0.55, "hum": 0.45, "lowpass": 0.7},
    "distant": {"snr_db": 10, "gain": 0.2, "hum": 0.2, "lowpass": 0.35},
    "telephone": {"snr_db": 8, "gain": 0.5, "hum": 0.1, "lowpass": 0.65},
}
FILES = {"catalog.json", "validation.jsonl", "test.jsonl", "clean_queries.jsonl", "recordings.jsonl", "transcriptions.jsonl", "asr_review.jsonl"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def seed(value: str) -> int:
    return int(sha(value.encode())[:16], 16)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def source_hashes(directory: Path) -> dict:
    return {name: file_sha(directory / name) for name in ("catalog.json", "validation.jsonl", "test.jsonl", "manifest.json")}


def prepare() -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    all_plan = []
    for kind, directory in DIRS.items():
        source = BASE[kind]
        require(source.is_dir(), f"Missing source benchmark: {source}")
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = source_hashes(source)
        existing = directory / "plan_manifest.json"
        if existing.exists():
            require(read_json(existing)["source_files"] == snapshot, "Source changed since recording plan was frozen")
        shutil.copyfile(source / "catalog.json", directory / "catalog.json")
        plan = []
        for split in ("validation", "test"):
            rows = read_rows(source / f"{split}.jsonl")
            for index, row in enumerate(rows):
                recording_id = row["id"].replace("launcher-basic-v1", "launcher-asr-v1")
                condition = list(CONDITIONS)[index % len(CONDITIONS)]
                plan.append({
                    "recording_id": recording_id, "source_example_id": row["id"],
                    "split": split, "clean_query": row["query"],
                    "expected_action": row["expected_action"], "expected_object_id": row["expected_object_id"],
                    "group_id": row["group_id"], "tags": row["tags"], "rationale": row["rationale"],
                    "speaker_id": "tts-irina" if split == "validation" else "tts-pavel",
                    "voice": SPEAKERS[split], "tts_rate": [-1, 0, 1][index % 3],
                    "condition": condition, "noise_seed": seed(f"noise:{recording_id}"),
                })
                all_plan.append({
                    "text": row["query"], "voice": SPEAKERS[split], "rate": [-1, 0, 1][index % 3],
                    "output": str(ARTIFACTS / kind / "audio/clean" / f"{recording_id}.wav"),
                })
        write_rows(directory / "clean_queries.jsonl", plan)
        write_json(existing, {
            "schema_version": 1, "dataset_id": f"launcher-asr-v1-{kind}",
            "created_on": "2026-10-04", "source_dataset": read_json(source / "manifest.json")["dataset_id"],
            "source_files": snapshot, "source_path": source.relative_to(ROOT).as_posix(),
            "clean_queries_sha256": file_sha(directory / "clean_queries.jsonl"),
            "asr": ASR, "asr_client_sha256": file_sha(ROOT / "src/jev_launcher/speech.py"),
            "audio_origin": "windows_tts_with_deterministic_noise", "noise_profiles": CONDITIONS,
            "speaker_policy": "Irina validation / Pavel test; shared across corresponding catalogs",
        })
    write_json(ARTIFACTS / "tts_plan.json", all_plan)
    return {"prepared": len(all_plan), "recordings": "TTS only; no microphone capture"}


def samples_from_wav(data: bytes) -> tuple[array.array, int]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        require(wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getcomptype() == "NONE", "Expected uncompressed mono 16-bit WAV")
        rate = wav.getframerate()
        samples = array.array("h", wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    require(bool(samples), "Empty waveform")
    return samples, rate


def wav_from_samples(samples: array.array, rate: int) -> bytes:
    samples = array.array("h", samples)
    if sys.byteorder != "little":
        samples.byteswap()
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return out.getvalue()


def degrade_audio(
    wav: bytes, condition: str, noise_seed: int, *, profile: dict | None = None
) -> tuple[bytes, dict]:
    samples, rate = samples_from_wav(wav)
    profile = CONDITIONS[condition] if profile is None else profile
    require(rate == 16000, "TTS recordings must use 16000 Hz")
    filtered, previous, highpass, lowpass = [], 0.0, 0.0, 0.0
    for sample in samples:
        value = float(sample)
        if condition == "telephone":
            highpass = 0.89 * (highpass + value - previous)
            previous = value
            value = highpass
        lowpass += profile["lowpass"] * (value - lowpass)
        filtered.append(lowpass * profile["gain"])
    # Speech-level RMS excludes silence; seeded noise remains present during pauses.
    active = [value for value in filtered if abs(value) > 20]
    require(bool(active), "Silent TTS recording")
    speech_rms = math.sqrt(sum(v * v for v in active) / len(active))
    randomizer = random.Random(noise_seed)
    noise = [
        randomizer.uniform(-1, 1) + profile["hum"] * math.sin(2 * math.pi * 100 * i / rate)
        for i in range(len(filtered))
    ]
    noise_rms = math.sqrt(sum(v * v for v in noise) / len(noise))
    scale = speech_rms / (10 ** (profile["snr_db"] / 20)) / noise_rms
    noisy = array.array("h")
    clipped = 0
    for voice, background in zip(filtered, noise, strict=True):
        value = round(voice + scale * background)
        clipped += not -32768 <= value <= 32767
        noisy.append(max(-32768, min(32767, value)))
    return wav_from_samples(noisy, rate), {
        **profile, "rate": rate, "channels": 1, "sample_width": 2,
        "duration_seconds": len(samples) / rate, "active_speech_rms": round(speech_rms, 3),
        "noise_rms": round(scale * noise_rms, 3), "clipped_samples": clipped,
        "snr_definition": "Filtered speech RMS over active samples / added noise RMS over all samples",
    }


def synthesize() -> dict:
    require((ARTIFACTS / "tts_plan.json").exists(), "Run prepare first")
    result = subprocess.run([
        shutil.which("pwsh.exe") or "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(ROOT / "scripts/synthesize_asr_audio.ps1"), "-Plan", str(ARTIFACTS / "tts_plan.json"),
    ], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.stdout:
        print(result.stdout.decode("utf-8", errors="replace"), end="", flush=True)
    if result.stderr:
        print(result.stderr.decode("utf-8", errors="replace"), file=sys.stderr, end="", flush=True)
    require(result.returncode == 0, "Windows TTS generation failed")
    total = 0
    for kind, directory in DIRS.items():
        records = []
        for row in read_rows(directory / "clean_queries.jsonl"):
            clean_path = ARTIFACTS / kind / "audio/clean" / f"{row['recording_id']}.wav"
            noisy_path = ARTIFACTS / kind / "audio/noisy" / clean_path.name
            noisy_path.parent.mkdir(parents=True, exist_ok=True)
            data, details = degrade_audio(clean_path.read_bytes(), row["condition"], row["noise_seed"])
            noisy_path.write_bytes(data)
            records.append({
                "recording_id": row["recording_id"], "audio_origin": "tts_noisy",
                "speaker_id": row["speaker_id"], "voice": row["voice"], "tts_rate": row["tts_rate"],
                "condition": row["condition"], "noise_seed": row["noise_seed"],
                "clean_audio_path": clean_path.relative_to(ROOT).as_posix(),
                "clean_audio_sha256": file_sha(clean_path),
                "audio_path": noisy_path.relative_to(ROOT).as_posix(), "audio_sha256": sha(data),
                "audio": details,
            })
            total += 1
            if total % 50 == 0:
                print(json.dumps({"noisy_audio_completed": total, "total": 400}), flush=True)
        write_rows(directory / "recordings.jsonl", records)
    return {"audio_completed": total, "conditions": CONDITIONS}


def load_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    database = Path(os.environ["LOCALAPPDATA"]) / "JevLauncher/catalog.sqlite3"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT value FROM settings WHERE key='groq_key'").fetchone()
    require(row is not None, "ASR key unavailable in environment and launcher settings")
    sys.path.insert(0, str(ROOT / "src"))
    from jev_launcher.config import _crypt
    return _crypt(base64.b64decode(json.loads(row[0])), decrypt=True).decode()


def transcribe_audio(*, max_new: int | None = None, interval: float = 3.1) -> dict:
    require(interval >= 3.05, "Use <=20 requests/minute for the configured Groq plan")
    sys.path.insert(0, str(ROOT / "src"))
    from jev_launcher.speech import SpeechError, transcribe
    key = load_api_key()
    total = sum(len(read_rows(directory / "recordings.jsonl")) for directory in DIRS.values())
    completed, next_request, newly_completed = 0, time.monotonic() + interval, 0
    for kind, directory in DIRS.items():
        plan = read_json(directory / "plan_manifest.json")
        require(plan["asr_client_sha256"] == file_sha(ROOT / "src/jev_launcher/speech.py"), "ASR client changed; create a new dataset revision")
        require(plan["source_files"] == source_hashes(BASE[kind]), "Source benchmark changed")
        path = directory / "transcriptions.jsonl"
        previous = read_rows(path) if path.exists() else []
        done = {row["recording_id"]: row for row in previous}
        require(len(done) == len(previous), "Duplicate transcription cache entries")
        recordings = read_rows(directory / "recordings.jsonl")
        with path.open("a", encoding="utf-8", newline="\n") as log:
            for recording in recordings:
                recording_id = recording["recording_id"]
                data = (ROOT / recording["audio_path"]).read_bytes()
                require(sha(data) == recording["audio_sha256"], "Recording changed before ASR")
                if recording_id in done:
                    require(done[recording_id]["audio_sha256"] == recording["audio_sha256"] and done[recording_id]["asr"] == ASR, "Stale transcription cache")
                    completed += 1
                    continue
                if max_new is not None and newly_completed >= max_new:
                    return {"completed": completed, "new": newly_completed, "remaining": total - completed}
                for attempt in range(6):
                    wait = max(0.0, next_request - time.monotonic())
                    if wait:
                        time.sleep(wait)
                    started = time.monotonic()
                    next_request = started + interval
                    try:
                        text = transcribe(data, key, model=ASR["model"])
                        status, error = "ok", None
                        break
                    except SpeechError as exc:
                        if "Речь не распознана" in str(exc):
                            text, status, error = "", "empty", str(exc)
                            break
                        cause = exc.__cause__
                        code = cause.code if isinstance(cause, urllib.error.HTTPError) else None
                        if code in (401, 403) or attempt == 5:
                            raise
                        retry = min(30.0, 3.0 * 2 ** attempt)
                        if code == 429 and cause.headers.get("Retry-After"):
                            retry = min(30.0, max(retry, float(cause.headers["Retry-After"])))
                        print(json.dumps({"asr_retry": recording_id, "attempt": attempt + 1, "http_status": code, "wait_seconds": retry}), flush=True)
                        next_request = time.monotonic() + retry
                response = {
                    "recording_id": recording_id, "audio_sha256": recording["audio_sha256"],
                    "asr": ASR, "transcript": text, "status": status, "error": error,
                    "measured_at": datetime.now(timezone.utc).isoformat(),
                    "request_seconds": round(time.monotonic() - started, 4),
                    "attempts": attempt + 1,
                }
                log.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
                log.flush()
                completed += 1
                newly_completed += 1
                print(json.dumps({"asr_completed": completed, "total": total, "status": status, "kind": kind}), flush=True)
    return {"completed": completed, "new": newly_completed, "remaining": total - completed}


def finalize() -> dict:
    reports = []
    for kind, directory in DIRS.items():
        plan = read_json(directory / "plan_manifest.json")
        require(source_hashes(BASE[kind]) == plan["source_files"], "Source changed before finalization")
        clean = read_rows(directory / "clean_queries.jsonl")
        recordings = read_rows(directory / "recordings.jsonl")
        responses = read_rows(directory / "transcriptions.jsonl")
        by_response = {r["recording_id"]: r for r in responses}
        require(len(responses) == len(by_response) == len(clean) == len(recordings) == 200, "Need one actual ASR result for each of 200 recordings")
        review_source = ARTIFACTS / kind / "author_asr_review.jsonl"
        annotations = read_rows(review_source) if review_source.exists() else []
        author_review = {r["recording_id"]: r for r in annotations}
        require(len(author_review) == len(annotations) and set(author_review) <= set(by_response), "Duplicate or unknown author ASR review ID")
        splits = {"validation": [], "test": []}
        review = []
        for source in clean:
            response = by_response[source["recording_id"]]
            same = normalize_query(source["clean_query"]) == normalize_query(response["transcript"])
            row = {
                "id": source["recording_id"], "query": response["transcript"],
                "expected_action": "open", "expected_object_id": source["expected_object_id"],
                "input_type": "asr_text", "group_id": source["group_id"],
                "tags": list(dict.fromkeys([*source["tags"], "tts_audio", "noisy_audio", source["condition"]])),
                "source": "asr", "rationale": source["rationale"],
            }
            splits[source["split"]].append(row)
            review.append({
                "recording_id": source["recording_id"],
                "text_status": "unchanged" if same else "empty" if response["status"] == "empty" else "changed",
                "recoverability": "preserved" if same else "lost" if response["status"] == "empty" else "not_reviewed",
                "reviewer": "automatic_text_comparison" if same or response["status"] == "empty" else None,
                "note": "Comparison ignores case, ё/е, whitespace and terminal punctuation; changed text requires semantic review.",
            })
            if source["recording_id"] in author_review:
                annotation = author_review[source["recording_id"]]
                require(annotation["transcript_sha256"] == sha(response["transcript"].encode()), "Semantic review refers to a different ASR transcript")
                require(annotation["recoverability"] in ("preserved", "lost", "uncertain", "not_reviewed"), "Invalid author ASR review")
                review[-1].update(annotation)
        for split, rows in splits.items():
            write_rows(directory / f"{split}.jsonl", rows)
        write_rows(directory / "asr_review.jsonl", review)
        catalog = read_json(directory / "catalog.json")
        manifest = {
            **plan, "revision": "tts-noisy-asr", "difficulty": "asr",
            "privacy": "shareable_fictional" if kind == "synthetic" else "private",
            "catalog_hash": catalog["catalog_hash"], "query_source": "actual_api_asr",
            "files": {name: file_sha(directory / name) for name in sorted(FILES)},
            "splits": {split: {"rows": len(rows), "actions": {"open": len(rows)}} for split, rows in splits.items()},
            "asr_statistics": {
                "status": dict(Counter(r["status"] for r in responses)),
                "text_status": dict(Counter(r["text_status"] for r in review)),
                "recoverability": dict(Counter(r["recoverability"] for r in review)),
                "audio_seconds": round(sum(r["audio"]["duration_seconds"] for r in recordings), 3),
                "conditions": dict(Counter(r["condition"] for r in recordings)),
            },
            "annotation_review": {
                "status": "original_intent_author_reviewed", "independent": False,
                "reviewer": "Codex (original dataset author)",
                "transcript_semantics": "Author assessments are recorded in asr_review.jsonl; unchanged normalized text is inherited from original intent review. No transcripts repaired or excluded.",
                "author_reviewed_transcripts": len(author_review),
            },
            "protocol": {
                "allowed_actions": ["open"], "require_unique_target": True,
                "unique_target_scope": "original_spoken_request",
                "model_input": ["query", "full_catalog.objects"],
                "metric": "top1_accuracy_on_all_original_requests",
                "metric_definition": "Match intended object ID / all split rows; empty ASR is an upstream failure, no exclusions or deduplication.",
                "paired_control": "Evaluate the same intended objects on clean_queries.clean_query separately to locate ASR-induced loss.",
                "post_asr_uncertainty": "Noise can remove identifying words; original intent remains the label, but text-only recovery may become impossible.",
                "independent_of_basic_profile": bool(plan.get("independent_of_previous_datasets", False)),
                "independent_of_paired_clean": False,
                "test_usage": "Held out from decision-model settings; no model predictions used to filter transcripts.",
            },
            "limitations": [
                ("Synthetic Windows TTS" if plan["audio_origin"] == "windows_tts_with_deterministic_noise" else "Synthetic neural Edge TTS")
                + " plus synthetic acoustic degradation; no human speakers or actual microphone/room recordings.",
                "Two synthetic voices with voice and split confounded; no claim of human-speaker generalization.",
                "ASR text is exactly the application client return value (outer whitespace stripped); no clean text or catalog prompt sent to ASR.",
                "Request IDs and raw HTTP response bodies are not available from the reused client; model revision is the hosted model ID at measured_at.",
                "Forced open only; this dataset does not evaluate clarification/refusal policy.",
            ],
        }
        write_json(directory / "manifest.json", manifest)
        reports.append(validate(directory))
    return {"valid": True, "datasets": reports}


def validate(directory: Path, *, check_audio: bool = True) -> dict:
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("revision") == "tts-noisy-asr", "Invalid ASR revision")
    require(set(manifest["files"]) == FILES, "Invalid ASR file list")
    for name, expected in manifest["files"].items():
        require(file_sha(directory / name) == expected, f"{name}: file hash mismatch")
    catalog = read_json(directory / "catalog.json")
    objects = {obj["object_id"]: obj for obj in catalog["objects"]}
    require(len(objects) == 177 and all(set(o) == OBJECT_FIELDS for o in objects.values()), "Invalid model-safe catalog")
    require(sha(json.dumps(catalog["objects"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()) == catalog["catalog_hash"] == manifest["catalog_hash"], "Catalog hash mismatch")
    source = ROOT / manifest["source_path"]
    require(source_hashes(source) == manifest["source_files"], "Frozen clean benchmark changed")
    require(file_sha(directory / "catalog.json") == manifest["source_files"]["catalog.json"], "ASR catalog differs from paired clean catalog")
    require(manifest["asr"] == ASR and manifest["audio_origin"] == AUDIO_ORIGIN, "Invalid ASR/TTS provenance")
    require(
        manifest["protocol"]["model_input"] == ["query", "full_catalog.objects"]
        and manifest["protocol"]["allowed_actions"] == ["open"]
        and manifest["protocol"]["unique_target_scope"] == "original_spoken_request",
        "Invalid ASR protocol or clean-text leakage into model input",
    )
    clean = read_rows(directory / "clean_queries.jsonl")
    recordings = read_rows(directory / "recordings.jsonl")
    transcripts = read_rows(directory / "transcriptions.jsonl")
    reviews = read_rows(directory / "asr_review.jsonl")
    maps = [{r["recording_id"]: r for r in rows} for rows in (clean, recordings, transcripts, reviews)]
    require(all(len(mapping) == len(rows) == 200 for mapping, rows in zip(maps, (clean, recordings, transcripts, reviews), strict=True)), "Duplicate/missing recording provenance")
    require(all(set(mapping) == set(maps[0]) for mapping in maps), "Provenance ID mismatch")
    by_clean, by_audio, by_transcript, by_review = maps
    groups, speakers, example_ids, missing_audio = {}, {}, set(), 0
    summary = {}
    for split, count in (("validation", 50), ("test", 150)):
        original = {r["id"]: r for r in read_rows(source / f"{split}.jsonl")}
        rows = read_rows(directory / f"{split}.jsonl")
        require(len(rows) == count, "Invalid ASR split size")
        for row in rows:
            require(set(row) == ROW_FIELDS and row["id"] not in example_ids, "Invalid ASR row fields/ID")
            example_ids.add(row["id"])
            require(row["id"] in by_clean, "Unknown recording ID")
            base, audio, transcript, review = (mapping[row["id"]] for mapping in maps)
            require(base["split"] == split and base["source_example_id"] in original, "Recording crossed source split")
            gold = original[base["source_example_id"]]
            require(base["clean_query"] == gold["query"] and base["expected_object_id"] == gold["expected_object_id"], "Clean query/gold label changed")
            require(row["expected_action"] == "open" and row["expected_object_id"] == gold["expected_object_id"] and row["expected_object_id"] in objects, "Wrong intended object ID")
            require(row["input_type"] == "asr_text" and row["source"] == "asr", "Not actual ASR input")
            require(row["query"] == transcript["transcript"], "ASR transcript was edited")
            require(transcript["status"] in ("ok", "empty") and bool(row["query"].strip()) == (transcript["status"] == "ok"), "Invalid ASR status/text")
            require(transcript["audio_sha256"] == audio["audio_sha256"] and transcript["asr"] == ASR, "ASR/audio provenance mismatch")
            require(audio["audio_sha256"] != audio["clean_audio_sha256"], "Noisy recording equals clean audio")
            require(
                audio["audio_origin"] == "tts_noisy" and audio["speaker_id"] == base["speaker_id"]
                and audio["voice"] == base["voice"] and audio["tts_rate"] == base["tts_rate"]
                and audio["condition"] == base["condition"] and audio["noise_seed"] == base["noise_seed"],
                "Acoustic metadata differs from recording plan",
            )
            require(row["group_id"] == base["group_id"] == gold["group_id"], "Wrong target family")
            require(row["group_id"] not in groups or groups[row["group_id"]] == split, "Target family crosses splits")
            require(audio["speaker_id"] not in speakers or speakers[audio["speaker_id"]] == split, "Speaker crosses splits")
            groups[row["group_id"]], speakers[audio["speaker_id"]] = split, split
            require(review["recoverability"] in ("preserved", "lost", "uncertain", "not_reviewed"), "Invalid transcript review")
            same_text = normalize_query(base["clean_query"]) == normalize_query(row["query"])
            expected_status = "unchanged" if same_text else "empty" if transcript["status"] == "empty" else "changed"
            require(review["text_status"] == expected_status, "ASR text review status mismatch")
            if "transcript_sha256" in review:
                require(review["transcript_sha256"] == sha(row["query"].encode()), "ASR semantic review transcript hash mismatch")
            require(audio["condition"] in CONDITIONS and audio["audio"]["snr_db"] == CONDITIONS[audio["condition"]]["snr_db"], "Unknown acoustic condition")
            for path_key, hash_key in (("audio_path", "audio_sha256"), ("clean_audio_path", "clean_audio_sha256")):
                path = ROOT / audio[path_key]
                require(path.resolve().is_relative_to(ARTIFACTS.resolve()), "Audio outside benchmark artifacts")
                if not path.exists():
                    missing_audio += 1
                    require(not check_audio, "Original audio missing; use --text-only for portable transcript validation")
                elif check_audio:
                    require(file_sha(path) == audio[hash_key], "Audio file hash mismatch")
        summary[split] = {"rows": len(rows), "empty_asr": sum(not r["query"].strip() for r in rows)}
    require(example_ids == set(by_clean), "Dropped ASR example")
    actual_statistics = {
        "status": dict(Counter(r["status"] for r in transcripts)),
        "text_status": dict(Counter(r["text_status"] for r in reviews)),
        "recoverability": dict(Counter(r["recoverability"] for r in reviews)),
        "audio_seconds": round(sum(r["audio"]["duration_seconds"] for r in recordings), 3),
        "conditions": dict(Counter(r["condition"] for r in recordings)),
    }
    require(actual_statistics == manifest["asr_statistics"], "ASR statistics mismatch")
    for train_name in ("launcher-train-v1", "launcher-stt-train-v1"):
        train = ROOT / "data/training" / train_name
        if train.is_dir():
            training_rows = read_rows(train / "train.jsonl")
            from_training = {r["expected_object_id"] for r in training_rows}
            require(not from_training & {r["expected_object_id"] for r in clean}, "ASR targets leaked into training")
            train_queries = {normalize_query(r["query"]) for r in training_rows}
            require(not train_queries & {normalize_query(r["transcript"]) for r in transcripts}, "ASR query overlaps with training")
    return {
        "dataset_id": manifest["dataset_id"], "valid": True, "splits": summary,
        "asr_statistics": manifest["asr_statistics"], "audio_hashes_checked": check_audio,
        "missing_audio_files": missing_audio,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "synthesize", "transcribe", "finalize", "validate"))
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--interval", type=float, default=3.1)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--dataset", type=Path, help="Validate one dataset, including a portable public copy")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare()
        elif args.command == "synthesize":
            result = synthesize()
        elif args.command == "transcribe":
            result = transcribe_audio(max_new=args.max_new, interval=args.interval)
        elif args.command == "finalize":
            result = finalize()
        else:
            directories = [args.dataset] if args.dataset else DIRS.values()
            result = {"valid": True, "datasets": [validate(d, check_audio=not args.text_only) for d in directories]}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
