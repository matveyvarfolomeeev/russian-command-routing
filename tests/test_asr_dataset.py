"""Offline ASR provenance tests; fixtures are explicit stubs, never shipped data."""

import array
import math
import shutil
from pathlib import Path

import pytest

from scripts import asr_dataset as asr
from scripts.validate_benchmark import BenchmarkError

ROOT = Path(__file__).resolve().parents[1]


def tone():
    samples = array.array("h", (int(10000 * math.sin(2 * math.pi * 700 * n / 16000)) for n in range(1600)))
    return asr.wav_from_samples(samples, 16000)


def test_noise_is_reproducible_and_changes_the_actual_waveform():
    original = tone()
    first, details = asr.degrade_audio(original, "fan", 42)
    second, again = asr.degrade_audio(original, "fan", 42)
    other, _ = asr.degrade_audio(original, "fan", 43)
    assert first == second and details == again
    assert first != original and first != other
    samples, rate = asr.samples_from_wav(first)
    assert rate == 16000 and len(samples) == 1600
    assert details["snr_db"] == 12 and details["clipped_samples"] == 0
    ratio = details["active_speech_rms"] / details["noise_rms"]
    assert 20 * math.log10(ratio) == pytest.approx(12, abs=0.001)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    source = tmp_path / "data/benchmarks/launcher-basic-v1/synthetic"
    shutil.copytree(ROOT / "data/benchmarks/launcher-basic-v1/synthetic", source)
    client = tmp_path / "src/jev_launcher/speech.py"
    client.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "src/jev_launcher/speech.py", client)
    artifacts = tmp_path / "artifacts/benchmarks/launcher-asr-v1"
    directory = tmp_path / "data/benchmarks/launcher-asr-v1/synthetic"
    monkeypatch.setattr(asr, "ROOT", tmp_path)
    monkeypatch.setattr(asr, "ARTIFACTS", artifacts)
    monkeypatch.setattr(asr, "DIRS", {"synthetic": directory})
    monkeypatch.setattr(asr, "BASE", {"synthetic": source})
    asr.prepare()
    records, responses = [], []
    for index, row in enumerate(asr.read_rows(directory / "clean_queries.jsonl")):
        clean_path = artifacts / "synthetic/audio/clean" / f"{row['recording_id']}.wav"
        noisy_path = artifacts / "synthetic/audio/noisy" / clean_path.name
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        noisy_path.parent.mkdir(parents=True, exist_ok=True)
        clean_wav = tone()
        noisy_wav, details = asr.degrade_audio(clean_wav, row["condition"], row["noise_seed"])
        clean_path.write_bytes(clean_wav)
        noisy_path.write_bytes(noisy_wav)
        records.append({
            "recording_id": row["recording_id"], "audio_origin": "tts_noisy",
            "speaker_id": row["speaker_id"], "voice": row["voice"], "tts_rate": row["tts_rate"],
            "condition": row["condition"], "noise_seed": row["noise_seed"],
            "clean_audio_path": clean_path.relative_to(tmp_path).as_posix(),
            "clean_audio_sha256": asr.sha(clean_wav),
            "audio_path": noisy_path.relative_to(tmp_path).as_posix(), "audio_sha256": asr.sha(noisy_wav),
            "audio": details,
        })
        # One changed transcript and one empty result protect raw/failed ASR handling.
        text = "Открой архасферу." if index == 0 else "" if index == 1 else row["clean_query"] + "."
        responses.append({
            "recording_id": row["recording_id"], "audio_sha256": asr.sha(noisy_wav),
            "asr": asr.ASR, "transcript": text, "status": "ok" if text else "empty",
            "measured_at": "test-stub", "error": None,
        })
    asr.write_rows(directory / "recordings.jsonl", records)
    asr.write_rows(directory / "transcriptions.jsonl", responses)
    asr.finalize()
    return directory


def rehash(directory, name, rows):
    asr.write_rows(directory / name, rows)
    manifest = asr.read_json(directory / "manifest.json")
    manifest["files"][name] = asr.file_sha(directory / name)
    asr.write_json(directory / "manifest.json", manifest)


def test_asr_keeps_original_intent_empty_results_and_paired_splits(dataset):
    report = asr.validate(dataset)
    assert report["splits"]["validation"] == {"rows": 50, "empty_asr": 1}
    assert report["splits"]["test"] == {"rows": 150, "empty_asr": 0}
    assert report["asr_statistics"]["recoverability"]["not_reviewed"] == 1
    assert report["audio_hashes_checked"]


def test_semantic_review_is_bound_to_the_actual_transcript(dataset):
    first = asr.read_rows(dataset / "transcriptions.jsonl")[0]
    annotations = [{
        "recording_id": first["recording_id"], "transcript_sha256": asr.sha(first["transcript"].encode()),
        "recoverability": "uncertain", "reviewer": "test author", "note": "explicit test annotation",
    }]
    path = asr.ARTIFACTS / "synthetic/author_asr_review.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    asr.write_rows(path, annotations)
    asr.finalize()
    assert asr.validate(dataset)["asr_statistics"]["recoverability"]["uncertain"] == 1
    annotations[0]["transcript_sha256"] = "0" * 64
    asr.write_rows(path, annotations)
    with pytest.raises(BenchmarkError, match="different ASR transcript"):
        asr.finalize()


def test_repairing_asr_with_clean_text_is_rejected(dataset):
    rows = asr.read_rows(dataset / "validation.jsonl")
    rows[0]["query"] = asr.read_rows(dataset / "clean_queries.jsonl")[0]["clean_query"]
    rehash(dataset, "validation.jsonl", rows)
    with pytest.raises(BenchmarkError, match="transcript was edited"):
        asr.validate(dataset)


def test_wrong_target_is_rejected_even_when_the_target_exists(dataset):
    rows = asr.read_rows(dataset / "validation.jsonl")
    rows[0]["expected_object_id"] = rows[6]["expected_object_id"]
    rehash(dataset, "validation.jsonl", rows)
    with pytest.raises(BenchmarkError, match="Wrong intended object"):
        asr.validate(dataset)


def test_an_empty_asr_result_cannot_be_silently_dropped(dataset):
    rows = asr.read_rows(dataset / "validation.jsonl")
    rehash(dataset, "validation.jsonl", [row for row in rows if row["query"]])
    with pytest.raises(BenchmarkError, match="split size"):
        asr.validate(dataset)


def test_swapping_audio_after_transcription_is_rejected(dataset):
    recording = asr.read_rows(dataset / "recordings.jsonl")[0]
    path = asr.ROOT / recording["audio_path"]
    path.write_bytes(tone())
    with pytest.raises(BenchmarkError, match="Audio file hash mismatch"):
        asr.validate(dataset)


def test_clean_text_cannot_be_declared_as_an_additional_model_input(dataset):
    path = dataset / "manifest.json"
    manifest = asr.read_json(path)
    manifest["protocol"]["model_input"].append("clean_query")
    asr.write_json(path, manifest)
    with pytest.raises(BenchmarkError, match="clean-text leakage"):
        asr.validate(dataset)


def test_portable_transcripts_explicitly_report_unavailable_audio(dataset):
    recording = asr.read_rows(dataset / "recordings.jsonl")[0]
    (asr.ROOT / recording["audio_path"]).unlink()
    with pytest.raises(BenchmarkError, match="Original audio missing"):
        asr.validate(dataset)
    report = asr.validate(dataset, check_audio=False)
    assert not report["audio_hashes_checked"] and report["missing_audio_files"] == 1
