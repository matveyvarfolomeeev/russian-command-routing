"""Offline checks for a fresh holdout and its overlap/provenance safeguards."""

import array
import shutil

import pytest

from scripts import asr_dataset as asr
from scripts import independent_dataset as independent
from scripts.validate_benchmark import BenchmarkError


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    data = tmp_path / "data/benchmarks/launcher-independent-v1"
    data.mkdir(parents=True)
    shutil.copyfile(independent.DATA / "authoring.json", data / "authoring.json")
    client = tmp_path / "src/jev_launcher/speech.py"
    client.parent.mkdir(parents=True)
    shutil.copyfile(independent.ROOT / "src/jev_launcher/speech.py", client)
    prior = tmp_path / "data/training/prior"
    prior.mkdir(parents=True)
    asr.write_json(prior / "catalog.json", {"objects": [{
        "object_id": "old-object", "name": "Ёлочный журнал", "kind": "app",
        "description": "Старое назначение.", "aliases": ["Ёлка"],
    }]})
    asr.write_rows(prior / "train.jsonl", [{"id": "old-row", "query": "Открой ёлочный журнал!"}])
    asr.write_json(prior / "manifest.json", {"files": {
        name: asr.file_sha(prior / name) for name in ("catalog.json", "train.jsonl")
    }})
    for module in (independent, asr):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(independent, "DATA", data)
    monkeypatch.setattr(independent, "CLEAN", data / "clean")
    monkeypatch.setattr(independent, "SPEECH", data / "asr")
    monkeypatch.setattr(independent, "ARTIFACTS", tmp_path / "artifacts/benchmarks/launcher-independent-v1")
    independent.prepare()
    return tmp_path


def test_fresh_catalog_and_target_families_are_separate_between_splits(workspace):
    report = independent.audit(check_audio=False)
    assert report["clean"]["catalog_objects"] == 177
    assert report["clean"]["distinct_open_targets"] == 40
    assert set(report["overlaps"].values()) == {0}
    validation = asr.read_rows(independent.CLEAN / "validation.jsonl")
    test = asr.read_rows(independent.CLEAN / "test.jsonl")
    assert len(validation) == 50 and len(test) == 150
    for field in ("id", "group_id", "expected_object_id"):
        assert not {r[field] for r in validation} & {r[field] for r in test}
    assert independent.VOICES["validation"] != independent.VOICES["test"]


def test_audit_uses_frozen_sources_after_later_experiments_are_added(workspace):
    later = workspace / "artifacts/benchmarks/later"
    later.mkdir(parents=True)
    asr.write_json(later / "manifest.json", {"files": {"unrelated.sqlite3": "not-a-source"}})
    assert independent.audit(check_audio=False)["audit_scope"] == "all_frozen_sources"


def test_public_audit_reports_excluded_private_sources(workspace):
    private = workspace / "artifacts/benchmarks/private/manifest.json"
    private.parent.mkdir(parents=True)
    asr.write_json(private, {"query": "прошлая личная команда"})
    name = private.relative_to(workspace).as_posix()
    frozen = asr.read_json(independent.DATA / "overlap_sources.json")
    frozen[name] = asr.file_sha(private)
    asr.write_json(independent.DATA / "overlap_sources.json", frozen)
    plan_path = independent.SPEECH / "plan_manifest.json"
    plan = asr.read_json(plan_path)
    plan["guard_source_files"] = frozen
    asr.write_json(plan_path, plan)
    private.unlink()
    with pytest.raises(BenchmarkError, match="Missing frozen prior source"):
        independent.audit(check_audio=False)
    report = independent.audit(check_audio=False, public_only=True)
    assert report["audit_scope"] == "public_frozen_sources"
    assert report["excluded_private_prior_files"] == 1


def test_public_audit_still_rejects_changed_public_sources(workspace):
    path = workspace / "data/training/prior/catalog.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="Prior corpus changed"):
        independent.audit(check_audio=False, public_only=True)


@pytest.mark.parametrize(("field", "new_value", "prior_field", "prior_value"), [
    ("object_id", "old-object", "ids", "old-object"),
    ("name", "ЕЛКА!!!", "names", "елка"),
    ("aliases", ["ЕЛКА!!!"], "names", "елка"),
    ("description", "Старое назначение!", "descriptions", "старое назначение"),
])
def test_old_objects_names_aliases_and_descriptions_cannot_reenter(field, new_value, prior_field, prior_value):
    previous = {k: set() for k in ("queries", "ids", "names", "descriptions", "audio", "voices")}
    previous[prior_field].add(prior_value)
    obj = {"object_id": "fresh-id", "name": "Свежий объект", "kind": "app",
           "description": "Новое назначение", "aliases": [], field: new_value}
    with pytest.raises(BenchmarkError, match="Overlap with previous data"):
        independent.check_overlap([obj], [], previous)


@pytest.mark.parametrize("asr_text", [False, True])
def test_punctuation_case_and_yo_cannot_hide_a_previous_query(workspace, asr_text):
    previous = independent.inventory(independent.prior_sources())
    rows = [] if asr_text else [{"id": "fresh", "group_id": "fresh", "query": "  ОТКРОЙ, елочный журнал. "}]
    transcripts = [{"transcript": "ОТКРОЙ елочный журнал?"}] if asr_text else []
    with pytest.raises(BenchmarkError, match="Overlap with previous data"):
        independent.check_overlap([], rows, previous, transcripts=transcripts)


def test_previous_audio_or_voice_cannot_be_reused():
    previous = {k: set() for k in ("queries", "ids", "names", "descriptions", "audio", "voices")}
    recordings = [{"audio_sha256": "noisy", "clean_audio_sha256": "clean", "voice": "new-voice"}]
    previous["audio"].add("clean")
    with pytest.raises(BenchmarkError, match="audio_hashes"):
        independent.check_overlap([], [], previous, recordings=recordings)
    previous["audio"].clear()
    previous["voices"].add("new-voice")
    with pytest.raises(BenchmarkError, match="voices"):
        independent.check_overlap([], [], previous, recordings=recordings)


def test_changed_or_new_prior_data_invalidates_overlap_evidence(workspace):
    prior = workspace / "data/training/prior/train.jsonl"
    asr.write_rows(prior, [{"id": "new-old", "query": "Изменённый старый набор"}])
    with pytest.raises(BenchmarkError, match="Prior corpus changed"):
        independent.audit(check_audio=False)
    with pytest.raises(BenchmarkError, match="Prior data changed"):
        independent.prepare()


@pytest.mark.parametrize("parameter", ["PROFILES", "VOICES", "PARAMETERS"])
def test_frozen_acoustic_and_api_parameters_cannot_be_silently_changed(workspace, monkeypatch, parameter):
    monkeypatch.setattr(independent, parameter, {"changed": True})
    with pytest.raises(BenchmarkError, match="Frozen .* changed"):
        independent.audit(check_audio=False)
    with pytest.raises(BenchmarkError, match="Frozen .* changed"):
        independent.prepare()


def test_rehashing_an_edited_gold_label_does_not_bypass_authoring(workspace):
    path = independent.CLEAN / "validation.jsonl"
    rows = asr.read_rows(path)
    rows[0]["expected_object_id"] = rows[5]["expected_object_id"]
    asr.write_rows(path, rows)
    manifest = asr.read_json(independent.CLEAN / "manifest.json")
    manifest["files"][path.name] = asr.file_sha(path)
    asr.write_json(independent.CLEAN / "manifest.json", manifest)
    with pytest.raises(BenchmarkError, match="Authored query or intended object changed"):
        independent.audit(check_audio=False)


def test_neural_context_restores_original_asr_even_after_an_error():
    old = {k: getattr(asr, k) for k in ("ASR", "DIRS", "BASE", "CONDITIONS", "AUDIO_ORIGIN")}
    with pytest.raises(RuntimeError, match="fixture"):
        with independent.speech_context():
            assert asr.ASR["model"] == "whisper-large-v3"
            assert asr.AUDIO_ORIGIN == "new_neural_tts_with_deterministic_mild_noise"
            raise RuntimeError("fixture")
    assert old == {k: getattr(asr, k) for k in old}


def test_transcription_resumes_without_changing_audio_or_model_and_counts_actual_rows(workspace, monkeypatch):
    from jev_launcher import speech

    samples = array.array("h", [1000, -1000] * 800)
    wav = asr.wav_from_samples(samples, 16000)
    audio = independent.ARTIFACTS / "test-stub.wav"
    audio.write_bytes(wav)
    recordings = [{"recording_id": f"test-stub-{i}", "audio_path": audio.relative_to(workspace).as_posix(),
                   "audio_sha256": asr.sha(wav)} for i in range(2)]
    asr.write_rows(independent.SPEECH / "recordings.jsonl", recordings)
    calls = []

    def transcribe(data, key, *, model):
        assert data == wav and key == "test-stub-key"
        calls.append(model)
        return "Тестовый ответ API"

    monkeypatch.setattr(speech, "transcribe", transcribe)
    monkeypatch.setattr(asr, "load_api_key", lambda: "test-stub-key")
    monkeypatch.setattr(asr.time, "sleep", lambda _: None)
    with independent.speech_context():
        first = asr.transcribe_audio(max_new=1)
        second = asr.transcribe_audio()
    assert first == {"completed": 1, "new": 1, "remaining": 1}
    assert second == {"completed": 2, "new": 1, "remaining": 0}
    assert calls == ["whisper-large-v3", "whisper-large-v3"]
    responses = asr.read_rows(independent.SPEECH / "transcriptions.jsonl")
    assert [r["transcript"] for r in responses] == ["Тестовый ответ API"] * 2
