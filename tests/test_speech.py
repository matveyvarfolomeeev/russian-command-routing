import array
import io
import json
import urllib.error
import wave
from contextlib import contextmanager

import pytest
from PySide6.QtMultimedia import QAudioFormat

from jev_launcher import speech
from jev_launcher.audio import pcm_to_wav
from jev_launcher.catalog import Catalog
from jev_launcher.config import api_key, save_api_key


def wav_bytes():
    return pcm_to_wav(b"\x00\x00" * 160, QAudioFormat.SampleFormat.Int16, 16000, 1)


def test_request_uses_expected_model_file_and_language(monkeypatch):
    requests = []

    @contextmanager
    def fake(request, timeout):
        requests.append((request, timeout))
        yield io.BytesIO(json.dumps({"text": "  открой блокнот  "}).encode())

    monkeypatch.setattr(speech.urllib.request, "urlopen", fake)
    assert speech.transcribe(wav_bytes(), "example-test-key") == "открой блокнот"
    request, timeout = requests[0]
    assert request.full_url == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert request.get_header("Authorization") == "Bearer example-test-key"
    assert request.get_header("User-agent") == "JevLauncher/0.1"
    assert b"whisper-large-v3-turbo" in request.data
    assert b'name="language"\r\n\r\nru' in request.data
    assert wav_bytes() in request.data
    assert timeout == 30


@pytest.mark.parametrize("code", [401, 403, 413, 429, 500])
def test_http_errors_hide_credentials(monkeypatch, code):
    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, code, "example-test-key", {}, None)

    monkeypatch.setattr(speech.urllib.request, "urlopen", fail)
    with pytest.raises(speech.SpeechError) as caught:
        speech.transcribe(wav_bytes(), "example-test-key")
    assert "example-test-key" not in str(caught.value)


@pytest.mark.parametrize("payload", [b"{}", b'{"text": ""}', b"null", b"not-json"])
def test_invalid_or_empty_response(monkeypatch, payload):
    @contextmanager
    def fake(request, timeout):
        yield io.BytesIO(payload)

    monkeypatch.setattr(speech.urllib.request, "urlopen", fake)
    with pytest.raises(speech.SpeechError):
        speech.transcribe(wav_bytes(), "example-test-key")


def test_missing_key_empty_recording_and_timeout(monkeypatch):
    with pytest.raises(speech.SpeechError):
        speech.transcribe(wav_bytes(), "")
    with pytest.raises(speech.SpeechError):
        speech.transcribe(b"", "test")

    def fail(request, timeout):
        raise TimeoutError()

    monkeypatch.setattr(speech.urllib.request, "urlopen", fail)
    with pytest.raises(speech.SpeechError):
        speech.transcribe(wav_bytes(), "test")


@pytest.mark.parametrize(
    ("format", "values", "typecode"),
    [
        (QAudioFormat.SampleFormat.Float, [-1, 0, 1], "f"),
        (QAudioFormat.SampleFormat.Int32, [-2147483648, 0, 2147483647], "i"),
        (QAudioFormat.SampleFormat.UInt8, [0, 128, 255], "B"),
    ],
)
def test_microphone_pcm_conversion(format, values, typecode):
    data = pcm_to_wav(array.array(typecode, values).tobytes(), format, 48000, 1)
    with wave.open(io.BytesIO(data)) as wav:
        assert (wav.getsampwidth(), wav.getframerate(), wav.getnchannels(), wav.getnframes()) == (
            2,
            48000,
            1,
            3,
        )


def test_key_encrypted_at_rest_and_env_priority(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    save_api_key(catalog, "example-test-key")
    assert "example-test-key" not in catalog.setting("groq_key")
    assert api_key(catalog) == "example-test-key"
    monkeypatch.setenv("GROQ_API_KEY", "environment-test-key")
    assert api_key(catalog) == "environment-test-key"
    monkeypatch.delenv("GROQ_API_KEY")
    save_api_key(catalog, "")
    assert api_key(catalog) == ""
