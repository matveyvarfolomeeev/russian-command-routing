"""Groq transcription over HTTPS. No GUI, audio-device or model dependencies."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from uuid import uuid4


class SpeechError(RuntimeError):
    pass


def transcribe(
    wav: bytes, api_key: str, *, timeout: float = 30, model: str = "whisper-large-v3"
) -> str:
    if not api_key.strip():
        raise SpeechError("Укажите ключ Groq в настройках или GROQ_API_KEY в окружении.")
    if not wav or len(wav) <= 44:
        raise SpeechError("Запись пустая. Запишите голос ещё раз.")
    if len(wav) > 20 * 1024 * 1024:
        raise SpeechError("Запись слишком большая. Используйте запись до 60 секунд.")
    boundary = "jev-" + uuid4().hex
    parts = []
    for key, value in {
        "model": model,
        "language": "ru",
        "response_format": "json",
    }.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"'
            "\r\nContent-Type: audio/wav\r\n\r\n"
        ).encode()
    )
    parts.extend([wav, f"\r\n--{boundary}--\r\n".encode()])
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        data=b"".join(parts),
        headers={
            "Authorization": "Bearer " + api_key.strip(),
            "Content-Type": "multipart/form-data; boundary=" + boundary,
            "User-Agent": "JevLauncher/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read(1024 * 1024))
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SpeechError("Речь не распознана. Попробуйте ещё раз.")
        return text.strip()
    except urllib.error.HTTPError as exc:
        messages = {
            401: "Groq отклонил API-ключ. Проверьте его в настройках.",
            403: "Groq отклонил запрос. Проверьте доступ к сервису и попробуйте ещё раз.",
            413: "Запись слишком большая для сервиса.",
            429: "Достигнут лимит Groq. Попробуйте позже.",
        }
        raise SpeechError(
            messages.get(exc.code, "Сервис распознавания временно недоступен.")
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SpeechError("Нет ответа от сервиса. Проверьте сеть и попробуйте ещё раз.") from exc
    except (ValueError, TypeError, AttributeError) as exc:
        raise SpeechError("Сервис вернул некорректный ответ.") from exc
