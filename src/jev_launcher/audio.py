"""Push-to-record microphone capture; PCM and WAV stay in memory."""

from __future__ import annotations

import array
import io
import sys
import wave
from dataclasses import dataclass

from PySide6.QtCore import QBuffer, QIODevice, QObject, QTimer, Signal, Slot
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices, QtAudio


def pcm_to_wav(
    raw: bytes, sample_format: QAudioFormat.SampleFormat, rate: int, channels: int
) -> bytes:
    if sample_format == QAudioFormat.SampleFormat.Int16:
        pcm = raw[: len(raw) // (2 * channels) * (2 * channels)]
    else:
        widths = {
            QAudioFormat.SampleFormat.Float: ("f", 4),
            QAudioFormat.SampleFormat.Int32: ("i", 4),
            QAudioFormat.SampleFormat.UInt8: ("B", 1),
        }
        if sample_format not in widths:
            raise ValueError("Микрофон использует неподдерживаемый формат")
        code, width = widths[sample_format]
        samples = array.array(code)
        samples.frombytes(raw[: len(raw) // (width * channels) * (width * channels)])
        if sys.byteorder != "little" and width > 1:
            samples.byteswap()
        if code == "f":
            converted = (max(-32768, min(32767, int(value * 32767))) for value in samples)
        elif code == "i":
            converted = (value >> 16 for value in samples)
        else:
            converted = ((value - 128) << 8 for value in samples)
        output = array.array("h", converted)
        if sys.byteorder != "little":
            output.byteswap()
        pcm = output.tobytes()
    if channels > 1:
        samples = array.array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        mono = array.array(
            "h",
            (
                round(sum(samples[index : index + channels]) / channels)
                for index in range(0, len(samples), channels)
            ),
        )
        if sys.byteorder != "little":
            mono.byteswap()
        pcm = mono.tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


@dataclass(frozen=True)
class AudioRecording:
    raw: bytes
    sample_format: QAudioFormat.SampleFormat
    rate: int
    channels: int

    def wav(self) -> bytes:
        return pcm_to_wav(self.raw, self.sample_format, self.rate, self.channels)


class Recorder(QObject):
    changed = Signal()
    failed = Signal(str)
    limitReached = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.buffer = None
        self.format = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.limitReached)

    @staticmethod
    def devices() -> list[dict]:
        return [
            {"name": device.description(), "id": bytes(device.id()).hex()}
            for device in QMediaDevices.audioInputs()
        ]

    @property
    def recording(self):
        return self.source is not None

    def start(self, device_id: str = ""):
        inputs = QMediaDevices.audioInputs()
        device = next(
            (value for value in inputs if bytes(value.id()).hex() == device_id),
            QMediaDevices.defaultAudioInput(),
        )
        if device.isNull():
            raise RuntimeError(
                "Микрофон не найден. Подключите устройство и проверьте настройки Windows."
            )
        # WASAPI may advertise 16 kHz mono but reject it at stream initialization.
        # Capture the actual native format and convert/downmix off the GUI thread.
        format = device.preferredFormat()
        self.format = format
        self.buffer = QBuffer(self)
        self.buffer.open(QIODevice.OpenModeFlag.ReadWrite)
        self.source = QAudioSource(device, format, self)
        self.source.stateChanged.connect(self._state_changed)
        self.source.start(self.buffer)
        if self.source is not None and self.source.error() != QtAudio.Error.NoError:
            self.cancel()
            raise RuntimeError(
                "Не удалось включить микрофон. Проверьте доступ в настройках Windows."
            )
        self.timer.start(60000)
        self.changed.emit()

    @Slot()
    def _state_changed(self):
        if (
            self.source
            and self.source.state() == QtAudio.State.StoppedState
            and self.source.error() != QtAudio.Error.NoError
        ):
            self.cancel()
            self.failed.emit("Запись прервана. Проверьте микрофон и доступ к нему.")

    def stop(self) -> AudioRecording:
        if self.source is None:
            raise ValueError("Запись уже остановлена")
        source, self.source = self.source, None
        source.stop()
        source.deleteLater()
        self.timer.stop()
        raw = bytes(self.buffer.data())
        self.buffer.close()
        self.buffer.deleteLater()
        self.buffer = None
        self.changed.emit()
        return AudioRecording(
            raw, self.format.sampleFormat(), self.format.sampleRate(), self.format.channelCount()
        )

    def cancel(self):
        if self.source:
            source, self.source = self.source, None
            source.stop()
            source.deleteLater()
        self.timer.stop()
        if self.buffer:
            self.buffer.close()
            self.buffer.deleteLater()
            self.buffer = None
        self.changed.emit()
