import io
import wave

from PySide6.QtCore import QObject, Signal
from PySide6.QtMultimedia import QAudioFormat, QtAudio

from jev_launcher import audio


class FakeDevice:
    def isNull(self):
        return False

    def preferredFormat(self):
        format = QAudioFormat()
        format.setSampleRate(48000)
        format.setChannelCount(4)
        format.setSampleFormat(QAudioFormat.SampleFormat.Float)
        return format


class FakeSource(QObject):
    stateChanged = Signal()

    def __init__(self, device, format, parent):
        super().__init__(parent)
        self._error = QtAudio.Error.NoError
        self._state = QtAudio.State.StoppedState

    def start(self, buffer):
        self._state = QtAudio.State.ActiveState
        buffer.write(b"\x00" * (4 * 4 * 480))
        self.stateChanged.emit()

    def stop(self):
        self._state = QtAudio.State.StoppedState
        self.stateChanged.emit()

    def state(self):
        return self._state

    def error(self):
        return self._error


def fake_microphone(monkeypatch):
    monkeypatch.setattr(audio.QMediaDevices, "audioInputs", lambda: [])
    monkeypatch.setattr(audio.QMediaDevices, "defaultAudioInput", lambda: FakeDevice())
    monkeypatch.setattr(audio, "QAudioSource", FakeSource)


def test_native_format_and_qtaudio_enums(qapp, monkeypatch):
    fake_microphone(monkeypatch)
    recorder = audio.Recorder()
    recorder.start()
    assert recorder.recording
    recording = recorder.stop()
    assert not recorder.recording
    assert recording.rate == 48000 and recording.channels == 4
    with wave.open(io.BytesIO(recording.wav())) as wav:
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 480


def test_device_failure_stops_and_discards_recording(qapp, monkeypatch):
    fake_microphone(monkeypatch)
    recorder = audio.Recorder()
    errors = []
    recorder.failed.connect(errors.append)
    recorder.start()
    recorder.source._error = QtAudio.Error.OpenError
    recorder.source._state = QtAudio.State.StoppedState
    recorder.source.stateChanged.emit()
    assert not recorder.recording
    assert recorder.buffer is None
    assert len(errors) == 1


def test_recording_cancel_and_limit_signal(qapp, monkeypatch):
    fake_microphone(monkeypatch)
    recorder = audio.Recorder()
    reached = []
    recorder.limitReached.connect(lambda: reached.append(True))
    recorder.start()
    recorder.timer.timeout.emit()
    assert reached
    recorder.cancel()
    assert not recorder.recording
    assert recorder.buffer is None
    assert not recorder.timer.isActive()
