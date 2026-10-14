import threading
import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, QPointF, Qt
from PySide6.QtTest import QTest

from jev_launcher import controller as controller_module
from jev_launcher import quick as quick_module
from jev_launcher.app import create_window
from jev_launcher.catalog import Catalog
from jev_launcher.controller import Controller


def wait_until(predicate, timeout=4000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(20)
    assert predicate()


@pytest.fixture
def session(qapp, tmp_path, monkeypatch):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.set_setting("search_model", "filter")
    for name in ("Editor", "Notes"):
        folder = tmp_path / name
        folder.mkdir()
        catalog.add_path(folder)
    controller = Controller(catalog)
    calls = []
    monkeypatch.setattr(quick_module, "launch_by_id",
                        lambda db, value: calls.append(value) or db.get(value))
    monkeypatch.setattr(controller_module.config, "api_key", lambda db: "test-key")
    quick = controller.quick

    def start(device):
        quick.recorder.recording = True

    def stop():
        quick.recorder.recording = False
        return SimpleNamespace(wav=lambda: b"test-wav")

    quick.recorder = SimpleNamespace(recording=False, start=start, stop=stop,
                                     cancel=lambda: setattr(quick.recorder, "recording", False))
    yield controller, quick, catalog, calls
    controller.closeModels()
    controller.jobs.pool.waitForDone(5000)
    QTest.qWait(30)


def manual_jobs(controller, monkeypatch):
    pending = []
    monkeypatch.setattr(controller.jobs, "submit", lambda work, done: pending.append((work, done)))
    return pending


def complete(job, result=None, error="", run=False):
    work, done = job
    if run:
        try:
            result = work()
        except Exception as exc:
            error = str(exc)
    done(result, error)


def local_model(controller):
    controller.catalog.set_setting("search_model", "laya")


def ranked(catalog, name="Editor"):
    item = next(item for item in catalog.items() if item.name == name)
    return {"object_ids": [item.object_id], "elapsed_ms": 2}


def test_text_enter_opens_top_one_once_and_keeps_hub_query(session):
    controller, quick, catalog, calls = session
    controller.setQuery("Notes")
    controller.showQuick("text")
    quick.setQuery("Editor")
    assert quick.bestResult["name"] == "Editor"
    controller.submitQuick()
    controller.submitQuick()
    wait_until(lambda: not quick.visible)
    assert calls == ranked(catalog)["object_ids"]
    assert controller.query == "Notes"


def test_enter_during_search_waits_for_exact_submitted_query(session, monkeypatch):
    controller, quick, catalog, calls = session
    local_model(controller)
    jobs = manual_jobs(controller, monkeypatch)
    quick.show("text")
    quick.setQuery("old")
    quick.timer.stop()
    quick._search()
    quick.setQuery("Editor")
    quick.submit()
    quick.setQuery("Notes")
    quick.submit()
    assert quick.query == "Editor"
    assert quick.stage == "searching" and quick.submitted
    complete(jobs.pop(0), ranked(catalog, "Notes"))
    assert quick.bestResult == {} and calls == []
    assert len(jobs) == 1
    complete(jobs.pop(0), ranked(catalog))
    QTest.qWait(30)
    assert quick.stage == "opening" and len(jobs) == 1
    complete(jobs.pop(0), run=True)
    assert calls == ranked(catalog)["object_ids"]
    assert not quick.visible


@pytest.mark.parametrize("stop_first", [False, True])
def test_voice_enter_transcribes_selects_and_opens_once(session, monkeypatch, stop_first):
    controller, quick, catalog, calls = session
    received = []
    monkeypatch.setattr(quick_module, "transcribe",
                        lambda audio, key: received.append((audio, key)) or "Editor")
    jobs = manual_jobs(controller, monkeypatch)
    quick.show("voice")
    assert quick.recording and quick.stage == "recording"
    if stop_first:
        quick.stopRecording()
        assert not quick.submitted
        complete(jobs.pop(0), run=True)
        assert quick.query == "Editor" and quick.bestResult["name"] == "Editor"
        assert calls == []
    quick.submit()
    quick.submit()
    if not stop_first:
        assert quick.stage == "transcribing" and quick.submitted
        complete(jobs.pop(0), run=True)
    QTest.qWait(30)
    assert quick.stage == "opening"
    complete(jobs.pop(0), run=True)
    assert received == [(b"test-wav", "test-key")]
    assert calls == ranked(catalog)["object_ids"] and not quick.visible


def test_enter_while_preview_transcription_is_pending(session, monkeypatch):
    controller, quick, catalog, calls = session
    jobs = manual_jobs(controller, monkeypatch)
    quick.show("text")
    quick.startRecording()
    quick.stopRecording()
    quick.submit()
    assert quick.submitted and quick.stage == "transcribing"
    complete(jobs.pop(0), "Editor")
    QTest.qWait(20)
    complete(jobs.pop(0), run=True)
    assert calls == ranked(catalog)["object_ids"]


@pytest.mark.parametrize("stage", ["recording", "transcribing", "searching", "opening"])
def test_cancel_prevents_late_result_and_queued_opening(session, monkeypatch, stage):
    controller, quick, catalog, calls = session
    jobs = manual_jobs(controller, monkeypatch)
    if stage in ("recording", "transcribing"):
        quick.show("voice")
        if stage == "transcribing":
            quick.submit()
    else:
        if stage == "searching":
            local_model(controller)
        quick.show("text")
        quick.setQuery("Editor")
        quick.submit()
        if stage == "opening":
            QTest.qWait(20)
    assert quick.stage == stage
    quick.cancel()
    for job in list(jobs):
        if stage == "transcribing":
            complete(job, "Editor")
        elif stage == "searching":
            complete(job, ranked(catalog))
        else:
            complete(job, run=True)
    QTest.qWait(20)
    assert not quick.visible and not quick.recording and calls == []


def test_cancelled_session_cannot_overwrite_next_session(session, monkeypatch):
    controller, quick, _, calls = session
    jobs = manual_jobs(controller, monkeypatch)
    quick.show("voice")
    quick.submit()
    quick.cancel()
    quick.show("text")
    quick.setQuery("Notes")
    complete(jobs.pop(0), "Editor")
    assert quick.query == "Notes" and quick.bestResult["name"] == "Notes" and calls == []


@pytest.mark.parametrize("change", ["model", "catalog"])
def test_change_invalidates_submitted_model_response(session, monkeypatch, change):
    controller, quick, catalog, calls = session
    local_model(controller)
    jobs = manual_jobs(controller, monkeypatch)
    quick.show("text")
    quick.setQuery("Editor")
    quick.submit()
    if change == "model":
        controller.setSearchModel("e5_trained")
        controller._prepare_timer.stop()
    else:
        controller.reload()
        controller._prepare_timer.stop()
    complete(jobs.pop(0), ranked(catalog))
    QTest.qWait(20)
    assert quick.visible and not quick.submitted and quick.error and quick.query == "Editor"
    assert calls == []


def test_empty_input_no_duplicate_launch_and_hotkey_toggles(session):
    controller, quick, _, calls = session
    quick.show("text")
    quick.setQuery("   ")
    quick.submit()
    assert not quick.submitted and calls == []
    quick.show("text")
    assert not quick.visible
    quick.show("voice")
    assert quick.recording
    quick.show("voice")
    assert not quick.recording and not quick.visible
    quick.show("text")
    quick.show("voice")
    assert quick.visible and quick.mode == "voice" and quick.recording


def test_focus_loss_cancels_before_enter_but_processing_continues_after(session, monkeypatch):
    controller, quick, catalog, calls = session
    quick.show("voice")
    quick.focusLost()
    assert not quick.visible and not quick.recording
    jobs = manual_jobs(controller, monkeypatch)
    quick.show("text")
    quick.setQuery("Editor")
    quick.submit()
    quick.focusLost()
    assert quick.visible and quick.submitted
    QTest.qWait(20)
    complete(jobs.pop(0), run=True)
    assert calls == ranked(catalog)["object_ids"]


@pytest.mark.parametrize("error", ["microphone", "key", "asr", "empty_asr", "model", "open", "catalog"])
def test_errors_expand_and_keep_available_text(session, monkeypatch, error):
    controller, quick, catalog, calls = session
    jobs = manual_jobs(controller, monkeypatch)
    quick.show("text")
    quick.setQuery("Editor")
    if error == "microphone":
        def missing(device):
            raise RuntimeError("Микрофон не найден")
        quick.recorder.start = missing
        quick.startRecording()
    elif error == "key":
        monkeypatch.setattr(controller_module.config, "api_key", lambda db: "")
        quick.startRecording()
    elif error in ("asr", "empty_asr"):
        quick.startRecording()
        quick.submit()
        complete(jobs.pop(0), "" if error == "empty_asr" else None,
                 "ASR unavailable" if error == "asr" else "")
    elif error == "model":
        local_model(controller)
        quick.setQuery("Editor model")
        quick.submit()
        complete(jobs.pop(0), error="Нет весов модели")
    elif error == "open":
        quick.submit()
        QTest.qWait(20)
        complete(jobs.pop(0), error="Windows отказала в доступе")
    else:
        for item in catalog.items():
            catalog.remove(item.object_id)
        controller.reload()
        quick.submit()
    assert quick.visible and not quick.submitted and quick.error and not quick.recording
    assert quick.query.startswith("Editor") and calls == []


def test_background_preload_is_responsive_and_reused(session, monkeypatch):
    controller, quick, catalog, calls = session
    entered, release = threading.Event(), threading.Event()
    prepared = []

    def prepare(model, objects, **kwargs):
        entered.set()
        assert release.wait(3)
        prepared.append((model, objects))
        return {"event": "ready"}

    monkeypatch.setattr(controller.model_runtime, "prepare", prepare)
    monkeypatch.setattr(controller.model_runtime, "search", lambda *args, **kwargs: ranked(catalog))
    controller.setSearchModel("laya")
    try:
        wait_until(entered.is_set)
        assert controller.modelPreparing and not controller.modelReady
        quick.show("text")
        quick.setQuery("Editor")
        assert quick.query == "Editor"
        release.set()
        wait_until(lambda: controller.modelReady and not controller.modelPreparing)
        quick.cancel()
        quick.show("text")
        quick.setQuery("Editor")
        quick.submit()
        wait_until(lambda: not quick.visible)
        assert len(prepared) == 1 and calls == ranked(catalog)["object_ids"]
    finally:
        release.set()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_quick_qml_keyboard_capsule_center_and_escape(qapp, tmp_path, monkeypatch, theme):
    folder = tmp_path / "Editor"
    folder.mkdir()
    catalog = Catalog(tmp_path / "ui.sqlite3")
    catalog.set_setting("search_model", "filter")
    catalog.set_setting("theme", theme)
    catalog.add_path(folder)
    engine, controller, hub = create_window(qapp, catalog)
    warnings, calls = [], []
    engine.warnings.connect(lambda values: warnings.extend(v.toString() for v in values))
    quick, window = controller.quick, controller.quick_window
    monkeypatch.setattr(quick_module, "launch_by_id", lambda db, value: calls.append(value) or db.get(value))
    jobs = manual_jobs(controller, monkeypatch)
    try:
        # Let the desktop manager finish placing the freshly created Hub before activation.
        QTest.qWait(300)
        controller.showQuick("text")
        QTest.qWait(250)
        assert hub.isVisible() and window.isVisible()
        assert window.flags() & Qt.WindowType.FramelessWindowHint
        assert window.flags() & Qt.WindowType.WindowStaysOnTopHint
        assert window.property("dark") == (theme == "dark")
        assert window.property("transientParent") is None
        window.requestActivate()
        field = window.findChild(QObject, "quickInput")
        field.forceActiveFocus()
        for key in (Qt.Key.Key_E, Qt.Key.Key_D, Qt.Key.Key_I, Qt.Key.Key_T, Qt.Key.Key_O, Qt.Key.Key_R):
            QTest.keyClick(window, key)
        assert quick.query.lower() == "editor" and quick.bestResult["name"] == "Editor"
        center = QPointF(window.x() + window.width() / 2, window.y() + window.height() / 2)
        QTest.keyClick(window, Qt.Key.Key_Return)
        QTest.keyClick(window, Qt.Key.Key_Return)
        wait_until(lambda: window.width() == 220 and window.height() == 84)
        assert QPointF(window.x() + window.width() / 2, window.y() + window.height() / 2) == center
        assert window.findChild(QObject, "quickStageLabel").property("text") == "Открываю"
        assert len(jobs) == 1
        QTest.keyClick(window, Qt.Key.Key_Escape)
        complete(jobs.pop(0), run=True)
        assert not quick.visible and calls == []
    finally:
        controller.closeModels()
        controller.jobs.pool.waitForDone(5000)
        hub.hide()
        window.hide()
        engine.deleteLater()
        QTest.qWait(50)
    assert warnings == []
