import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, QPointF, Qt
from PySide6.QtQuick import QQuickItem
from PySide6.QtTest import QTest

from jev_launcher import controller as controller_module
from jev_launcher.app import create_window
from jev_launcher.catalog import Catalog


def wait_until(predicate, timeout=3000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(20)
    assert predicate()


@pytest.fixture
def ui(qapp, tmp_path):
    folder = tmp_path / "Проект"
    folder.mkdir()
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    item = catalog.add_path(folder)
    catalog.update(item.object_id, name="Editor", description="Редактор кода", aliases=["код"])
    engine, controller, window = create_window(qapp, catalog)
    warnings = []
    engine.warnings.connect(
        lambda values: warnings.extend(str(value.toString()) for value in values)
    )
    QTest.qWait(250)
    yield controller, window, catalog, warnings
    controller.cancelSpeech()
    controller.jobs.pool.waitForDone(5000)
    window.hide()
    engine.deleteLater()
    QTest.qWait(100)
    assert not warnings


def test_qml_keyboard_filter_and_explicit_enter_launch(ui, monkeypatch):
    controller, window, catalog, _ = ui
    calls = []
    monkeypatch.setattr(
        controller_module,
        "launch_by_id",
        lambda catalog, object_id: calls.append(object_id) or catalog.get(object_id),
    )
    window.requestActivate()
    field = window.findChild(QObject, "searchField")
    field.forceActiveFocus()
    for key in (Qt.Key.Key_E, Qt.Key.Key_D, Qt.Key.Key_I, Qt.Key.Key_T, Qt.Key.Key_O, Qt.Key.Key_R):
        QTest.keyClick(window, key)
    wait_until(lambda: controller.query.casefold() == "editor")
    assert len(controller.results) == 1
    assert calls == []
    QTest.keyClick(window, Qt.Key.Key_Return)
    wait_until(lambda: bool(calls))
    assert calls == [catalog.items()[0].object_id]
    wait_until(lambda: not controller.opening)
    assert not controller.isError


def test_editor_saves_description_and_favorite(ui):
    controller, window, catalog, _ = ui
    item = catalog.items()[0]
    assert controller.saveObject(item.object_id, "Мой проект", "Описание", "проект,мл", True, True)
    assert catalog.get(item.object_id).favorite
    assert catalog.get(item.object_id).aliases == ("проект", "мл")
    controller.setQuery("описание")
    assert len(controller.results) == 1
    assert window.findChild(QObject, "searchField").property("text") == "описание"
    assert not controller.saveObject(item.object_id, "", "Описание", "", True, False)
    assert controller.isError


def test_qml_editor_and_theme_controls(ui):
    controller, window, catalog, _ = ui
    navigation = window.findChild(QObject, "navigation")
    navigation.setProperty("currentIndex", 1)
    page = window.findChild(QObject, "catalogPage")
    page.editObject(controller.objectDetails(catalog.items()[0].object_id))
    QTest.qWait(50)
    window.findChild(QObject, "descriptionField").setProperty("text", "Описание из окна")
    window.findChild(QObject, "saveObjectButton").clicked.emit()
    assert catalog.items()[0].description == "Описание из окна"
    navigation.setProperty("currentIndex", 2)
    window.findChild(QObject, "themeBox").setProperty("currentIndex", 1)
    window.findChild(QObject, "saveSettingsButton").clicked.emit()
    assert controller.theme == "light"
    assert window.property("dark") is False


def test_scan_runs_off_gui_thread_and_preserves_responsiveness(ui, monkeypatch):
    controller, _, _, _ = ui
    from jev_launcher.objects import ScanResult

    def slow():
        time.sleep(0.2)
        return ScanResult([], ["start_apps"])

    monkeypatch.setattr(controller_module, "scan_windows", slow)
    controller.refresh()
    assert controller.scanning
    controller.setQuery("редактор")
    assert len(controller.results) == 1
    wait_until(lambda: not controller.scanning)
    assert not controller.isError


@pytest.mark.parametrize("cancel_method", ["cancel", "edit"])
def test_late_transcription_cannot_replace_new_input(ui, monkeypatch, cancel_method):
    controller, _, _, _ = ui
    callbacks = []
    monkeypatch.setattr(controller_module.config, "api_key", lambda catalog: "test-only-key")
    monkeypatch.setattr(controller.jobs, "submit", lambda work, done: callbacks.append(done))
    controller.recorder = SimpleNamespace(
        recording=True,
        stop=lambda: b"audio",
        cancel=lambda: None,
    )
    controller.setQuery("original")
    controller.toggleRecording()
    assert controller.transcribing
    if cancel_method == "cancel":
        controller.cancelSpeech()
    else:
        controller.setQuery("new input")
    assert not controller.transcribing
    callbacks[0]("stale transcription", "")
    assert controller.query == ("original" if cancel_method == "cancel" else "new input")


def test_missing_voice_key_is_a_user_visible_error(ui, monkeypatch):
    controller, _, _, _ = ui
    monkeypatch.setattr(controller_module.config, "api_key", lambda catalog: "")
    controller.toggleRecording()
    assert controller.isError
    assert not controller.recording


@pytest.mark.parametrize("size", [(720, 560), (1080, 760)])
def test_search_catalog_and_editor_remain_inside_resized_window(ui, size):
    controller, window, catalog, _ = ui
    window.setWidth(size[0])
    window.setHeight(size[1])
    item = catalog.items()[0]
    catalog.update(item.object_id, name="Очень длинное название проекта " * 8,
                   description="Подробное описание папки " * 16)
    controller.reload()
    QTest.qWait(100)

    def assert_inside(name):
        control = window.findChild(QQuickItem, name)
        assert control is not None
        position = control.mapToScene(QPointF(0, 0))
        assert control.width() > 24 and control.height() > 20
        assert position.x() >= -1 and position.y() >= -1
        assert position.x() + control.width() <= window.width() + 1
        assert position.y() + control.height() <= window.height() + 1

    for name in ("searchField", "voiceButton", "resultsList", "openButton"):
        assert_inside(name)
    navigation = window.findChild(QObject, "navigation")
    navigation.setProperty("currentIndex", 1)
    QTest.qWait(100)
    for name in ("catalogSearchField", "catalogList", "refreshButton"):
        assert_inside(name)
    window.findChild(QObject, "catalogPage").editObject(controller.objectDetails(item.object_id))
    QTest.qWait(100)
    dialog = window.findChild(QObject, "objectEditor")
    assert dialog.property("x") >= 0 and dialog.property("y") >= 0
    assert dialog.property("x") + dialog.property("width") <= window.width()
    assert dialog.property("y") + dialog.property("height") <= window.height()
    assert_inside("saveObjectButton")
