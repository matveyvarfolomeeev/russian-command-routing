"""Render the real QML UI with fictional objects; never launches programs."""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import QObject
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from jev_launcher import config
from jev_launcher.app import create_window
from jev_launcher.catalog import Catalog
from jev_launcher.hotkey import DEFAULT_HOTKEY
from jev_launcher.objects import DiscoveredObject, ScanResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/preview"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = Catalog("artifacts/demo-preview/catalog.sqlite3")
    catalog.set_setting("hotkey", DEFAULT_HOTKEY)
    catalog.set_setting("theme", "dark")
    catalog.set_setting("search_model", "filter")
    examples = [
        (
            "Проект по ML",
            "folder",
            "folder",
            r"C:\Examples\ML Project",
            "Код, данные и результаты экспериментов",
            ("машинное обучение", "исследование"),
        ),
        (
            "Visual Studio Code",
            "app",
            "executable",
            r"C:\Examples\Code.exe",
            "Редактор кода и работа с проектами",
            ("вскод", "код"),
        ),
        (
            "Блокнот",
            "app",
            "executable",
            r"C:\Windows\System32\notepad.exe",
            "Простые текстовые заметки",
            ("заметки",),
        ),
        (
            "Материалы курса",
            "folder",
            "folder",
            r"C:\Examples\Course",
            "Конспекты, задачи и учебные материалы",
            ("учёба",),
        ),
        (
            "План проекта",
            "file",
            "file",
            r"C:\Examples\Plan.pdf",
            "План работ и описание экспериментов",
            ("документ",),
        ),
    ]
    catalog.merge_scan(
        ScanResult(
            [
                DiscoveredObject(
                    name=name,
                    kind=kind,
                    launch_type=launch_type,
                    launch_target=target,
                    description=description,
                    description_source="user",
                    aliases=aliases,
                    source="demo",
                    source_key="demo:" + name,
                )
                for name, kind, launch_type, target, description, aliases in examples
            ],
            ["demo"],
        )
    )
    favorite = next(item for item in catalog.items() if item.kind == "folder")
    catalog.update(favorite.object_id, favorite=True)
    application = QApplication([])
    engine, controller, window = create_window(application, catalog)
    QTest.qWait(600)
    navigation = window.findChild(QObject, "navigation")
    for index, name in [(0, "search"), (1, "catalog"), (2, "settings")]:
        navigation.setProperty("currentIndex", index)
        QTest.qWait(100)
        assert window.grabWindow().save(str(args.output_dir / (name + ".png")))
    navigation.setProperty("currentIndex", 1)
    window.findChild(QObject, "catalogPage").editObject(
        controller.objectDetails(favorite.object_id)
    )
    QTest.qWait(100)
    assert window.grabWindow().save(str(args.output_dir / "editor.png"))
    window.findChild(QObject, "objectEditor").close()
    controller.saveSettings(DEFAULT_HOTKEY, "light", "", "", False)
    navigation.setProperty("currentIndex", 0)
    QTest.qWait(100)
    assert window.grabWindow().save(str(args.output_dir / "light.png"))
    quick, quick_window = controller.quick, controller.quick_window
    def capture_quick(name, compact=False):
        deadline = time.monotonic() + 4
        surface = quick_window.findChild(QObject, "quickSurface")
        while time.monotonic() < deadline:
            QTest.qWait(30)
            if (quick.visible and surface.property("opacity") >= 0.99
                    and quick_window.width() == (220 if compact else 680)
                    and quick_window.height() == (84 if compact else 250 if quick.error else 218)):
                QTest.qWait(50)
                assert quick_window.grabWindow().save(str(args.output_dir / name))
                return
        raise RuntimeError("Preview did not reach the requested visible state: " + name)

    # Record/ASR/opening previews use substitutes; no microphone or programs are started.
    def start(device):
        quick.recorder.recording = True

    quick.recorder = SimpleNamespace(
        recording=False, start=start,
        stop=lambda: setattr(quick.recorder, "recording", False),
        cancel=lambda: setattr(quick.recorder, "recording", False),
    )
    for theme in ("dark", "light"):
        controller.saveSettings(DEFAULT_HOTKEY, theme, "", "", False)
        quick.show("text")
        quick.setQuery("код")
        capture_quick(f"quick-{theme}.png")
        quick.cancel()
        with patch.object(config, "api_key", return_value="preview-only"):
            quick.show("voice")
        capture_quick(f"voice-{theme}.png")
        quick.cancel()
        quick.show("text")
        quick.setQuery("код")
        with patch.object(controller.jobs, "submit", return_value=0):
            quick.submit()
            capture_quick(f"capsule-{theme}.png", compact=True)
        quick._fail("Не удалось открыть объект. Проверьте путь в каталоге.")
        capture_quick(f"error-{theme}.png")
        quick.cancel()
    controller.closeModels()
    controller.jobs.pool.waitForDone(5000)
    window.hide()
    quick_window.hide()
    engine.deleteLater()
    QTest.qWait(100)
    print("Saved Hub and quick-launch previews with fictional objects; no programs launched.")


if __name__ == "__main__":
    main()
