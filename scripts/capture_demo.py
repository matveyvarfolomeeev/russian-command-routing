"""Render the real QML UI with fictional objects; never launches programs."""

import argparse
from pathlib import Path

from PySide6.QtCore import QObject
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from jev_launcher.app import create_window
from jev_launcher.catalog import Catalog
from jev_launcher.objects import DiscoveredObject, ScanResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/preview"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = Catalog("artifacts/demo-preview/catalog.sqlite3")
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
    controller.saveSettings("Ctrl+Alt+Space", "light", "", "", False)
    navigation.setProperty("currentIndex", 0)
    QTest.qWait(100)
    assert window.grabWindow().save(str(args.output_dir / "light.png"))
    window.hide()
    engine.deleteLater()
    QTest.qWait(100)
    print("Saved five UI previews with fictional objects.")


if __name__ == "__main__":
    main()
