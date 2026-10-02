"""Execution boundary: only a stored catalog ID can be opened."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .catalog import Catalog
from .objects import CatalogObject


class LaunchError(RuntimeError):
    pass


def open_object(item: CatalogObject) -> None:
    if os.name != "nt":
        raise LaunchError("Открытие объектов поддерживается только в Windows")
    if not item.enabled or not item.available:
        raise LaunchError("Объект отключён или больше недоступен. Обновите каталог.")
    try:
        if item.launch_type == "app_id":
            os.startfile("shell:AppsFolder\\" + item.launch_target)
        else:
            target = Path(item.launch_target)
            if not target.exists():
                raise LaunchError("Объект не найден по сохранённому пути. Обновите каталог.")
            if item.launch_type == "executable":
                subprocess.Popen(
                    [str(target), *item.arguments],
                    cwd=item.working_directory or None,
                    shell=False,
                )
            elif item.launch_type in ("shortcut", "folder", "file"):
                os.startfile(str(target))
            else:
                raise LaunchError("Неизвестный способ открытия объекта")
    except OSError as exc:
        messages = {
            2: "Объект не найден. Обновите каталог.",
            5: "Windows отказала в доступе к объекту.",
            1155: "Для этого типа файла не назначено приложение.",
        }
        raise LaunchError(
            messages.get(getattr(exc, "winerror", None), "Windows не смогла открыть объект.")
        ) from exc


def launch_by_id(catalog: Catalog, object_id: str) -> CatalogObject:
    try:
        item = catalog.get(object_id)
    except KeyError as exc:
        raise LaunchError("Объект больше не существует в каталоге") from exc
    open_object(item)
    return item
