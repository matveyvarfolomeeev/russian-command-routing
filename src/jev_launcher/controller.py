"""Qt adapter. Domain modules remain callable without this controller."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from . import config
from .audio import Recorder
from .discovery import scan_windows
from .hotkey import DEFAULT_HOTKEY
from .jobs import Jobs
from .launch import launch_by_id
from .search import filter_catalog
from .speech import transcribe

KIND_NAMES = {"app": "Приложение", "folder": "Папка", "file": "Файл"}
SOURCE_NAMES = {
    "manual": "Добавлено вручную",
    "start_apps": "Приложения Windows",
    "start_menu_user": "Меню Пуск",
    "start_menu_common": "Общее меню Пуск",
}


class Controller(QObject):
    stateChanged = Signal()
    queryChanged = Signal()
    catalogChanged = Signal()
    showRequested = Signal()
    hideRequested = Signal()

    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.jobs = Jobs(self)
        self.recorder = Recorder(self)
        self.recorder.failed.connect(self._recording_failed)
        self.recorder.changed.connect(self.stateChanged)
        self.recorder.limitReached.connect(self.toggleRecording)
        self._query = ""
        self._catalog_query = ""
        self._all = []
        self._revision = 0
        self._status = ""
        self._error = False
        self._scanning = False
        self._opening = False
        self._transcribing = False
        self._speech_token = 0
        self.hotkey = None
        self.reload()

    def reload(self):
        self._all = self.catalog.items()
        self._revision += 1
        self.catalogChanged.emit()
        self.stateChanged.emit()

    def refreshIcons(self):
        self._revision += 1
        self.stateChanged.emit()

    def _message(self, message, error=False):
        self._status, self._error = message, error
        self.stateChanged.emit()

    def _view(self, item):
        value = item.to_dict()
        for key in ("aliases", "arguments", "overrides"):
            value[key] = list(value[key])
        value.update(
            kindLabel=KIND_NAMES[item.kind],
            sourceLabel=SOURCE_NAMES.get(item.source, item.source),
            icon=f"image://catalog/{item.object_id}?v={self._revision}",
        )
        return value

    @Property(str, notify=queryChanged)
    def query(self):
        return self._query

    @Slot(str)
    def setQuery(self, query):
        if query != self._query:
            if self._transcribing:
                self._speech_token += 1
                self._transcribing = False
                self._status = "Ввод изменён. Результат распознавания отменён."
                self._error = False
            self._query = query
            self.queryChanged.emit()
            self.stateChanged.emit()

    @Slot(str)
    def setCatalogQuery(self, query):
        self._catalog_query = query
        self.stateChanged.emit()

    @Property("QVariantList", notify=stateChanged)
    def results(self):
        return [self._view(item) for item in filter_catalog(self._all, self._query)]

    @Property("QVariantList", notify=stateChanged)
    def catalogItems(self):
        return [
            self._view(item)
            for item in filter_catalog(self._all, self._catalog_query, active_only=False)
        ]

    @Property(int, notify=stateChanged)
    def totalCount(self):
        return len(self._all)

    @Property(int, notify=stateChanged)
    def activeCount(self):
        return sum(item.enabled and item.available for item in self._all)

    @Property(str, notify=stateChanged)
    def status(self):
        return self._status

    @Property(bool, notify=stateChanged)
    def isError(self):
        return self._error

    @Property(bool, notify=stateChanged)
    def scanning(self):
        return self._scanning

    @Property(bool, notify=stateChanged)
    def opening(self):
        return self._opening

    @Property(bool, notify=stateChanged)
    def recording(self):
        return self.recorder.recording

    @Property(bool, notify=stateChanged)
    def transcribing(self):
        return self._transcribing

    @Property(str, notify=stateChanged)
    def hotkeyText(self):
        return self.catalog.setting("hotkey", DEFAULT_HOTKEY)

    @Property(bool, notify=stateChanged)
    def hotkeyRegistered(self):
        return bool(self.hotkey and self.hotkey.registered)

    @Property(str, notify=stateChanged)
    def hotkeyStatus(self):
        return self.hotkey.status_text if self.hotkey else "Горячая клавиша отключена"

    @Property(str, notify=stateChanged)
    def theme(self):
        return self.catalog.setting("theme", "dark")

    @Property(bool, notify=stateChanged)
    def keyConfigured(self):
        return bool(config.api_key(self.catalog))

    @Property("QVariantList", notify=stateChanged)
    def microphones(self):
        return [{"name": "По умолчанию в Windows", "id": ""}, *Recorder.devices()]

    @Property(str, notify=stateChanged)
    def microphoneId(self):
        return self.catalog.setting("microphone", "")

    @Property(str, notify=stateChanged)
    def lastScan(self):
        stamp = self.catalog.setting("last_scan", {}).get("at", "")
        if not stamp:
            return "Ещё не обновлялся"
        try:
            return datetime.fromisoformat(stamp).astimezone().strftime("%d.%m.%Y %H:%M")
        except (TypeError, ValueError):
            return "Ещё не обновлялся"

    @Property(bool, notify=stateChanged)
    def catalogCached(self):
        return bool(self._all)

    @Property(str, notify=stateChanged)
    def catalogState(self):
        if self._scanning:
            return "Обновляется в фоне"
        return "Локальный каталог" if self._all else "Каталог пуст"

    @Slot()
    def refreshIfStale(self):
        if self.catalog.needs_refresh():
            self.refresh()

    @Slot()
    def refresh(self):
        if self._scanning:
            return
        self._scanning = True
        self._message("Собираем приложения из Windows…")

        def work():
            result = scan_windows()
            self.catalog.merge_scan(result)
            return result

        def done(result, error):
            self._scanning = False
            self.reload()
            if error:
                self._message(error, True)
            elif result.errors:
                self._message(
                    f"Каталог обновлён: {self.activeCount} объектов. "
                    f"Не удалось прочитать часть записей ({len(result.errors)}).",
                    True,
                )
            else:
                self._message(f"Каталог обновлён: {self.activeCount} объектов.")

        self.jobs.submit(work, done)

    @Slot(str)
    def openObject(self, object_id):
        if self._opening:
            return
        self._opening = True
        self._message("Открываем…")

        def done(item, error):
            self._opening = False
            if error:
                self._message(error, True)
            else:
                self._message("Команда открытия отправлена: " + item.name)

        self.jobs.submit(lambda: launch_by_id(self.catalog, object_id), done)

    @Slot(bool)
    def addPath(self, folder=False):
        if folder:
            path = QFileDialog.getExistingDirectory(None, "Добавить папку")
        else:
            path, _ = QFileDialog.getOpenFileName(None, "Добавить приложение или файл")
        if not path:
            return
        try:
            item = self.catalog.add_path(path)
            self.reload()
            self._message("Добавлено: " + item.name)
        except (OSError, ValueError) as exc:
            self._message(str(exc), True)

    @Slot(str, result="QVariantMap")
    def objectDetails(self, object_id):
        try:
            return self._view(self.catalog.get(object_id))
        except KeyError:
            return {}

    @Slot(str, str, str, str, bool, bool, result=bool)
    def saveObject(self, object_id, name, description, aliases, enabled, favorite):
        try:
            self.catalog.update(
                object_id,
                name=name,
                description=description,
                aliases=aliases.split(","),
                enabled=enabled,
                favorite=favorite,
            )
            self.reload()
            self._message("Изменения сохранены.")
            return True
        except (ValueError, KeyError) as exc:
            self._message(str(exc), True)
            return False

    @Slot(str)
    def toggleFavorite(self, object_id):
        item = self.catalog.get(object_id)
        self.catalog.update(object_id, favorite=not item.favorite)
        self.reload()

    @Slot(str)
    def removeObject(self, object_id):
        self.catalog.remove(object_id)
        self.reload()
        self._message(
            "Объект удалён."
            if object_id not in {i.object_id for i in self._all}
            else "Объект отключён. Его можно включить в каталоге."
        )

    @Slot()
    def exportCatalog(self):
        path, _ = QFileDialog.getSaveFileName(
            None, "Экспорт для экспериментов", "catalog.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            from .cli import catalog_snapshot

            Path(path).write_text(
                json.dumps(catalog_snapshot(self.catalog), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._message("Полный доступный каталог экспортирован.")
        except OSError:
            self._message("Не удалось сохранить файл экспорта.", True)

    @Slot(str, str, str, str, bool, result=bool)
    def saveSettings(self, hotkey, theme, microphone, new_key, clear_key=False):
        try:
            if self.hotkey:
                self.hotkey.register(hotkey)
            if theme not in ("dark", "light"):
                raise ValueError("Неизвестная тема")
            if clear_key:
                config.save_api_key(self.catalog, "")
            elif new_key.strip():
                config.save_api_key(self.catalog, new_key)
            self.catalog.set_setting("hotkey", hotkey)
            self.catalog.set_setting("theme", theme)
            self.catalog.set_setting("microphone", microphone)
            self._message("Настройки сохранены.")
            return True
        except (ValueError, RuntimeError, OSError) as exc:
            self._message(str(exc), True)
            return False

    @Slot()
    def toggleRecording(self):
        if self._transcribing:
            self.cancelSpeech()
            return
        if not self.recording:
            if not self.keyConfigured:
                self._message("Настройте ключ Groq на вкладке «Настройки».", True)
                return
            try:
                self.recorder.start(self.microphoneId)
                self._message("Идёт запись. Нажмите кнопку ещё раз, чтобы распознать.")
            except (ValueError, RuntimeError) as exc:
                self._message(str(exc), True)
            return
        try:
            recording = self.recorder.stop()
        except ValueError as exc:
            self._message(str(exc), True)
            return
        self._speech_token += 1
        token = self._speech_token
        self._transcribing = True
        self._message("Распознаём речь…")
        key = config.api_key(self.catalog)

        def done(text, error):
            if token != self._speech_token:
                return
            self._transcribing = False
            if error:
                self._message(error, True)
            else:
                self.setQuery(text)
                self._message("Речь распознана. Проверьте текст и выберите объект.")

        self.jobs.submit(lambda: transcribe(recording.wav(), key), done)

    @Slot()
    def cancelSpeech(self):
        self._speech_token += 1
        self._transcribing = False
        self.recorder.cancel()
        self._message("Запись или распознавание отменены.")

    def _recording_failed(self, error):
        self._message(error, True)

    @Slot()
    def showWindow(self):
        self.showRequested.emit()

    @Slot()
    def hideWindow(self):
        self.hideRequested.emit()
