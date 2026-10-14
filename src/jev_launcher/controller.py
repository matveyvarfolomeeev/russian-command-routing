"""Qt adapter. Domain modules remain callable without this controller."""

from __future__ import annotations

import ctypes
import json
import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from . import config
from .audio import Recorder
from .discovery import scan_windows
from .hotkey import DEFAULT_HOTKEY, DEFAULT_VOICE_HOTKEY, parse_hotkey, register_pair
from .jobs import Jobs
from .launch import launch_by_id
from .model_search import DEFAULT_MODEL, MODELS, ModelRuntime
from .quick import QuickSession
from .search import filter_catalog
from .speech import transcribe

KIND_NAMES = {"app": "Приложение", "folder": "Папка", "file": "Файл"}
SOURCE_NAMES = {
    "manual": "Добавлено вручную",
    "start_apps": "Приложения Windows",
    "start_menu_user": "Меню Пуск",
    "start_menu_common": "Общее меню Пуск",
    "known_folders": "Папки Windows",
    "project_folders": "Папки проектов",
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
        self.model_runtime = ModelRuntime()
        self._model_ids = []
        self._model_generation = 0
        self._model_job_active = False
        self._model_searching = False
        self._model_timer = QTimer(self)
        self._model_timer.setSingleShot(True)
        self._model_timer.setInterval(250)
        self._model_timer.timeout.connect(self._run_model_search)
        self._prepare_generation = 0
        self._prepare_active = False
        self._model_ready = False
        self._model_preparing = False
        self._model_error = ""
        self._prepare_timer = QTimer(self)
        self._prepare_timer.setSingleShot(True)
        self._prepare_timer.timeout.connect(self._prepare_model)
        self._closing = False
        self.hotkey = None
        self.voice_hotkey = None
        self.quick = QuickSession(self)
        self.reload()

    def reload(self):
        self._all = self.catalog.items()
        self._revision += 1
        self._schedule_model_search()
        self._schedule_preparation()
        self.quick.invalidate()
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
            self._schedule_model_search()
            self.queryChanged.emit()
            self.stateChanged.emit()

    @Slot(str)
    def setCatalogQuery(self, query):
        self._catalog_query = query
        self.stateChanged.emit()

    @Property("QVariantList", notify=stateChanged)
    def results(self):
        if self.searchModel != "filter" and self._query.strip():
            by_id = {item.object_id: item for item in self._all if item.enabled and item.available}
            return [self._view(by_id[value]) for value in self._model_ids if value in by_id]
        return [self._view(item) for item in filter_catalog(self._all, self._query)]

    @Property("QVariantList", notify=stateChanged)
    def searchModels(self):
        return [{"id": key, "name": name} for key, name in MODELS]

    @Property(str, notify=stateChanged)
    def searchModel(self):
        value = self.catalog.setting("search_model", DEFAULT_MODEL)
        if value == "e5":
            return "e5_trained"
        return value if value in dict(MODELS) else DEFAULT_MODEL

    @Property(bool, notify=stateChanged)
    def modelSearching(self):
        return self._model_searching

    @Property(bool, notify=stateChanged)
    def modelReady(self):
        return self._model_ready

    @Property(bool, notify=stateChanged)
    def modelPreparing(self):
        return self._model_preparing

    @Property(str, notify=stateChanged)
    def modelStatus(self):
        if self._model_error:
            return "Модель недоступна: " + self._model_error
        if self._model_preparing:
            return "Подготавливаю модель и каталог…"
        return "Модель готова" if self._model_ready else "Модель не подготовлена"

    def modelCompleted(self, error):
        self._model_error = error
        self._model_ready = not bool(error)
        self.stateChanged.emit()

    def _schedule_preparation(self):
        self._prepare_generation += 1
        self._model_ready = self.searchModel == "filter"
        self._model_preparing = not self._model_ready
        self._model_error = ""
        self._prepare_timer.start(0)

    def _prepare_model(self):
        if self._closing or self._prepare_active or not self._model_preparing:
            return
        token, model = self._prepare_generation, self.searchModel
        runtime_generation = self.model_runtime.generation
        objects = [item.for_model() for item in self._all if item.enabled and item.available]
        self._prepare_active = True

        def work():
            if self._closing or token != self._prepare_generation:
                return None
            return self.model_runtime.prepare(model, objects, expected_generation=runtime_generation)

        def done(result, error):
            self._prepare_active = False
            if self._closing:
                return
            if token != self._prepare_generation:
                self._prepare_model()
                return
            self._model_preparing = False
            self._model_ready = not bool(error)
            self._model_error = error
            self.stateChanged.emit()

        self.jobs.submit(work, done)

    @Slot(str)
    def setSearchModel(self, model):
        if model not in dict(MODELS):
            self._message("Неизвестная модель.", True)
            return
        if model == self.searchModel:
            return
        self.catalog.set_setting("search_model", model)
        self.model_runtime.close()
        self._schedule_preparation()
        self._schedule_model_search()
        self.quick.invalidate(model_changed=True)
        self._message("Выбрана модель: " + dict(MODELS)[model])

    def _schedule_model_search(self):
        self._model_generation += 1
        self._model_ids = []
        self._model_timer.stop()
        self._model_searching = self.searchModel != "filter" and bool(self._query.strip())
        if self._model_searching:
            self._model_timer.start()

    def _run_model_search(self):
        if self._closing or self._model_job_active or not self._model_searching:
            return
        token, query, model = self._model_generation, self._query, self.searchModel
        runtime_generation = self.model_runtime.generation
        objects = [item.for_model() for item in self._all if item.enabled and item.available]
        if not objects:
            self._model_searching = False
            self.stateChanged.emit()
            return
        self._model_job_active = True
        self._message("Ищем подходящий объект…")

        def done(result, error):
            self._model_job_active = False
            if self._closing:
                return
            if token != self._model_generation:
                if not self._model_timer.isActive():
                    self._run_model_search()
                return
            self._model_searching = False
            if error:
                self.modelCompleted(error)
                self._message("Не удалось выполнить поиск: " + error, True)
            else:
                self.modelCompleted("")
                self._model_ids = result["object_ids"]
                self._message(
                    f"{dict(MODELS)[model]}: {len(self._model_ids)} вариантов, "
                    f"{result['elapsed_ms']:.0f} мс. Выберите объект и нажмите Enter."
                )

        def work():
            if self._closing or token != self._model_generation:
                return None
            return self.model_runtime.search(model, query, objects,
                                             expected_generation=runtime_generation)

        self.jobs.submit(work, done)

    def closeModels(self):
        self._closing = True
        self.quick.close()
        self._prepare_generation += 1
        self._prepare_timer.stop()
        self._model_generation += 1
        self._model_timer.stop()
        self._model_searching = False
        self.model_runtime.close()

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
    def voiceHotkeyText(self):
        return self.catalog.setting("voice_hotkey", DEFAULT_VOICE_HOTKEY)

    @Property(bool, notify=stateChanged)
    def voiceHotkeyRegistered(self):
        return bool(self.voice_hotkey and self.voice_hotkey.registered)

    @Property(str, notify=stateChanged)
    def voiceHotkeyStatus(self):
        return self.voice_hotkey.status_text if self.voice_hotkey else "Горячая клавиша отключена"

    @Property(str, notify=stateChanged)
    def theme(self):
        return self.catalog.setting("theme", "dark")

    @Property(bool, constant=True)
    def animationsEnabled(self):
        if os.name == "nt":
            enabled = ctypes.c_int(1)
            if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0):
                return bool(enabled.value)
        return True

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
        self._message("Собираем приложения и папки из Windows…")

        def work():
            roots = self.catalog.setting("folder_roots", [])
            result = scan_windows(roots) if roots else scan_windows()
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
    @Slot(str, str, str, str, bool, str, result=bool)
    def saveSettings(self, hotkey, theme, microphone, new_key, clear_key=False, voice_hotkey=None):
        voice_hotkey = voice_hotkey if voice_hotkey is not None else self.voiceHotkeyText
        try:
            if theme not in ("dark", "light"):
                raise ValueError("Неизвестная тема")
            if parse_hotkey(hotkey) == parse_hotkey(voice_hotkey):
                raise ValueError("Для текста и голоса нужны разные сочетания клавиш.")
            if self.hotkey and self.voice_hotkey:
                register_pair(self.hotkey, self.voice_hotkey, hotkey, voice_hotkey)
            elif self.hotkey:
                self.hotkey.register(hotkey)
            if clear_key:
                config.save_api_key(self.catalog, "")
            elif new_key.strip():
                config.save_api_key(self.catalog, new_key)
            self.catalog.set_setting("hotkey", hotkey)
            self.catalog.set_setting("voice_hotkey", voice_hotkey)
            self.catalog.set_setting("theme", theme)
            self.catalog.set_setting("microphone", microphone)
            self._message("Настройки сохранены.")
            return True
        except (ValueError, RuntimeError, OSError) as exc:
            self._message(str(exc), True)
            return False

    @Slot()
    def toggleRecording(self):
        self.quick.cancel()
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

    @Slot(str)
    def showQuick(self, mode):
        self.quick.show(mode)

    @Slot()
    def submitQuick(self):
        self.quick.submit()

    @Slot()
    def cancelQuick(self):
        self.quick.cancel()
