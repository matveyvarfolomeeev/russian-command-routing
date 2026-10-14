"""A compact launch session. Queries and cancellation are independent of the Hub."""

from __future__ import annotations

import logging

from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot

from . import config
from .audio import Recorder
from .launch import launch_by_id
from .search import filter_catalog
from .speech import transcribe


class QuickSession(QObject):
    changed = Signal()
    showRequested = Signal()
    focusRequested = Signal()

    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self.recorder = Recorder(self)
        self.recorder.failed.connect(self._fail)
        self.recorder.changed.connect(self.changed)
        self.recorder.limitReached.connect(self.stopRecording)
        self._visible = False
        self._mode = "text"
        self._stage = "idle"
        self._query = ""
        self._error = ""
        self._result = {}
        self._submitted = False
        self._session = 0
        self._request = 0
        self._resolved = None
        self._search_active = False
        self._closing = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(controller._model_timer.interval())
        self.timer.timeout.connect(self._search)

    @Property(bool, notify=changed)
    def visible(self):
        return self._visible

    @Property(str, notify=changed)
    def mode(self):
        return self._mode

    @Property(str, notify=changed)
    def stage(self):
        return self._stage

    @Property(str, notify=changed)
    def query(self):
        return self._query

    @Property(str, notify=changed)
    def error(self):
        return self._error

    @Property(bool, notify=changed)
    def submitted(self):
        return self._submitted

    @Property(bool, notify=changed)
    def recording(self):
        return self.recorder.recording

    @Property("QVariantMap", notify=changed)
    def bestResult(self):
        return self._result

    @Property(int, notify=changed)
    def requestId(self):
        return self._request

    def _token(self):
        return self._session, self._request, self.controller.searchModel

    def _valid(self, token):
        return self._visible and not self._closing and token == self._token()

    @Slot(str)
    def show(self, mode):
        if mode not in ("text", "voice") or self._closing:
            return
        if self._visible and self._mode == mode:
            self.cancel()
            return
        self.cancel()
        if self.controller.recording or self.controller.transcribing:
            self.controller.cancelSpeech()
        self._mode = mode
        self._query = ""
        self._error = ""
        self._result = {}
        self._visible = True
        self.changed.emit()
        self.showRequested.emit()
        if mode == "voice":
            self.startRecording()

    @Slot()
    def cancel(self):
        self._session += 1
        self._request += 1
        self.timer.stop()
        self._visible = False
        self._submitted = False
        self._resolved = None
        self._stage = "idle"
        self.recorder.cancel()
        self.changed.emit()

    @Slot()
    def focusLost(self):
        if self._visible and not self._submitted:
            logging.info("Quick session cancelled on focus loss: mode=%s", self._mode)
            self.cancel()

    @Slot(str)
    def setQuery(self, query):
        if not self._visible or self._submitted or query == self._query:
            return
        if self._stage == "transcribing" or self.recording:
            self.recorder.cancel()
        self._query = query
        self._error = ""
        self._schedule_search()

    def invalidate(self, model_changed=False):
        if not self._visible:
            return
        if model_changed:
            self._request += 1
            self._resolved = None
            self._result = {}
            self._fail("Модель изменена. Нажмите Enter, чтобы отправить запрос заново.")
        elif self._submitted:
            self._request += 1
            self._result = {}
            self._fail("Каталог изменён. Проверьте запрос и нажмите Enter заново.")
        elif self.recording or self._stage == "transcribing":
            return
        else:
            self._schedule_search()

    def _schedule_search(self):
        self._request += 1
        self._resolved = None
        self._result = {}
        self.timer.stop()
        self._stage = "searching" if self._query.strip() else "idle"
        self.changed.emit()
        if self._query.strip():
            if self._submitted or self.controller.searchModel == "filter":
                self._search()
            else:
                self.timer.start()

    def _search(self):
        if not self._visible or not self._query.strip() or self._closing:
            return
        if self.recording or self._stage == "transcribing":
            return
        token, query = self._token(), self._query
        runtime_generation = self.controller.model_runtime.generation
        objects = [item for item in self.controller._all if item.enabled and item.available]
        if not objects:
            self._resolved = token
            self._fail("Каталог пуст. Добавьте объекты в большом окне.")
            return
        if self.controller.searchModel == "filter":
            self._searched(token, filter_catalog(objects, query), "")
            return
        if self._search_active:
            return
        self._search_active = True
        safe = [item.for_model() for item in objects]

        def work():
            if not self._valid(token):
                return None
            return self.controller.model_runtime.search(token[2], query, safe,
                                                       expected_generation=runtime_generation)

        def done(result, error):
            self._search_active = False
            if not self._valid(token):
                if not self.timer.isActive():
                    self._search()
                return
            by_id = {item.object_id: item for item in objects}
            ranked = [by_id[value] for value in result["object_ids"]] if not error else []
            self._searched(token, ranked, error)

        self.controller.jobs.submit(work, done)

    def _searched(self, token, items, error):
        if not self._valid(token):
            return
        self._resolved = token
        self._stage = "idle"
        self._result = self.controller._view(items[0]) if items else {}
        if error:
            self.controller.modelCompleted(error)
            self._fail("Не удалось выполнить поиск: " + error)
            return
        if token[2] != "filter":
            self.controller.modelCompleted("")
        if self._submitted:
            if not self._result:
                self._fail("Подходящий объект не найден. Измените запрос.")
            else:
                self._queue_open(token)
        else:
            self.changed.emit()

    @Slot()
    def submit(self):
        if not self._visible or self._submitted:
            return
        if self.recording:
            self._submitted = True
            self.stopRecording()
            return
        if self._stage == "transcribing":
            self._submitted = True
            self.changed.emit()
            return
        if not self._query.strip():
            return
        self._submitted = True
        self.timer.stop()
        self._stage = "searching"
        self.changed.emit()
        if self._resolved == self._token() and self._result:
            self._queue_open(self._token())
        else:
            self._error = ""
            self._search()

    def _queue_open(self, token):
        self._stage = "opening"
        object_id = self._result["object_id"]
        self.changed.emit()

        def send():
            if not self._valid(token) or not self._submitted:
                return

            def work():
                # Cancellation before sending the Windows command prevents the side effect.
                if self._valid(token) and self._submitted:
                    return launch_by_id(self.controller.catalog, object_id)
                return None

            def done(item, error):
                if not self._valid(token):
                    return
                if error:
                    self._fail("Не удалось открыть объект: " + error)
                elif item is not None:
                    self.cancel()

            self.controller.jobs.submit(work, done)

        # Give Esc/cancel a chance to invalidate a queued opening; animations never gate work.
        QTimer.singleShot(0, send)

    @Slot()
    def startRecording(self):
        if not self._visible or self._submitted or self.recording:
            return
        if not self.controller.keyConfigured:
            self._fail("Настройте ключ Groq в большом окне на вкладке «Настройки».")
            return
        self._request += 1
        self.timer.stop()
        self._resolved = None
        self._result = {}
        self._error = ""
        self._stage = "recording"
        try:
            self.recorder.start(self.controller.microphoneId)
        except (ValueError, RuntimeError) as exc:
            self._fail(str(exc))
        self.changed.emit()

    @Slot()
    def stopRecording(self):
        if not self._visible or not self.recording:
            return
        try:
            audio = self.recorder.stop()
        except (ValueError, RuntimeError) as exc:
            self._fail(str(exc))
            return
        token = self._token()
        key = config.api_key(self.controller.catalog)
        self._stage = "transcribing"
        self.changed.emit()

        def done(text, error):
            if not self._valid(token):
                return
            if error:
                self._fail("Не удалось распознать речь: " + error)
            elif not text or not text.strip():
                self._fail("Речь не распознана. Повторите запись или введите запрос.")
            else:
                self._query = text
                self._schedule_search()

        self.controller.jobs.submit(lambda: transcribe(audio.wav(), key), done)

    def _fail(self, error):
        if not self._visible:
            return
        self._request += 1
        self.timer.stop()
        self._submitted = False
        self._stage = "idle"
        self._error = error
        self.recorder.cancel()
        self.changed.emit()
        self.focusRequested.emit()

    def close(self):
        self._closing = True
        self.cancel()
