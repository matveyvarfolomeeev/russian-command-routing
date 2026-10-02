"""A bounded Qt worker pool with results delivered on the GUI thread."""

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class _Signals(QObject):
    done = Signal(int, object, str)


class _Task(QRunnable):
    def __init__(self, token, work):
        super().__init__()
        self.token, self.work = token, work
        self.signals = _Signals()

    @Slot()
    def run(self):
        try:
            result = self.work()
        except Exception as exc:
            self.signals.done.emit(self.token, None, str(exc))
        else:
            self.signals.done.emit(self.token, result, "")


class Jobs(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(3)
        self.pending = {}
        self.counter = 0

    def submit(self, work, callback):
        self.counter += 1
        token = self.counter
        task = _Task(token, work)
        self.pending[token] = (task, callback)
        task.signals.done.connect(self._done)
        self.pool.start(task)
        return token

    @Slot(int, object, str)
    def _done(self, token, result, error):
        pending = self.pending.pop(token, None)
        if pending:
            pending[1](result, error)
