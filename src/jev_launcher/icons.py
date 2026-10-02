"""Native icons prepared by Qt jobs; the QML provider only reads the ready cache."""

import ctypes
import os
from ctypes import wintypes

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtQuick import QQuickImageProvider


def native_icon(path: str) -> QImage:
    if os.name != "nt" or not path:
        return QImage()

    class FileInfo(ctypes.Structure):
        _fields_ = [
            ("icon", wintypes.HANDLE),
            ("index", ctypes.c_int),
            ("attributes", wintypes.DWORD),
            ("display_name", wintypes.WCHAR * 260),
            ("type_name", wintypes.WCHAR * 80),
        ]

    ole = ctypes.WinDLL("ole32")
    initialized = ole.CoInitializeEx(None, 2) in (0, 1)
    info = FileInfo()
    shell = ctypes.WinDLL("shell32")
    shell.SHGetFileInfoW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(FileInfo),
        wintypes.UINT,
        wintypes.UINT,
    ]
    shell.SHGetFileInfoW.restype = ctypes.c_void_p
    try:
        result = shell.SHGetFileInfoW(
            os.path.expandvars(path), 0, ctypes.byref(info), ctypes.sizeof(info), 0x100 | 0x1
        )
        if not result or not info.icon:
            return QImage()
        image = QImage.fromHICON(int(info.icon)).copy()
        user = ctypes.WinDLL("user32")
        user.DestroyIcon.argtypes = [wintypes.HANDLE]
        user.DestroyIcon(info.icon)
        return image
    finally:
        if initialized:
            ole.CoUninitialize()


class CatalogIcons(QQuickImageProvider):
    def __init__(self):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self.cache = {}
        self.targets = {}
        self.kinds = {}
        self.generation = 0
        self.placeholders = {kind: self._fallback(kind) for kind in ("app", "folder", "file")}

    def prepare(self, items, jobs, ready):
        self.generation += 1
        token = self.generation
        self.kinds = {item.object_id: item.kind for item in items}
        self.cache = {key: image for key, image in self.cache.items() if key in self.kinds}
        targets = {
            item.object_id: item.icon_target or item.launch_target
            for item in items
            if item.launch_type != "app_id"
        }
        pending = {
            object_id: target
            for object_id, target in targets.items()
            if object_id not in self.cache or self.targets.get(object_id) != target
        }
        if not pending:
            return

        def work():
            images = {}
            for object_id, target in pending.items():
                if token != self.generation:
                    return {}
                if target.startswith(("\\\\", "shell:")):
                    continue
                image = native_icon(target)
                if not image.isNull():
                    images[object_id] = image
            return images

        def done(images, error):
            if token != self.generation or error:
                return
            self.cache.update(images)
            self.targets = targets
            ready()

        jobs.submit(work, done)

    def requestImage(self, identifier, size, requested_size):
        object_id = identifier.split("?", 1)[0]
        image = self.cache.get(object_id, self.placeholders[self.kinds.get(object_id, "app")])
        size.setWidth(image.width())
        size.setHeight(image.height())
        return image

    @staticmethod
    def _fallback(kind):
        image = QImage(QSize(48, 48), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#283852"))
        painter.drawRoundedRect(2, 2, 44, 44, 12, 12)
        painter.setPen(QColor("#aecaff"))
        painter.setFont(QFont("Segoe UI", 20))
        painter.drawText(
            QRect(0, 0, 48, 48),
            Qt.AlignmentFlag.AlignCenter,
            {"folder": "▰", "file": "▤"}.get(kind, "◈"),
        )
        painter.end()
        return image
