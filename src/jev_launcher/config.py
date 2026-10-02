"""Settings helpers. API keys use Windows DPAPI for the current user."""

from __future__ import annotations

import base64
import ctypes
import os
from ctypes import wintypes

from .catalog import Catalog


class _Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _crypt(data: bytes, *, decrypt: bool = False) -> bytes:
    if os.name != "nt":
        raise RuntimeError("Хранение API-ключа поддерживается только в Windows")
    buffer = ctypes.create_string_buffer(data)
    source = _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = _Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [
        ctypes.POINTER(_Blob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_Blob),
    ]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree(ctypes.cast(output.data, ctypes.c_void_p))


def api_key(catalog: Catalog) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    protected = catalog.setting("groq_key", "")
    if not protected:
        return ""
    try:
        return _crypt(base64.b64decode(protected), decrypt=True).decode("utf-8")
    except (OSError, ValueError, UnicodeError):
        return ""


def save_api_key(catalog: Catalog, value: str) -> None:
    protected = base64.b64encode(_crypt(value.strip().encode())).decode() if value.strip() else ""
    catalog.set_setting("groq_key", protected)
