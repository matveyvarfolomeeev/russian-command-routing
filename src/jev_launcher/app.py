"""Application assembly: Qt window, tray, hotkey and background controller."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QLockFile, QTimer, QUrl
from PySide6.QtGui import QAction, QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from .catalog import Catalog, default_data_dir
from .controller import Controller
from .hotkey import GlobalHotkey
from .icons import CatalogIcons

PACKAGE_DIR = Path(__file__).parent


def create_window(application, catalog):
    if QQuickStyle.name() != "Basic":
        QQuickStyle.setStyle("Basic")
    controller = Controller(catalog)
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("controller", controller)
    icons = CatalogIcons()
    engine.addImageProvider("catalog", icons)
    controller.catalogChanged.connect(
        lambda: icons.prepare(controller._all, controller.jobs, controller.refreshIcons)
    )
    icons.prepare(controller._all, controller.jobs, controller.refreshIcons)
    engine.load(QUrl.fromLocalFile(str(PACKAGE_DIR / "ui" / "Main.qml")))
    if not engine.rootObjects():
        raise RuntimeError("Не удалось загрузить интерфейс QML")
    window = engine.rootObjects()[0]
    return engine, controller, window


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Jev Launcher for Windows")
    parser.add_argument("--data-dir", type=Path, help="Isolated catalog/settings directory")
    parser.add_argument("--no-scan", action="store_true", help="Skip first-run automatic discovery")
    parser.add_argument("--no-hotkey", action="store_true")
    parser.add_argument("--no-tray", action="store_true")
    parser.add_argument("--hidden", action="store_true")
    args = parser.parse_args(argv)
    data_dir = args.data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        data_dir / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    application = QApplication(sys.argv[:1])
    application.setApplicationName("Jev Launcher")
    application.setOrganizationName("JevLauncher")
    application.setWindowIcon(QIcon(str(PACKAGE_DIR / "resources" / "launcher.svg")))
    lock = QLockFile(str(data_dir / "app.lock"))
    server_name = (
        "jev-launcher-" + hashlib.sha256(str(data_dir.resolve()).encode()).hexdigest()[:20]
    )
    if not lock.tryLock(0):
        client = QLocalSocket()
        client.connectToServer(server_name)
        if client.waitForConnected(1000):
            client.write(b"show")
            client.waitForBytesWritten(1000)
            client.disconnectFromServer()
            return 0
        QMessageBox.information(
            None, "Jev Launcher", "Приложение уже запускается. Попробуйте ещё раз."
        )
        return 1
    try:
        catalog = Catalog(data_dir / "catalog.sqlite3")
        engine, controller, window = create_window(application, catalog)
    except Exception as exc:
        logging.exception("Startup failed")
        QMessageBox.critical(None, "Jev Launcher", str(exc))
        lock.unlock()
        return 1

    server = QLocalServer(application)
    server.listen(server_name)

    def show_window():
        window.show()
        window.raise_()
        window.requestActivate()
        window.focusSearch()

    def new_connection():
        while server.hasPendingConnections():
            connection = server.nextPendingConnection()
            # A second instance only requests activation, never passes executable commands.
            show_window()
            connection.disconnectFromServer()
            connection.deleteLater()

    server.newConnection.connect(new_connection)
    controller.showRequested.connect(show_window)
    tray = None
    tray_available = not args.no_tray and QSystemTrayIcon.isSystemTrayAvailable()
    application.setQuitOnLastWindowClosed(False)

    def hide_window():
        if controller.recording:
            controller.cancelSpeech()
        if tray_available or controller.hotkey and controller.hotkey.current:
            window.hide()
        else:
            application.quit()

    controller.hideRequested.connect(hide_window)
    if tray_available:
        tray = QSystemTrayIcon(application.windowIcon(), application)
        tray.setToolTip("Jev Launcher")
        menu = QMenu()
        show_action = QAction("Открыть", menu)
        show_action.triggered.connect(show_window)
        quit_action = QAction("Выйти", menu)
        quit_action.triggered.connect(application.quit)
        menu.addAction(show_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        tray.setContextMenu(menu)
        tray.activated.connect(
            lambda reason: (
                show_window()
                if reason
                in (
                    QSystemTrayIcon.ActivationReason.Trigger,
                    QSystemTrayIcon.ActivationReason.DoubleClick,
                )
                else None
            )
        )
        tray.show()

    if not args.no_hotkey:
        controller.hotkey = GlobalHotkey(
            application, lambda: hide_window() if window.isVisible() else show_window()
        )
        try:
            controller.hotkey.register(controller.hotkeyText)
        except (ValueError, RuntimeError) as exc:
            controller._message(str(exc), True)
    if args.hidden and (tray_available or controller.hotkey and controller.hotkey.current):
        window.hide()
    if not args.no_scan and not catalog.items():
        QTimer.singleShot(100, controller.refresh)

    def cleanup():
        controller.cancelSpeech()
        engine.imageProvider("catalog").generation += 1
        if controller.hotkey:
            controller.hotkey.close()
        if tray:
            tray.hide()
        server.close()
        controller.jobs.pool.waitForDone(50000)
        lock.unlock()

    application.aboutToQuit.connect(cleanup)
    # Keep the engine, tray menu, lock and server alive for the entire Qt event loop.
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
