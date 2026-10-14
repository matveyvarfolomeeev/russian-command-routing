"""Application assembly: Qt window, tray, hotkey and background controller."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QLockFile, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QCursor, QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from .catalog import Catalog, default_data_dir
from .controller import Controller
from .hotkey import VOICE_HOTKEY_ID, GlobalHotkey, register_pair
from .icons import CatalogIcons

PACKAGE_DIR = Path(__file__).parent


def create_window(application, catalog):
    if QQuickStyle.name() != "Basic":
        QQuickStyle.setStyle("Basic")
    controller = Controller(catalog)
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("controller", controller)
    engine.rootContext().setContextProperty("quick", controller.quick)
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
    engine.load(QUrl.fromLocalFile(str(PACKAGE_DIR / "ui" / "Quick.qml")))
    if len(engine.rootObjects()) != 2:
        raise RuntimeError("Не удалось загрузить компактную панель QML")
    controller.quick_window = engine.rootObjects()[1]

    def show_quick():
        quick_window = controller.quick_window
        screen = application.screenAt(QCursor.pos()) or application.primaryScreen()
        area = screen.availableGeometry()
        quick_window.setScreen(screen)
        quick_window.setProperty("availableWidth", area.width())
        quick_window.setProperty("availableHeight", area.height())
        quick_window.setProperty("centerX", area.x() + area.width() / 2)
        quick_window.setProperty("centerY", area.y() + area.height() / 2)
        quick_window.show()
        quick_window.raise_()
        quick_window.requestActivate()
        quick_window.focusInput()

    def focus_quick():
        if controller.quick.visible:
            controller.quick_window.raise_()
            controller.quick_window.requestActivate()
            controller.quick_window.focusInput()

    controller.quick.showRequested.connect(show_quick)
    controller.quick.focusRequested.connect(focus_quick)
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
        # A minimized Qt window remains minimized after show(). Restore it first.
        if window.windowState() & Qt.WindowState.WindowMinimized:
            window.showNormal()
        else:
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
        if (tray_available or controller.hotkeyRegistered or controller.voiceHotkeyRegistered):
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
            application,
            lambda: controller.showQuick("text"),
            controller.stateChanged.emit,
        )
        controller.voice_hotkey = GlobalHotkey(
            application,
            lambda: controller.showQuick("voice"),
            controller.stateChanged.emit,
            hotkey_id=VOICE_HOTKEY_ID,
        )
        try:
            register_pair(controller.hotkey, controller.voice_hotkey,
                          controller.hotkeyText, controller.voiceHotkeyText)
        except (ValueError, RuntimeError) as exc:
            controller._message(str(exc), True)
            startup_hotkey_error = str(exc)

            def retry_hotkey():
                # A previous instance can briefly retain its registration while exiting.
                # Retry once, keeping the user's configured shortcut unchanged.
                if controller.hotkey.registered and controller.voice_hotkey.registered:
                    return
                try:
                    register_pair(controller.hotkey, controller.voice_hotkey,
                                  controller.hotkeyText, controller.voiceHotkeyText)
                    if controller.status == startup_hotkey_error:
                        controller._message("Горячие клавиши для текста и голоса активны.")
                except (ValueError, RuntimeError):
                    logging.warning("Global hotkey retry failed; change it in Settings")

            QTimer.singleShot(1500, retry_hotkey)
    if args.hidden and (tray_available or controller.hotkeyRegistered or controller.voiceHotkeyRegistered):
        window.hide()
    refresh_timer = QTimer(application)
    refresh_timer.setInterval(3_600_000)
    if not args.no_scan:
        # Render the stored catalog immediately; refresh stale data in background.
        QTimer.singleShot(100, controller.refreshIfStale)
        refresh_timer.timeout.connect(controller.refreshIfStale)
        refresh_timer.start()

    def cleanup():
        controller.cancelSpeech()
        controller.closeModels()
        engine.imageProvider("catalog").generation += 1
        if controller.hotkey:
            controller.hotkey.close()
        if controller.voice_hotkey:
            controller.voice_hotkey.close()
        if tray:
            tray.hide()
        server.close()
        controller.jobs.pool.waitForDone(50000)
        controller.model_runtime.close(wait=True)
        lock.unlock()

    application.aboutToQuit.connect(cleanup)
    # Keep the engine, tray menu, lock and server alive for the entire Qt event loop.
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
