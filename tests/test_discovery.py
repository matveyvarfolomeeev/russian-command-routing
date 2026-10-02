import subprocess

import pytest

from jev_launcher import discovery
from jev_launcher.catalog import Catalog
from jev_launcher.hotkey import parse_hotkey


def test_windows_argument_split_preserves_quotes_and_metacharacters():
    assert discovery.split_windows_arguments('--profile "Имя с пробелами" --value "x & y"') == (
        "--profile",
        "Имя с пробелами",
        "--value",
        "x & y",
    )


def test_fixed_helper_hidden_and_bounded(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args, 0, '{"entries": [], "successful_sources": ["start_apps"], "errors": {}}', ""
        )

    monkeypatch.setattr(discovery.subprocess, "run", run)
    result = discovery.scan_windows()
    assert result.successful_sources == ["start_apps"]
    args, kwargs = calls[0]
    assert "-EncodedCommand" in args
    assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
    assert kwargs["timeout"] == 45
    assert "shell" not in kwargs


def test_desktop_startapp_path_is_an_executable(tmp_path):
    target = tmp_path / "Application.exe"
    target.touch()
    result = discovery.parse_scan(
        {
            "entries": [
                {
                    "name": "Application",
                    "launch_type": "app_id",
                    "launch_target": str(target),
                    "source": "start_apps",
                    "source_key": "start_apps:test",
                }
            ],
            "successful_sources": ["start_apps"],
        }
    )
    item = result.objects[0]
    assert item.launch_type == "executable"
    assert item.working_directory == str(tmp_path)
    assert item.icon_target == str(target)


def test_telegram_shortcuts_and_startapps_share_exact_command(tmp_path):
    target = tmp_path / "Telegram.exe"
    target.touch()
    entries = []
    for index, app_id in enumerate((str(target), "Telegram.TelegramDesktop")):
        shortcut = tmp_path / f"Telegram{index}.lnk"
        shortcut.touch()
        entries.append(
            {
                "name": "Telegram",
                "launch_type": "shortcut",
                "launch_target": str(shortcut),
                "source": "start_menu_user",
                "source_key": f"shortcut:{index}",
                "resolved_target": str(target),
                "app_id": app_id,
                "working_directory": str(tmp_path),
            }
        )
        entries.append(
            {
                "name": "Telegram",
                "launch_type": "app_id",
                "launch_target": app_id,
                "source": "start_apps",
                "source_key": f"app:{index}",
            }
        )
    entries.append(
        {
            "name": "Деинсталляция Telegram",
            "launch_type": "shortcut",
            "launch_target": str(tmp_path / "uninstall.lnk"),
            "resolved_target": str(tmp_path / "unins000.exe"),
            "source": "start_menu_user",
            "source_key": "shortcut:uninstall",
        }
    )
    result = discovery.parse_scan(
        {"entries": entries, "successful_sources": ["start_menu_user", "start_apps"]}
    )
    assert len(result.objects) == 4
    assert len({item.identity for item in result.objects}) == 1
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.merge_scan(result)
    assert len(catalog.items(active_only=True)) == 1
    assert "телеграмм" in catalog.items()[0].aliases


def test_shortcut_profiles_are_not_merged_by_appid(tmp_path):
    target = tmp_path / "Telegram.exe"
    target.touch()
    entries = [
        {
            "name": "Telegram",
            "launch_type": "shortcut",
            "launch_target": str(tmp_path / f"profile{profile}.lnk"),
            "source": "start_menu_user",
            "source_key": str(profile),
            "resolved_target": str(target),
            "app_id": "Telegram.TelegramDesktop",
            "raw_arguments": f"--workdir profile{profile}",
        }
        for profile in (1, 2)
    ]
    entries.append(
        {
            "name": "Telegram",
            "launch_type": "app_id",
            "launch_target": "Telegram.TelegramDesktop",
            "source": "start_apps",
            "source_key": "app:telegram",
        }
    )
    result = discovery.parse_scan({"entries": entries})
    assert len({item.identity for item in result.objects}) == 3


def test_opaque_startapp_executable_uses_unique_shortcut_command(tmp_path):
    target = tmp_path / "7zFM.exe"
    target.touch()
    shortcut = tmp_path / "7-Zip.lnk"
    shortcut.touch()
    result = discovery.parse_scan(
        {
            "entries": [
                {
                    "name": "7-Zip File Manager",
                    "launch_type": "shortcut",
                    "launch_target": str(shortcut),
                    "resolved_target": str(target),
                    "working_directory": "",
                    "source": "start_menu_common",
                    "source_key": "shortcut:7zip",
                },
                {
                    "name": "7-Zip File Manager",
                    "launch_type": "app_id",
                    "launch_target": str(target),
                    "source": "start_apps",
                    "source_key": "start_apps:7zip",
                },
            ],
            "successful_sources": ["start_menu_common", "start_apps"],
        }
    )
    assert len({item.identity for item in result.objects}) == 1
    assert result.objects[1].working_directory == ""
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.merge_scan(result)
    assert len(catalog.items()) == 1
    assert catalog.items()[0].launch_type == "shortcut"


def test_opaque_startapp_does_not_collapse_explicit_cwd_variants(tmp_path):
    target = tmp_path / "App.exe"
    target.touch()
    entries = [
        {
            "name": "App",
            "launch_type": "shortcut",
            "launch_target": str(tmp_path / f"{cwd}.lnk"),
            "resolved_target": str(target),
            "working_directory": str(tmp_path / cwd),
            "source": "start_menu_common",
            "source_key": f"shortcut:{cwd}",
        }
        for cwd in ("one", "two")
    ]
    entries.append(
        {
            "name": "App",
            "launch_type": "app_id",
            "launch_target": str(target),
            "source": "start_apps",
            "source_key": "app:direct",
        }
    )
    result = discovery.parse_scan({"entries": entries})
    assert len({item.identity for item in result.objects}) == 3


@pytest.mark.parametrize(
    "name,target,arguments",
    [
        ("Uninstall App", "app.exe", ""),
        ("Деинсталляция Telegram", "app.exe", ""),
        ("App", "unins000.exe", ""),
        ("App", "app.exe", "--uninstall"),
        ("App", "setup.exe", "/repair"),
        ("App", "msiexec.exe", "/x {package}"),
    ],
)
def test_installer_actions_are_not_candidates(name, target, arguments):
    result = discovery.parse_scan(
        {
            "entries": [
                {
                    "name": name,
                    "launch_type": "shortcut",
                    "launch_target": "app.lnk",
                    "resolved_target": target,
                    "raw_arguments": arguments,
                    "source": "start_menu_user",
                    "source_key": "test",
                }
            ]
        }
    )
    assert result.objects == []


@pytest.mark.parametrize("name", ["Everything", "Remove.bg", "Total Uninstall"])
def test_ordinary_application_names_are_retained(name):
    result = discovery.parse_scan(
        {
            "entries": [
                {
                    "name": name,
                    "launch_type": "shortcut",
                    "launch_target": "app.lnk",
                    "resolved_target": "app.exe",
                    "source": "start_menu_user",
                    "source_key": "test",
                }
            ]
        }
    )
    assert len(result.objects) == 1


def test_timeout_preserves_catalog_by_not_returning_partial_result(monkeypatch):
    def run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(discovery.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="времени"):
        discovery.scan_windows()


@pytest.mark.parametrize("text", ["Space", "Ctrl+?", "Ctrl++Space", "Ctrl+F13", "Meta+A"])
def test_invalid_hotkeys(text):
    with pytest.raises(ValueError):
        parse_hotkey(text)


def test_hotkey_modifiers_and_function_key():
    assert parse_hotkey("Ctrl+Alt+Space") == (0x4003, 0x20)
    assert parse_hotkey("Ctrl+Shift+F12") == (0x4006, 0x7B)
