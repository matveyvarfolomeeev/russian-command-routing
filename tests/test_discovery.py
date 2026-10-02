import subprocess

import pytest

from jev_launcher import discovery
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
