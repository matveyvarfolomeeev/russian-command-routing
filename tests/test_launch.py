from dataclasses import replace

import pytest

from jev_launcher import launch
from jev_launcher.catalog import Catalog
from jev_launcher.objects import CatalogObject


def item(path, launch_type="file", **kwargs):
    return CatalogObject(
        object_id="known-id",
        name="Тест",
        kind="file",
        source="manual",
        identity="test",
        launch_type=launch_type,
        launch_target=str(path),
        **kwargs,
    )


@pytest.mark.parametrize("launch_type", ["shortcut", "folder", "file"])
def test_shell_opens_exact_stored_path(tmp_path, monkeypatch, launch_type):
    target = tmp_path / "Объект с пробелами"
    target.mkdir()
    calls = []
    monkeypatch.setattr(launch.os, "startfile", lambda path: calls.append(path))
    launch.open_object(item(target, launch_type))
    assert calls == [str(target)]


def test_executable_has_separate_arguments_and_no_shell(tmp_path, monkeypatch):
    target = tmp_path / "test.exe"
    target.touch()
    calls = []
    monkeypatch.setattr(
        launch.subprocess, "Popen", lambda args, **kwargs: calls.append((args, kwargs))
    )
    launch.open_object(
        item(
            target,
            "executable",
            arguments=("--name", "x & calc.exe"),
            working_directory=str(tmp_path),
        )
    )
    assert calls == [
        ([str(target), "--name", "x & calc.exe"], {"cwd": str(tmp_path), "shell": False})
    ]


def test_app_id_shell_namespace(monkeypatch):
    calls = []
    monkeypatch.setattr(launch.os, "startfile", lambda path: calls.append(path))
    launch.open_object(item("Microsoft.WindowsCalculator_example!App", "app_id"))
    assert calls == [r"shell:AppsFolder\Microsoft.WindowsCalculator_example!App"]


def test_missing_disabled_and_unknown_id_cannot_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(launch.os, "startfile", lambda path: pytest.fail("Unexpected launch"))
    with pytest.raises(launch.LaunchError):
        launch.open_object(item(tmp_path / "missing"))
    target = tmp_path / "exists"
    target.touch()
    with pytest.raises(launch.LaunchError):
        launch.open_object(replace(item(target), enabled=False))
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    with pytest.raises(launch.LaunchError):
        launch.launch_by_id(catalog, "calc.exe & anything")


def test_os_error_becomes_user_error(tmp_path, monkeypatch):
    target = tmp_path / "file"
    target.touch()

    def fail(path):
        raise PermissionError("Internal system detail")

    monkeypatch.setattr(launch.os, "startfile", fail)
    with pytest.raises(launch.LaunchError, match="Windows"):
        launch.open_object(item(target))
