import json
import subprocess
from pathlib import Path

import pytest

from jev_launcher import app_metadata
from jev_launcher.catalog import Catalog
from jev_launcher.objects import DiscoveredObject, ScanResult


def app(name="Example", **values):
    return DiscoveredObject(
        name=name, kind="app", launch_type="executable", launch_target="Example.exe",
        source="start_apps", source_key="app:" + name, **values,
    )


@pytest.mark.parametrize("found,label", [("Found", "Description"), ("Найден", "Описание")])
def test_package_parser_keeps_only_exact_description(found, label):
    output = (f"{found} Example [Vendor.Example]\nVersion: 1.2\n{label}:\n"
              "  Editor for photographs.\n  Supports layers.\nLicense: MIT\n"
              "Release Notes:\n  Do not put this into the description.\n")
    result = app_metadata.parse_package_description(output, "example")
    assert result["package_id"] == "Vendor.Example"
    assert result["description"] == "Editor for photographs. Supports layers."
    assert app_metadata.parse_package_description(output, "Other") is None
    assert app_metadata.parse_package_description(output + "Found Other [Vendor.Other]\n", "Example") is None


def test_descriptions_are_bounded():
    assert len(app_metadata.short_description("words " * 300)) <= 400


def test_enrichment_preserves_fields_and_skips_browser_host_metadata(tmp_path, monkeypatch):
    normal = app()
    browser = app("Website", resolved_target="Browser.exe", arguments=("--app-id=site",))
    described = app("Edited", description="My description")
    folder = app("Folder")
    folder.kind = "folder"
    before = [vars(item).copy() for item in (normal, browser, described, folder)]
    monkeypatch.setattr(app_metadata, "executable_metadata", lambda target: {
        "FileDescription": "Photo editor", "ProductName": "Example", "CompanyName": "Vendor",
    })
    result = app_metadata.enrich_descriptions(
        [normal, browser, described, folder], cache_path=tmp_path / "cache.json", package_budget=0,
    )
    assert result["windows"] == 1
    assert normal.description == "Photo editor"
    assert normal.description_source == "windows_metadata"
    assert browser.description == ""
    for item, prior in zip((normal, browser, described, folder), before):
        assert {k: v for k, v in vars(item).items() if k not in ("description", "description_source")} == {
            k: v for k, v in prior.items() if k not in ("description", "description_source")
        }
    assert described.description == "My description"


def test_package_cache_works_without_network_budget(tmp_path, monkeypatch):
    import time

    cache = tmp_path / "cache.json"
    versions = {"Vendor.Example": "1.2"}
    import hashlib

    signature = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
    package = {"name": "Example", "description": "Code editor", "package_id": "Vendor.Example"}
    cache.write_text(json.dumps({"schema_version": 2, "packages": {
        "example": {"checked_at": time.time(), "package": package, "installed_signature": signature},
    }}))
    monkeypatch.setattr(app_metadata, "executable_metadata", lambda target: {})
    monkeypatch.setattr(app_metadata, "fetch_package_description", lambda *args: pytest.fail("network"))
    item = app()
    stats = app_metadata.enrich_descriptions([item], cache_path=cache, package_budget=0,
                                           installed_versions=versions)
    assert stats["cache_hits"] == 1
    assert item.description == "Code editor"
    assert item.description_source == "winget:Vendor.Example"


def test_duplicate_exe_description_does_not_add_information(tmp_path, monkeypatch):
    monkeypatch.setattr(app_metadata, "executable_metadata", lambda target: {
        "FileDescription": "Example", "ProductName": "Example",
    })
    item = app()
    app_metadata.enrich_descriptions([item], cache_path=tmp_path / "cache.json", package_budget=0)
    assert item.description == ""


def test_winget_is_hidden_bounded_and_exact(monkeypatch):
    monkeypatch.setattr(app_metadata.shutil, "which", lambda name: "winget.exe")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "Found Example [Vendor.Example]\nVersion: 1.2\nDescription: Editor\n", "")

    monkeypatch.setattr(app_metadata.subprocess, "run", run)
    result, cacheable = app_metadata.fetch_package_description("Example", 2, {"Vendor.Example": "1.2"})
    args, options = calls[0]
    assert "--exact" in args and "--disable-interactivity" in args
    assert "--accept-source-agreements" not in args
    assert 0 < options["timeout"] <= 2
    assert options["creationflags"] == subprocess.CREATE_NO_WINDOW
    assert "shell" not in options
    assert result["description"] == "Editor" and cacheable


def test_network_timeout_is_not_cached_as_missing(monkeypatch):
    monkeypatch.setattr(app_metadata.shutil, "which", lambda name: "winget.exe")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("winget", 2)

    monkeypatch.setattr(app_metadata.subprocess, "run", timeout)
    assert app_metadata.fetch_package_description("Example", 2, {}) == (None, False)


def test_same_name_uninstalled_package_is_rejected(monkeypatch):
    monkeypatch.setattr(app_metadata.shutil, "which", lambda name: "winget.exe")
    monkeypatch.setattr(app_metadata.subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(
        args, 0, "Found Weather [Other.Weather]\nVersion: 1.0\nDescription: A weather app\n", ""
    ))
    assert app_metadata.fetch_package_description("Weather", 2, {"Microsoft.Weather": "1.0"}) == (None, True)


def test_description_uses_installed_version(monkeypatch):
    monkeypatch.setattr(app_metadata.shutil, "which", lambda name: "winget.exe")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        version = "1.1" if "--version" in args else "2.0"
        return subprocess.CompletedProcess(
            args, 0, f"Found Example [Vendor.Example]\nVersion: {version}\nDescription: Editor {version}\n", ""
        )

    monkeypatch.setattr(app_metadata.subprocess, "run", run)
    result, cacheable = app_metadata.fetch_package_description("Example", 2, {"Vendor.Example": "1.1"})
    assert result["version"] == "1.1" and result["description"] == "Editor 1.1" and cacheable
    assert calls[1][calls[1].index("--id") + 1] == "Vendor.Example"
    assert calls[1][calls[1].index("--version") + 1] == "1.1"


def test_cache_round_trip(tmp_path):
    cache = tmp_path / "descriptions.json"
    app_metadata._save_cache(cache, {"sample": {"checked_at": 1, "package": None}})
    assert app_metadata._load_cache(cache) == {"sample": {"checked_at": 1, "package": None}}


def test_catalog_apply_preserves_concurrent_manual_edit(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.merge_scan(ScanResult(objects=[app()], successful_sources=["start_apps"]))
    original = catalog.items()[0]
    update = {original.object_id: {"description": "Generated", "description_source": "winget:V.App"}}
    catalog.update(original.object_id, description="My manual edit")
    assert catalog.apply_descriptions(update) == 0
    assert catalog.get(original.object_id).description == "My manual edit"


def test_scan_does_not_downgrade_cached_package_description(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.merge_scan(ScanResult(objects=[app()], successful_sources=["start_apps"]))
    item = catalog.items()[0]
    assert catalog.apply_descriptions({item.object_id: {
        "description": "Editor for code", "description_source": "winget:V.App",
    }}) == 1
    fallback = app(description="Generic executable", description_source="windows_metadata")
    catalog.merge_scan(ScanResult(objects=[fallback], successful_sources=["start_apps"]))
    assert catalog.get(item.object_id).description == "Editor for code"


def test_windows_version_resources_are_readable():
    import os

    target = Path(os.environ["SystemRoot"]) / "System32/notepad.exe"
    metadata = app_metadata.executable_metadata(str(target))
    assert metadata.get("FileDescription")
