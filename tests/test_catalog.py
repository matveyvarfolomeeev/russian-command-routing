from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from jev_launcher.catalog import WINDOWS_SOURCES, Catalog
from jev_launcher.cli import catalog_snapshot, main
from jev_launcher.objects import DiscoveredObject, ScanResult, launch_identity
from jev_launcher.search import filter_catalog


@pytest.fixture
def catalog(tmp_path):
    return Catalog(tmp_path / "catalog.sqlite3")


def discovered(name="Редактор", target=r"C:\Apps\editor.exe", key="user:editor", **kwargs):
    return DiscoveredObject(
        name=name,
        launch_target=target,
        launch_type="executable",
        source="start_menu_user",
        source_key=key,
        **kwargs,
    )


def scan(*items, sources=("start_menu_user",), errors=None):
    return ScanResult(list(items), list(sources), errors or {})


def test_refresh_preserves_id_and_user_overrides(catalog):
    first = discovered(description="Исходное описание")
    catalog.merge_scan(scan(first))
    old = catalog.items()[0]
    catalog.update(
        old.object_id,
        name="Мой редактор",
        description="",
        aliases=["код", "код"],
        favorite=True,
        enabled=False,
    )
    catalog.merge_scan(scan(replace(first, name="Новое название", description="Обновление")))
    item = catalog.items()[0]
    assert item.object_id == old.object_id
    assert item.name == "Мой редактор"
    assert item.description == ""
    assert item.description_source == "user"
    assert item.aliases == ("код",)
    assert item.favorite and not item.enabled


def test_missing_source_and_failed_source_are_different(catalog):
    catalog.merge_scan(scan(discovered()))
    catalog.merge_scan(scan(sources=(), errors={"start_menu_user": "Unavailable"}))
    assert catalog.items()[0].available
    catalog.merge_scan(scan())
    assert not catalog.items()[0].available
    object_id = catalog.items()[0].object_id
    catalog.merge_scan(scan(discovered()))
    assert catalog.items()[0].available
    assert catalog.items()[0].object_id == object_id


def test_deduplicate_target_but_keep_distinct_profiles(catalog):
    first = discovered(arguments=("--profile", "one"))
    copy = replace(first, source="start_menu_common", source_key="common:editor")
    second = replace(first, source_key="user:other", arguments=("--profile", "two"), identity="")
    second.__post_init__()
    catalog.merge_scan(scan(first, copy, second, sources=("start_menu_user", "start_menu_common")))
    assert len(catalog.items()) == 2
    catalog.merge_scan(scan(sources=("start_menu_user",)))
    assert sum(item.available for item in catalog.items()) == 1


def test_same_name_different_targets_do_not_merge(catalog):
    catalog.merge_scan(scan(discovered(), discovered(target=r"C:\Other\editor.exe", key="other")))
    assert len(catalog.items()) == 2


def test_source_identity_survives_target_update(catalog):
    catalog.merge_scan(scan(discovered()))
    old_id = catalog.items()[0].object_id
    catalog.merge_scan(scan(discovered(target=r"C:\Updated\editor.exe")))
    assert catalog.items()[0].object_id == old_id
    assert catalog.items()[0].launch_target == r"C:\Updated\editor.exe"


def test_manual_folder_survives_refresh_and_marks_missing(catalog, tmp_path):
    folder = tmp_path / "Проект с пробелами"
    folder.mkdir()
    item = catalog.add_path(folder)
    catalog.update(item.object_id, description="Эксперимент с моделями")
    catalog.merge_scan(scan())
    assert catalog.get(item.object_id).available
    assert catalog.get(item.object_id).description == "Эксперимент с моделями"
    folder.rmdir()
    catalog.merge_scan(scan())
    assert not catalog.get(item.object_id).available


def test_unknown_path_does_not_enter_catalog(catalog, tmp_path):
    with pytest.raises(ValueError):
        catalog.add_path(tmp_path / "missing")
    assert catalog.items() == []


def test_delete_manual_disable_discovered(catalog, tmp_path):
    manual = catalog.add_path(tmp_path)
    catalog.remove(manual.object_id)
    with pytest.raises(KeyError):
        catalog.get(manual.object_id)
    catalog.merge_scan(scan(discovered()))
    item = catalog.items()[0]
    catalog.remove(item.object_id)
    catalog.merge_scan(scan(discovered()))
    assert not catalog.get(item.object_id).enabled


def test_reopen_database_and_unicode(catalog):
    catalog.merge_scan(scan(discovered()))
    item = catalog.items()[0]
    catalog.update(item.object_id, description="ёлка и русский текст", aliases=["мой редактор"])
    reopened = Catalog(catalog.path)
    assert reopened.get(item.object_id).description == "ёлка и русский текст"
    assert reopened.get(item.object_id).aliases == ("мой редактор",)
    with pytest.raises(ValueError):
        reopened.update(item.object_id, launch_target="anything")
    with pytest.raises(ValueError):
        reopened.update(item.object_id, name="  ")


def test_full_snapshot_independent_of_search_and_contains_no_paths(catalog):
    catalog.merge_scan(
        scan(discovered(), discovered(name="Калькулятор", key="calc", target=r"C:\calc.exe"))
    )
    assert len(filter_catalog(catalog.items(), "редактор")) == 1
    snapshot = catalog_snapshot(catalog)
    assert len(snapshot["objects"]) == 2
    assert all(
        set(item) == {"object_id", "name", "kind", "description", "aliases"}
        for item in snapshot["objects"]
    )
    assert catalog_snapshot(Catalog(catalog.path)) == snapshot
    catalog.update(catalog.items()[0].object_id, enabled=False)
    assert len(catalog_snapshot(catalog)["objects"]) == 1
    assert catalog_snapshot(catalog)["catalog_hash"] != snapshot["catalog_hash"]


def test_cli_export_and_search_never_import_execution(catalog, tmp_path, capsys, monkeypatch):
    catalog.merge_scan(scan(discovered(description="Редактирование кода", aliases=("вскод",))))
    import jev_launcher.launch

    monkeypatch.setattr(
        jev_launcher.launch, "open_object", lambda item: pytest.fail("Unexpected launch")
    )
    assert main(["--db", str(catalog.path), "search", "вскод"]) == 0
    assert "Редактор" in capsys.readouterr().out
    output = tmp_path / "export" / "catalog.json"
    assert main(["--db", str(catalog.path), "export", "--output", str(output)]) == 0
    assert output.is_file()
    assert "launch_target" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "query",
    ["РЕДАКТОР", "редактирование", "вскод", " елка ", ""],
)
def test_text_filter(catalog, query):
    catalog.merge_scan(
        scan(discovered(description="Редактирование кода и ёлка", aliases=("вскод",)))
    )
    assert len(filter_catalog(catalog.items(), query)) == 1
    catalog.update(catalog.items()[0].object_id, enabled=False)
    assert filter_catalog(catalog.items(), query) == []
    assert len(filter_catalog(catalog.items(), query, active_only=False)) == 1


def test_identity_preserves_case_in_arguments_and_working_directory():
    assert launch_identity("executable", r"C:\App.exe") == launch_identity(
        "executable", r"c:\APP.exe"
    )
    assert launch_identity("executable", r"C:\App.exe", ("A",)) != launch_identity(
        "executable", r"C:\App.exe", ("a",)
    )
    assert launch_identity("executable", r"C:\App.exe", (), r"C:\Work") != launch_identity(
        "executable", r"C:\App.exe", (), r"C:\Other"
    )


def test_manual_add_does_not_fake_windows_refresh(catalog, tmp_path):
    catalog.add_path(tmp_path)
    assert catalog.setting("last_scan") is None
    assert catalog.needs_refresh()


def test_freshness_only_advances_after_successful_windows_sources(catalog):
    catalog.merge_scan(scan(sources=WINDOWS_SOURCES))
    now = datetime.now(timezone.utc)
    stamp = catalog.setting("last_successful_scan")
    assert not catalog.needs_refresh(now)
    assert catalog.needs_refresh(now + timedelta(hours=25))
    catalog.merge_scan(scan(sources=("start_apps",), errors={"start_menu_user": "Failed"}))
    assert catalog.setting("last_successful_scan") == stamp
    catalog.set_setting("last_successful_scan", "broken")
    assert catalog.needs_refresh(now)


def test_legacy_upgrade_removes_uninstaller_and_merges_exact_command(catalog, tmp_path):
    target = tmp_path / "Telegram.exe"
    target.touch()
    cwd = str(tmp_path)
    shortcuts = [
        DiscoveredObject(
            name="Telegram",
            launch_type="shortcut",
            launch_target=str(tmp_path / f"telegram{index}.lnk"),
            source="start_menu_user",
            source_key=f"telegram:{index}",
            icon_target=str(target),
            identity=launch_identity("app_id", app_id),
            working_directory=cwd,
        )
        for index, app_id in enumerate((str(target), "Telegram.TelegramDesktop"))
    ]
    direct = discovered(
        name="Telegram", target=str(target), key="telegram:path", working_directory=cwd
    )
    uninstall = discovered(
        name="Деинсталляция Telegram", target=str(tmp_path / "unins000.exe"), key="uninstall"
    )
    # Insert schema-1 records directly: current merge correctly refuses the uninstaller.
    from jev_launcher.objects import CatalogObject

    with catalog.connection() as db:
        for index, item in enumerate([*shortcuts, direct, uninstall]):
            values = {key: value for key, value in vars(item).items() if key != "source_key"}
            stored = CatalogObject(object_id=f"id{index}", **values)
            catalog._save(db, stored)
            db.execute(
                "INSERT INTO sources VALUES (?, ?, ?, 1)",
                (item.source_key, item.source, stored.object_id),
            )
        db.execute("PRAGMA user_version=1")
    catalog.update(
        "id1",
        name="Мой Telegram",
        description="Мой мессенджер",
        aliases=["телеграмм"],
        favorite=True,
    )
    upgraded = Catalog(catalog.path)
    items = upgraded.items()
    assert len(items) == 1
    assert items[0].object_id == "id1"
    assert items[0].description == "Мой мессенджер"
    assert items[0].aliases == ("телеграмм",)
    assert items[0].favorite
    with upgraded.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM sources WHERE object_id='id1'").fetchone()[0] == 3
    assert upgraded.needs_refresh()
    assert len(Catalog(catalog.path).items()) == 1


def test_schema2_opaque_startapp_merges_without_overriding_shortcut_cwd(catalog, tmp_path):
    target = tmp_path / "7zFM.exe"
    target.touch()
    shortcut = tmp_path / "7zip.lnk"
    shortcut.touch()
    link = DiscoveredObject(
        name="7-Zip",
        launch_target=str(shortcut),
        launch_type="shortcut",
        source="start_menu_common",
        source_key="7zip:link",
        resolved_target=str(target),
        identity=launch_identity("executable", str(target)),
        working_directory="",
    )
    direct = DiscoveredObject(
        name="7-Zip",
        launch_target=str(target),
        launch_type="executable",
        source="start_apps",
        source_key="7zip:direct",
        resolved_target=str(target),
        working_directory=str(tmp_path),
    )
    catalog.merge_scan(ScanResult([link, direct], ["start_menu_common", "start_apps"]))
    direct_id = next(item.object_id for item in catalog.items() if item.launch_type == "executable")
    catalog.update(direct_id, description="Мой архиватор", aliases=["архивы"], favorite=True)
    with catalog.connection() as db:
        db.execute("PRAGMA user_version=2")
    upgraded = Catalog(catalog.path)
    assert len(upgraded.items()) == 1
    item = upgraded.items()[0]
    assert item.object_id == direct_id
    assert item.description == "Мой архиватор" and item.aliases == ("архивы",)
    assert item.launch_type == "shortcut" and item.launch_target == str(shortcut)
    assert item.working_directory == ""


def test_schema2_preserves_distinct_shortcut_cwd_variants(catalog, tmp_path):
    target = tmp_path / "App.exe"
    target.touch()
    shortcuts = [
        DiscoveredObject(
            name="App",
            launch_target=str(tmp_path / f"{cwd}.lnk"),
            launch_type="shortcut",
            source="start_menu_common",
            source_key=f"app:{cwd}",
            resolved_target=str(target),
            identity=launch_identity("executable", str(target), (), str(tmp_path / cwd)),
            working_directory=str(tmp_path / cwd),
        )
        for cwd in ("one", "two")
    ]
    direct = DiscoveredObject(
        name="App",
        launch_target=str(target),
        launch_type="executable",
        source="start_apps",
        source_key="app:direct",
        resolved_target=str(target),
        working_directory=str(tmp_path),
    )
    catalog.merge_scan(ScanResult([*shortcuts, direct], ["start_menu_common", "start_apps"]))
    with catalog.connection() as db:
        db.execute("PRAGMA user_version=2")
    upgraded = Catalog(catalog.path)
    assert len(upgraded.items()) == 3
