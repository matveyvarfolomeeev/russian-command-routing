"""SQLite catalog, independent of GUI, discovery and decision models."""

from __future__ import annotations

import json
import ntpath
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from .objects import CatalogObject, DiscoveredObject, ScanResult, is_maintenance, launch_identity

WINDOWS_SOURCES = {
    "start_menu_user",
    "start_menu_common",
    "start_apps",
    "known_folders",
    "project_folders",
}
REFRESH_INTERVAL = timedelta(hours=24)


def default_data_dir() -> Path:
    override = os.environ.get("JEV_LAUNCHER_DATA_DIR")
    return (
        Path(override)
        if override
        else Path(os.environ.get("LOCALAPPDATA", Path.home())) / "JevLauncher"
    )


class Catalog:
    """Short-lived connections allow a worker to refresh while the GUI reads."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_data_dir() / "catalog.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3, 4):
                raise RuntimeError(f"Unsupported catalog schema: {version}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    object_id TEXT PRIMARY KEY,
                    identity TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    source_key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    object_id TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
                    present INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS source_object ON sources(object_id);
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            db.execute("PRAGMA journal_mode=WAL")
            if version < 3:
                self._clean_legacy_catalog(db)
            if version < 4:
                # Apply enriched Shell metadata and automatic folders on the next scan.
                db.execute(
                    "INSERT OR REPLACE INTO settings VALUES ('last_successful_scan', 'null')"
                )
                db.execute("PRAGMA user_version=4")

    def _clean_legacy_catalog(self, db):
        """One cheap upgrade removes old uninstallers and joins exact desktop commands."""
        items = [self._decode(row) for row in db.execute("SELECT payload FROM objects")]
        commands: dict[str, set[str]] = {}
        desktop_commands: dict[str, set[str]] = {}
        identities = {}
        for item in items:
            if item.source == "manual":
                continue
            resolved = item.resolved_target
            if (
                not resolved
                and item.launch_type == "shortcut"
                and item.icon_target.lower().endswith(".exe")
            ):
                # Schema 1's discovery stored the actual shortcut target in icon_target.
                resolved = item.icon_target
            if is_maintenance(item.name, resolved or item.launch_target, item.arguments):
                db.execute("DELETE FROM objects WHERE object_id=?", (item.object_id,))
                continue
            if item.launch_type == "shortcut" and resolved:
                identity = launch_identity(
                    "executable" if Path(resolved).suffix.lower() == ".exe" else "file",
                    resolved,
                    item.arguments,
                    item.working_directory,
                )
                old = json.loads(item.identity)
                if old[0] == "app_id" and not item.arguments:
                    commands.setdefault(old[1].casefold(), set()).add(identity)
                if not item.arguments and Path(resolved).suffix.lower() == ".exe":
                    desktop_commands.setdefault(
                        ntpath.normcase(ntpath.normpath(resolved)), set()
                    ).add(identity)
                item.resolved_target = resolved
                identities[item.object_id] = identity
        groups: dict[str, list[CatalogObject]] = {}
        for item in items:
            if (
                is_maintenance(
                    item.name,
                    item.resolved_target or item.icon_target or item.launch_target,
                    item.arguments,
                )
                and item.source != "manual"
            ):
                continue
            identity = identities.get(item.object_id, item.identity)
            if item.launch_type == "app_id" and item.source != "manual":
                matches = commands.get(item.launch_target.casefold(), set())
                if len(matches) == 1:
                    identity = next(iter(matches))
            elif (
                item.launch_type == "executable"
                and item.source == "start_apps"
                and not item.arguments
            ):
                matches = desktop_commands.get(
                    ntpath.normcase(ntpath.normpath(item.launch_target)), set()
                )
                if len(matches) == 1:
                    identity = next(iter(matches))
                    item.working_directory = json.loads(identity)[3]
            groups.setdefault(identity, []).append(item)
        for identity, copies in groups.items():
            self._merge_copies(db, copies, identity)
        # Refresh once after an upgrade: new discovery policy and builtin aliases need applying.
        if items:
            db.execute("INSERT OR REPLACE INTO settings VALUES ('last_successful_scan', 'null')")

    def _merge_copies(self, db, copies, identity) -> CatalogObject:
        """Join aliases physically, preserving edited IDs and every source reference."""
        copies.sort(
            key=lambda item: (
                not bool(item.overrides),
                not item.favorite,
                item.enabled,
                item.source != "manual",
                item.launch_type != "shortcut",
            )
        )
        keeper = copies[0]
        shortcut = next(
            (
                item
                for item in copies
                if item.launch_type == "shortcut" and Path(item.launch_target).exists()
            ),
            None,
        )
        if shortcut and keeper.launch_type != "shortcut":
            for key in (
                "launch_type",
                "launch_target",
                "arguments",
                "working_directory",
                "icon_target",
                "resolved_target",
            ):
                setattr(keeper, key, getattr(shortcut, key))
        for copy in copies[1:]:
            for key in copy.overrides:
                if key not in keeper.overrides:
                    setattr(keeper, key, getattr(copy, key))
            keeper.overrides = tuple(sorted(set(keeper.overrides) | set(copy.overrides)))
            keeper.favorite |= copy.favorite
            keeper.enabled &= copy.enabled
            keeper.available |= copy.available
            db.execute(
                "UPDATE sources SET object_id=? WHERE object_id=?",
                (keeper.object_id, copy.object_id),
            )
            db.execute("DELETE FROM objects WHERE object_id=?", (copy.object_id,))
        keeper.identity = identity
        command_type = json.loads(identity)[0]
        keeper.kind = (
            "app" if command_type in ("executable", "shortcut", "app_id") else command_type
        )
        self._save(db, keeper)
        return keeper

    def _clean_orphaned_copies(self, db):
        """Older scans could move a source reference and leave an unused duplicate behind."""
        rows = db.execute(
            "SELECT payload FROM objects o WHERE NOT EXISTS "
            "(SELECT 1 FROM sources s WHERE s.object_id=o.object_id)"
        ).fetchall()
        for row in rows:
            orphan = self._decode(row)
            if orphan.source == "manual":
                continue
            command = launch_identity(
                orphan.launch_type, orphan.launch_target, orphan.arguments, orphan.working_directory
            )
            candidates = [
                self._decode(row)
                for row in db.execute(
                    "SELECT payload FROM objects o WHERE EXISTS "
                    "(SELECT 1 FROM sources s WHERE s.object_id=o.object_id)"
                )
            ]
            matches = [
                item
                for item in candidates
                if launch_identity(
                    item.launch_type, item.launch_target, item.arguments, item.working_directory
                )
                == command
            ]
            if len({item.identity for item in matches}) == 1:
                self._merge_copies(db, [*matches, orphan], matches[0].identity)
            elif not orphan.overrides and not orphan.favorite and orphan.enabled:
                db.execute("DELETE FROM objects WHERE object_id=?", (orphan.object_id,))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _decode(row) -> CatalogObject:
        return CatalogObject.from_dict(json.loads(row["payload"]))

    @staticmethod
    def _save(db, item: CatalogObject) -> None:
        db.execute(
            "INSERT INTO objects VALUES (?, ?, ?) "
            "ON CONFLICT(object_id) DO UPDATE SET identity=excluded.identity, payload=excluded.payload",
            (item.object_id, item.identity, json.dumps(item.to_dict(), ensure_ascii=False)),
        )

    def items(self, *, active_only: bool = False) -> list[CatalogObject]:
        with self.connection() as db:
            items = [self._decode(row) for row in db.execute("SELECT payload FROM objects")]
        if active_only:
            items = [item for item in items if item.enabled and item.available]
        return sorted(
            items, key=lambda item: (not item.favorite, item.name.casefold(), item.object_id)
        )

    def get(self, object_id: str) -> CatalogObject:
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM objects WHERE object_id=?", (object_id,)
            ).fetchone()
        if row is None:
            raise KeyError("Объект больше не существует в каталоге")
        return self._decode(row)

    def merge_scan(self, result: ScanResult) -> int:
        """Only successful sources can mark old entries missing; edits always survive."""
        now = datetime.now(timezone.utc).isoformat()
        successful = set(result.successful_sources)
        with self.connection() as db:
            for source in successful:
                db.execute("UPDATE sources SET present=0 WHERE source=?", (source,))
            for found in result.objects:
                if found.source not in successful:
                    continue
                if found.source != "manual" and is_maintenance(
                    found.name,
                    found.resolved_target or found.icon_target or found.launch_target,
                    found.arguments,
                ):
                    continue
                row = db.execute(
                    "SELECT payload FROM objects WHERE identity=?", (found.identity,)
                ).fetchone()
                source_row = db.execute(
                    "SELECT o.payload FROM objects o JOIN sources s ON o.object_id=s.object_id "
                    "WHERE s.source_key=?",
                    (found.source_key,),
                ).fetchone()
                if row and source_row:
                    existing, previous = self._decode(row), self._decode(source_row)
                    if existing.object_id != previous.object_id:
                        self._merge_copies(db, [existing, previous], found.identity)
                        row = db.execute(
                            "SELECT payload FROM objects WHERE identity=?", (found.identity,)
                        ).fetchone()
                row = row or source_row
                if row:
                    item = self._decode(row)
                    preserve_shortcut = (
                        found.launch_type in ("app_id", "executable", "file", "folder")
                        and item.launch_type == "shortcut"
                        and Path(item.launch_target).exists()
                    )
                    for key in (
                        "name",
                        "kind",
                        "description",
                        "description_source",
                        "aliases",
                        "launch_type",
                        "launch_target",
                        "arguments",
                        "working_directory",
                        "identity",
                        "icon_target",
                        "resolved_target",
                    ):
                        if preserve_shortcut and key in (
                            "launch_type",
                            "launch_target",
                            "arguments",
                            "working_directory",
                            "icon_target",
                            "resolved_target",
                        ):
                            continue
                        if (
                            preserve_shortcut
                            and key == "identity"
                            and found.launch_type == "app_id"
                            and not found.resolved_target
                        ):
                            # Missing Shell metadata must not split an already resolved alias.
                            continue
                        if key in ("description", "description_source", "aliases") and not getattr(
                            found, key
                        ):
                            continue
                        if key not in item.overrides:
                            setattr(item, key, getattr(found, key))
                    item.available = found.available
                    item.last_seen = now
                else:
                    values = {
                        key: value for key, value in vars(found).items() if key != "source_key"
                    }
                    item = CatalogObject(object_id=str(uuid4()), last_seen=now, **values)
                self._save(db, item)
                db.execute(
                    "INSERT INTO sources VALUES (?, ?, ?, 1) "
                    "ON CONFLICT(source_key) DO UPDATE SET source=excluded.source, "
                    "object_id=excluded.object_id, present=1",
                    (found.source_key, found.source, item.object_id),
                )
            self._clean_orphaned_copies(db)
            for row in db.execute("SELECT payload FROM objects").fetchall():
                item = self._decode(row)
                has_source = db.execute(
                    "SELECT 1 FROM sources WHERE object_id=? AND present=1", (item.object_id,)
                ).fetchone()
                if not has_source and item.source != "manual":
                    item.available = False
                    self._save(db, item)
                elif item.source == "manual" and item.launch_type != "app_id":
                    item.available = Path(item.launch_target).exists()
                    self._save(db, item)
            if successful & WINDOWS_SOURCES or any(
                key.split(":", 1)[0] in WINDOWS_SOURCES for key in result.errors
            ):
                db.execute(
                    "INSERT INTO settings VALUES ('last_scan', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (
                        json.dumps(
                            {"at": now, "errors": result.errors, "sources": sorted(successful)}
                        ),
                    ),
                )
            if WINDOWS_SOURCES <= successful:
                db.execute(
                    "INSERT INTO settings VALUES ('last_successful_scan', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (json.dumps(now),),
                )
        return len(self.items())

    def needs_refresh(self, now: datetime | None = None) -> bool:
        """A cached catalog is usable immediately; refresh only when Windows metadata is stale."""
        stamp = self.setting("last_successful_scan")
        if not stamp:
            return True
        try:
            last = datetime.fromisoformat(stamp)
            if last.tzinfo is None:
                return True
        except (TypeError, ValueError):
            return True
        current = now or datetime.now(timezone.utc)
        age = current - last
        return age >= REFRESH_INTERVAL or age < timedelta(0)

    def add_path(self, path: Path | str, kind: str | None = None) -> CatalogObject:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            raise ValueError("Выбранный путь не существует")
        if target.is_dir():
            launch_type, object_kind = "folder", "folder"
        elif target.suffix.lower() == ".lnk":
            launch_type, object_kind = "shortcut", "app"
        elif target.suffix.lower() == ".exe":
            launch_type, object_kind = "executable", "app"
        else:
            launch_type, object_kind = "file", "file"
        if kind is not None and kind != object_kind:
            raise ValueError("Тип объекта не соответствует выбранному пути")
        found = DiscoveredObject(
            name=target.stem if object_kind == "app" else target.name,
            kind=object_kind,
            launch_type=launch_type,
            launch_target=str(target),
            source="manual",
            source_key="manual:" + str(target).casefold(),
            icon_target=str(target),
        )
        self.merge_scan(ScanResult([found], ["manual"]))
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM objects WHERE identity=?", (found.identity,)
            ).fetchone()
        return self._decode(row)

    def update(self, object_id: str, **changes) -> CatalogObject:
        allowed = {"name", "description", "aliases", "enabled", "favorite"}
        if set(changes) - allowed:
            raise ValueError("Unsupported editable fields")
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM objects WHERE object_id=?", (object_id,)
            ).fetchone()
            if row is None:
                raise KeyError(object_id)
            item = self._decode(row)
            if "name" in changes and not str(changes["name"]).strip():
                raise ValueError("Введите название")
            for key, value in changes.items():
                if key in ("enabled", "favorite"):
                    value = bool(value)
                elif key == "aliases":
                    value = tuple(dict.fromkeys(str(v).strip() for v in value if str(v).strip()))
                else:
                    value = str(value).strip()
                setattr(item, key, value)
            edited = set(item.overrides) | (set(changes) - {"enabled", "favorite"})
            if "description" in changes:
                item.description_source = "user"
                edited.add("description_source")
            item.overrides = tuple(sorted(edited))
            self._save(db, item)
        return item

    def remove(self, object_id: str) -> None:
        """Remove manual entries. Discovered entries are disabled to survive a refresh."""
        item = self.get(object_id)
        if item.source != "manual":
            self.update(object_id, enabled=False)
        else:
            with self.connection() as db:
                db.execute("DELETE FROM objects WHERE object_id=?", (object_id,))

    def setting(self, key: str, default=None):
        with self.connection() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key: str, value) -> None:
        with self.connection() as db:
            db.execute(
                "INSERT INTO settings VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def model_catalog(self) -> list[dict]:
        """Always the full enabled/available catalog, independent of GUI filtering."""
        return [item.for_model() for item in self.items(active_only=True)]
