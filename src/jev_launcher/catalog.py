"""SQLite catalog, independent of GUI, discovery and decision models."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .objects import CatalogObject, DiscoveredObject, ScanResult


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
            if version not in (0, 1):
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
                PRAGMA user_version=1;
            """)
            db.execute("PRAGMA journal_mode=WAL")

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
                row = db.execute(
                    "SELECT payload FROM objects WHERE identity=?", (found.identity,)
                ).fetchone()
                if row is None:
                    row = db.execute(
                        "SELECT o.payload FROM objects o JOIN sources s ON o.object_id=s.object_id "
                        "WHERE s.source_key=?",
                        (found.source_key,),
                    ).fetchone()
                if row:
                    item = self._decode(row)
                    preserve_shortcut = (
                        found.launch_type in ("app_id", "executable", "file")
                        and item.launch_type == "shortcut"
                        and Path(item.launch_target).exists()
                    )
                    for key in (
                        "name",
                        "description",
                        "description_source",
                        "aliases",
                        "launch_type",
                        "launch_target",
                        "arguments",
                        "working_directory",
                        "identity",
                        "icon_target",
                    ):
                        if preserve_shortcut and key in (
                            "launch_type",
                            "launch_target",
                            "arguments",
                            "working_directory",
                            "icon_target",
                        ):
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
            db.execute(
                "INSERT INTO settings VALUES ('last_scan', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (json.dumps({"at": now, "errors": result.errors}),),
            )
        return len(self.items())

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
