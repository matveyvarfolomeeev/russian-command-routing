"""Apply the reviewed enrichment to the live catalog, preserving every other field."""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FOLDER = ROOT / "artifacts/benchmarks/app-metadata-v2"


def run():
    from jev_launcher.catalog import Catalog, default_data_dir

    path = default_data_dir() / "catalog.sqlite3"
    backup_path = FOLDER / "catalog-before-apply.sqlite3"
    if backup_path.exists():
        raise ValueError("This apply operation already has a backup")
    updates = json.loads((FOLDER / "description-updates.json").read_text(encoding="utf-8"))
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        with sqlite3.connect(backup_path) as backup:
            db.backup(backup)
    catalog = Catalog(path)
    before = {item.object_id: item.to_dict() for item in catalog.items()}
    applied = catalog.apply_descriptions(updates)
    after = {item.object_id: item.to_dict() for item in catalog.items()}
    assert set(before) == set(after)
    changed = []
    for object_id, first in before.items():
        second = after[object_id]
        assert {k: v for k, v in first.items() if k not in ("description", "description_source")} == {
            k: v for k, v in second.items() if k not in ("description", "description_source")
        }
        if first != second:
            assert object_id in updates and not first["description"]
            assert "description" not in first["overrides"]
            assert second["description"] == updates[object_id]["description"]
            changed.append(object_id)
    assert len(changed) == applied
    with backup_path.open("rb") as stream:
        backup_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"applied": applied, "requested": len(updates), "changed_object_ids": changed,
              "only_missing_descriptions_and_provenance_changed": True, "manual_overrides_preserved": True,
              "apps_with_description": sum(bool(item["description"]) for item in after.values() if item["kind"] == "app"),
              "backup": backup_path.relative_to(ROOT).as_posix(), "backup_sha256": backup_sha}
    (FOLDER / "application-readback.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    run()
