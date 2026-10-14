"""Enrich the real catalog without exposing evaluation queries to the collector."""

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    from jev_launcher.app_metadata import enrich_descriptions
    from jev_launcher.catalog import default_data_dir
    from jev_launcher.objects import CatalogObject

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    database = default_data_dir() / "catalog.sqlite3"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        live = [CatalogObject.from_dict(json.loads(row[0])) for row in db.execute("SELECT payload FROM objects")]
        with sqlite3.connect(args.output / "catalog-before.sqlite3") as backup:
            db.backup(backup)
    current = {item.object_id: item for item in live}
    source = ROOT / "artifacts/benchmarks/launcher-asr-v1/real/catalog.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    objects = original["objects"]
    assert len(objects) == len({item["object_id"] for item in objects}) == 177
    # Metadata is collected from installed apps, without looking at any query/target.
    for item in objects:
        assert current[item["object_id"]].for_model() == item, item["name"]
    cache_path = default_data_dir() / "app-descriptions.json"
    installed = None
    if args.installed_packages:
        exported = json.loads(args.installed_packages.read_text(encoding="utf-8-sig"))
        installed = {package["PackageIdentifier"]: package.get("Version", "")
                     for source in exported["Sources"] for package in source["Packages"]}
    stats = enrich_descriptions(live, cache_path=cache_path, package_budget=args.package_budget,
                               installed_versions=installed)
    updates = {}
    new_objects = []
    for before in objects:
        after = current[before["object_id"]].for_model()
        assert {k: v for k, v in before.items() if k != "description"} == {
            k: v for k, v in after.items() if k != "description"
        }
        if after != before:
            assert before["kind"] == "app" and not before["description"]
            updates[before["object_id"]] = {
                "name": before["name"], "description": after["description"],
                "description_source": current[before["object_id"]].description_source,
            }
        new_objects.append(after)
    write(args.output / "before-catalog.json", original)
    # The original catalog hash is not carried over to modified card contents.
    write(args.output / "after-catalog.json", {"schema_version": 1, "objects": new_objects})
    write(args.output / "description-updates.json", updates)
    if cache_path.exists():
        (args.output / "package-cache.json").write_bytes(cache_path.read_bytes())
    if args.installed_packages:
        (args.output / "installed-packages.json").write_bytes(args.installed_packages.read_bytes())
    # Only known real validation scenarios; the sealed old test is not opened.
    query_source = ROOT / "artifacts/kaggle/e5-evaluation-v1/input/full-evaluation.jsonl"
    rows = [json.loads(line) for line in query_source.read_text(encoding="utf-8").splitlines()]
    data = [row for row in rows if row["profile"].endswith("/real")]
    assert len(data) == len({row["id"] for row in data}) == 150
    (args.output / "evaluation.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8"
    )
    sources = [source, query_source, Path(__file__).resolve(), ROOT / "src/jev_launcher/app_metadata.py"]
    manifest = {"role": "Known real validation basic50/difficult50/STT50; fixed weights; no training or selection",
                "query_rows": 150, "objects": 177, "apps": 139, "collection_stats": stats,
                "changed_cards": len(updates), "collector_received_queries": False,
                "apps_described_before": sum(bool(item["description"]) for item in objects if item["kind"] == "app"),
                "apps_described_after": sum(bool(item["description"]) for item in new_objects if item["kind"] == "app"),
                "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
                "files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}}
    write(args.output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("changed_cards", "apps_described_before", "apps_described_after", "collection_stats")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/app-metadata-v1")
    parser.add_argument("--package-budget", type=float, default=300)
    parser.add_argument("--installed-packages", type=Path)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(parser.parse_args())
