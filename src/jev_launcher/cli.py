"""Headless catalog access for experiments. Search and export never launch apps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .catalog import Catalog
from .search import filter_catalog


def catalog_snapshot(catalog: Catalog) -> dict:
    objects = catalog.model_catalog()
    canonical = json.dumps(objects, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "catalog_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "objects": objects,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, help="Catalog database; default is the application catalog"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="Refresh Windows applications and folders")
    scan.add_argument(
        "--folder-root",
        action="append",
        type=Path,
        help="Save project roots to index (root and immediate subfolders); repeatable",
    )
    commands.add_parser("list", help="Print the complete enabled/available catalog")
    export = commands.add_parser("export", help="Export a stable model-facing snapshot")
    export.add_argument("--output", type=Path, required=True)
    search = commands.add_parser("search", help="Run the text baseline; never opens a program")
    search.add_argument("query")
    add = commands.add_parser("add", help="Add a local app, folder or file")
    add.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    catalog = Catalog(args.db)
    try:
        if args.command == "scan":
            from .discovery import scan_windows

            if args.folder_root is not None:
                roots = [str(path.expanduser().resolve()) for path in args.folder_root]
                if any(not Path(root).is_dir() for root in roots):
                    raise ValueError("Каждая папка проектов должна существовать")
                catalog.set_setting("folder_roots", list(dict.fromkeys(roots)))
                catalog.set_setting("last_successful_scan", None)
            roots = catalog.setting("folder_roots", [])
            result = scan_windows(roots) if roots else scan_windows()
            catalog.merge_scan(result)
            output = {
                "objects": len(catalog.items()),
                "active": len(catalog.items(active_only=True)),
                "successful_sources": result.successful_sources,
                "errors": result.errors,
            }
        elif args.command == "export":
            output = catalog_snapshot(catalog)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Exported {len(output['objects'])} objects to {args.output}")
            return 0
        elif args.command == "search":
            output = [item.for_model() for item in filter_catalog(catalog.items(), args.query)]
        elif args.command == "add":
            output = catalog.add_path(args.path).for_model()
        else:
            output = catalog.model_catalog()
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
