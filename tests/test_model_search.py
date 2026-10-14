import json
import sys
from pathlib import Path

import pytest

from jev_launcher.model_search import ModelRuntime


@pytest.fixture
def runtime(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, os, sys\n"
        "print(json.dumps({'event': 'loaded'}), flush=True)\n"
        "for line in sys.stdin:\n"
        " r = json.loads(line)\n"
        " if r.get('operation') == 'prepare':\n"
        "  print(json.dumps({'event': 'ready', 'pid': os.getpid()}), flush=True)\n"
        "  continue\n"
        " ids = [o['object_id'] for o in r['objects']][::-1]\n"
        " if r['query'] == 'foreign': ids = ['foreign-id']\n"
        " print(json.dumps({'object_ids': ids, 'elapsed_ms': 2, 'pid': os.getpid(),\n"
        "   'received_fields': sorted(r['objects'][0])}), flush=True)\n",
        encoding="utf-8",
    )
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({
        "worker": str(worker),
        "profiles": {"laya": {"python": sys.executable}, "e5_trained": {"python": sys.executable}},
    }), encoding="utf-8")
    service = ModelRuntime(config, timeout=5)
    yield service
    service.close()


def objects():
    return [
        {"object_id": value, "name": value, "kind": "app", "description": "",
         "aliases": [], "launch_target": "PRIVATE-EXE", "arguments": ["PRIVATE-ARG"]}
        for value in ["one", "two"]
    ]


def test_worker_is_reused_and_receives_only_catalog_metadata(runtime):
    first = runtime.search("laya", "first", objects())
    second = runtime.search("laya", "second", objects())
    assert first["pid"] == second["pid"]
    assert first["object_ids"] == ["two", "one"]
    assert first["received_fields"] == ["aliases", "description", "kind", "name", "object_id"]
    switched = runtime.search("e5_trained", "third", objects())
    assert switched["pid"] != first["pid"]


def test_foreign_object_id_is_rejected_and_worker_can_restart(runtime):
    with pytest.raises(ValueError, match="ID вне"):
        runtime.search("laya", "foreign", objects())
    assert runtime.search("laya", "valid", objects())["object_ids"] == ["two", "one"]


def test_missing_runtime_is_visible_instead_of_silent_filter_fallback(tmp_path):
    service = ModelRuntime(Path(tmp_path) / "missing.json")
    with pytest.raises(RuntimeError, match="не настроены"):
        service.search("laya", "query", objects())


def test_preparation_is_cached_and_search_uses_the_loaded_worker(runtime):
    prepared = runtime.prepare("laya", objects())
    again = runtime.prepare("laya", objects())
    assert prepared["event"] == "ready"
    assert again["cached"]
    assert runtime.search("laya", "query", objects())["pid"] == prepared["pid"]
    changed = objects()
    changed[0]["description"] = "Изменённое описание"
    assert not runtime.prepare("laya", changed).get("cached")
    process = runtime._process
    runtime.close(wait=True)
    assert process.poll() is not None


def test_work_scheduled_before_model_switch_cannot_restart_old_weights(runtime):
    runtime.prepare("laya", objects())
    generation = runtime.generation
    runtime.close(wait=True)
    current = runtime.prepare("e5_trained", objects())
    with pytest.raises(RuntimeError, match="отменён"):
        runtime.search("laya", "obsolete", objects(), expected_generation=generation)
    with pytest.raises(RuntimeError, match="отменён"):
        runtime.prepare("laya", objects(), expected_generation=generation)
    assert runtime.search("e5_trained", "current", objects())["pid"] == current["pid"]
