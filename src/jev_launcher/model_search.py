"""Model choices and a persistent local worker; the Qt process stays lightweight."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

DEFAULT_MODEL = "laya"
MODELS = (
    ("laya", "Laya — основная"),
    ("modernbert", "RuModernBERT — быстрый reranker"),
    ("e5_trained", "E5 — дообученная, быстрый поиск"),
    ("filter", "Текстовый фильтр"),
)


def runtime_config() -> Path:
    override = os.environ.get("JEV_MODEL_RUNTIME", "")
    if override:
        return Path(override)
    package = Path(__file__).resolve().parent
    candidates = (
        package.parents[1] / "models" / "launcher-runtime.json",
        Path(sys.executable).parent / "models" / "launcher-runtime.json",
        package / "resources" / "launcher-runtime.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


class ModelRuntime:
    """One worker and one in-flight request, owned by the controller's background job."""

    def __init__(self, config_path=None, timeout=180):
        self.config_path = Path(config_path) if config_path else runtime_config()
        self.timeout = timeout
        self._process = None
        self._model = ""
        self._responses = None
        self._prepared = ""
        self._generation = 0
        self._operation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._workers = set()

    @property
    def generation(self):
        return self._generation

    def close(self, wait=False):
        with self._state_lock:
            self._generation += 1
            process, self._process = self._process, None
            self._model = ""
            self._prepared = ""
        if process and process.poll() is None:
            process.terminate()
        if wait:
            for worker in list(self._workers):
                if worker.poll() is None:
                    worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)

    def _discard(self, process):
        with self._state_lock:
            if self._process is process:
                self._process = None
                self._model = ""
                self._prepared = ""
        if process.poll() is None:
            process.terminate()

    @staticmethod
    def _read(process, responses):
        try:
            for line in process.stdout:
                responses.put(line)
        finally:
            responses.put(None)
            process.wait()
            process.stdout.close()
            process.stdin.close()

    def _receive(self, process, responses):
        try:
            line = responses.get(timeout=self.timeout)
        except queue.Empty as exc:
            self._discard(process)
            raise RuntimeError("Модель не ответила вовремя. Повторите запрос.") from exc
        if line is None:
            raise RuntimeError("Процесс модели завершился. Повторите запрос.")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def _start(self, model, generation):
        if not self.config_path.is_file():
            raise RuntimeError(
                "Локальные модели не настроены. Выполните configure_model_runtime.py "
                "или выберите «Текстовый фильтр»."
            )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        profile = config["profiles"].get(model)
        if not profile:
            raise ValueError("Выбранная модель отсутствует в локальной конфигурации.")
        python, worker = Path(profile["python"]), Path(config["worker"])
        bundled_worker = Path(__file__).parent / "runtime" / "run_launcher_model.py"
        if getattr(sys, "frozen", False) and bundled_worker.is_file():
            worker = bundled_worker
        if not python.is_file() or not worker.is_file():
            raise RuntimeError("Не найдена среда модели. Проверьте локальную конфигурацию.")
        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "USE_TF": "0",
            "USE_FLAX": "0",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
        process = subprocess.Popen(
            [str(python), "-u", str(worker), "--config", str(self.config_path), "--model", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        responses = queue.Queue()
        with self._state_lock:
            self._workers = {worker for worker in self._workers if worker.poll() is None}
            self._workers.add(process)
            if generation != self._generation:
                process.terminate()
            else:
                self._process, self._responses, self._model = process, responses, model
                self._prepared = ""
        threading.Thread(target=self._read, args=(process, responses), daemon=True).start()
        try:
            if self._receive(process, responses).get("event") != "loaded":
                raise ValueError("Неверный ответ при запуске модели.")
            logging.info("Model weights loaded: model=%s pid=%s worker=%s", model, process.pid, worker)
        except Exception:
            self._discard(process)
            raise

    @staticmethod
    def _safe_objects(objects):
        # Only metadata enters the helper; paths and commands never leave the host.
        return [
            {key: item[key] for key in ("object_id", "name", "kind", "description", "aliases")}
            for item in objects
        ]

    def _worker(self, model, generation):
        if generation != self._generation:
            raise RuntimeError("Запрос модели отменён.")
        if model not in dict(MODELS) or model == "filter":
            raise ValueError("Неизвестная локальная модель.")
        with self._state_lock:
            previous, previous_model = self._process, self._model
        if previous and previous_model != model:
            self._discard(previous)
            previous = None
        if not previous or previous.poll() is not None:
            self._start(model, generation)
        with self._state_lock:
            if generation != self._generation or self._process is None:
                raise RuntimeError("Запрос модели отменён.")
            return self._process, self._responses

    def _send(self, process, responses, request):
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            return self._receive(process, responses)
        except Exception:
            self._discard(process)
            raise

    def _prepare(self, process, responses, safe):
        fingerprint = hashlib.sha256(
            json.dumps(safe, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if self._prepared == fingerprint:
            return {"event": "ready", "cached": True, "pid": process.pid}
        response = self._send(process, responses, {"operation": "prepare", "objects": safe})
        if response.get("event") != "ready":
            self._discard(process)
            raise ValueError("Модель не подтвердила подготовку индекса.")
        self._prepared = fingerprint
        logging.info("Model and catalog ready: model=%s pid=%s objects=%s elapsed_ms=%s",
                     self._model, process.pid, len(safe), response.get("elapsed_ms"))
        return response

    def prepare(self, model, objects, *, expected_generation=None):
        generation = self._generation if expected_generation is None else expected_generation
        with self._operation_lock:
            process, responses = self._worker(model, generation)
            return self._prepare(process, responses, self._safe_objects(objects))

    def search(self, model, query, objects, *, expected_generation=None):
        generation = self._generation if expected_generation is None else expected_generation
        with self._operation_lock:
            process, responses = self._worker(model, generation)
            safe = self._safe_objects(objects)
            self._prepare(process, responses, safe)
            response = self._send(process, responses, {"query": query, "objects": safe})
            ids = response["object_ids"]
            if (
                not isinstance(ids, list)
                or len(ids) != len(set(ids))
                or len(ids) > 20
                or not set(ids).issubset({item["object_id"] for item in safe})
                or (safe and not ids)
            ):
                self._discard(process)
                raise ValueError("Модель вернула ID вне доступного каталога.")
            elapsed = float(response["elapsed_ms"])
            if not math.isfinite(elapsed) or elapsed < 0:
                self._discard(process)
                raise ValueError("Модель вернула неверное время поиска.")
            return response
