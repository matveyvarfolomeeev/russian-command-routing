"""JSON-line CPU worker using the same pinned loaders and cards as the experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def choice_scores(logits, temperature=1.0):
    """Return probabilities and raw-logit order, even when sharp softmax rounds to one."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Температура должна быть положительным конечным числом.")
    if not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError("Модель вернула некорректные оценки.")
    maximum = max(logits)
    weights = [math.exp((value - maximum) / temperature) for value in logits]
    total = math.fsum(weights)
    return ([value / total for value in weights],
            sorted(range(len(logits)), key=lambda index: -logits[index]))


class Runtime:
    def __init__(self, config, key):
        self.config, self.profile, self.key = config, config["profiles"][key], key
        self.model = self.tokenizer = None
        self.torch = None
        self.temperature = 1.0
        self.calibration_candidate_count = None
        sys.path.insert(0, str(Path(config["worktree"]) / "src"))
        from benchmark_choice_quantization import configure_cpu
        from jev_launcher.semantic import E5Search

        self.policy = configure_cpu(SimpleNamespace(
            threads=6, performance_cores=os.name == "nt", no_e5_spin=True
        ))
        import onnxruntime as ort

        original_options = ort.SessionOptions

        def options():
            value = original_options()
            value.add_session_config_entry("session.intra_op.allow_spinning", "0")
            value.add_session_config_entry("session.inter_op.allow_spinning", "0")
            return value

        self.weights_sha = None
        if key == "e5_trained":
            from tokenizers import Tokenizer

            for name in ("onnx", "tokenizer"):
                if file_sha(self.profile[name]) != self.profile[f"{name}_sha256"]:
                    raise ValueError("Файлы дообученной E5 изменились.")
            self.weights_sha = self.profile["checkpoint_sha256"]
            # A weight-specific directory prevents reusing embeddings from the original E5.
            self.e5 = E5Search(Path(self.profile["index_directory"]) /
                               self.profile["onnx_sha256"], offline=True)
            session_options = options()
            session_options.intra_op_num_threads, session_options.inter_op_num_threads = 4, 1
            self.e5._session = ort.InferenceSession(
                self.profile["onnx"], sess_options=session_options, providers=["CPUExecutionProvider"]
            )
            self.e5._tokenizer = Tokenizer.from_file(self.profile["tokenizer"])
            self.e5._tokenizer.enable_truncation(max_length=512)
            self.e5._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            return
        self.e5 = E5Search(config["e5_directory"], offline=True)
        with patch.object(ort, "SessionOptions", side_effect=options):
            self.e5._load()
        self.e5.cache_dir = Path(config["index_directory"])
        if key == "e5":
            return
        import torch

        self.torch = torch
        torch.set_num_threads(6)
        checkpoint = Path(self.profile["checkpoint"])
        with (checkpoint / "model.safetensors").open("rb") as stream:
            self.weights_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if self.weights_sha != self.profile["checkpoint_sha256"]:
            raise ValueError("Веса изменились относительно выбранного эксперимента.")
        if key == "laya" and self.profile.get("calibration"):
            calibration_path = self.profile["calibration"]
            if file_sha(calibration_path) != self.profile["calibration_sha256"]:
                raise ValueError("Файл калибровки Laya изменился.")
            calibration = json.loads(Path(calibration_path).read_text("utf-8"))
            if (calibration["checkpoint_sha256"] != self.weights_sha
                    or calibration["retriever"] != "original_e5"
                    or calibration["question_type"] != "choice"
                    or calibration["candidate_count"] != 20):
                raise ValueError("Калибровка не соответствует весам и кандидатам Laya.")
            self.temperature = float(calibration["temperature"])
            choice_scores([0.0], self.temperature)
            self.calibration_candidate_count = calibration["candidate_count"]
        if key.startswith("laya"):
            from train_laya_kaggle import load_model
        elif key == "nano":
            from continue_nanojev_kaggle import load_model
        else:
            from train_modernbert_kaggle import load_model
        self.model, self.tokenizer = load_model(checkpoint, "cpu")
        self.model.requires_grad_(False)
        if key == "laya_int8":
            from benchmark_choice_quantization import quantize_choice_linear

            self.model, _ = quantize_choice_linear(self.model, torch, "laya")
            state = Path(self.profile["state"])
            with state.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != self.profile["state_sha256"]:
                    raise ValueError("Сохранённый INT8 state изменился.")
            self.model.load_state_dict(
                torch.load(state, weights_only=True, mmap=True), strict=True, assign=True
            )

    def _logits(self, query, documents):
        from train_nanojev_kaggle import INSTRUCTION

        if self.key.startswith("laya"):
            from laya import common
            from train_laya_kaggle import batch, encode_all

            if len(documents) == 20:
                encoded, _ = encode_all(self.tokenizer, [
                    {"id": "launcher", "query": query, "documents": documents}
                ])
            else:
                question = {
                    "t": "choice", "ins": INSTRUCTION,
                    "crit": {f"o{i}": text for i, text in enumerate(documents)},
                }
                ids, markers, options, state = common.build_sequence(
                    self.tokenizer, query, question, max_len=8192, head_max_len=8192,
                    return_stats=True, return_truncation_stats=True,
                )
                if state["truncated"] or len(markers) != len(documents):
                    raise ValueError("Запрос слишком длинный для выбранной модели.")
                if options["options_distinct"] != len(documents):
                    raise ValueError("Не все варианты попали во вход модели.")
                encoded = [{"ids": ids, "markers": markers, "qtype": common.QTYPES["choice"]}]
            logits, _ = self.model(**batch(self.tokenizer, encoded, [0], "cpu"))
            logits = logits[0].float()
        else:
            if self.key == "nano":
                first = [f"question: {INSTRUCTION} option: {text}" for text in documents]
                second, limit = [query] * len(documents), 512
            else:
                first, second, limit = [query] * len(documents), documents, 2048
            encoded = self.tokenizer(first, second, padding=False, truncation=False)
            if max(map(len, encoded["input_ids"])) > limit:
                raise ValueError("Запрос или описания слишком длинные для выбранной модели.")
            values = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
            logits = self.model(**values).logits.reshape(len(documents)).float()
        if not self.torch.isfinite(logits).all():
            raise ValueError("Модель вернула некорректные оценки.")
        return logits.cpu().tolist()

    def search(self, request):
        from jev_launcher.semantic import passage_text

        query, objects = request["query"], request["objects"]
        began = time.perf_counter()
        ranked = self.e5.rank(query, objects, limit=20)
        candidates = list(ranked["results"])
        if ranked["query_truncated"]:
            raise ValueError("Запрос слишком длинный. Сократите его и повторите поиск.")
        if not candidates:
            return {"object_ids": [], "elapsed_ms": (time.perf_counter() - began) * 1000}
        temperature = None
        if self.key in {"e5", "e5_trained"}:
            scores = [item["score"] for item in candidates]
            order = list(range(len(candidates)))
        else:
            # A stable shuffle prevents the E5 position from becoming a choice shortcut.
            if "candidate_ids" in request:  # Known-input parity checks, never used by the GUI.
                by_id = {item["object_id"]: item for item in candidates}
                order = request["candidate_ids"]
                if len(order) != len(by_id) or set(order) != set(by_id):
                    raise ValueError("Контрольный shortlist не совпадает с E5.")
                candidates = [by_id[value] for value in order]
            else:
                random.Random("launcher-choice-v1:" + query.strip().casefold()).shuffle(candidates)
            documents = [passage_text(item).removeprefix("passage: ") for item in candidates]
            with self.torch.inference_mode():
                logits = self._logits(query, documents)
            temperature = (self.temperature
                           if len(candidates) == self.calibration_candidate_count else 1.0)
            scores, order = choice_scores(logits, temperature)
        if len(scores) != len(candidates) or not all(math.isfinite(x) for x in scores):
            raise ValueError("Некорректное число оценок модели.")
        return {
            "object_ids": [candidates[index]["object_id"] for index in order],
            "candidate_ids": [item["object_id"] for item in candidates],
            "scores": scores,
            "elapsed_ms": (time.perf_counter() - began) * 1000,
            "checkpoint_sha256": self.weights_sha,
            "score_type": ("cosine_similarity" if self.key in {"e5", "e5_trained"}
                           else "within_shortlist_softmax"),
            "temperature": temperature,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = sys.stdout
    runtime = None

    def send(value):
        protocol.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        protocol.flush()

    # Loading completes before the first handshake; the GUI still needs a prepared index.
    try:
        with redirect_stdout(sys.stderr):
            runtime = Runtime(config, args.model)
        send({"event": "loaded"})
    except Exception as exc:
        send({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            # Native libraries may print startup information; keep it outside the protocol.
            with redirect_stdout(sys.stderr):
                if request.get("operation") == "prepare":
                    began = time.perf_counter()
                    objects = request["objects"]
                    if objects:
                        # Prepare catalog embeddings and the real inference path, once per index.
                        runtime.search({"query": "открыть приложение", "objects": objects})
                    result = {"event": "ready", "objects_count": len(objects),
                              "elapsed_ms": (time.perf_counter() - began) * 1000}
                else:
                    result = runtime.search(request)
            send(result)
        except Exception as exc:
            send({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    raise SystemExit(main())
