"""Connect the already downloaded checkpoints and existing CPU environments to the GUI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def configure(worktree, decision_python=None, reranker_python=None):
    root = Path(__file__).resolve().parents[1]
    worktree = Path(worktree).resolve()
    decision_python = Path(decision_python or worktree / ".venv-models/Scripts/python.exe")
    reranker_python = Path(
        reranker_python or root / "artifacts/kaggle/.venv-modernbert/Scripts/python.exe"
    )
    saved = json.loads(
        (root / "artifacts/kaggle/final-model-comparison-v1/comparison.json").read_text("utf-8")
    )
    runs = {
        "laya": "laya-clean-first-v1",
        "modernbert": "modernbert-clean-first-v1",
    }
    profiles = {}
    for key, run in runs.items():
        checkpoint = root / "artifacts/kaggle/outputs" / run / "results/checkpoint"
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint)
        profiles[key] = {
            "python": str(reranker_python if key == "modernbert" else decision_python),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": saved["training"][run]["checkpoint_sha256"],
        }
    export_dir = root / "artifacts/kaggle/e5-evaluation-v1/selected-export"
    export = json.loads((export_dir / "export.json").read_text("utf-8"))
    profiles["e5_trained"] = {
        "python": str(decision_python),
        "onnx": str(export_dir / "model.onnx"), "onnx_sha256": export["onnx_sha256"],
        "tokenizer": str(root / "artifacts/kaggle/outputs/e5-small-clean-first-v1"
                         "/results/checkpoint/tokenizer.json"),
        "tokenizer_sha256": export["tokenizer_sha256"],
        "checkpoint_sha256": export["checkpoint_sha256"],
        "index_directory": str(root / "models/e5-trained-launcher-index"),
    }
    calibration_path = root / "artifacts/benchmarks/laya-temperature-v1/temperature.json"
    with calibration_path.open("rb") as stream:
        calibration_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    calibration = json.loads(calibration_path.read_text("utf-8"))
    if calibration["checkpoint_sha256"] != profiles["laya"]["checkpoint_sha256"]:
        raise ValueError("Калибровка не соответствует выбранным весам Laya.")
    profiles["laya"].update(calibration=str(calibration_path), calibration_sha256=calibration_sha)
    for path in [
        decision_python, reranker_python, worktree / "src/jev_launcher/semantic.py",
        Path(profiles["e5_trained"]["onnx"]), Path(profiles["e5_trained"]["tokenizer"]),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    result = {
        "schema_version": 1,
        "default_model": "laya",
        "worker": str(root / "scripts/run_launcher_model.py"),
        "worktree": str(worktree),
        "e5_directory": str(worktree / "models/e5"),
        "index_directory": str(root / "models/e5-launcher-index"),
        "profiles": profiles,
    }
    target = root / "models/launcher-runtime.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-worktree", type=Path, required=True)
    parser.add_argument("--decision-python", type=Path)
    parser.add_argument("--reranker-python", type=Path)
    args = parser.parse_args()
    print(configure(args.model_worktree, args.decision_python, args.reranker_python))
