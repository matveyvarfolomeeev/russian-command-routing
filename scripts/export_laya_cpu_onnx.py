"""Export selected Laya with its author's exporter and the validated CPU loader."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from train_kev_kaggle import sha

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/kaggle/quantization-v1"


def author():
    source = ARTIFACTS / "author-export_onnx.py"
    spec = importlib.util.spec_from_file_location("laya_author_export", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def main(args):
    os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if args.quantize:
        source = args.output / "laya.onnx"
        if not source.exists():
            raise ValueError("Export FP32 first")
        exporter, exporter_path = author()
        destination = args.output / "laya.int8.onnx"
        if destination.exists():
            raise ValueError("Do not overwrite an existing quantized graph")
        exporter.quantize_model(str(source), str(destination), per_channel=False)
        manifest_path = args.output / "export.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["int8"] = {"file": destination.name, "sha256": sha(destination),
                            "bytes": destination.stat().st_size, "per_channel": False,
                            "operations": ["MatMul"], "calibration_rows": 0,
                            "author_script_sha256": sha(exporter_path)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"stage": "quantized", **manifest["int8"]}), flush=True)
        return
    import evaluate_laya_locally as api

    args.checkpoint = ROOT / "artifacts/kaggle/outputs/laya-clean-first-v1/results/checkpoint"
    args.input = ROOT / "artifacts/kaggle/independent-v1/cpu-parity-input.jsonl"

    def export(model, *_values, **_kwargs):
        import torch

        exporter, exporter_path = author()
        parameters = sum(p.numel() for p in model.parameters())
        model.eval().requires_grad_(False)
        torch.manual_seed(20261006)
        with patch.object(exporter, "Agent", return_value=SimpleNamespace(model=model)), torch.inference_mode():
            exporter.export_to_onnx(str(args.checkpoint), str(args.output / "laya.onnx"))
        graph_files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                       for p in args.output.glob("laya.onnx*") if p.is_file()}
        manifest = {"source_checkpoint_sha256": sha(args.checkpoint / "model.safetensors"),
                    "author_script_url": "https://github.com/NandhaKishorM/laya/blob/main/scripts/export_onnx.py",
                    "author_script_sha256": sha(exporter_path), "parameters": parameters,
                    "torch": torch.__version__, "source_input_format": "native Laya, unchanged",
                    "fp32": graph_files, "optimizer_updates": 0}
        (args.output / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"stage": "export_complete", "files": graph_files}), flush=True)
        return {"export": manifest}

    with patch.object(api, "evaluate", side_effect=export):
        api.score(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path,
                        default=Path("C:/Users/matva/.codex/worktrees/c58b/jev_pet"))
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "laya-onnx")
    parser.add_argument("--quantize", action="store_true")
    main(parser.parse_args())
