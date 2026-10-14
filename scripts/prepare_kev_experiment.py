"""Stage the pinned Kev base/adapter/code and fictional frozen training rows."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from build_nanojev_notebooks import cell
from train_kev_kaggle import sha
from train_nanojev_kaggle import rows, validate_inputs

ROOT = Path(__file__).resolve().parents[1]


def prepare(worktree):
    source = ROOT / "artifacts/kaggle/training-data"
    output = ROOT / "artifacts/kaggle/kev-data"
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((source / "experiment-input-manifest.json").read_text())
    for name, expected in previous["files"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Frozen training input changed: {name}")
        shutil.copy2(source / name, output / name)
    validate_inputs(rows(output / "prepared-train.jsonl"), rows(output / "prepared-validation.jsonl"))
    original = worktree / "models/rerank/kev-08"
    spec = json.loads((original / "manifest.json").read_text())
    checkpoint_files = {}
    for name, recorded in spec["files"].items():
        relative = name.removeprefix("kev-08/")
        actual = sha(original / relative)
        if actual != recorded["sha256"]:
            raise ValueError(f"Pinned file changed: {relative}")
        checkpoint_files[relative] = actual
    archive_path, manifest_path = output / "kev-original.zip", output / "kev-input-manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    reuse = (archive_path.exists() and old.get("checkpoint_files") == checkpoint_files
             and sha(archive_path) == old.get("initial_model_archive_sha256"))
    if not reuse:
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for relative in checkpoint_files:
                archive.write(original / relative, relative)
    scripts = ("train_kev_kaggle.py", "train_laya_kaggle.py", "train_nanojev_kaggle.py")
    for name in scripts:
        shutil.copy2(ROOT / "scripts" / name, output / name)
    manifest = {"model": spec["spec"], "checkpoint_files": checkpoint_files,
                "initial_model_archive_sha256": sha(output / "kev-original.zip"),
                "files": {name: sha(output / name) for name in
                          ("prepared-train.jsonl", "prepared-validation.jsonl", *scripts)},
                "retrieval_revision": previous["retrieval_revision"],
                "train_rows": previous["train_rows"], "validation_rows": previous["validation_rows"],
                "train_positive_injections": previous["train_positive_injections"],
                "evaluation_positive_injections": 0,
                "dependencies": {"peft": "0.19.0", "transformers": "5.6.2", "safetensors": "0.6.2",
                                 "accelerate": "1.15.0", "pydantic": "2.12.5"},
                "privacy": "Fictional train and synthetic validation only; no personal catalog or test rows."}
    (output / "kev-input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = {"id": "matveyvarfolomeev/launcher-kev-training-v1", "title": "Launcher Kev training v1",
                "licenses": [{"name": "other"}], "description": "Private fictional launcher training and "
                "synthetic validation. Original pinned Kev, Qwen base and native code retain their license notices. "
                "No personal catalog or test queries included."}
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    notebook_path = ROOT / "notebooks/kev-clean-first.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"] = [
        cell("markdown", "# Kev-0.8B: чистый старт и остановка по валидации\n\n"
             "Продолжаем обучение закреплённых исходных LoRA и pointer head. Базовая Qwen заморожена. "
             "Те же 22 656 строк и E5 top-20: первые 2 272 clean, затем остаток clean/STT; максимум один проход. "
             "Это один запуск адаптации, без сравнения двух curriculum. Личный каталог и test остаются локально."),
        cell("markdown", "## Среда\n\nNative Kev code закреплён в архиве; исходные base, adapter и head "
             "проверяются по SHA-256. LoRA не сливается с базой. Training, weights и validation FP32: "
             "на T4 FP16 backward переполнился до первого обновления; его результаты не используются."),
        cell("code", "import os, sys, json, subprocess\nfrom pathlib import Path\n"
             "os.environ['USE_TF'] = '0'\nos.environ['USE_FLAX'] = '0'\n"
             "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n"
             "# Kaggle ships torchao 0.10; PEFT 0.19 rejects it even for plain FP32 LoRA.\n"
             "# This experiment uses no torchao quantization. Remove only that optional package.\n"
             "subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'])\n"
             "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', "
             "'peft==0.19.0', 'transformers==5.6.2', 'safetensors==0.6.2', "
             "'accelerate==1.15.0', 'pydantic==2.12.5'])\n"
             "import torch\nassert torch.cuda.is_available(), 'Нужен GPU'\n"
             "print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})\n"),
        cell("markdown", "## Обучение\n\nLoRA и pointer head: AdamW lr=2e-5, effective batch 16 "
             "(microbatch 1 × accumulation 16), warmup 6%, gradient clipping 1.0. Полный запрос и 20 кандидатов "
             "сохраняются; overflow останавливает запуск. Проверка каждые 142 updates: group-balanced conditional "
             "NLL с равным весом basic/difficult/ASR. min_delta=0.005, patience=3; baseline может остаться лучшей."),
        cell("code", "matches = list(Path('/kaggle/input').rglob('kev-input-manifest.json'))\n"
             "assert len(matches) == 1\ninput_dir = matches[0].parent\n"
             "import hashlib\n"
             "assert hashlib.sha256((input_dir / 'kev-input-manifest.json').read_bytes()).hexdigest() == "
             f"'{sha(output / 'kev-input-manifest.json')}', 'Unexpected input dataset version'\n"
             "output_dir = Path('/kaggle/working/results')\n"
             "command = [sys.executable, str(input_dir / 'train_kev_kaggle.py'), "
             "'--input', str(input_dir), '--output', str(output_dir)]\n"
             "with Path('/kaggle/working/training.log').open('w') as log:\n"
             "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
             "    for line in process.stdout:\n        print(line, end='')\n        log.write(line)\n        log.flush()\n"
             "    assert process.wait() == 0, 'См. training.log'\n"),
        cell("markdown", "## Выбранные веса\n\ncheckpoint — выбранные LoRA/head; last-checkpoint — последние. "
             "После скачивания нужна парная CPU-проверка. Сохранён temperature=1.0: confidence не откалиброван. "
             "Прочие typed-задачи Kev, GUI и CPU latency этим запуском не проверены."),
        cell("code", "result = json.loads((output_dir / 'result.json').read_text())\n"
             "print({key: result[key] for key in ('steps', 'best_step', 'stop_reason', 'elapsed_seconds')})\n"
             "for profile in result['baseline']['metrics']:\n"
             "    before = result['baseline']['metrics'][profile]['accuracy']\n"
             "    after = result['best']['metrics'][profile]['accuracy']\n"
             "    print(f'{profile}: {before:.1%} -> {after:.1%}')\n"),
    ]
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    stage = ROOT / "artifacts/kaggle/notebooks/kev-clean-first"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, stage / notebook_path.name)
    kernel = {"id": "matveyvarfolomeev/launcher-kev-clean-first-v1", "title": "Launcher Kev clean first v1",
              "code_file": notebook_path.name, "language": "python", "kernel_type": "notebook",
              "is_private": True, "enable_gpu": True, "enable_internet": True,
              "dataset_sources": [metadata["id"]], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (stage / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n")
    print(json.dumps({"input": str(output), "kernel": kernel["id"], "pinned_files": len(checkpoint_files),
                      "train_rows": manifest["train_rows"], "validation_rows": manifest["validation_rows"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    prepare(parser.parse_args().worktree)
