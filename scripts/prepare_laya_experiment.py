"""Stage pinned Laya weights and the existing frozen fictional E5 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from build_nanojev_notebooks import cell
from train_nanojev_kaggle import rows, validate_inputs

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(worktree):
    source = ROOT / "artifacts/kaggle/training-data"
    output = ROOT / "artifacts/kaggle/laya-data"
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((source / "experiment-input-manifest.json").read_text())
    for name, expected in previous["files"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Frozen Nano-Jev data changed: {name}")
        shutil.copy2(source / name, output / name)
    validate_inputs(rows(output / "prepared-train.jsonl"), rows(output / "prepared-validation.jsonl"))
    checkpoint = worktree / "models/rerank/laya-multi"
    spec = json.loads((checkpoint / "manifest.json").read_text())
    if sha(checkpoint / "model.safetensors") != spec["weight_lfs_sha256"]:
        raise ValueError("Pinned Laya weights changed")
    checkpoint_files = {}
    with ZipFile(output / "laya-multilingual.zip", "w", compression=ZIP_DEFLATED) as archive:
        for name, recorded in spec["files"].items():
            relative = name.removeprefix("laya-multi/")
            actual = sha(checkpoint / relative)
            if actual != recorded["sha256"]:
                raise ValueError(f"Pinned file changed: {relative}")
            archive.write(checkpoint / relative, relative)
            checkpoint_files[relative] = actual
    for name in ("train_laya_kaggle.py", "train_nanojev_kaggle.py"):
        shutil.copy2(ROOT / "scripts" / name, output / name)
    manifest = {"model": spec["spec"], "original_weight_sha256": spec["weight_lfs_sha256"],
                "checkpoint_files": checkpoint_files,
                "initial_model_archive_sha256": sha(output / "laya-multilingual.zip"),
                "files": {name: sha(output / name) for name in
                          ("prepared-train.jsonl", "prepared-validation.jsonl",
                           "train_laya_kaggle.py", "train_nanojev_kaggle.py")},
                "retrieval_revision": previous["retrieval_revision"],
                "train_rows": previous["train_rows"], "validation_rows": previous["validation_rows"],
                "train_positive_injections": previous["train_positive_injections"],
                "evaluation_positive_injections": 0,
                "dependencies": {"laya": "0.3.26", "transformers": "5.6.2",
                                 "safetensors": "0.6.2", "accelerate": "1.15.0"},
                "privacy": "Fictional data only. No personal catalog or test rows."}
    (output / "laya-input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = {"id": "matveyvarfolomeev/launcher-laya-training-v1",
                "title": "Launcher Laya training v1", "licenses": [{"name": "other"}],
                "description": "Private fictional launcher training and synthetic validation. "
                "Data remains copyright of its authors; model files retain their original Apache 2.0 "
                "license notices. No personal catalogs or test examples included."}
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    notebook_path = ROOT / "notebooks/laya-clean-first.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"] = [
        cell("markdown", "# Laya multilingual: чистый старт и контроль переобучения\n\n"
             "Исходные закреплённые веса Laya; те же E5 top-20 и 22 656 обучающих строк, что у Nano-Jev. "
             "Первые 2 272 строки — clean, затем остаток clean/STT в смеси; максимум один проход.\n\n"
             "Каждые 142 обновления измеряется только синтетическая валидация. Сохраняются лучшие "
             "веса по NLL достижимого правильного ответа: сначала среднее внутри group_id, затем "
             "внутри профиля, затем равный вес трём профилям. Остановка после трёх проверок без "
             "улучшения минимум 0.005. Baseline тоже участвует в выборе. Точность basic/difficult/ASR "
             "публикуется отдельно, включая промахи E5. Личный каталог и test остаются локально."),
        cell("markdown", "## Среда\n\nРодной SDK Laya, без генерации текста и замены головы. "
             "AMP FP16 для обучения; параметры и AdamW FP32. Валидация FP32."),
        cell("code", "import os, sys, json, subprocess\nfrom pathlib import Path\n"
             "os.environ['USE_TF'] = '0'\nos.environ['USE_FLAX'] = '0'\n"
             "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n"
             "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', "
             "'laya==0.3.26', 'transformers==5.6.2', 'safetensors==0.6.2', 'accelerate==1.15.0'])\n"
             "import torch\nassert torch.cuda.is_available(), 'Нужен GPU'\n"
             "print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})\n"),
        cell("markdown", "## Обучение\n\nEncoder lr=1e-5, native choice head lr=3e-5; effective batch 16 "
             "через накопление градиентов. Gradient checkpointing уменьшает память. Act/escalate head "
             "не обучается. Сохраняется нативное ограничение 48 токенов описания варианта; "
             "запрос не обрезается, потери метаданных записываются."),
        cell("code", "matches = list(Path('/kaggle/input').rglob('laya-input-manifest.json'))\n"
             "assert len(matches) == 1\ninput_dir = matches[0].parent\n"
             "output_dir = Path('/kaggle/working/results')\n"
             "command = [sys.executable, str(input_dir / 'train_laya_kaggle.py'), "
             "'--input', str(input_dir), '--output', str(output_dir)]\n"
             "with Path('/kaggle/working/training.log').open('w') as log:\n"
             "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
             "    for line in process.stdout:\n        print(line, end='')\n        log.write(line)\n        log.flush()\n"
             "    assert process.wait() == 0, 'См. training.log'\n"),
        cell("markdown", "## Выбранные веса\n\ncheckpoint — лучшие по синтетической валидации; "
             "last-checkpoint — последние. После скачивания нужна локальная оценка на реальном "
             "каталоге. Эти вероятности не откалиброваны; оценка точности не подтверждает CPU latency."),
        cell("code", "result = json.loads((output_dir / 'result.json').read_text())\n"
             "print({key: result[key] for key in ('steps', 'best_step', 'stop_reason', 'elapsed_seconds')})\n"
             "for profile in result['baseline']['metrics']:\n"
             "    before = result['baseline']['metrics'][profile]['accuracy']\n"
             "    after = result['best']['metrics'][profile]['accuracy']\n"
             "    print(f'{profile}: {before:.1%} -> {after:.1%}')\n"),
    ]
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    stage = ROOT / "artifacts/kaggle/notebooks/laya-clean-first"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, stage / notebook_path.name)
    kernel = {"id": "matveyvarfolomeev/launcher-laya-clean-first-v1", "title": "Launcher Laya clean first v1",
              "code_file": notebook_path.name, "language": "python", "kernel_type": "notebook",
              "is_private": True, "enable_gpu": True, "enable_internet": True,
              "dataset_sources": [metadata["id"]], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (stage / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n")
    print(json.dumps({"input": str(output), "kernel": kernel["id"], "manifest": manifest}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    prepare(parser.parse_args().worktree)
