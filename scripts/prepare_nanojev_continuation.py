"""Freeze trained Nano weights and the existing data for one private continuation."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from build_nanojev_notebooks import cell
from continue_nanojev_kaggle import sha
from train_nanojev_kaggle import rows, validate_inputs

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    source = ROOT / "artifacts/kaggle/training-data"
    parent_run = ROOT / "artifacts/kaggle/outputs/clean-first-v1/results"
    checkpoint = parent_run / "checkpoint"
    output = ROOT / "artifacts/kaggle/nano-continuation-data"
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((source / "experiment-input-manifest.json").read_text())
    parent = json.loads((parent_run / "result.json").read_text())
    if sha(checkpoint / "model.safetensors") != parent["checkpoint_sha256"]:
        raise ValueError("Previously trained Nano changed")
    for name, expected in previous["files"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Frozen data changed: {name}")
        shutil.copy2(source / name, output / name)
    data, validation = rows(output / "prepared-train.jsonl"), rows(output / "prepared-validation.jsonl")
    validate_inputs(data, validation)
    if len(data) != 22656 or len(validation) != 150:
        raise ValueError("Frozen input sizes changed")
    if Counter(r["source"] for r in data) != {"clean": 5664, "stt": 16992}:
        raise ValueError("Training composition changed")
    if sha(source / "nanojev-v01.zip") != previous["initial_model_archive_sha256"]:
        raise ValueError("Original Nano package changed")
    checkpoint_files = {}
    with ZipFile(output / "nanojev-trained.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "nanojev_config.json", "calibration.json"):
            bundle.write(checkpoint / name, name)
            checkpoint_files[name] = sha(checkpoint / name)
        with ZipFile(source / "nanojev-v01.zip") as original:
            for name, destination in (("native_model.py", "native_model.py"),
                                      ("README.md", "ORIGINAL_MODEL_CARD.md")):
                content = original.read(name)
                if hashlib.sha256(content).hexdigest() != previous["checkpoint_files"][name]:
                    raise ValueError(f"Original source changed: {name}")
                bundle.writestr(destination, content)
                checkpoint_files[destination] = hashlib.sha256(content).hexdigest()
    trainers = ("continue_nanojev_kaggle.py", "train_nanojev_kaggle.py", "train_laya_kaggle.py")
    for name in trainers:
        shutil.copy2(ROOT / "scripts" / name, output / name)
    manifest = {
        "model": previous["model"], "initial_weight_sha256": parent["checkpoint_sha256"],
        "parent_run": "matveyvarfolomeev/launcher-nanojev-clean-first-v1",
        "parent_result_sha256": sha(parent_run / "result.json"), "parent_updates": 1416,
        "optimizer_state_present": False, "initial_model_archive_sha256": sha(output / "nanojev-trained.zip"),
        "checkpoint_files": checkpoint_files,
        "files": {name: sha(output / name) for name in
                  ("prepared-train.jsonl", "prepared-validation.jsonl", *trainers)},
        "retrieval_revision": previous["retrieval_revision"], "shortlist_size": 20,
        "train_rows": 22656, "validation_rows": 150,
        "train_positive_injections": previous["train_positive_injections"],
        "evaluation_positive_injections": 0,
        "dependencies": {"transformers": "5.6.2", "safetensors": "0.6.2"},
        "privacy": "Fictional training and original synthetic validation only; no real catalogs or test rows",
    }
    manifest_path = output / "nano-continuation-input-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    dataset = {"id": "matveyvarfolomeev/launcher-nanojev-continuation-training-v1",
               "title": "Launcher NanoJev continuation training v1", "licenses": [{"name": "other"}],
               "description": "Private continuation from the trained Nano-Jev clean-first weights. "
               "Only frozen fictional training and original synthetic validation; no test or personal data. "
               "Pinned Nano source and original model card included."}
    (output / "dataset-metadata.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    notebook_path = ROOT / "notebooks/nanojev-continuation.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell("markdown", "# Nano-Jev: один дополнительный проход\n\n"
             "Проверяем, улучшится ли уже дообученная Nano clean-first при продолжении с меньшей скоростью обучения. "
             "Веса стартуют из завершённого первого запуска; состояние AdamW там не сохранялось, поэтому оптимизатор новый.\n\n"
             "Те же 22 656 строк, кандидаты и исходная синтетическая validation из 150 строк. "
             "Чистые и шумные строки перемешаны с начала (seed=20261007); чистый старт уже пройден. "
             "FP32, lr=1e-5, effective batch=16, максимум ещё один проход / 1 416 обновлений.\n\n"
             "Validation NLL каждые 142 обновления, patience=3, min_delta=0.005; стартовые веса участвуют в выборе. "
             "Никаких новых данных, поиска гиперпараметров или выбора по третьему каталогу."),
        cell("markdown", "## Среда\n\nИспользуем ту же версию Transformers, что в первом обучении Nano. "
             "GPU-время обучения не является замером задержки CPU."),
        cell("code", "import os, sys, json, hashlib, subprocess\nfrom pathlib import Path\n"
             "os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')\n"
             "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', "
             "'transformers==5.6.2', 'safetensors==0.6.2'])\n"
             "import torch\nassert torch.cuda.is_available(), 'Нужен GPU Kaggle'\n"
             "print({'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0)})\n"),
        cell("markdown", "## Проверка пакета и обучение\n\n"
             "Модель получает прежние пары question/option и query; ID ответа служит только меткой loss. "
             "Промахи E5 на валидации остаются ошибками цепочки, но не дают ложную метку выбора."),
        cell("code", "matches = list(Path('/kaggle/input').rglob('nano-continuation-input-manifest.json'))\n"
             "assert len(matches)==1, f'Ожидался один пакет данных: {len(matches)}'\n"
             "input_dir = matches[0].parent\n"
             f"assert hashlib.sha256(matches[0].read_bytes()).hexdigest()=='{sha(manifest_path)}', 'Другой пакет данных'\n"
             "output_dir = Path('/kaggle/working/results')\n"
             "command = [sys.executable, str(input_dir/'continue_nanojev_kaggle.py'), "
             "'--input', str(input_dir), '--output', str(output_dir)]\n"
             "with Path('/kaggle/working/training.log').open('w') as log:\n"
             "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
             "    for line in process.stdout:\n"
             "        print(line, end='')\n        log.write(line)\n        log.flush()\n"
             "    assert process.wait()==0, 'Ошибка обучения; см. training.log'\n"),
        cell("markdown", "## Выбранные и последние веса\n\n"
             "Сохраняем обе версии и полный журнал. Если best_step=0, стартовые веса оказались лучше "
             "по выбранному критерию. Остальные известные наборы сравним локально после фиксации checkpoint; "
             "это повторная проверка качества, а не новый слепой тест."),
        cell("code", "result = json.loads((output_dir/'result.json').read_text())\n"
             "print({k:result[k] for k in ('steps','best_step','stop_reason','elapsed_seconds')})\n"
             "print({'baseline_nll':result['baseline']['selection_loss'], 'best_nll':result['best']['selection_loss']})\n"
             "for profile in result['baseline']['metrics']:\n"
             "    before = result['baseline']['metrics'][profile]['accuracy']\n"
             "    after = result['best']['metrics'][profile]['accuracy']\n"
             "    print(f'{profile}: {before:.1%} → {after:.1%}')\n"),
    ]
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    stage = ROOT / "artifacts/kaggle/notebooks/nanojev-continuation"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, stage / notebook_path.name)
    kernel = {"id": "matveyvarfolomeev/launcher-nanojev-continuation-v1",
              "title": "Launcher NanoJev continuation v1", "code_file": notebook_path.name,
              "language": "python", "kernel_type": "notebook", "is_private": True,
              "enable_gpu": True, "enable_internet": True, "dataset_sources": [dataset["id"]],
              "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (stage / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"input": str(output), "manifest_sha256": sha(manifest_path), "kernel": kernel["id"]}))


if __name__ == "__main__":
    prepare()
