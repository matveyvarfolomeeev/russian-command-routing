"""Stage one RuBERT-tiny2 choice experiment using frozen fictional launcher data."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from build_nanojev_notebooks import cell
from train_modernbert_kaggle import sha
from train_nanojev_kaggle import rows, validate_inputs

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    source = ROOT/"artifacts/kaggle/training-data"
    original = ROOT/"artifacts/kaggle/rubert-source-v1/checkpoint"
    output = ROOT/"artifacts/kaggle/rubert-data"
    output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((source/"experiment-input-manifest.json").read_text())
    origin = json.loads((original.parent/"source.json").read_text())
    for name, expected in frozen["files"].items():
        if sha(source/name) != expected:
            raise ValueError(f"Frozen data changed: {name}")
        shutil.copy2(source/name, output/name)
    data, validation = rows(output/"prepared-train.jsonl"), rows(output/"prepared-validation.jsonl")
    validate_inputs(data, validation)
    assert len(data) == 22656 and len(validation) == 150
    assert Counter(r["source"] for r in data) == {"clean": 5664, "stt": 16992}
    assert sha(original/"model.safetensors") == origin["initial_choice_weight_sha256"]
    files = {}
    with ZipFile(output/"rubert-original.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(original.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
                files[path.name] = sha(path)
    trainers = ("train_modernbert_kaggle.py", "train_laya_kaggle.py", "train_nanojev_kaggle.py")
    for name in trainers:
        shutil.copy2(ROOT/"scripts"/name, output/name)
    manifest = {
        "family": "rubert", "model": origin,
        "original_weight_sha256": files["model.safetensors"],
        "initial_model_archive_sha256": sha(output/"rubert-original.zip"),
        "checkpoint_files": files,
        "files": {name: sha(output/name) for name in
                  ("prepared-train.jsonl", "prepared-validation.jsonl", *trainers)},
        "retrieval_revision": frozen["retrieval_revision"], "shortlist_size": 20,
        "train_rows": 22656, "validation_rows": 150,
        "train_positive_injections": frozen["train_positive_injections"],
        "evaluation_positive_injections": 0,
        "input_format": "native BERT query/card pairs with segment IDs; labels/IDs excluded",
        "head": "one scalar per pair; global softmax and listwise cross-entropy across twenty choices",
        "training": "all encoder weights plus newly initialized scalar classifier",
        "dependencies": {"transformers": "5.12.1", "safetensors": "0.6.2"},
        "privacy": "Fictional train and synthetic validation only; no personal catalogs or test data",
    }
    manifest_path = output/"rubert-input-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    metadata = {
        "id": "matveyvarfolomeev/launcher-rubert-tiny2-choice-training-v1",
        "title": "Launcher RuBERT tiny2 choice training v1",
        "licenses": [{"name": "other"}],
        "description": "Private frozen fictional launcher data and synthetic validation. "
                       "Pinned MIT cointegrated/rubert-tiny2 with one seeded scalar head. "
                       "No personal catalogs, known transfer data or test rows.",
    }
    (output/"dataset-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf8")
    notebook_path = ROOT/"notebooks/rubert-tiny2-choice.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf8"))
    notebook["cells"] = [
        cell("markdown", "# RuBERT-tiny2: обучение головы выбора\n\n"
             "Проверяем русский трёхслойный encoder с новой скалярной головой: "
             "запрос → E5 top-20 → выбранный ID. Готового reranker здесь нет: "
             "голова создана с seed 20261007, исходные encoder-веса сохранены точно.\n\n"
             "Цель: измерить качество на сложном STT и локальную скорость после обучения. "
             "Laya остаётся основной моделью приложения. Предположение о скорости проверяется на CPU."),
        cell("markdown", "## Протокол до просмотра результатов\n\n"
             "Те же 22 656 train: 5 664 clean и 16 992 STT-искажения. "
             "Для выбора весов — прежние 150 synthetic validation, разделённые по группам.\n\n"
             "Первые 2 272 train-строки чистые, затем смесь остальных; максимум один проход. "
             "FP32, AdamW lr=2e-5, effective batch=16, micro=2, warmup=6%, weight decay=.01. "
             "Validation NLL каждые 142 обновления, patience=3, min_delta=.005. "
             "Исходные веса с новой головой также участвуют в выборе.\n\n"
             "Личные каталоги и test не загружаются. Известные 800 представлений будут "
             "проверены локально после выбора checkpoint; они не будут новым слепым тестом."),
        cell("code", "import os, sys, json, hashlib, subprocess\nfrom pathlib import Path\n"
             "os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')\n"
             "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', "
             "'transformers==5.12.1', 'safetensors==0.6.2'])\n"
             "import torch\nassert torch.cuda.is_available(), 'Нужен GPU Kaggle'\n"
             "print({'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0)})\n"),
        cell("markdown", "## Пары и динамический выбор\n\n"
             "BERT получает запрос и одну карточку с token_type_ids 0/1. "
             "Голова выдаёт одно число для каждой пары. Двадцать чисел сравниваются "
             "через cross-entropy. Target ID нужен только для индекса правильного варианта, "
             "без класса под каждое приложение. Порядок кандидатов закреплён исходными данными.\n\n"
             "Валидационные промахи E5 сохраняются в accuracy цепочки. "
             "Условный NLL для выбора весов считается только там, где цель есть среди кандидатов."),
        cell("code", "matches = list(Path('/kaggle/input').rglob('rubert-input-manifest.json'))\n"
             "assert len(matches)==1, f'Ожидался один пакет, найдено {len(matches)}'\n"
             "input_dir = matches[0].parent\n"
             f"assert hashlib.sha256(matches[0].read_bytes()).hexdigest() == '{sha(manifest_path)}'\n"
             "manifest = json.loads(matches[0].read_text())\n"
             "for name, expected in manifest['files'].items():\n"
             "    assert hashlib.sha256((input_dir/name).read_bytes()).hexdigest() == expected, name\n"
             "sys.path.insert(0, str(input_dir))\n"
             "from types import SimpleNamespace\nfrom train_modernbert_kaggle import run\n"
             "result = run(SimpleNamespace(input=input_dir, output=Path('/kaggle/working/results'), "
             "seed=20261005, family='rubert', smoke=False))\n"),
        cell("markdown", "## Результаты и дальнейшая проверка\n\n"
             "Выбранные и последние веса сохраняются отдельно. Журналы фиксируют реальные "
             "обновления, состав clean/STT и каждую проверку NLL. Больший training progress "
             "сам по себе не означает лучшие веса. Локально проверим все SHA, GPU/CPU parity, "
             "10 сценариев и два полных замера E5 → 20 кандидатов → ID."),
        cell("code", "print({k:result[k] for k in ('steps','best_step','stop_reason','elapsed_seconds')})\n"
             "for profile in result['best']['metrics']:\n"
             "    before = result['baseline']['metrics'][profile]['accuracy']\n"
             "    after = result['best']['metrics'][profile]['accuracy']\n"
             "    print(f'{profile}: {before:.1%} → {after:.1%}')\n"),
    ]
    for index, item in enumerate(notebook["cells"]):
        item["id"] = f"rubert-{index}"
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"rubert-cell-{index}", "exec")
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf8")
    stage = ROOT/"artifacts/kaggle/notebooks/rubert-tiny2-choice"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, stage/notebook_path.name)
    kernel = {
        "id": "matveyvarfolomeev/launcher-rubert-tiny2-choice-v1",
        "title": "Launcher RuBERT tiny2 choice v1", "code_file": notebook_path.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [metadata["id"]], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (stage/"kernel-metadata.json").write_text(json.dumps(kernel, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"input": str(output), "manifest_sha256": sha(manifest_path),
                      "kernel": kernel["id"]}), flush=True)


if __name__ == "__main__":
    prepare()
