"""Stage a private Russian reranker experiment using the existing frozen data."""

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
    original = ROOT/"artifacts/kaggle/modernbert-source-v1/checkpoint"
    output = ROOT/"artifacts/kaggle/modernbert-data"
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((source/"experiment-input-manifest.json").read_text())
    origin = json.loads((original.parent/"source.json").read_text())
    for name, expected in previous["files"].items():
        if sha(source/name) != expected:
            raise ValueError(f"Frozen data changed: {name}")
        shutil.copy2(source/name, output/name)
    data, validation = rows(output/"prepared-train.jsonl"), rows(output/"prepared-validation.jsonl")
    validate_inputs(data, validation)
    if len(data) != 22656 or len(validation) != 150:
        raise ValueError("Wrong frozen input sizes")
    if Counter(r["source"] for r in data) != {"clean": 5664, "stt": 16992}:
        raise ValueError("Training composition changed")
    checkpoint_files = {}
    with ZipFile(output/"modernbert-original.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(original.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
                checkpoint_files[path.name] = sha(path)
    trainer_files = ("train_modernbert_kaggle.py", "train_laya_kaggle.py", "train_nanojev_kaggle.py")
    for name in trainer_files:
        shutil.copy2(ROOT/"scripts"/name, output/name)
    manifest = {"model": origin, "original_weight_sha256": checkpoint_files["model.safetensors"],
                "initial_model_archive_sha256": sha(output/"modernbert-original.zip"),
                "checkpoint_files": checkpoint_files,
                "files": {name: sha(output/name) for name in
                          ("prepared-train.jsonl", "prepared-validation.jsonl", *trainer_files)},
                "retrieval_revision": previous["retrieval_revision"], "shortlist_size": 20,
                "train_rows": 22656, "validation_rows": 150,
                "train_positive_injections": previous["train_positive_injections"],
                "evaluation_positive_injections": 0,
                "input_format": "native query/card pairs; labels and object IDs excluded",
                "dependencies": {"transformers": "5.12.1", "safetensors": "0.6.2"},
                "privacy": "Fictional training and synthetic validation only; no personal catalogs or test rows"}
    manifest_path = output/"modernbert-input-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    metadata = {"id": "matveyvarfolomeev/launcher-modernbert-training-v1",
                "title": "Launcher Russian ModernBERT training v1", "licenses": [{"name": "other"}],
                "description": "Private fictional launcher training and synthetic validation. "
                "Model originates from ARGA100/ru-reranker-modernbert-small under MIT; "
                "original model card and pinned revision retained. No personal catalogs or test data."}
    (output/"dataset-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf8")
    notebook_path = ROOT/"notebooks/modernbert-clean-first.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf8"))
    notebook["cells"] = [
        cell("markdown", "# Русский ModernBERT: дообучение выбора приложения\n\n"
             "Проверяем компактный русский reranker на нашей задаче: запрос → E5 top-20 → выбранный ID. "
             "Модель уже знает русский; адаптируем выбор приложения, файла или папки.\n\n"
             "Те же 22 656 обучающих строк (5 664 clean, 16 992 STT) и 150 синтетических validation. "
             "Связанные группы не пересекаются. Test и личные каталоги не входят в этот пакет.\n\n"
             "Короткий чистый старт: первые 2 272 строки, затем смесь оставшихся; максимум один проход. "
             "Выбор checkpoint по validation NLL каждые 142 обновления; patience=3, min_delta=0.005. "
             "Исходные веса также участвуют в выборе. Это один исследовательский запуск, не поиск гиперпараметров."),
        cell("markdown", "## Среда\n\nЗакреплённые веса и зависимости; родная скалярная голова, "
             "без генерации текста. FP32. Скорость на GPU не подтверждает скорость CPU."),
        cell("code", "import os, sys, json, hashlib, subprocess\nfrom pathlib import Path\n"
             "os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')\n"
             "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', "
             "'transformers==5.12.1', 'safetensors==0.6.2'])\n"
             "import torch\nassert torch.cuda.is_available(), 'Нужен GPU Kaggle'\n"
             "print({'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0)})\n"),
        cell("markdown", "## Закреплённые данные и обучение\n\n"
             "На входе пары запроса и карточки. Двадцать оценок сравниваются cross-entropy; "
             "target ID используется только для вычисления loss. Порядок кандидатов не исправляется. "
             "Промахи E5 остаются ошибками цепочки и не превращаются в ложные метки для stopping."),
        cell("code", "matches = list(Path('/kaggle/input').rglob('modernbert-input-manifest.json'))\n"
             "assert len(matches)==1, f'Ожидался один пакет данных, найдено {len(matches)}'\n"
             "input_dir = matches[0].parent\n"
             f"assert hashlib.sha256(matches[0].read_bytes()).hexdigest()=='{sha(manifest_path)}', 'Другой пакет данных'\n"
             "output_dir = Path('/kaggle/working/results')\n"
             "command = [sys.executable, str(input_dir/'train_modernbert_kaggle.py'), "
             "'--input', str(input_dir), '--output', str(output_dir)]\n"
             "with Path('/kaggle/working/training.log').open('w') as log:\n"
             "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
             "    for line in process.stdout:\n"
             "        print(line, end='')\n        log.write(line)\n        log.flush()\n"
             "    assert process.wait()==0, 'Ошибка обучения; см. training.log'\n"),
        cell("markdown", "## Результаты\n\n"
             "Сравниваем исходную и выбранную модель по каждому профилю. Если лучший шаг равен нулю, "
             "улучшения дообучением не подтверждены. После фиксации весов измеряем остальные наборы и CPU latency."),
        cell("code", "result = json.loads((output_dir/'result.json').read_text())\n"
             "print({k:result[k] for k in ('steps','best_step','stop_reason','elapsed_seconds')})\n"
             "for profile in result['baseline']['metrics']:\n"
             "    before = result['baseline']['metrics'][profile]['accuracy']\n"
             "    after = result['best']['metrics'][profile]['accuracy']\n"
             "    print(f'{profile}: {before:.1%} → {after:.1%}')\n"),
    ]
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf8")
    stage = ROOT/"artifacts/kaggle/notebooks/modernbert-clean-first"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, stage/notebook_path.name)
    kernel = {"id": "matveyvarfolomeev/launcher-russian-modernbert-clean-first-v1",
              "title": "Launcher Russian ModernBERT clean first v1", "code_file": notebook_path.name,
              "language": "python", "kernel_type": "notebook", "is_private": True,
              "enable_gpu": True, "enable_internet": True, "dataset_sources": [metadata["id"]],
              "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (stage/"kernel-metadata.json").write_text(json.dumps(kernel, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"input": str(output), "manifest_sha256": sha(manifest_path), "kernel": kernel["id"]}))


if __name__ == "__main__":
    prepare()
