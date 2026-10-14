"""Fill the two experiment-template notebooks and stage private Kaggle metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, text):
    result = {"cell_type": kind, "id": hashlib.sha256((kind + text).encode()).hexdigest()[:12],
              "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    source = ROOT / "scripts/train_nanojev_kaggle.py"
    data = ROOT / "artifacts/kaggle/training-data"
    shutil.copy2(source, data / "train_nanojev_kaggle.py")
    manifest_path = data / "experiment-input-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_code_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["initial_model_archive_sha256"] = hashlib.sha256((data / "nanojev-v01.zip").read_bytes()).hexdigest()
    with ZipFile(data / "nanojev-v01.zip") as archive:
        manifest["checkpoint_files"] = {name: hashlib.sha256(archive.read(name)).hexdigest()
                                        for name in archive.namelist()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for schedule, slug in (("mixed", "nanojev-mixed"), ("clean_first", "nanojev-clean-first")):
        path = ROOT / "notebooks" / f"{slug}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        notebook["cells"] = [
            cell("markdown", f"# Nano-Jev v0.1 — {schedule}\n\n"
                 "Проверяем две предложенные схемы подачи примеров: смешивание с первого шага и чистый старт.\n\n"
                 "Одинаковые исходные веса, 22 656 примеров (5 664 clean + 16 992 simulated STT), "
                 "E5 top-20, кандидаты, seed, один проход и 1 416 обновлений весов. "
                 "В clean-first первые 142 обновления используют чистые примеры; "
                 "после них перемешивается оставшийся состав. В обоих запусках каждая строка встречается один раз.\n\n"
                 "Валидация: 50 синтетических примеров каждого из basic, difficult и ASR. "
                 "Профили оцениваются отдельно, test и личные каталоги остаются локально. "
                 "Промахи E5 остаются ошибками всей цепочки.\n\n"
                 "После обучения проверяем checkpoint локально; GPU-время не подтверждает CPU-лимит 3 секунды."),
            cell("markdown", "## Среда\n\nИспользуем GPU Kaggle и заранее закреплённые веса. "
                 "Интернет нужен только для Python-зависимостей."),
            cell("code", "import os, sys, subprocess, json\nfrom pathlib import Path\n"
                 "os.environ['USE_TF'] = '0'\nos.environ['USE_FLAX'] = '0'\n"
                 "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n"
                 "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', "
                 "'transformers==5.6.2', 'safetensors==0.6.2'])\n"
                 "import torch\nassert torch.cuda.is_available(), 'Нужен GPU Kaggle'\n"
                 "print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})\n"),
            cell("markdown", "## Дообучение\n\nВеса encoder и скалярной головы обновляются по cross-entropy выбора. "
                 "Для обучения правильный объект может быть добавлен в shortlist; на валидации "
                 "shortlist не исправляется. Baseline оценивается перед первым обновлением весов."),
            cell("code", "matches = list(Path('/kaggle/input').rglob('experiment-input-manifest.json'))\n"
                 "assert len(matches) == 1, f'Ожидался один пакет данных, найдено {len(matches)}'\n"
                 "input_dir = matches[0].parent\noutput_dir = Path('/kaggle/working/results')\n"
                 f"command = [sys.executable, str(input_dir / 'train_nanojev_kaggle.py'), "
                 f"'--input', str(input_dir), '--output', str(output_dir), '--schedule', '{schedule}']\n"
                 "with Path('/kaggle/working/training.log').open('w') as log:\n"
                 "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
                 "    for line in process.stdout:\n        print(line, end='')\n        log.write(line)\n        log.flush()\n"
                 "    assert process.wait() == 0, 'Обучение завершилось ошибкой; см. training.log'\n"),
            cell("markdown", "## Результаты\n\nСравниваем baseline и checkpoint по каждому профилю. "
                 "Один seed — первоначальный эксперимент; небольшая разница требует повторного подтверждения."),
            cell("code", "result = json.loads((output_dir / 'result.json').read_text())\n"
                 "for profile in result['baseline']:\n"
                 "    before = result['baseline'][profile]['accuracy']\n"
                 "    after = result['trained'][profile]['accuracy']\n"
                 "    print(f'{profile}: {before:.1%} -> {after:.1%}; delta={(after-before)*100:+.1f} pp')\n"
                 "print({'steps': result['steps'], 'elapsed_seconds': result['elapsed_seconds'], "
                 "'checkpoint': str(output_dir / 'checkpoint')})\n"),
        ]
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        stage = ROOT / "artifacts/kaggle/notebooks" / schedule
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, stage / path.name)
        metadata = {"id": f"matveyvarfolomeev/launcher-{slug}-v1", "title": f"Launcher {slug} v1",
                    "code_file": path.name, "language": "python", "kernel_type": "notebook",
                    "is_private": True, "enable_gpu": True, "enable_internet": True,
                    "dataset_sources": ["matveyvarfolomeev/launcher-decision-training-v1"],
                    "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
