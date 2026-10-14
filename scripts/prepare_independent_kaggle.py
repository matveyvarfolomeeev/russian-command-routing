"""Stage a private inference-only notebook; do not change any training dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from build_nanojev_notebooks import cell
from train_kev_kaggle import sha

ROOT = Path(__file__).resolve().parents[1]


def main(kev_only=False):
    source = ROOT / "artifacts/kaggle/independent-v1"
    output = ROOT / "artifacts/kaggle/independent-data"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "input-manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        if sha(source / filename) != expected:
            raise ValueError("Frozen independent input changed")
        shutil.copy2(source / filename, output / filename)
    scripts = ("evaluate_decision_kaggle.py", "train_nanojev_kaggle.py", "train_laya_kaggle.py", "train_kev_kaggle.py")
    for filename in scripts:
        shutil.copy2(ROOT / "scripts" / filename, output / filename)
        manifest["files"][filename] = sha(output / filename)
    (output / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "dataset-metadata.json").write_text(json.dumps({
        "id": "matveyvarfolomeev/launcher-independent-evaluation-v1", "title": "Launcher independent evaluation v1",
        "licenses": [{"name": "other"}], "description": "Private fictional clean/ASR evaluation. "
        "Previously selected checkpoints only; no training, checkpoint selection, calibration or personal catalog."}, indent=2) + "\n")
    notebook_path = ROOT / "notebooks/independent-evaluation.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"] = [
        cell("markdown", "# Проверка закреплённых моделей на launcher-independent-v1\n\n"
             "200 новых команд × clean/ASR. Validation 50 и test 150 показаны отдельно. "
             "E5 уже сформировал одинаковые top-20; целевые кандидаты не добавляются. "
             "Это только inference: веса, stopping и пороги по новым данным не меняются. "
             "Исходные и выбранные Nano/Laya/Kev загружаются из закреплённых приватных источников."),
        cell("code", "import os, sys, subprocess, json, hashlib\nfrom pathlib import Path\n"
             "os.environ['USE_TF'] = '0'\nos.environ['USE_FLAX'] = '0'\n"
             "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n"
             "subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'])\n"
             "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', "
             "'laya==0.3.26', 'peft==0.19.0', 'transformers==5.6.2', 'safetensors==0.6.2', "
             "'accelerate==1.15.0', 'pydantic==2.12.5'])\n"),
        cell("code", "matches = list(Path('/kaggle/input').rglob('fixed-checkpoints.json'))\n"
             "assert len(matches) == 1, 'Evaluation dataset is missing or ambiguous'\n"
             "input_dir = matches[0].parent\n"
             "assert hashlib.sha256((input_dir / 'input-manifest.json').read_bytes()).hexdigest() == "
             f"'{sha(output / 'input-manifest.json')}', 'Unexpected evaluation version'\n"
             "output_dir = Path('/kaggle/working/results')\n"
             "command = [sys.executable, str(input_dir / 'evaluate_decision_kaggle.py'), "
             "'--input', str(input_dir), '--output', str(output_dir)]\n"
             + ("command += ['--models', 'original-kev', 'trained-kev']\n" if kev_only else "") +
             "with Path('/kaggle/working/evaluation.log').open('w') as log:\n"
             "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
             "    for line in process.stdout:\n        print(line, end='')\n        log.write(line)\n        log.flush()\n"
             "    assert process.wait() == 0, 'Inspect evaluation.log'\n"),
        cell("markdown", "## Проверка результата\n\nВсе строки остаются в знаменателях. "
             "ASR — шумная TTS-речь после настоящего STT; человеческих голосов здесь нет. "
             "Вероятности не откалиброваны. GPU inference не измеряет скорость локального CPU. "
             "После скачивания проводится аудит SHA-256, ID, вероятностей, метрик и парных различий."),
        cell("code", "completion = json.loads((output_dir / 'completion.json').read_text())\n"
             f"assert len(completion['completed_models']) == {2 if kev_only else 7} and completion['optimizer_updates'] == 0\n"
             "print(completion)\n"),
    ]
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    stage = ROOT / "artifacts/kaggle/notebooks/independent-evaluation"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, stage / notebook_path.name)
    kernel = {"id": "matveyvarfolomeev/launcher-independent-models-v1", "title": "Launcher independent models v1",
              "code_file": notebook_path.name, "language": "python", "kernel_type": "notebook", "is_private": True,
              "enable_gpu": True, "enable_internet": True, "competition_sources": [], "model_sources": [],
              "dataset_sources": [f"matveyvarfolomeev/{name}" for name in ("launcher-independent-evaluation-v1",
                  "launcher-decision-training-v1", "launcher-laya-training-v1", "launcher-kev-training-v1")],
              "kernel_sources": [f"matveyvarfolomeev/{name}" for name in ("launcher-nanojev-mixed-v1",
                  "launcher-nanojev-clean-first-v1", "launcher-laya-clean-first-v1", "launcher-kev-clean-first-v1")]}
    if kev_only:
        kernel["dataset_sources"] = ["matveyvarfolomeev/launcher-independent-evaluation-v1",
                                     "matveyvarfolomeev/launcher-kev-training-v1"]
        kernel["kernel_sources"] = ["matveyvarfolomeev/launcher-kev-clean-first-v1"]
        notebook["cells"][0] = cell("markdown", "# Повтор только Kev на закреплённых новых входах\n\n"
            "Пять Nano/Laya расчётов уже завершены в версии 2 и сохранены локально. "
            "Повторяем original Kev и выбранный шаг710 после исправления имени базового shard в метаданных отчёта. "
            "Входы, веса и FP32 inference неизменны; optimizer не создаётся. "
            "Подключены только два нужных датасета и завершённый training notebook Kev.")
        notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        shutil.copy2(notebook_path, stage / notebook_path.name)
    (stage / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n")
    print(json.dumps({"staged": True, "manifest_sha256": sha(output / "input-manifest.json"),
                      "private": True, "optimizer_updates": 0}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kev-only", action="store_true", help="Reuse the already completed five Nano/Laya results")
    main(parser.parse_args().kev_only)
