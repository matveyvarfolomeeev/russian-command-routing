"""Stage a private E5 run: frozen train negatives, synthetic full-catalog stopping."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

if __package__:
    from .build_nanojev_notebooks import cell
    from .train_modernbert_kaggle import sha
    from .train_nanojev_kaggle import rows, validate_inputs
else:
    from build_nanojev_notebooks import cell
    from train_modernbert_kaggle import sha
    from train_nanojev_kaggle import rows, validate_inputs

ROOT = Path(__file__).resolve().parents[1]


def passage_text(item):
    """The card format used to freeze the original retrieval inputs."""
    kinds = {"app": "Приложение", "folder": "Папка", "file": "Файл"}
    fields = [item["name"], kinds.get(item["kind"], item["kind"]),
              item.get("description", ""), ", ".join(item.get("aliases", []))]
    return "passage: " + ". ".join(field.strip() for field in fields if field.strip())


def prepare(source, model_source, output, *, write_notebook=True):
    original = model_source / "original"
    output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((source / "experiment-input-manifest.json").read_text(encoding="utf-8"))
    origin = json.loads((original.parent / "source.json").read_text(encoding="utf-8"))
    for name, expected in frozen["files"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Changed frozen input: {name}")
        shutil.copy2(source / name, output / name)
    data, validation = rows(output / "prepared-train.jsonl"), rows(output / "prepared-validation.jsonl")
    validate_inputs(data, validation)
    assert len(data) == 22656 and len(validation) == 150
    assert Counter(row["source"] for row in data) == {"clean": 5664, "stt": 16992}
    catalogs, full_rows, catalog_sources = {}, [], {}
    for profile in sorted({row["profile"] for row in validation}):
        directory = ROOT / "data/benchmarks" / profile / "synthetic"
        catalog_path = directory / "catalog.json"
        local = rows(directory / "validation.jsonl")
        by_id = {row["id"]: row for row in local}
        objects = sorted(json.loads(catalog_path.read_text(encoding="utf-8"))["objects"], key=lambda item: item["object_id"])
        assert len(objects) == 177
        catalogs[profile] = {"candidate_ids": [item["object_id"] for item in objects],
                             "documents": [passage_text(item).removeprefix("passage: ") for item in objects]}
        catalog_sources[catalog_path.relative_to(ROOT).as_posix()] = sha(catalog_path)
        catalog_sources[(directory / "validation.jsonl").relative_to(ROOT).as_posix()] = sha(directory / "validation.jsonl")
        for row in [row for row in validation if row["profile"] == profile]:
            raw = by_id[row["id"]]
            assert row["query"] == raw["query"] and row["target"] == raw["expected_object_id"]
            assert row["group_id"] == raw["group_id"]
            assert row["target"] in catalogs[profile]["candidate_ids"]
            full_rows.append({key: row[key] for key in ("id", "query", "group_id", "target", "profile")} | {"catalog_key": profile})
    (output / "full-validation.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in full_rows), encoding="utf-8")
    (output / "validation-catalogs.json").write_text(json.dumps(catalogs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint_files = {}
    with ZipFile(output / "e5-original.zip", "w", compression=ZIP_DEFLATED) as archive:
        for name, expected in origin["files"].items():
            assert sha(original / name) == expected, name
            archive.write(original / name, name)
            checkpoint_files[name] = expected
    trainers = ("train_e5_kaggle.py", "train_modernbert_kaggle.py", "train_laya_kaggle.py", "train_nanojev_kaggle.py")
    for name in trainers:
        shutil.copy2(ROOT / "scripts" / name, output / name)
    manifest = {
        "family": "e5", "model": origin, "original_weight_sha256": checkpoint_files["model.safetensors"],
        "initial_model_archive_sha256": sha(output / "e5-original.zip"), "checkpoint_files": checkpoint_files,
        "files": {name: sha(output / name) for name in
                  ("prepared-train.jsonl", "prepared-validation.jsonl", "full-validation.jsonl", "validation-catalogs.json", *trainers)},
        "catalog_sources": catalog_sources,
        "card_renderer_sha256": sha(Path(__file__)),
        "train_rows": 22656, "train_positive_injections": frozen["train_positive_injections"],
        "training": "all E5 encoder weights; original top20 hard negatives; per-query contrastive CE, temperature .01",
        "head_added": False, "validation_rows": 150, "validation_candidates": 177,
        "selection": "full-catalog synthetic150 group/profile NLL; no shortlist target repair or conditional censoring",
        "privacy": "Fictional train and original synthetic validation only; no real/transfer/independent/test uploads",
        "dependencies": {"transformers": "5.12.1", "safetensors": "0.6.2"},
    }
    manifest_path = output / "e5-input-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metadata = {"id": "matveyvarfolomeev/launcher-e5-small-training-v1", "title": "Launcher E5 small training v1",
                "licenses": [{"name": "other"}],
                "description": "Private fictional launcher data; pinned MIT multilingual-e5-small. "
                               "Full-catalog original synthetic validation. No personal catalogs or test rows."}
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not write_notebook:
        print(json.dumps({"manifest_sha256": sha(manifest_path), "output": str(output),
                          "train": len(data), "validation": len(full_rows)}), flush=True)
        return
    notebook_path = ROOT / "notebooks/e5-small-clean-first.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell("markdown", "# E5-small: дообучение поиска приложений\n\n"
             "Сравниваем готовую E5 с одной адаптацией на нашем датасете. "
             "Encoder отдельно читает `query:` и `passage:`, строит mean-pooled L2 embeddings. "
             "Новой головы нет; меняются общие веса encoder. Карточки после обучения нужно индексировать заново."),
        cell("markdown", "## Протокол до просмотра результатов\n\n"
             "22 656 прежних train (5 664 clean + 16 992 STT); первые 2 272 чистые, затем смесь. "
             "Максимум один проход; all weights FP32, lr=1e-5, decay=.01, effective16/micro2, warmup6%, clip1. "
             "Train сравнивает цель с 19 закреплёнными соседями исходной E5: per-query CE по cosine/0.01. "
             "Чужие train-запросы не создают дополнительные ложные negatives.\n\n"
             "Выбор весов: прежние150 synthetic validation, но **весь177-объектный каталог**. "
             "Группы/профили имеют равный вес; NLL каждые142 обновления, patience3/min_delta.005. "
             "Исходные веса тоже eligible. Никакой real, transfer или independent test на Kaggle нет."),
        cell("code", "import os, sys, json, hashlib, subprocess\nfrom pathlib import Path\n"
             "os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')\n"
             "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', "
             "'transformers==5.12.1', 'safetensors==0.6.2'])\n"
             "import torch\nassert torch.cuda.is_available(), 'Нужен GPU Kaggle'\n"
             "print({'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0)})\n"),
        cell("markdown", "## Что проверяет validation\n\n"
             "E5 ранжирует весь каталог; accuracy, recall@20 и MRR считаются по всем177 кандидатам. "
             "Softmax с temperature=.01 нужен для NLL, а не как проверенная вероятность правильного открытия. "
             "Тексты длиннее512 токенов прерывают запуск; metadata, ID и target не добавляются к запросу."),
        cell("code", "matches = list(Path('/kaggle/input').rglob('e5-input-manifest.json'))\n"
             "assert len(matches)==1, f'Ожидался один пакет, найдено {len(matches)}'\n"
             "input_dir = matches[0].parent\n"
             f"assert hashlib.sha256(matches[0].read_bytes()).hexdigest() == '{sha(manifest_path)}'\n"
             "manifest = json.loads(matches[0].read_text())\n"
             "for name, expected in manifest['files'].items():\n"
             "    assert hashlib.sha256((input_dir/name).read_bytes()).hexdigest()==expected, name\n"
             "sys.path.insert(0, str(input_dir))\nfrom types import SimpleNamespace\nfrom train_e5_kaggle import run\n"
             "result = run(SimpleNamespace(input=input_dir, output=Path('/kaggle/working/results'), "
             "seed=20261005, smoke=False))\n"),
        cell("markdown", "## Результат и локальная проверка\n\n"
             "Сохраняем выбранные/последние веса и ledgers отдельно. После выбора — известные800 представлений "
             "с полными каталогами, парные ошибки исходной/новой E5 и CPU-скорость с пересчитанным индексом. "
             "В приложении Laya остаётся основной. Известный test не становится новым слепым подтверждением."),
        cell("code", "print({k:result[k] for k in ('steps','best_step','stop_reason','elapsed_seconds')})\n"
             "for profile, values in result['best']['metrics'].items():\n"
             "    print(profile, values)\n"),
    ]
    for index, item in enumerate(notebook["cells"]):
        item["id"] = f"e5-{index}"
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"e5-cell-{index}", "exec")
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    stage = ROOT / "artifacts/kaggle/notebooks/e5-small-clean-first"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, stage / notebook_path.name)
    kernel = {"id": "matveyvarfolomeev/launcher-e5-small-clean-first-v1", "title": "Launcher E5 small clean first v1",
              "code_file": notebook_path.name, "language": "python", "kernel_type": "notebook", "is_private": True,
              "enable_gpu": True, "enable_internet": True, "dataset_sources": [metadata["id"]],
              "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (stage / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_sha256": sha(manifest_path), "kernel": kernel["id"],
                      "dataset": metadata["id"], "train": len(data), "validation": len(full_rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-inputs", type=Path, default=ROOT / "artifacts/kaggle/training-data")
    parser.add_argument("--model-source", type=Path, default=ROOT / "artifacts/kaggle/e5-source-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kaggle/e5-data")
    parser.add_argument("--no-notebook", action="store_true",
                        help="Stage a portable bundle without rewriting the historical Kaggle notebook")
    args = parser.parse_args()
    prepare(args.training_inputs, args.model_source, args.output, write_notebook=not args.no_notebook)
