# Reproduction guide

There are three levels of reproduction: verify published evidence, validate/test the source checkout, and run new model training/inference. The first two do not need a GPU or API key. The recorded GPU training is not rerun by CI.

Run commands from the repository root with Python 3.12. The standard-library commands also work without installing the Qt application.

## 1. Recompute public evidence

```console
python scripts/verify_public_results.py
```

Expected: `verified: true`, 11 model variants, 4,400 predictions, 200 commands, 400 calibration representations and 22,656 frozen training rows. The script verifies file hashes and matches predictions to public labels, checks candidates/argmax, recomputes top-1/recall and recalculates the scalar temperature transformation, conditional NLL and ECE.

This verifies saved outputs. It does not reload checkpoints, run new inference, bootstrap the retained confidence intervals or recreate ASR audio. Details and original-source hashes are in [results](../results/README.md).

## 2. Validate datasets

```console
python scripts/validate_benchmark.py data/benchmarks/launcher-basic-v1/synthetic data/benchmarks/launcher-v1/synthetic data/benchmarks/launcher-independent-v1/clean
python scripts/training_dataset.py validate
python scripts/stt_training_dataset.py validate
python scripts/asr_dataset.py validate --dataset data/benchmarks/launcher-asr-v1/synthetic --text-only
python scripts/independent_dataset.py validate --text-only --public-only
```

The ASR profiles have their own transcript/provenance validator; the generic clean-text validator is not interchangeable with it. `--text-only` verifies stored transcripts and acoustic metadata without requiring excluded WAV files.

The public new-catalog audit checks **28 frozen prior files** and reports **19 excluded private prior files**. It checks overlaps against the public corpus and validates the new clean/ASR data. The original all-source evidence remains recorded, but a public checkout cannot independently recheck excluded personal data or raw audio. The full local audit uses `validate --text-only` without `--public-only`, when all frozen sources are available. Subsequent diagnostic artifacts are not silently added to that earlier frozen corpus.

Training validators report missing private guards as `evaluation_unavailable`; public guards remain mandatory. Do not run `build`, `prepare`, `finalize`, `synthesize` or `transcribe` just to verify a release: those commands create data or call services.

## 3. Run the source tests

Windows x64, Python 3.12; PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
$env:QT_QPA_PLATFORM = 'offscreen'
$env:QT_QUICK_BACKEND = 'software'
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_quick.py --ignore=tests/test_ui.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_quick.py tests/test_ui.py
```

The base environment runs dataset, training-contract, model-worker, launch-boundary and Qt/UI checks. Four tensor/gradient tests in `test_e5_training.py` additionally need PyTorch. `test_laya_cpu_weights.py` needs PyTorch, safetensors and the historical native loader on `PYTHONPATH`. These optional skips are reported explicitly; base CI does not claim GPU or native-weight parity.

The [workflow](../.github/workflows/ci.yml) has a standard-library public-evidence job on Linux and the base test suite on Windows.

Headless Qt tests explicitly use the software scene-graph backend, avoiding dependence on the desktop window manager and GPU renderer. This setting applies to tests; the application retains its ordinary rendering configuration. [Qt software adaptation](https://doc.qt.io/qt-6/qtquick-visualcanvas-adaptations-software.html).

The two pytest commands cover every test. Session/QML integration runs in a separate process because the verified PySide6 runtime showed a headless QML initialization crash after the preceding non-UI suite, while the isolated session/UI suite passed. This is a test-process isolation measure; a monolithic headless run is not reported as passing.

## 4. Build a portable E5 training bundle

This is the training path that no longer depends on the author's separate model worktree. It uses the original frozen candidate rows; selection validation is rebuilt over all 177 objects. It does not access the independent test set.

Create a separate ML environment and install PyTorch for your CPU/CUDA platform, plus the recorded E5 preparation dependencies:

```console
python -m pip install transformers==5.12.1 safetensors==0.6.2 huggingface-hub
python -m zipfile -e data/prepared/launcher-choice-v1.zip artifacts/public-training-inputs
python scripts/prepare_e5_source.py
python scripts/prepare_e5_experiment.py --training-inputs artifacts/public-training-inputs --output artifacts/public-e5-input --no-notebook
```

`prepare_e5_source.py` downloads the pinned original E5 revision, validates loading and writes `artifacts/kaggle/e5-source-v1/source.json`. It refuses to replace an existing source snapshot. If it is already present, reuse it. `--model-source PATH` lets bundle preparation use another source snapshot directory.

`prepare_e5_experiment.py` verifies frozen input hashes, checks all 22,656 training rows and 150 validation rows, renders full validation catalogs and packages original weights plus the trainers. It does not submit to Kaggle. `--no-notebook` preserves the historical notebook; a newly built bundle has its own manifest hash.

On a GPU machine, run:

```console
python scripts/train_e5_kaggle.py --input artifacts/public-e5-input --output artifacts/public-e5-run --seed 20261005
```

For a small CPU/GPU wiring check, use another output directory and append `--smoke`. Smoke output is not a full experiment and is excluded from reported results. Use a fresh output directory for each run.

The recorded full runs used Tesla T4 and Torch `2.11.0+cu128`; E5's recorded Transformers version is `5.12.1`. A different library build, device or kernel can change numerical results. Reproducing the inputs and recipe does not guarantee bit-identical weights. The public release does not bundle the trained checkpoints.

## 5. Other training families and historical notebooks

The [notebooks](../notebooks/) preserve original private Kaggle dataset IDs and exact input-manifest hashes. Opening them publicly does not make the old Kaggle datasets accessible. Rebuilt input bundles require a new notebook manifest hash; keep the original protocols as historical records.

| Family | Trainer | Extra prerequisite / recorded protocol |
| --- | --- | --- |
| Laya | [train_laya_kaggle.py](../scripts/train_laya_kaggle.py) | Native `laya` library, original multilingual checkpoint and [Laya input protocol](KAGGLE_LAYA_FINETUNING.md) |
| Kev | [train_kev_kaggle.py](../scripts/train_kev_kaggle.py) | Pinned native Kev code, Qwen base, original adapter/head and [warm-start protocol](KAGGLE_KEV_FINETUNING.md) |
| Nano-Jev | [train_nanojev_kaggle.py](../scripts/train_nanojev_kaggle.py) | Pinned upstream model/native loader and [mixed/clean-first protocol](KAGGLE_NANOJEV_FINETUNING.md) |
| RuModernBERT / RuBERT | [train_modernbert_kaggle.py](../scripts/train_modernbert_kaggle.py) | Pinned original reranker or seeded RuBERT head; [ModernBERT](KAGGLE_MODERNBERT_FINETUNING.md) / [RuBERT](KAGGLE_RUBERT_TINY2_CHOICE_RU.md) |

Some original local evaluators/preparers use `--worktree` or historical machine paths for native loaders and cached models. They are research records, not promised one-command installers. Pinned sources are in [source-models.json](../results/source-models.json). The frozen public archive contains candidate data, not those upstream libraries or weights.

Recreating audio uses external TTS/ASR services; the application defaults to full Whisper Large V3, whereas the older noisy-ASR profile preserves its original Turbo transcripts. Never overwrite the old transcripts to make those conditions look identical.

## 6. Windows demonstration

Install the base environment and run `python -m jev_launcher`. The text filter can run without model weights. Neural profiles require configured local Python environments and checkpoints; see [the launcher guide](LAUNCHER_GUIDE_RU.md) and [application model setup](APPLICATION_MODELS_RU.md). The release's source archive is not a pretrained portable EXE.
