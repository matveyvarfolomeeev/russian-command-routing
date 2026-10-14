# Setup and training

Use Python 3.12. Stored-data checks need only the standard library.

## Data checks

```console
python scripts/verify_public_results.py
python scripts/training_dataset.py validate
python scripts/stt_training_dataset.py validate
python scripts/asr_dataset.py validate --dataset data/benchmarks/launcher-asr-v1/synthetic --text-only
python scripts/independent_dataset.py validate --text-only --public-only
```

The first command checks 4,400 predictions across 11 variants, calibration probabilities and frozen training-input hashes. It recalculates saved metrics, rather than running inference. The overlap audit checks 28 public prior files and reports 19 excluded private files. `--text-only` uses stored transcripts without raw audio.

## Source tests

Windows x64, PowerShell:

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

Qt session/UI checks run in a separate process because a combined headless run crashed during QML initialization. Together the commands cover all tests. Four E5 tensor checks need PyTorch; the file-backed Laya weight check also needs safetensors and the native loader. Base CI reports these skips.

## E5 training

Use a separate environment with PyTorch for your CUDA/CPU platform:

```console
python -m pip install transformers==5.12.1 safetensors==0.6.2 huggingface-hub
python -m zipfile -e data/prepared/launcher-choice-v1.zip artifacts/public-training-inputs
python scripts/prepare_e5_source.py
python scripts/prepare_e5_experiment.py --training-inputs artifacts/public-training-inputs --output artifacts/public-e5-input --no-notebook
python scripts/train_e5_kaggle.py --input artifacts/public-e5-input --output artifacts/public-e5-run --seed 20261005
```

The source downloader pins E5 and refuses to replace an existing source directory; reuse it if already present. Bundle preparation checks frozen inputs and builds full-catalog validation. Training writes selected/last weights and logs to the output directory. Append `--smoke` for a small wiring check, using another output directory.

Recorded runs used Tesla T4, Torch 2.11.0+cu128 and, for E5, Transformers 5.12.1. Other builds or devices can change numerical results.

## Other models and application

Laya and Kev trainers require their pinned native libraries and original checkpoints. Nano requires its native loader; RuModernBERT/RuBERT use Transformers. [Source revisions](../results/source-models.json) and [training plans](../results/training-runs.json) specify the inputs and hyperparameters.

The Kaggle notebooks retain original private input IDs and manifest hashes. Rebuilt bundles need their own manifest hashes. Some local evaluators require `--worktree` and cached weights. The release includes source, datasets and recorded results; weights and raw audio are separate.

After the base install, `python -m jev_launcher` starts the Windows application. The text filter works without weights. Neural profiles use `models/launcher-runtime.json`, prepared with `scripts/configure_model_runtime.py` for local checkpoints and Python environments.
