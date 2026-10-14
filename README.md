# Russian Command Routing

Russian text and voice commands mapped to applications, files and folders. The project compares E5 retrieval, cross-encoder reranking and decision models, with task-specific fine-tuning and a Windows application.

[Architecture](docs/ARCHITECTURE.md) · [Experiments](docs/EXPERIMENTS.md) · [Setup and training](docs/REPRODUCIBILITY.md)

## Results

Test: 150 commands on a new catalog of 177 objects. Clean text and ASR are paired versions of the same commands; ASR comes from noisy TTS transcribed by Whisper Large V3.

| Model | Clean top-1 | ASR top-1 |
| --- | ---: | ---: |
| E5 original | 143/150 | 136/150 |
| E5 fine-tuned * | 145/150 | 143/150 |
| E5 → Laya original | 40/150 | 45/150 |
| E5 → Laya fine-tuned | 141/150 | 141/150 |
| E5 → Kev original | 143/150 | 143/150 |
| E5 → Kev fine-tuned | **149/150** | **146/150** |
| E5 → RuModernBERT fine-tuned * | 144/150 | 143/150 |

The initial comparison fixed the checkpoints before evaluating the new catalog. Rows marked `*` were evaluated later, after the test set had been inspected. Labels were reviewed by the author; each training recipe has one recorded seed.

## Training

The training pool has 22,656 synthetic examples: 5,664 clean and 16,992 simulated STT variants, built from 708 queries on eight catalogs. E5 and Laya use encoder fine-tuning; Kev trains rank-16 LoRA adapters and a pointer head over a frozen Qwen backbone. Nano-Jev, RuModernBERT and RuBERT-tiny2 provide additional comparisons.

[Training plans and logs](results/training-runs.json), [notebooks](notebooks/) and [experiment results](docs/EXPERIMENTS.md) include checkpoint selection, training-order comparisons, quantization and calibration.

## Check the results

Python 3.12+, no model download:

```console
git clone https://github.com/matveyvarfolomeeev/russian-command-routing.git
cd russian-command-routing
python scripts/verify_public_results.py
```

The script recalculates metrics from 4,400 saved predictions, verifies data hashes and checks Laya calibration. [Model training and source tests](docs/REPRODUCIBILITY.md) run separately.

## Code and data

- [data](data/): benchmark datasets, training examples and frozen top-20 inputs.
- [scripts](scripts/): training, evaluation and dataset checks.
- [results](results/): predictions, run logs, checkpoint hashes and model revisions.
- [notebooks](notebooks/): Kaggle execution protocols.
- [src/jev_launcher](src/jev_launcher/): Windows application, catalog and model worker.

The application passes catalog metadata to the model and validates the returned `object_id` before opening its saved target. It supports text search and Whisper voice input. Model weights and raw audio are not included in the release.

[MIT license](LICENSE) · [Upstream models and runtimes](THIRD_PARTY_NOTICES.md)
