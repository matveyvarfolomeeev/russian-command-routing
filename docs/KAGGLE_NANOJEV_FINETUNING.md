# Nano-Jev: mixed versus clean-first training

First experiment selected by the user on 2026-10-05. The broader training pool still includes Laya multilingual and Kev-0.8B.

Two private notebooks:
- [Mixed](https://www.kaggle.com/code/matveyvarfolomeev/launcher-nanojev-mixed-v1)
- [Clean first](https://www.kaggle.com/code/matveyvarfolomeev/launcher-nanojev-clean-first-v1)

## Controlled difference

Both runs start from `sdmlai/nano-jev@b9f855e8253f177d162b50807c99d80507ec1e61`, weight SHA-256 `aa62d33d0c6cd886609c910b245267d956bc20a5c1e95a687c9cc8ddd4261ace`. The author's scalar scoring head and encoder are retained. The prompt matches the existing launcher adapter.

Each run sees all 5,664 clean and 16,992 simulated-STT examples once: 22,656 rows, 1,416 optimizer steps, effective batch 16, seed 20261005. Mixed shuffles the complete list. Clean-first presents 2,272 clean rows first (142 steps), then shuffles the remaining clean and distorted examples. Each row has exactly the same candidates in both runs; only presentation order changes.

GPU micro-batch is four queries, accumulated over four batches per optimizer step. FP32, AdamW lr 3e-5, weight decay 0.01, six-percent learning-rate warmup, linear decay, gradient clipping at 1.0. No early stopping or checkpoint selection from test.

## Inputs and evaluation

The private [Kaggle dataset](https://www.kaggle.com/datasets/matveyvarfolomeev/launcher-decision-training-v1) contains fictional training data, pinned model files, prepared E5 top-20 examples and 150 synthetic validation rows (50 basic, 50 difficult, 50 ASR). Kaggle expands ZIPs into directories, so the runtime checks individual checkpoint file hashes as well as input/trainer hashes.

The training shortlist includes the labelled positive: E5 needed this repair in five clean and 46 STT cases. Evaluation does not insert missing targets. Synthetic validation E5 recall is 100 percent basic and 94 percent difficult/ASR; these are ceilings for this frozen shortlist, not model accuracy.

Personal catalogs and all test rows remain local. Local evaluation uses 300 validation rows: 50 per profile for each of the real and synthetic catalogs. Each such subset contains 13 related scenario groups. Profiles and catalogs are reported separately; paired uncertainty resamples entire groups, not individual rows. One training seed does not measure training-seed variability. The existing test data has been examined in prior benchmarking, so it cannot provide a new independent confirmation.

Baseline is evaluated before any weight update. Outputs include input/sample hashes, the training ledger, per-row validation predictions, result JSON and a reloadable checkpoint. After fine-tuning the choice scores remain uncalibrated.

Local E5-only validation provides an additional reference for whether the trained selection stage helps: real catalog basic 36/50, difficult 30/50, ASR 26/50; synthetic catalog basic 48/50, difficult 38/50, ASR 39/50. All 300 shortlist sets were checked against the frozen experiment inputs. Evidence is in `artifacts/kaggle/local-validation/e5-reference-v1` and its saved evaluation script.

## Verification and status

The five schedule/leakage tests pass. The full 22,656-row plans have matching sample multisets. A 32-row local CPU preflight completed two optimizer steps, changed the checkpoint weights, and the saved model reloaded and produced finite logits.

Earlier CPU runs with eight-query micro-batches crashed in native Torch training attention/dropout. Switching to eager attention did not resolve this. Reducing the local micro-batch to one query completed training; gradients were accumulated to preserve an effective batch of 16. These failures are retained under `artifacts/kaggle/smoke-*.log`. User apps were not closed.

Both private Kaggle notebooks completed successfully on Tesla T4, Torch 2.11.0+cu128. Each ledger contains all 1,416 updates with exactly the expected per-step clean/STT exposure. Input controls, initial weights, sample multisets, settings and baseline choices match. Both downloaded checkpoint hashes match the GPU results, and both models reloaded for all 300 local validation rows. GPU and CPU choices agree on all 150 synthetic rows for each model. Local batched accuracy evaluation does not qualify the 3000-ms complete-query latency requirement.

## Completed pilot — 2026-10-05

Accuracy below uses the same 50 validation rows in each cell. E5 is a separate retrieval-only reference; the other three columns use the identical frozen E5 top-20 shortlist.

| Catalog / profile | Original Nano-Jev | E5 only | Mixed | Clean first |
| --- | ---: | ---: | ---: | ---: |
| Real / basic | 19/50 (38%) | 36/50 (72%) | 43/50 (86%) | 44/50 (88%) |
| Real / difficult | 16/50 (32%) | 30/50 (60%) | 35/50 (70%) | 35/50 (70%) |
| Real / ASR | 10/50 (20%) | 26/50 (52%) | 31/50 (62%) | 34/50 (68%) |
| Synthetic / basic | 4/50 (8%) | 48/50 (96%) | 42/50 (84%) | 42/50 (84%) |
| Synthetic / difficult | 5/50 (10%) | 38/50 (76%) | 37/50 (74%) | 36/50 (72%) |
| Synthetic / ASR | 2/50 (4%) | 39/50 (78%) | 37/50 (74%) | 38/50 (76%) |

Fine-tuning substantially improves the original Nano-Jev on all six subsets. Both trained variants beat E5 on the real catalog in this pilot; E5 remains better on every synthetic subset. This difference between catalogs is preserved, not averaged away.

Clean-first has a small real-catalog advantage: one additional basic success and three net ASR successes; difficult accuracy is identical. In real ASR it fixes four mixed-model errors and loses one mixed-model success. The paired group bootstrap 95-percent interval for the clean-first-minus-mixed ASR difference is 0 to 12.24 percentage points. It includes zero. Synthetic differences are one example each in opposite directions. A single seed and 13 validation groups do not establish a general curriculum advantage or its transfer to Laya/Kev.

Trainer wall time was 2,987.67 seconds mixed and 3,360.37 seconds clean-first, including encoding, baseline and final validation. These are observed run durations, not a controlled speed comparison or GPU quota accounting.

Checkpoint SHA-256:
- Mixed: `ff828dfaf9594a07583d3bed4a5d12548051ca8e32bde71e8f4544be7a58accf`
- Clean first: `0df8d201084de2051c1beb6b7e7ec6d6891b3fd8f86318f4ef482d7286404095`

Artifacts stored on drive D:
- `artifacts/kaggle/outputs/{mixed-v1,clean-first-v1}/results`: downloaded checkpoints, complete GPU ledgers and predictions.
- `artifacts/kaggle/local-validation/{baseline-v1,e5-reference-v1,mixed-v1,clean-first-v1}`: local results and predictions.
- `artifacts/kaggle/comparison.json`: executed audit, separate metrics and paired group uncertainty.

Reproduce the final audit after downloading and evaluating both checkpoints:

```powershell
.venv\Scripts\python.exe -X utf8 scripts\compare_nanojev_runs.py --mixed artifacts\kaggle\outputs\mixed-v1\results --clean-first artifacts\kaggle\outputs\clean-first-v1\results
```

After reviewing this pilot, the user selected clean-first for Laya with overfitting control; see [Laya protocol](KAGGLE_LAYA_FINETUNING.md). Kev's later learner-selected clean-first/validation-stopping adaptation also completed; see [Kev protocol](KAGGLE_KEV_FINETUNING.md). The [new independent comparison](INDEPENDENT_MODEL_EVALUATION_RU.md) includes current results, a diagnostic of original Nano's poor transfer and CPU speed probes. No trained model has been connected to GUI selection.

## Files and commands

- `scripts/prepare_kaggle_training.py`: compact source bundle and private dataset metadata.
- `scripts/prepare_nanojev_experiment.py`: pinned E5 candidates and initial checkpoint archive.
- `scripts/train_nanojev_kaggle.py`: shared supervised trainer, baseline and validation.
- `scripts/build_nanojev_notebooks.py`: notebook cells and Kaggle metadata.
- `scripts/evaluate_nanojev_locally.py`: local validation and checkpoint evaluation.
- `scripts/compare_nanojev_runs.py`: audit complete ledgers and hashes, then compare paired validation predictions with group bootstrap intervals.
- `notebooks/nanojev-mixed.ipynb`, `notebooks/nanojev-clean-first.ipynb`: reviewable notebook sources.

Kaggle CLI lives in `artifacts/kaggle/.venv` on D:. The launcher GUI and Windows opening boundary are independent of this training experiment.

On 2026-10-07 the learner authorized a separate bounded continuation from the trained clean-first weights: one maximum extra pass, mixed clean/STT, lr1e-5 and ordinary-validation NLL stopping with baseline eligibility. The first-pass evidence above remains unchanged. See [continuation protocol and live state](KAGGLE_NANOJEV_CONTINUATION_RU.md).
