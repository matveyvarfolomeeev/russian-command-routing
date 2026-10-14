# Public experiment evidence

This compact export contains fictional catalog IDs and numeric/model metadata. Personal launch targets, credentials, raw audio and weights remain outside Git.

| File | Contents |
| --- | --- |
| [independent-v1/summary.json](independent-v1/summary.json) | Four separate clean/ASR, validation/test profiles for 11 variants; initial paired comparisons |
| [independent-v1/predictions.jsonl.gz](independent-v1/predictions.jsonl.gz) | 4,400 predictions, candidate lists, correctness and available scores |
| [training-runs.json](training-runs.json) | Eight plans, stopping decisions, selected/last hashes and validation results |
| [training](training/) | Compressed per-update training and available validation ledgers |
| [calibration/laya.json](calibration/laya.json) | Temperature and original/fitted metrics |
| [calibration/predictions.jsonl.gz](calibration/predictions.jsonl.gz) | 100 fitting and 300 test representations, raw/fitted probabilities |
| [source-models.json](source-models.json) | Pinned upstream model/native-code revisions |
| [manifest.json](manifest.json) | Published-file hashes and relative local-source provenance hashes |

The 400 representations per variant come from 200 commands. `initial_frozen_comparison` and `later_known_suite` identify different evaluation stages. The original Nano's zero result remains in the export.

From the repository root, run `python scripts/verify_public_results.py`. The standard-library verifier checks integrity, dataset labels, candidates, probability argmax and accuracy/recall. It recomputes calibration NLL/ECE and verifies frozen training-input hashes/counts. This is saved-evidence verification, not fresh training or inference. Bootstrap intervals are retained from the original study; the verifier does not resample them.

To export again on the original machine, run `python scripts/export_public_results.py`. That exporter requires ignored local `artifacts/` and does not upload anything. It exports only new-catalog predictions from later evaluations that also contain private-catalog rows locally. Unavailable local provenance is not treated as downloadable public evidence.

The [historical registry](../docs/EXPERIMENT_REGISTRY.json) and longer Russian reports contain local archive paths/checkpoints and describe earlier research snapshots. The published manifest describes this source/evidence release.
