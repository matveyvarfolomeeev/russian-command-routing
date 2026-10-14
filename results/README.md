# Results

| File | Content |
| --- | --- |
| [independent-v1/summary.json](independent-v1/summary.json) | Validation/test counts, recall@20 and paired comparisons for 11 variants |
| [independent-v1/predictions.jsonl.gz](independent-v1/predictions.jsonl.gz) | 4,400 predictions, candidate IDs and available probabilities |
| [training-runs.json](training-runs.json) | Eight training plans, selection losses and checkpoint hashes |
| [training](training/) | Per-update training and validation logs |
| [calibration](calibration/) | Laya temperature, metrics and original/fitted probabilities |
| [source-models.json](source-models.json) | Model and native-code revisions |
| [manifest.json](manifest.json) | SHA-256 file checksums |

Run `python scripts/verify_public_results.py` from the repository root to check the stored outputs. Each model has 400 representations of 200 paired clean/ASR commands. Summary entries distinguish the initial frozen-checkpoint evaluation from later runs on the known suite.

`export_public_results.py --registry PATH` rebuilds this directory from the original local run artifacts and experiment registry.
