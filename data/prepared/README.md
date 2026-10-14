# Frozen choice-model inputs

[launcher-choice-v1.zip](launcher-choice-v1.zip) contains exactly three files:

- `prepared-train.jsonl`: 22,656 fictional training examples, 5,664 clean + 16,992 simulated STT.
- `prepared-validation.jsonl`: 150 synthetic validation examples, 50 basic + 50 difficult + 50 actual-ASR transcripts.
- `experiment-input-manifest.json`: original hashes, revision metadata and preparation rules.

Each row contains query, target, group, 20 frozen candidate IDs and card text. Train targets may be inserted when retrieval misses them: 5 clean and 46 simulated-STT examples. Validation is never repaired. Candidate order is shared by the historical choice-model runs.

The archive contains no weights, external model libraries, personal catalogs, test rows or recordings. It preserves original prepared-file bytes. The initial Nano metadata records historical provenance; the same candidate examples also feed other trainers.

Extract into an ignored directory:

```console
python -m zipfile -e data/prepared/launcher-choice-v1.zip artifacts/public-training-inputs
```

[Reproduction guide](../../docs/REPRODUCIBILITY.md). Authored fictional data is covered by the repository's MIT license.
