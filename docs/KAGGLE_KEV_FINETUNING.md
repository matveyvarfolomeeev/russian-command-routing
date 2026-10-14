# Kev-0.8B launcher adaptation

The learner selected one clean-first run with validation-loss stopping. This continues the original Kev LoRA and pointer head; the Qwen base is frozen. It is not a matched curriculum comparison. Nano-Jev's comparison and Laya's adaptation do not establish which order is best for Kev.

## Fixed sources and input

- Kev: `jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f`.
- Qwen: `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Native code: `jaredpalmer/kev@fe64b1274ea7f80d4095866df90666abb03e9cf6`.
- Original adapter SHA: `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a`.
- Original pointer/head metadata SHA: `f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6`.
- Original base shard SHA: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`.

The input package reuses the same 22,656 frozen fictional training rows and E5 top-20 lists as Nano/Laya: 5,664 clean and 16,992 simulated-STT. Ordinary validation contains 150 fictional rows, 50 each for basic/difficult/ASR. Real-catalog evaluation stays local. The 51 training-only positive insertions remain recorded; evaluation has no insertion. Related train/validation groups are disjoint.

Native `SystemOneRequest`/`to_record` constructs the query state and a single choice question. Its criteria are ordinal keys `o0` through `o19` mapped to metadata strings. Native `encode` retains all query/option tokens with a strict 8,192-token state/row budget. The dummy native label is removed from network input. Cross-entropy receives the true position separately; host code maps the selected position to the unchanged candidate ID list.

All 22,656 train and 150 validation inputs passed the actual-tokenizer preflight. Maximum sequence lengths are 1,099 and 640; query maxima are 32 and 28. There are no query or metadata truncations and all 20 option markers remain distinct. Target/provenance perturbations and native label changes leave network inputs invariant. Caller text containing delimiter strings cannot forge another native option boundary. Evidence: `artifacts/kaggle/kev-input-preflight.json`.

## Parameters and stopping

The native text/pointer model has 763,740,480 parameters; 11,347,456 are trainable in 372 LoRA tensors and four pointer tensors. Original adapter and head are loaded through the author's strict architecture/key-coverage checks. LoRA is not merged into the base. Other Kev question types are not trained or validated here.

The ordering and maximum exposure are the same clean-first schedule: first 2,272 clean rows, 142 updates; then the remaining clean/STT rows shuffled without duplicating examples. Maximum is one pass, 1,416 updates. Early stopping may leave some planned rows unused.

AdamW uses lr=2e-5 for LoRA and pointer, weight decay 0.01, linear warmup 6%, gradient norm clipping 1.0. Microbatch one and accumulation 16 give an effective batch of 16 queries. The frozen base, trainable parameters, optimizer, activations and validation are FP32. The initial FP16-autocast attempt overflowed on its first backward and aborted before any optimizer update; its numerical behavior is not used for a completed adaptation. Gradient checkpointing uses the non-reentrant PyTorch path. Non-finite losses/gradients abort rather than count a skipped update as completed. The ledger retains `amp_scale=1.0` for compatibility; the FP32 run uses no GradScaler.

The pinned Qwen config has a nested text dtype set to BF16. The loader explicitly sets both outer and nested config dtype to FP32 and verifies every parameter. This prevents a nominal FP32 run from retaining BF16 base weights. CPU widening of stored BF16 is exact, by placing its 16 bits in the upper half of float32; no rounding or quantization is introduced.

The baseline and every 142 updates are checked using group-balanced conditional NLL: mean within a scenario group, mean across groups within each profile, then equal weight across the three profiles. Only reachable targets contribute to this selection loss; all retrieval misses remain in reported accuracy denominators. An improvement must exceed 0.005. Three consecutive checks without improvement stop the run. The baseline is eligible, and selected and last adapters/heads are saved separately. Temperature is reset to 1.0 for both comparison and saved checkpoints; scores are uncalibrated.

## Evidence state

CPU preflight loaded all original adapter tensors and the head, verified FP32 parameters, and returned finite probabilities for 20 choices. Five focused file-mapping tests pass, including exact F16/BF16 widening, inference equivalence, changed-cache rejection and truncated-file rejection. Existing schedule/group-weighting/stopping tests pass. The seven-cell notebook's code parses and its metadata is private.

The private dataset was uploaded and subsequently reached `ready`. Notebook version 1 was submitted while the dataset was still being prepared; Kaggle omitted the unavailable source and the input assertion stopped execution before training. Version 2 failed while its log service returned empty/500 responses. Version 3 exposed the environment problem: preinstalled torchao 0.10 is incompatible with PEFT 0.19, which checks it even for plain FP32 LoRA. That failure occurs during adapter construction, before a training example. Version 4 removes only this unused quantization package and successfully validates the original model, then aborts on a non-finite gradient norm in the first FP16 backward. Its downloaded empty training ledger confirms zero optimizer updates; its baseline is not a fine-tuning result. Fourteen failure outputs are preserved separately in `artifacts/kaggle/outputs/kev-failed-v4`.

Training numerical precision was changed to FP32 after that observed failure; model sources, data, learning rate, order and stopping remain unchanged. A one-training-example CPU numerical preflight has finite loss 0.090289 and gradient norm 18.40105 before clipping, changes all 376 private trainable tensors, and leaves source files unchanged. This checks the update mechanism, not GPU completion or generalization. Evidence: `artifacts/kaggle/kev-fp32-gradient-preflight/result.json`.

Dataset version 2 is ready and its remotely downloaded manifest matches the staged FP32 package: SHA256 `b9dafdb3b2cd861c7c8b82a35db8d98cec56c9f7bc0438d5cff40aef7fb5a057`. Notebook version 5 checks this exact manifest before execution, preventing accidental use of the earlier FP16 trainer. It completed on 2026-10-06: **1,136 finite optimizer steps**, selected **step 710**, stop reason **validation_patience**. Script time is 36,162.99 seconds (10 h 2 min 43 s), on Tesla T4 / Torch 2.11.0+cu128. All 31 outputs are downloaded; the GPU audit and trained CPU evaluation/audit on the canonical 300 rows passed, as reported below.

The selected checkpoint used 11,360 rows: 3,772 clean and 7,588 simulated-STT. The last checkpoint used 18,176: 4,909 clean and 13,267 STT. The remaining 4,480 planned rows were not used. These counts are replayed against every update in `training.jsonl`, not inferred from the maximum budget.

| Step | Validation NLL | Basic / difficult / ASR, correct out of 50 |
| ---: | ---: | --- |
| 0 | 0.27894043 | 46 / 41 / 41 |
| 142 | 0.22349184 | 50 / 43 / 44 |
| 284 | 0.20673135 | 49 / 45 / 45 |
| 426 | 0.22271050 | 50 / 45 / 45 |
| 568 | 0.18295118 | 50 / 45 / 45 |
| **710** | **0.12146429** | **50 / 46 / 45** |
| 852 | 0.19254208 | 50 / 46 / 45 |
| 994 | 0.19002033 | 50 / 45 / 45 |
| 1,136 | 0.19290215 | 50 / 45 / 45 |

![Actual Kev training losses and ordinary validation](../assets/experiments/kev-learning-curves.png)

Selected adapter SHA256: `8a24b7d1c1af1cab6c9b1b7f79d82967da8ac8a12352cd0c89ec594f8b395c4e`; selected head SHA256: `b15c3eb2032675de25a3cbefbd92aa7b2e60817b1d88839e9a3976805a795837`. Last adapter: `2291720e8639737e690147044552c3c75b1185b4cdc2054d7241ae8bb6135b01`; last head: `6850671eca0a57d3be80e16a333fef3c44d4801c7428984457cdb40ecd02eca2`. The portable adapter config hash is `c7fdcefbace4030fbb0d84218bb4624f92a06c507de9c5cfb671fbe9fd0ccb8e` for both. Evidence: `artifacts/kaggle/kev-gpu-audit.json`.

Steps 710 and 852 have identical accuracy counts but worse NLL at 852. Paired predictions show one changed top-1 ID, incorrect at both steps, and several unchanged errors becoming more confident. For the request for the active project folder rather than its archive, the wrong archive receives 0.999973 at 710 and 0.99999988 at 852; the true folder falls from 2.65e-5 to 1.46e-7. This is measured overconfidence, including in the selected model, not evidence of calibrated scores. The top three per-query weighted NLL increases account for almost all aggregate deterioration. Evidence: `artifacts/kaggle/kev-validation-drift.json`.

Original Kev completed all 300 canonical CPU rows. Real basic/difficult/ASR: **46/50, 42/50, 40/50**; fictional basic/difficult/ASR: **46/50, 41/50, 41/50**. The original model is already stronger than the trained smaller models on the real profiles; improvement from adaptation is not assumed. All 150 fictional top-1 choices match version 4's GPU baseline, with maximum probability difference 8.71e-6. GPU selection NLL on the three fictional profiles is 0.27894043; the CPU result's six-profile NLL 0.23924266 averages a different scope and must not be compared as a discrepancy. Evidence: `artifacts/kaggle/local-validation/kev-baseline-v1/result.json`.

Selected step 710 subsequently completed the same 300-row offline FP32 CPU evaluation and the full audit passed. Real basic/difficult/ASR: **48/50, 45/50, 42/50**; fictional: **50/50, 46/50, 45/50**. Paired with the original, the selected model fixes 2/3/2 real errors and 4/5/4 fictional errors, without losing any original successes on these rows. The real-profile gains are +4/+6/+4 percentage points; their 13-group bootstrap intervals include zero. All 150 fictional top-1 IDs match the selected GPU checkpoint, maximum probability difference **1.33813e-5**. Evidence: `artifacts/kaggle/kev-comparison.json`. These validation results do not replace a fresh catalog test or CPU latency qualification.

Paired by query ID against the E5-only reference on the same real profiles, original Kev fixes 12/13/15 E5 errors and loses 2/1/1 E5 successes. For real ASR, six targets are missing from E5 top-20 and four further queries are Kev choice failures: 40/50 overall, 40/44 conditional accuracy. With this frozen retriever the maximum is 44/50; longer Kev training cannot repair absent candidates. Evidence: `artifacts/kaggle/kev-baseline-e5-comparison.json`. These are validation comparisons, not a tested confidence cascade.

Private notebook: <https://www.kaggle.com/code/matveyvarfolomeev/launcher-kev-clean-first-v1>. Private input: <https://www.kaggle.com/datasets/matveyvarfolomeev/launcher-kev-training-v1>.

## Reproduction and boundaries

`scripts/prepare_kev_experiment.py` verifies the 53 pinned model/code files and stages the frozen data, native sources and reviewable notebook. `scripts/preflight_kev_inputs.py` checks full tokenization and supervision boundaries. `scripts/train_kev_kaggle.py` performs native warm start, training, validation stopping and portable adapter/head saving. `scripts/evaluate_kev_locally.py` performs offline unmerged FP32 evaluation.

On memory-constrained Windows, the local evaluator uses a read-only file map of hash-checked exact FP32 base widening, while the small adapter/head remain ordinary tensors. Safetensors header shapes, dtypes, offsets and total length are checked without reserving a writable whole-file map. The source shard is unchanged. This evaluation is for accuracy; it does not qualify the complete-query 3,000 ms product limit.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\prepare_kev_experiment.py --worktree C:\Users\matva\.codex\worktrees\c58b\jev_pet
# Upload from artifacts/kaggle/kev-data; default is private.
D:\ML\petprodj\jev_pet\artifacts\kaggle\.venv\Scripts\kaggle.exe datasets create -p . -r skip
# Wait for ready and confirm the dataset source before submitting from the main folder.
.\artifacts\kaggle\.venv\Scripts\kaggle.exe kernels push -p artifacts\kaggle\notebooks\kev-clean-first --accelerator NvidiaTeslaT4 --timeout 43200
```

Kev was not frozen among the five checkpoints before the already inspected third-catalog test. A later comparison there is a known benchmark, not fresh independent confirmation for Kev. No test selects Kev's checkpoint. The GUI still uses its existing manual selection flow; trained-model routing, uncertainty handling and latency qualification remain separate work.

Original Kev subsequently completed the unchanged known third-catalog benchmark: basic **48/50**, difficult **47/50**, with all targets present and no input truncation (maximum 816 tokens). E5 is 50/50 and 45/50. On difficult, Kev fixes all five E5 errors and loses three E5 successes; the +4 percentage-point difference has a broad group-bootstrap interval of -12 to +25.86 points across 38 groups. No reliable universal advantage follows from these 50 queries. Hashes, paired inputs, probabilities and counts passed the audit in `artifacts/kaggle/transfer-test-v1/kev-known-test-comparison.json`; the selected counterpart completed later, as recorded below.

All five original-Kev errors are saved in `original-kev-errors.json` within that artifact directory. For example, a request for an approved PDF selects the ZIP source archive of the same project: chosen probability 0.562125, target 0.437095. The target is in the shortlist and its description distinguishes the format, so this is a choice failure. These errors are examined for interpretation, without editing the test or the submitted training protocol.

Selected Kev subsequently completed the same 100-row known test on CPU: **50/50 basic, 50/50 difficult**, fixing the original's two basic and three difficult errors without losing a success. Input and checkpoint hashes, base/precision, all paired prediction IDs, probabilities and recomputed counts passed the audit. Evidence: `artifacts/kaggle/transfer-test-v1/kev-known-test-comparison.json`. Its previous `trained_audit=pending` state is superseded by `passed`; the test remains a known benchmark.

## Newly held-out launcher-independent-v1

After fixing the original and selected checkpoints, both completed all 400 FP32 GPU predictions on the user's new fictional catalog: 200 new commands with paired clean and actual Whisper Large V3 ASR representations. Validation has 50 commands / 10 groups; test has 150 / 30 other target groups. The full source/audio/overlap audit passed against 47 previous files. This checks explicit overlaps and author-reviewed labels, without an independent human annotation review or human-speaker recordings. No new data selects weights, stopping or thresholds.

| Kev state | Validation clean / 50 | Validation ASR / 50 | Test clean / 150 | Test ASR / 150 |
| --- | ---: | ---: | ---: | ---: |
| Original | 47 | 47 | 143 | 143 |
| Selected step 710 | 50 | 50 | 149 | 146 |

E5-only is 42/50, 41/50, 143/150 and 136/150 respectively. The selected model fixes seven original clean-test errors and loses one original success: +4 pp, 30-group bootstrap 95% CI [+0.67,+7.33] pp. On ASR it fixes four and loses one: +2 pp, CI [−0.67,+4.67] pp. The latter interval includes zero. Compared with E5, selected Kev gains +4 clean pp, CI [+0.67,+8], and +6.67 ASR pp, CI [+1.33,+11.33]. The intervals do not include training-seed variation or other catalogs.

Conditional group-balanced test NLL improves from 0.17492 / 0.12401 to 0.01646 / 0.02921, clean / ASR separately. One ASR target is absent from E5 top-20: it is excluded from conditional NLL but remains in the 150-row accuracy denominator. Selected Kev has one clean choice error and four ASR errors, three choice failures plus that retrieval miss. In the missing-target case it assigns the wrong object probability 0.998751; this is confidence within the supplied choices, without evidence that the target exists among them. No confidence threshold follows from this observation.

The completed private independent-evaluation notebook version 5 contains both original and selected Kev; the aggregate reuses five completed Nano/Laya model outputs from version 2 and records their separate origins. Full input/checkpoint/probability/count audit passed. The selected Kev also matches GPU top-1 on 12 CPU inputs fixed before inference; maximum probability difference is 6.32e-6. This is numerical parity on that sample, not a full CPU400 result or a latency qualification. [Russian report with all models and errors](INDEPENDENT_MODEL_EVALUATION_RU.md); machine evidence: `artifacts/kaggle/independent-v1/comparison.json` and `numerical-parity.json`. This test is now examined and must not be presented as fresh confirmation after future tuning on its results.

A later matched warm CPU speed probe used the same fixed 12 inputs for trained Kev and Laya, two measured repetitions each after two warmups. On i7-13700H / Torch 2.9.1+cpu / four Torch threads, unmerged FP32 Kev with 20 choices has full E5-to-ID median **4.502 s**, sample p95 **4.651 s**, maximum **4.784 s**, and **24/24 over 3 s**. Laya is **1.009 / 1.091 / 1.106 s**, 0/24 over 3 s. All measured top-1 IDs match the full GPU results; no optimizer or quantization is used. This excludes ASR/loading/index initialization and does not qualify Laya on the full benchmark. Old original-Kev INT8 top10 timing, median 1.376 s / maximum 3.346 s, is a different protocol. Exact rows and probe scope are in `artifacts/kaggle/independent-v1/speed-probe/kev/probe.json` and the Russian report; current FP32 Kev exceeds the existing 3-second limit in this sample.
