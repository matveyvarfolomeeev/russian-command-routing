# Results, evidence and limits

On a new 177-object catalog, fine-tuned Kev selected the correct object for 149/150 clean test commands and 146/150 ASR commands. Laya improved strongly over its original checkpoint. E5 was substantially faster and its later fine-tuned version was competitive on the known suite.

These are results of this authored benchmark. They do not establish accuracy on arbitrary computers or live human speech.

## New-catalog results

`launcher-independent-v1` contains 200 authored commands, with paired clean/ASR representations. Validation50 and test150 have disjoint target groups. ASR comes from neural TTS plus mild noise transcribed by Whisper Large V3; 161/200 normalized transcripts match clean text and 39 differ. All 200 ASR intentions were judged preserved by the author.

| Variant | Clean validation / 50 | ASR validation / 50 | Clean test / 150 | ASR test / 150 |
| --- | ---: | ---: | ---: | ---: |
| E5 original | 42 | 41 | 143 | 136 |
| Nano original | 0 | 0 | 0 | 0 |
| Nano mixed | 37 | 36 | 128 | 127 |
| Nano clean-first | 38 | 38 | 128 | 125 |
| Laya original | 16 | 14 | 40 | 45 |
| Laya fine-tuned, step 568 | 47 | 48 | 141 | 141 |
| Kev original | 47 | 47 | 143 | 143 |
| Kev fine-tuned, step 710 | 50 | 50 | **149** | **146** |
| E5 fine-tuned, step 142 * | 48 | 48 | 145 | 143 |
| RuModernBERT fine-tuned, step 426 * | 48 | 47 | 144 | 143 |
| RuBERT-tiny2 + choice head, step 142 * | 32 | 35 | 76 | 90 |

`*` Later adaptations: this suite was already inspected. Their checkpoint selection still uses the original synthetic validation, but this evaluation is descriptive. Initial Nano/Laya/Kev checkpoints were fixed before the new-catalog run.

The original E5 shortlist covers 150/150 clean and 149/150 ASR test targets. Fine-tuned Kev makes one clean error and four ASR errors; one ASR error is forced by a missing retrieval target. Fine-tuned E5's own top-20 covers all 150 test targets in each representation.

The original Nano's zero result was reproduced on all 400 CPU representations, with source/checkpoint hashes and GPU/CPU top-1 agreement checked. It is retained as an anomalous baseline, with no unsupported causal explanation. [Original detailed evaluation](INDEPENDENT_MODEL_EVALUATION_RU.md).

## Paired comparisons

The initial comparison resamples entire target groups 10,000 times. Each test profile has 30 groups. Differences are percentage points, with group bootstrap 95% intervals.

| Comparison | Clean test | ASR test |
| --- | --- | --- |
| Fine-tuned Kev − original E5 | +4.00 `[+0.67, +8.00]` | +6.67 `[+1.33, +11.33]` |
| Fine-tuned Kev − original Kev | +4.00 `[+0.67, +7.33]` | +2.00 `[−0.67, +4.67]` |
| Fine-tuned Laya − original E5 | −1.33 `[−6.00, +3.33]` | +3.33 `[−1.33, +8.00]` |
| Nano clean-first − Nano mixed | 0.00 `[−3.33, +4.00]` | −1.33 `[−4.67, +2.00]` |

Kev's ASR improvement over its original checkpoint does not exclude zero. A consistent Nano curriculum benefit or Laya advantage over E5 is not established. The intervals do not account for training-seed variability, authors or future catalogs. [Machine-readable comparison](../results/independent-v1/summary.json).

## Training records

Eight completed runs are published. The 22,656-example pool is shared where applicable; checkpoint selection and completed exposure are separate.

| Run | Selected update | Completed updates | Selection |
| --- | ---: | ---: | --- |
| Nano mixed / clean-first | 1,416 each | 1,416 each | Final one-pass weights |
| Laya | 568 | 994 | Group/profile conditional NLL |
| Kev | 710 | 1,136 | Group/profile conditional NLL |
| RuModernBERT | 426 | 852 | Group/profile conditional NLL |
| Nano continuation | 0 | 426 | Initial checkpoint remained best |
| RuBERT-tiny2 | 142 | 568 | Group/profile conditional NLL |
| E5 | 142 | 568 | Group/profile full-catalog NLL |

E5's step142 corresponds to 2,272 clean examples, before simulated STT. Its gain on STT evaluation cannot be attributed to training on those corruption variants. Nano continuation used a fresh optimizer because the first run had not preserved optimizer state.

Failed numerical or inference attempts remain in [the historical registry](EXPERIMENT_REGISTRY.json); they are not counted as completed successful training. [Public plans and ledgers](../results/training-runs.json) distinguish selected and last hashes.

## Calibration

Laya temperature `T=2.4938812077079957` is fitted separately from weight training. A later application-order replay uses 100 known validation representations / 10 groups for fitting and 300 test representations / 30 other groups for evaluation. These represent 50 and 150 paired commands, respectively.

| Test metric | Original, T=1 | Fitted temperature |
| --- | ---: | ---: |
| Top-1 correct | 283/300 | 283/300 |
| Conditional NLL, reachable targets | 0.404409 | 0.241047 |
| Top-label ECE, 10 equal-width bins | 4.5867% | 3.1101% |
| Conditional multiclass Brier | 0.093573 | 0.095755 |
| Top-label Brier | 0.048939 | 0.052220 |

The application-order replay differs from the initial Laya table, which has 282/300 correct test representations. Both records are retained. NLL/ECE improved, Brier worsened, and old noisy-STT NLL also worsened. The public verifier recomputes the transformation and reported NLL/ECE. [Calibration evidence](../results/calibration/laya.json), [original protocol](LAYA_TEMPERATURE_RU.md).

## Ablations and negative results

- **Retriever substitution:** on the known ten-scenario suite, trained E5 → fixed Laya changed equal-scenario accuracy from 88.15% to 87.95% and old noisy-STT from 80% to 78%. Retrieval coverage improved, but conditional selection degraded. [Report](E5_LAYA_PIPELINE_RU.md).
- **Standalone E5 adaptation:** known-suite equal-scenario accuracy increased from 80.5% to 89.4%; old noisy-STT two-catalog validation mean increased from 65% to 82%. Synthetic validation influenced checkpoint selection. [Report](KAGGLE_E5_FINETUNING_RU.md).
- **Dynamic INT8 Laya:** old noisy-STT validation mean dropped from 80% to 64%. Repeated CPU runs did not establish a consistent sub-500 ms pipeline. [Report](LAYA_QUANTIZATION_RU.md).
- **RuBERT-tiny2:** the scalar-head adaptation was fast but weak: 25% on old noisy-STT validation and 76/150 clean / 90/150 ASR on the known new-catalog test. [Report](KAGGLE_RUBERT_TINY2_CHOICE_RU.md).
- **Metadata:** removing descriptions reduced retrieval quality. Adding descriptions from Windows/package metadata helped some queries and harmed others. [Ablation](E5_METADATA_ABLATION_RU.md), [application metadata](APP_DESCRIPTIONS_RU.md).

## CPU latency

Recorded CPU: Intel Core i7-13700H, Windows. Timers cover warm selection, with ASR, loading and initial indexing excluded. Thread/core settings and sample sizes differ, so the values below do not form a single controlled hardware sweep.

| Variant | Recorded median | Scope |
| --- | ---: | --- |
| Fine-tuned E5 | ≈14–15 ms | Full known800 × 2, matching-weight index |
| RuModernBERT | 118–148 ms | 400 unique queries × 2, 6 performance-core threads |
| RuBERT-tiny2 | ≈66 ms | 400 unique queries × 2, 6 performance-core threads |
| Laya FP32 | ≈0.78 s | Later full application-order selection path |
| Laya INT8 | 435 ms / 647 ms | Two full400 runs; second exceeded 500 ms on 381/400 |
| Kev | ≈4.50 s | Small CPU probe: 12 unique queries × 2, 4 threads |

Kev leads the initial quality comparison, but its latency has only a small-probe result. Laya is the demo's educational decision-model default. Fine-tuned E5 and RuModernBERT are available as alternative architectural paths. Detailed protocols remain in [the final comparison](FINAL_MODEL_COMPARISON_RU.md).

## Public reproduction

The repository ships fictional source datasets, frozen candidates, training code, original notebook protocols, compact ledgers, 4,400 new-catalog predictions and calibration probabilities. `python scripts/verify_public_results.py` recalculates saved counts and calibration metrics without weights or network access.

Model weights, raw audio, private Kaggle inputs and personal catalogs are not included. The public evidence check is not a new inference run. [Reproduction instructions](REPRODUCIBILITY.md) explain the portable E5 bundle and extra native-library prerequisites for other models.
