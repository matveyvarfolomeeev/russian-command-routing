# Experiments

## Data and training

The training pool contains 5,664 clean examples and 16,992 simulated-STT variants, generated from 708 source queries on eight 177-object catalogs. Simulated STT is text corruption; ASR evaluation uses actual Whisper transcriptions of synthesized speech.

Runs used Tesla T4, effective batch size 16, AdamW, 6% warmup and gradient clipping. Clean-first begins with 2,272 clean examples, then shuffles the remainder. Periodic selection runs evaluate every 142 updates, with patience 3 and minimum improvement 0.005. Selection averages NLL by group and profile on the original synthetic validation150; E5 validates over the full catalog, other models over reachable top-20 targets.

| Run | Selected update | Completed updates |
| --- | ---: | ---: |
| Nano mixed / clean-first | 1,416 each | 1,416 each |
| Laya | 568 | 994 |
| Kev | 710 | 1,136 |
| RuModernBERT | 426 | 852 |
| Nano continuation | 0 | 426 |
| RuBERT-tiny2 | 142 | 568 |
| E5 | 142 | 568 |

The Nano pair uses final one-pass weights. For the other runs, selected and last weights are stored separately. E5's selected checkpoint saw only the clean prefix, so its STT gains cannot be attributed to training on simulated-STT examples. [Plans, losses and checkpoint hashes](../results/training-runs.json).

![Laya training and validation losses](../assets/experiments/laya-learning-curves.png)

## New-catalog evaluation

`launcher-independent-v1` has 200 commands: validation50 in 10 groups and test150 in 30 other groups. Each has clean and ASR versions. Initial checkpoints were fixed before this evaluation; subsequent E5 and reranker runs use the already inspected suite.

| Variant | Clean val / 50 | ASR val / 50 | Clean test / 150 | ASR test / 150 |
| --- | ---: | ---: | ---: | ---: |
| E5 original | 42 | 41 | 143 | 136 |
| Nano original | 0 | 0 | 0 | 0 |
| Nano mixed | 37 | 36 | 128 | 127 |
| Nano clean-first | 38 | 38 | 128 | 125 |
| Laya original | 16 | 14 | 40 | 45 |
| Laya fine-tuned | 47 | 48 | 141 | 141 |
| Kev original | 47 | 47 | 143 | 143 |
| Kev fine-tuned | 50 | 50 | 149 | 146 |
| E5 fine-tuned * | 48 | 48 | 145 | 143 |
| RuModernBERT fine-tuned * | 48 | 47 | 144 | 143 |
| RuBERT-tiny2 + head * | 32 | 35 | 76 | 90 |

`*` Later runs. The original Nano's zero result was reproduced on all 400 CPU representations; its cause is not established. Original E5 retrieves 150/150 clean and 149/150 ASR test targets, so one ASR error is unavoidable for every pipeline using that shortlist.

The initial study uses 10,000 paired bootstrap resamples of whole target groups. Fine-tuned Kev minus original E5: clean **+4.00 pp [0.67, 8.00]**, ASR **+6.67 pp [1.33, 11.33]**. Kev minus original Kev on ASR: **+2.00 pp [-0.67, 4.67]**. These intervals cover the sampled groups, not training-seed variability. [Predictions and comparisons](../results/independent-v1/summary.json).

## Additional experiments

- **Training order:** Nano mixed and clean-first both score 128/150 clean; ASR scores are 127/150 and 125/150. A consistent clean-first advantage was not found. Continuation at a lower learning rate kept the initial checkpoint as best.
- **Retrieval adaptation:** standalone E5 increased known-suite equal-scenario accuracy from 80.5% to 89.4%, and old noisy-STT validation mean from 65% to 82%. Trained E5 before fixed Laya reduced mean accuracy from 88.15% to 87.95%, despite better target coverage.
- **Quantization:** dynamic INT8 Laya reduced old noisy-STT validation accuracy from 80% to 64%. Two full warm CPU runs had medians of 435 ms and 647 ms.
- **Metadata:** removing descriptions reduced E5 retrieval quality. Added descriptions helped some queries and harmed others.

## Calibration and timing

Laya temperature `T=2.4938812077079957` was fitted on 100 validation representations from 10 groups. A later application-order replay on 300 test representations from 30 other groups gave NLL **0.4044 → 0.2410**, ECE **4.59% → 3.11%**, unchanged top-1 **283/300** and worse Brier scores. The initial comparison above used another candidate order and scored 282/300. [Calibration data](../results/calibration/laya.json).

Warm CPU selection on i7-13700H took roughly 14–15 ms for fine-tuned E5 and 118–148 ms for RuModernBERT in full repeated checks. Laya's later application path took about 0.78 s. Kev's roughly 4.5 s median is from a small 12-query × 2 probe. Timings exclude ASR, model loading and initial indexing; sample sizes and thread settings differ.

The labels are author-reviewed, speech is TTS and each recipe has one recorded seed. Clean/ASR pairs are correlated. Old noisy-STT numbers use validation data that influenced selection; later test-set runs are descriptive.
