# Laya multilingual: clean first with validation stopping

The user selected a clean start after the completed Nano-Jev comparison and requested continued training with attention to overfitting. Nano-Jev's small curriculum difference is a hypothesis for Laya, not evidence of transfer between architectures.

Private dataset: https://www.kaggle.com/datasets/matveyvarfolomeev/launcher-laya-training-v1

Private notebook: https://www.kaggle.com/code/matveyvarfolomeev/launcher-laya-clean-first-v1

## Frozen protocol

- Initial model: `convaiinnovations/laya-multilingual@1720e3e3357cfe1e281542e223f8273b0890ca34`. Original weight SHA-256: `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`.
- Native SDK `laya==0.3.26`, Transformers 5.6.2, safetensors 0.6.2, accelerate 1.15.0. No replacement classification head or generated answer text.
- Exactly the already frozen Nano-Jev training/validation inputs: 5,664 base and 16,992 simulated-STT rows with E5 top-20. Query/metadata and ordinal option keys enter the model; target IDs are labels and the final mapping only.
- Maximum one pass, 1,416 optimizer updates. Clean prefix: 2,272 rows / 142 updates. Remaining rows are shuffled; stopping can end exposure before the complete pass. It is not a matched full-budget curriculum comparison against Nano-Jev.
- Full choice-task fine-tuning: encoder lr 1e-5; native choice/type/head lr 3e-5; AdamW weight decay 0.01, 6% warmup then linear decay, gradient norm clipping 1.0.
- Microbatch one query, 16 accumulated queries per update. FP16 autocast and gradient scaling for training; FP32 parameters, optimizer states and validation. Encoder/head gradient checkpointing reduces activation memory.
- The act/escalate head is frozen. Its weights remain, but escalation is not trained or validated. Choice probabilities remain uncalibrated after fine-tuning.
- Native input matches the current model adapter: max_len/head_max_len 8192, 20 markers, 48 tokens per option description. Query truncation, missing/collapsed option slots, or an additional head-budget cut abort the run. Metadata losses are counted rather than silently called full-context input.

## Validation and stopping

Baseline and each 142nd update are evaluated on the same 150 synthetic validation rows, 50 per basic/difficult/ASR profile. Test and personal catalogs stay local. Missing E5 targets remain errors in accuracy and recall; there are no evaluation positive insertions.

Checkpoint selection uses conditional negative log probability of the target, clipped at probability 1e-12, on rows whose target exists in the shortlist. Loss is averaged inside each scenario group, then over groups in a profile, then equally over the three profiles. This is a predeclared optimization/stopping scalar. Accuracy and candidate recall remain separate per profile and catalog; the scalar is not reported as overall product quality. Baseline is eligible as the selected checkpoint.

Save a new best checkpoint only after a loss decrease greater than 0.005; otherwise increment patience. Stop after three consecutive checks without such improvement, or at one complete pass. Save both the selected and last weights. A finite stopping rule does not prove absence of overfitting, including adaptation to the validation set itself. New scenario families and subsequent independent evaluation are still needed.

## Verification and current status

Ten focused schedule/group weighting/stopping tests passed. Native SDK input token IDs, option marker positions and question types match the new trainer on six representative real/synthetic inputs. Original weights reload and return finite 20-option logits on CPU. Notebook code cells parse and private metadata is set. These local checks are separate from the GPU experiment evidence below.

Private input upload completed and all 12 expected expanded files are visible in the account. Original Laya was evaluated on all 300 local validation rows: real basic/difficult/ASR 24/50, 23/50, 20/50; synthetic 23/50, 18/50, 16/50. No validation query or metadata was truncated.

After discussion, the user selected validation loss for checkpoint choice and assigned a separate third catalog to test-only transfer assessment, with no stopping/selection influence. Private notebook version 1 completed on Tesla T4, Torch 2.11.0+cu128. Actual execution stopped after 994 finite optimizer updates because three consecutive checks did not improve the selected loss. Script runtime was 2,601.79 seconds; this excludes setup/download/local evaluation and is not inference latency.

The selected checkpoint is step 568, not the last step. Synthetic validation basic/difficult/ASR accuracy is 49/50, 43/50, 46/50 and selection NLL is 0.26327. At the last step 994, accuracy is 48/50, 42/50, 44/50 and NLL is 0.46160. Baseline NLL was 2.18034. All eight validation NLL values were recomputed from downloaded prediction files. These checks participated in checkpoint selection and are not independent test results.

Actual exposure at the selected step was 9,088 rows: 3,400 clean and 5,688 simulated-STT. At the last step, exposure was 15,904 rows: 4,519 clean and 11,385 simulated-STT. Early stopping left 6,752 rows unused. The selected weight SHA-256 is `acc8ee1848ae7c2fcca1b8e640b76823d8990482c0be2af592f987cd234f6bcc`; the last weight SHA-256 is `9ff569d29cb71972089a5dbc80e68529c9d9699d6d799c70fef86ef4d42055c1`. All 28 output files downloaded, and both local weight hashes match the remote result. The selected checkpoint reloaded for 300 paired local predictions; full exposure/input/stopping/probability audits passed. GPU and CPU differ in zero synthetic top-1 predictions out of 150.

The actual training/validation curves are saved in `artifacts/kaggle/laya-learning-curves.png` and `.svg`. Training batch cross-entropy and group-balanced validation NLL have different averaging and example composition, so their absolute heights are not directly comparable. Individual training batches approach zero, while the rolling 50-batch mean remains higher.

Native encoding recorded 18,927 capped training options out of 453,120 (4.18%), dropping 89,006 metadata tokens. Maximum sequence length was 995 training tokens and 634 validation tokens. Validation had zero capped options; all queries and 20 distinct option markers were retained. Shared raw examples do not imply identical tokenized inputs across Nano-Jev and Laya.

## Paired local validation

Each cell contains 50 unchanged query rows. Every retrieval miss remains an error; profiles and catalogs are reported separately. E5 and Nano numbers come from the already audited frozen-input comparison.

| Catalog / profile | Original Laya | E5 only | Nano clean-first | Selected Laya |
| --- | ---: | ---: | ---: | ---: |
| Real / basic | 24/50 (48%) | 36/50 (72%) | 44/50 (88%) | 43/50 (86%) |
| Real / difficult | 23/50 (46%) | 30/50 (60%) | 35/50 (70%) | 37/50 (74%) |
| Real / ASR | 20/50 (40%) | 26/50 (52%) | 34/50 (68%) | 34/50 (68%) |
| Synthetic / basic | 23/50 (46%) | 48/50 (96%) | 42/50 (84%) | 49/50 (98%) |
| Synthetic / difficult | 18/50 (36%) | 38/50 (76%) | 36/50 (72%) | 43/50 (86%) |
| Synthetic / ASR | 16/50 (32%) | 39/50 (78%) | 38/50 (76%) | 46/50 (92%) |

Trained-minus-original Laya improvement on real validation is +38/+28/+28 percentage points. Bootstrap resamples the 13 scenario groups in each subset, 10,000 times; the corresponding 95% intervals are approximately [+26,+49.02], [+13.73,+44], [+14.58,+41.18] percentage points. These intervals cover variation in the sampled validation groups, not another training seed. Close trained Nano/Laya scores are not evidence of a general architecture ranking.

Machine-readable checked evidence: `artifacts/kaggle/laya-comparison.json`. Inference used CPU FP32, four threads and one query per microbatch. This accuracy run does not qualify the complete-query 3,000 ms product limit. The GUI does not use the trained checkpoint yet.

## Third-catalog test, after checkpoint selection

The fixed selected step-568 checkpoint was evaluated on a new fictional 177-object catalog with 50 basic and 50 difficult text queries, grouped into 38 scenario families per profile. All five Nano/Laya checkpoint hashes were fixed before model inference. Query/group/target overlap guards passed against 23,856 existing rows. No test row entered training, validation stopping or checkpoint selection. The test was authored by Codex and lacks independent human label review; novel names do not establish coverage of entirely new concepts. It has no ASR profile or human voice, and all descriptions are filled, unlike the real catalog.

| Fixed model | Basic | Difficult |
| --- | ---: | ---: |
| E5 only | 50/50 | 45/50 |
| Original Nano-Jev | 16/50 | 10/50 |
| Nano mixed | 46/50 | 36/50 |
| Nano clean-first | 46/50 | 38/50 |
| Original Laya | 7/50 | 4/50 |
| Selected Laya clean-first | 50/50 | 43/50 |

E5 candidate recall is 100/100; all neural errors are choice errors on reachable targets. Laya adaptation transfers strongly relative to its original checkpoint, while adding its decision head does not beat E5 top-1 on this test. The difficult-profile difference from E5 is -4 percentage points; its group-bootstrap 95% interval is approximately [-26,+18.97] points, so these 50 rows do not establish a reliable ranking of the close scores. The test cannot establish absence of overfitting or a clean-first advantage, since Laya has no matched mixed run. Checked evidence is `artifacts/kaggle/transfer-test-v1/comparison.json`.

This test has now been inspected. Subsequent design changes informed by these results require another fresh confirmation sample. Kev was not one of the five pre-frozen checkpoints; evaluating it here later would be a comparison on a known benchmark, not the same fresh independent confirmation.

## Files and reproduction

- `scripts/train_laya_kaggle.py`: native model loading, tokenization, cross-entropy training, validation, stopping and checkpoint serialization. Reuses the existing frozen schedule/input validators.
- `scripts/prepare_laya_experiment.py`: pinned input/checkpoint hashes and the private notebook package.
- `scripts/evaluate_laya_locally.py`: offline FP32 evaluation on the existing 300-row local validation set.
- `scripts/laya_cpu_weights.py`: read-only file-backed FP32 weights for memory-constrained Windows evaluation. Original F16 weights are widened exactly into a hash-checked disk cache; selected trained F32 weights are read directly. Three inference-equivalence/source-integrity/cache-corruption tests passed in the model environment. Native weights, logits and the trained GPU script are preserved; inference latency remains unqualified.
- `scripts/audit_laya_run.py`: actual exposure/stopping/hash checks and paired results, including GPU/CPU choices.
- `scripts/plot_laya_training.py`: actual update/validation curves with the selected step and clean-stage boundary.
- `scripts/evaluate_transfer_test.py`: freezes selected checkpoint hashes before evaluating the new test-only catalog.
- `notebooks/laya-clean-first.ipynb`: reviewable Kaggle notebook.

On Windows, run the Kaggle dataset upload from the data directory with `-p .`; CLI 2.2.4's resume-marker path mishandles the nested relative upload directory. No system CLI files were changed.

CLI 2.2.4's notebook-output path buffers each complete response, even with `stream=True`, and writes its combined log with the Windows default encoding. `scripts/download_kaggle_outputs.py` uses the existing SDK streaming/resume implementation, atomic `.part` files and UTF-8 logs, without changing installed CLI code. It avoids holding each 1.29 GB FP32 weight file in private memory.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\prepare_laya_experiment.py --worktree C:\Users\matva\.codex\worktrees\c58b\jev_pet
# Working directory: D:\ML\petprodj\jev_pet\artifacts\kaggle\laya-data
D:\ML\petprodj\jev_pet\artifacts\kaggle\.venv\Scripts\kaggle.exe datasets create -p . -r skip
# Main project directory:
.\artifacts\kaggle\.venv\Scripts\kaggle.exe kernels push -p artifacts\kaggle\notebooks\laya-clean-first --accelerator NvidiaTeslaT4 --timeout 14400
# Read-only live logs; Windows stdout and log-file writes must support UTF-8.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
.\artifacts\kaggle\.venv\Scripts\kaggle.exe kernels logs matveyvarfolomeev/launcher-laya-clean-first-v1 --follow
```

The input dataset contains fictional data and the pinned original model/license notice. Outputs and personal evaluations belong under `artifacts/kaggle`, outside Git. No trained model is connected to the GUI by this experiment.
