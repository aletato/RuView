# WiFlow-STD (DY2434) Benchmark Results — ADR-152 §2.2

Upstream: <https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling>
pinned at `06899d29` (2026-04-05), Apache-2.0. Dataset: Kaggle `kaka2434/wiflow-dataset`
(12.8 GB archive → 15.5 GB extracted; 360,000 windows of 540×20 CSI + 15-keypoint 2D labels).

Published claims (README "Setting 1"): PCK@20 97.25%, PCK@30 98.63%, PCK@40 99.16%,
PCK@50 99.48%, MPJPE 0.007 m, 2.23M params, 0.07 GFLOPs.

## Measurement (a): their model on their data

### Artifact verification (MEASURED, 2026-06-10, this repo `eval_repro.py`)

| Check | Result |
|---|---|
| Parameter count | **2,225,042 (2.23M) — matches claim** |
| FLOPs (torch profiler, batch 1) | ~0.055 GFLOPs — consistent with 0.07B claim |
| CPU latency (Windows box, torch 2.12 CPU) | 13.2 ms/window @ batch 1 (76/s); 2.48 ms/sample @ batch 64 (403/s) |
| Checkpoint load | `weights_only=True` (no pickle code execution) |

### Released checkpoint does NOT reproduce the claims — REFUTED as shipped

Running the released `best_pose_model.pth` through the released code on the released
dataset with the released split procedure (seed-42 file-level 70/15/15; 54,000 test
samples) yields:

| Metric | Published | Measured (shipped checkpoint) |
|---|---|---|
| PCK@20 | 97.25% | **0.08%** |
| PCK@30 | 98.63% | 0.78% |
| PCK@40 | 99.16% | 5.53% |
| PCK@50 | 99.48% | 15.42% |
| MPJPE | 0.007 | **NaN** (dataset contains NaN CSI windows) |

Raw output: `results/repro_a.json`.

Diagnostics (on 2,000 NaN-free windows from the first files of the dataset, i.e.
mostly would-be *training* data — so this is not a split mismatch):

- Predictions correlate with targets (Pearson r ≈ 0.76) — the checkpoint is a trained
  model, but in a **different keypoint normalization/order** than the released data.
- Best-case post-hoc global per-axis affine correction: PCK@20 ≈ 20%.
- Best-case per-keypoint affine correction (15×2 fitted transforms — generous
  cheating): PCK@20 ≈ 72%, still far below 97.25%.
- Pred↔target keypoint correspondence matrix is degenerate (multiple predicted
  keypoints best-match the same target joint) — keypoint convention mismatch.

### Reproducibility defects in the released artifacts

1. `models/__init__.py` imports `TemporalConvNet`, which `models/tcn.py` does not
   define — **the published code does not import/run as-is**.
2. The released root checkpoint uses pre-rename module names (`att.*`, `final_conv.*`)
   vs the published code (`attention.*`, `decoder.*`) — same shapes/param count, but
   confirms the checkpoint predates the published code.
3. The second shipped checkpoint (`cross_dataset_test/WiFlow/best_pose_model.pth`) is
   a **different architecture** (342-channel input = MM-Fi layout, 3 TCN layers,
   3-channel/3D decoder) — not usable on their own dataset.
4. `run.py` ignores `--data_dir` and hardcodes `../preprocessed_csi_data`.
5. The released dataset's final 13 files (indices 487–499; 9,072 windows, 2.52%)
   are corrupted: NaN values plus garbage amplitudes up to 3.4e38 (float32 max) in
   data that is otherwise [0,1]-normalized. Upstream code has no NaN/inf handling;
   training as published on this download diverges — the first corrupted batch
   overflows fp16 autocast and permanently poisons BatchNorm running statistics
   (GradScaler step-skipping does not protect BN). The authors' training curves
   show normal convergence, so their local data evidently differed from the
   Kaggle upload. Window masks: `results/nan_windows_mask.npy`,
   `results/big_windows_mask.npy`.

### Retraining result (MEASURED, 2026-06-10): claims APPROXIMATELY REPRODUCED

Since the shipped checkpoint is unusable, measurement (a) fell back to retraining
with upstream code + defaults (seed 42, batch 64, early-stopped at epoch 41 of 50,
best epoch 36, ~75 s/epoch) on ruvultra (RTX 5080). Deviations, all forced and
documented: one-line fix for defect (1); torch 2.x+cu128 instead of pinned 2.3.1
(Blackwell sm_120 unsupported); the 9,072 corrupted windows (defect 5) zeroed
entirely — without this the published pipeline produces NaN from epoch 1 (observed).
Scripts mirrored in `remote/`; raw metrics in `results/eval_retrained.json`.

| Metric | Published | Retrained (full test, 54,000) | Retrained (corruption-free, 52,560) |
|---|---|---|---|
| PCK@20 | 97.25% | **96.09%** | **96.61%** |
| PCK@30 | 98.63% | 97.89% | 98.23% |
| PCK@40 | 99.16% | 98.58% | 98.79% |
| PCK@50 | 99.48% | 98.99% | 99.11% |
| MPJPE | 0.007 | 0.0098 | 0.0094 |

Within ~0.6–1.2 PCK points of every published figure (single run, corrupted train
windows zeroed, different torch/GPU). **Verdict: the accuracy claims are credible
and approximately reproducible — but only after repairing the released dataset and
code.** Val best: PCK@20 96.99%, MPJPE 0.0086 (epoch 36).

One more defect found during the run:

6. `train.py` calls `plot_training_history`, which is not defined anywhere — the
   built-in post-training test evaluation is unreachable as published (crashes
   with NameError after training completes).

## ADR-152 §2.2 citation rule

Evidence grade for the WiFlow-STD accuracy claims after measurement (a):
**MEASURED-EQUIVALENT (96.1–96.6% PCK@20 reproduced by retraining; shipped
checkpoint REFUTED; dataset/code require repairs)**. RuView docs may cite
"~96% PCK@20 (our reproduction)" — still **not comparable** to our 17-keypoint
ESP32 numbers (different hardware, 5 subjects, in-domain random split,
15 keypoints).

## Pending

- (b) fine-tune on our ESP32 17-keypoint eval set.
- (c) our internal WiFlow on their dataset (15-keypoint subset mapping).
