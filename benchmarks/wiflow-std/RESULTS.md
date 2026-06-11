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

## Edge optimization (measured)

ADR-152 "optimize beyond SOTA" track, 2026-06-10, this Windows box (Windows 11,
16 torch threads, torch 2.12.0+cpu, onnxruntime 1.26.0). Subject: the retrained
checkpoint `results/retrained_best_pose_model.pth` (2,225,042 fp32 params).
Scripts: `quantize_bench.py`, `onnx_bench.py`, `eval_ort_accuracy.py`.
Raw numbers: `results/edge_optimization.json`.

Accuracy is on a **10,000-window seed-42 random subset** of the corruption-free
test split (same seed-42 file-level 70/15/15 split as `eval_repro.py`; 54,000
test windows, 1,440 corrupted excluded via `results/nan_windows_mask.npy` |
`results/big_windows_mask.npy`, leaving 52,560; subset drawn with
`np.random.default_rng(42)`). The fp32 subset PCK@20 (96.68%) matches the full
clean-test figure (96.61%), so the subset is representative.

Latency is CPU ms/window, median of repeated runs, 3 interleaved repetitions
per variant (medians below; run-to-run spread on this box is large, roughly
±20-40% at batch 1 — reps are in the JSON).

| Variant | Disk size | Batch 1 (ms/win) | Batch 64 (ms/win) | PCK@20 | PCK@50 | MPJPE |
|---|---|---|---|---|---|---|
| torch fp32 (baseline) | 9.07 MB | 11.0 | 2.27 | 96.68% | 99.15% | 0.00936 |
| torch fp16 (`.half()`) | **4.58 MB** | 24.3 | 2.42 | 96.68% | 99.15% | 0.00946 |
| torch int8 dynamic | 9.07 MB (unchanged) | 15.6 | 2.06 | 96.68% (identical) | 99.15% | 0.00936 |
| ONNX fp32 (onnxruntime) | 8.97 MB | **3.2** | **2.0** | 96.68% | 99.15% | 0.00936 |
| ONNX int8 (ORT dynamic, supplementary) | **2.44 MB** | 6.5 | 5.8 | 96.52% | 99.15% | 0.01108 |

Findings:

- **torch dynamic INT8 quantizes nothing on this model.** The architecture has
  **zero `nn.Linear` layers** — it is entirely Conv1d (21) + Conv2d (22) +
  BatchNorm. `torch.ao.quantization.quantize_dynamic` (requested over
  `{Linear, Conv1d, Conv2d}`) converted **0 modules / 0.0% of params**: dynamic
  quantization only has kernels for Linear/RNN-family modules and silently
  skips convolutions. The "int8" model is bit-identical to fp32 (same outputs,
  same 9.07 MB). Conv quantization would require static (PTQ) quantization
  with calibration — out of scope here; the ORT dynamic path below is the
  honest int8 datapoint.
- **fp16 halves size for free accuracy-wise** (PCK@20 −0.005 pt, MPJPE
  +0.0001) but is *slower* on CPU at batch 1 (~2.2×) — torch CPU fp16 conv
  kernels are emulated. fp16 is a storage/transport format here, not a CPU
  runtime win.
- **ONNX Runtime is the real batch-1 latency win: ~3.4× faster than torch**
  (3.2 vs 11.0 ms/window) at identical accuracy (parity 2.4e-7).

### Verdict on the paper's "~2.2 MB int8" claim

**Plausible but not free, and unreachable by the obvious PyTorch route.**
2,225,042 params × 1 byte ≈ 2.2 MB assumes *every* parameter quantizes.
PyTorch dynamic quantization — the one-liner most readers would reach for —
yields **9.07 MB (0% quantized)** because the model has no Linear layers.
ONNX Runtime dynamic quantization, which does have int8 conv weight support,
gets **2.44 MB** (close to the claim; the overhead is BatchNorm params/buffers
and quantization scales kept in fp32) at a measurable accuracy cost:
PCK@20 96.68 → 96.52% (−0.16 pt) and MPJPE 0.00936 → 0.01108 (+18%), and
~2× slower inference than ONNX fp32 (ConvInteger kernels). The paper does not
state a method or an int8 accuracy; treat "2.2 MB" as a weight-arithmetic
estimate, achievable in practice only via conv-capable quantization toolchains
and with a small accuracy penalty.

### ONNX export status

**Works.** Exported via the TorchScript exporter (`dynamo=False`), opset 17,
with a dynamic batch axis — `results/retrained_fp32_dynamic.onnx` (8.97 MB),
verified to run at batch 1/2/64. The axial attention's
`view(N*W, C, H)` reshape traced correctly (sizes recorded as graph ops, not
baked constants). The dynamo exporter also captures the graph but crashed on
this box writing a ✅ to a cp1252 console (cosmetic Windows encoding issue, not
a model blocker). Parity vs torch on the stored fixture
(`results/parity_fixture.npz`, batch 2, seed 42): **max abs diff 2.4e-7 —
PASS** (< 1e-4). ORT-quantized int8 model: `results/retrained_int8_ort_dynamic.onnx`.

## Measurement (b): BLOCKED-ON-DATA (attempted 2026-06-10)

The fine-tune-on-ESP32 measurement stopped at dataset characterization, per the
pre-registered stop rule (<2,000 paired windows). Findings (MEASURED):

- **Only one trainable paired dataset exists**: `ruvultra:~/work/cog-pose-train/paired.jsonl`
  — 1,077 windows (one subject, one room, one 29.9-min session, single node;
  CSI [56, 20]; 17 COCO keypoints, MediaPipe confidence mean 0.44 — only 264
  windows pass ADR-079's own conf>0.5 training filter). Prior measured attempts
  on this exact set: 0–3% torso-PCK@20 (temporal splits, three independent
  pipelines). Fine-tuning a 2.23M-param model on ~860 train windows would
  measure memorization, not transfer.
- **The April session behind the old "92.9% PCK@20" claim is lost** (345
  samples, 35 subcarriers; raw CSI gone from ruvzen/ruvultra/cognitum-v0; only
  a 69-sample predictions+GT holdout survives at `models/wiflow-real/eval-holdout.jsonl`).
- **Forensic recheck of that holdout RETRACTS the 92.9% figure**: the trainer's
  `pck()` used an absolute 0.2 image-unit threshold (not torso-normalized) and
  the model output a **constant pose** (pred std 0.0000 across 69 near-static
  frames; a mean predictor scores 100% under the same protocol). The
  torso-normalized PCK@20 on the same holdout is 19.1%. This corroborates the
  2026-05-11 audit retraction (CHANGELOG, PR #535); stale doc citations were
  removed 2026-06-10 (user-guide, readme-details, ADR-152 §2.1.3). The §2.2
  no-citation rule now applies to ADR-079 accuracy claims.

Unblock criteria: a paired collection session of ≥2k windows (≈35+ min at the
observed stride; multi-pose, conf>0.5, ideally with the §2.1.3 two-checkerboard
calibration), plus a re-baselined our-pipeline number under torso-PCK@20 on the
same split. WiFlow-STD assets stand ready on ruvultra (`~/wiflow-std-bench/`).
Also worth investigating: ADR-079's protocol predicts ~9k windows per 30 min;
the May session under-delivered ~8× (aligner drop rate?).

## Pending

- (b) fine-tune on our ESP32 17-keypoint eval set — **BLOCKED-ON-DATA**, see above.
- (c) our internal WiFlow on their dataset (15-keypoint subset mapping) — also
  affected: there is currently no validated internal pose model to compare
  (the 92.9% artifact is retracted; the MM-Fi SOTA models in ADR-150 §3 are a
  different input domain).
