# PERO — pixel-level box refinement for infrared small target detection

Two interchangeable, independently trained refinement heads that correct the
**box coordinates** of a **frozen** DETR-style infrared small-target detector
(PERO-R, D-FINE-S / DEIM) without touching its weights, its classification
scores or its candidate set. No NMS, no new proposals, no threshold tuning:

* **PlainReg + fixed damping** — the paper's **main model**: a direct xyxy
  regression head whose boxes are blended with the detector's boxes by fixed
  centre/size constants. Best AP50.
* **PERO-CBP v5** — distributional head: four 65-bin edge distributions plus
  four per-edge validity logits, decoded by expected position with
  fall-back-to-original gating. Best AP75 / AP50:95.

Both heads take the same four inputs (box crop, stride-4 feature crop, decoder
query, box geometry), are trained on the same data and budget, and are
alternatives on the same frozen base — they are **not** stacked.

| | |
|---|---|
| Base detector | PERO-R, 50 epochs EMA — frozen, 10,884,985 params |
| PlainReg head | 121,124 params (+1.11 %) |
| PERO-CBP head | 155,560 params (+1.43 %) |
| Dataset | IRDST, 42 odd sequences train / 43 even sequences eval |

---

## Results — IRDST, 43-even (20,258 images / 20,763 GT)

One-shot evaluation with `pycocotools` COCOeval; `F1@0.5` is a fixed-confidence
(0.5) greedy P/R/F1 diagnostic on the same predictions. All rows below were
re-produced by `eval.py` in this repository.

| State | AP50 | AP75 | AP50:95 | ΔAP50 | ΔAP75 | F1@0.5 |
|---|---|---|---|---|---|---|
| PERO-R (frozen) | 0.9374 | 0.5572 | 0.5262 | — | — | 0.9377 |
| + PlainReg (λ = 1, no damping) | 0.9621 | 0.5863 | 0.5532 | +2.47 | +2.91 | 0.9527 |
| **+ PlainReg + damping (0.5, 0.75) — main model** | **0.9629** | **0.6103** | **0.5648** | **+2.55** | **+5.31** | **0.9527** |
| + PERO-CBP v5 | 0.9526 | **0.6331** | **0.5674** | +1.52 | +7.59 | 0.9459 |

Paper-caliber P/R/F1 of the **main model** (TDCNet `test.py` precision-array
formula on the frozen predictions): **P 0.9865 / R 0.9884 / F1 0.9875**.

**Which row is "best"?**

* highest **AP50** and highest paper P/R/F1 → **PlainReg + damping** (main model);
* highest **AP75 / AP50:95** → **PERO-CBP** (it fixes boundary location more
  aggressively; that helps strict-IoU metrics, costs a little AP50);
* at a fixed confidence 0.5 both give the same F1, with plain regression
  moving 19653 TP / 840 FP / 1110 FN vs 19343 / 1150 / 1420 for the base.

> **Scope note.** The 43-even split has been consumed by earlier development
> rounds, so these are exploratory one-shot numbers, not a blind test. Deltas
> are the *total* effect of "new head + 50 epochs of head-only training" on top
> of the frozen detector.

---

## Method

### Frozen base (unchanged in both methods)

PERO-R = D-FINE-S / DEIM detector (1 class, 640×640 anamorphic input) with the
PERO relation read-out (`nets/rmtq_fine/pero.py`, `pero_jagg.py`,
`pero_r.py`). Weights, BN statistics, decoder and post-processor are frozen;
scores are bitwise identical in every state.

### Inputs (per candidate, 300 candidates per image)

| Input | Shape | Source |
|---|---|---|
| pixel patch | 3×32×32 | `roi_align` of the 640×640 input on the predicted box |
| C2 patch | 64×16×16 | `roi_align` of the frozen stride-4 backbone feature |
| query vector | 256 | final decoder query of that candidate |
| box geometry | 4 | `(cx/640, cy/640, log w, log h)` |

### PlainReg head (main model)

```text
2-conv pixel stem (32×32→8×8) + 1-conv C2 stem (16×16→8×8)
+ query projection + geometry projection  →  concat 160 channels @ 8×8
→ global average pool → MLP 256 → 128 → 4 deltas
edge_i = base_edge_i + delta_i * 0.5 * (box width or height)
```

Deployment blends the refined box `(c1, s1)` with the base box `(c0, s0)`:

```text
centre = c0 + lc * (c1 - c0)        # IRDST: lc = 0.5
size   = s0 + ls * (s1 - s0)        # IRDST: ls = 0.75
```

The constants are fixed deployment constants selected once on a
sequence-isolated dev split (no per-image selector, no test-time tuning). They
are dataset-specific: on IRSTD-UAV the selected constants are (0.25, 0.5), i.e.
they do **not** transfer — see the note below.

### PERO-CBP v5 head

```text
same stems and same 8×8 fused map
→ 4 position heads (65 bins per edge: left / right / top / bottom)
→ 4 validity heads (1 logit per edge)
decode: expected edge = Σ_k softmax(bins)_k · centre_k          (soft-argmax)
        if validity ≤ 0.5 on any edge        → keep original box
        if new box < 1 px or outside [0,640] → keep original box
        otherwise replace the four edges only (score untouched)
```

v5 differs from the earlier v2 decode only in the read-out (expected position
instead of central-80 % interval projection); the interval rule projected edges
to bin endpoints and threw away the improvement on already-accurate boxes.

---

## Training recipes (both are head-only, base frozen)

| Setting | PlainReg (main) | PERO-CBP v5 |
|---|---|---|
| Data | IRDST 42 odd sequences (20,398 images) | same |
| Steps | 50 epochs × 1,275 iters, batch 16 | same |
| Optimizer | AdamW, lr 1e-4 → 1e-5 cosine, wd 1e-4 | same |
| Warmup | 1,000 iters (reference schedule: restarts each epoch — kept verbatim) | 1,000 iters, global |
| EMA | 0.9999 (1,000 warmup updates) | same |
| Seed | 1234 | 1234 |
| Sampling | Hungarian-matched queries (IoU ≥ 0.1) + 2 geometric perturbations each | same (+ top-score / random nulls) |
| Loss | L1 on xyxy, normalised per edge by the base box side | dual-bin soft-label CE (positions) + balanced BCE (validity) + L2 consistency (0.1) |
| Time | ≈ 167 min on one RTX 4090 | ≈ 200 min |

---

## Quick start

### 1. Environment

```bash
git clone https://github.com/wujiaxuans/PERO-CBP.git
cd PERO-CBP
pip install -r requirements.txt        # torch, torchvision, pycocotools, scipy, numpy
```

Python ≥ 3.9, CUDA GPU recommended. DEIM/D-FINE is vendored in
`third_party/DEIM/`; nothing else to install.

### 2. Data

```
data/IRDST/images/
├── 1/           # odd sequence  → train
│   ├── 0.png
│   └── ...
├── 2/           # even sequence → eval
│   └── ...
└── ...
```

Sequence-level COCO annotations are included: `splits/train_42odd_coco.json`
(20,398 images / 21,038 boxes) and `splits/val_43even_coco.json` (20,258 /
20,763). Image paths are relative to the image root (`2/0.png`, …).

### 3. Weights

| File | What | SHA-256 (16) |
|---|---|---|
| `weights/peroR_ep50_ema.pth` | frozen PERO-R base (44 MB) | `91431f6792fc5285` |
| `weights/plainreg_ep50_ema.pth` | PlainReg head (0.5 MB) | `7acfa5e184f21e3f` |
| `weights/perocbp_v5_ep50_ema.pth` | PERO-CBP v5 head (0.6 MB) | `fde809844cd94efd` |

### 4. Evaluate

```bash
# main model (PlainReg + damping) and PERO-CBP, one traversal each state
python eval.py \
    --data_root ./data/IRDST/images \
    --peror_weights    ./weights/peroR_ep50_ema.pth \
    --plainreg_weights ./weights/plainreg_ep50_ema.pth \
    --cbp_weights      ./weights/perocbp_v5_ep50_ema.pth \
    --lc 0.5 --ls 0.75 \
    --output_dir ./outputs
```

Only one head is needed if you just want that method (drop the other flag).
Output (`outputs/eval_results.json` + stdout):

```text
  state                       AP50    AP75  AP50:95   dAP50   dAP75  F1@0.5
  PERO-R                    0.9374  0.5572   0.5262   +0.00   +0.00  0.9377
  PlainReg (raw)            0.9621  0.5863   0.5532   +2.47   +2.91  0.9527
  PlainReg + damping        0.9629  0.6103   0.5648   +2.55   +5.31  0.9527
  PERO-CBP                  0.9526  0.6331   0.5674   +1.52   +7.59  0.9459
```

### 5. Train a head from scratch (optional)

```bash
# main model
python train.py --method plainreg \
    --data_root ./data/IRDST/images \
    --peror_weights ./weights/peroR_ep50_ema.pth \
    --output_dir ./outputs

# PERO-CBP v5
python train.py --method cbp \
    --data_root ./data/IRDST/images \
    --peror_weights ./weights/peroR_ep50_ema.pth \
    --output_dir ./outputs
```

The base detector stays frozen (SHA-checkable, no gradients reach it). EMA
checkpoints are written as `outputs/plainreg_ep50_ema.pth` /
`outputs/perocbp_ep50_ema.pth`.

---

## Repository structure

```
.
├── train.py                        # head-only training (--method plainreg | cbp)
├── eval.py                         # one-shot eval of base + any provided head(s)
├── nets/rmtq_fine/
│   ├── pero_plainreg.py            # ★ PlainReg head + damping (main model)
│   ├── pero_cbp_v2.py              # ★ PERO-CBP v5 module + v5 / v2 decoders
│   ├── pero_cbp.py                 # crop / bin / soft-label utilities
│   ├── pero_r.py                   # PERORAdapter — frozen base wrapper
│   ├── pero.py                     # PEROCore (relation read-out of the base)
│   └── pero_jagg.py                # JointAggregationCore used by the base
├── configs/deim_dfine_s.yml        # D-FINE-S config (1 class, 640×640)
├── splits/                         # 42-odd train / 43-even eval annotations
├── weights/                        # frozen base + both released heads
├── third_party/DEIM/               # vendored DEIM / D-FINE framework (Apache-2.0)
├── requirements.txt
└── README.md
```

---

## Notes for reproduction

* **Equivalence checks performed for this release.** (a) `eval.py` reproduces
  the frozen table above to the last reported digit; (b) the PlainReg boxes
  computed by this repository are **bit-identical** (max abs diff `0.0`) to the
  frozen paired prediction cache that produced the main-model table;
  (c) the PERO-CBP state keeps the base scores bitwise and reproduces
  .9526 / .6331 / .5674.
* **Score handling.** The post-processor returns the top-300 queries **sorted by
  score**; `eval.py` reproduces that permutation and maps each returned
  detection back to its query index before applying a head. Scores are never
  modified, so every state shares one candidate ranking.
* **Damping constants are deployment constants**, chosen once on a
  sequence-isolated dev split (`lc = 0.5`, `ls = 0.75` for IRDST). On IRSTD-UAV
  the selected constants are different (0.25, 0.5) and the IRDST pair degrades
  there — do not transfer them silently.
* **One checkpoint, one evaluation.** No epoch selection, no threshold sweep, no
  test-time augmentation, no NMS.
* **PlainReg LR schedule.** The released head was trained with the reference
  schedule, whose warmup restarts at the beginning of every epoch;
  `train.py --method plainreg` keeps that behaviour verbatim (documented in
  `lr_at_plainreg`) so the released checkpoint can be reproduced.
* **Portability.** All paths are relative to the repository root; the vendored
  DEIM transforms are patched at runtime (`_patch_transforms`) to work with
  torchvision ≥ 0.17, where the v2 `Transform.transform` contract changed.

---

## Citation

```bibtex
@misc{pero2025,
  title  = {PERO: Pixel-Level Box Refinement for Infrared Small Target Detection},
  author = {wujiaxuans},
  year   = {2025},
  url    = {https://github.com/wujiaxuans/PERO-CBP}
}
```

## Acknowledgements & license

The frozen detector builds on [DEIM](https://github.com/ShihuaHuang95/DEIM) and
[D-FINE](https://github.com/Peterande/D-FINE); the vendored copy in
`third_party/DEIM/` keeps its original Apache-2.0 license
(`third_party/DEIM/LICENSE`). The IRDST dataset is **not** redistributed here —
please obtain it from its original release and respect its license.
