"""PERO-CBP v2 — independent Pixel Boundary Correction for the frozen PERO-R.

Paper main method candidate (2026-10-12): v5 decode = expected position.

Architecture (155,560 params, <= 0.30M):
  Inputs per candidate (all read-only w.r.t. the frozen detector):
    - pixel patch: roi_align of the 640 input image on the predicted box, 32x32
    - C2 patch: the same box on the frozen stride-4 feature, 16x16
    - query: the decoder's final query vector (256-d)
    - b0 geometry: the detector's predicted box (cx/640, cy/640, log w, log h)
  Stems: 2-conv pixel stem + 1-conv C2 stem + query/geom projections
  Heads: 4 independent position heads (65-bin distribution per edge) +
         4 independent validity heads (1 logit per edge)

Decodes:
  v2 (interval projection): central-80% [q10,q90] of the position
     distribution; original edge inside -> keep; else project to nearest
     endpoint. (Superseded by v5 — kept for ablation reference.)
  v5 (expected position — PAPER METHOD): soft-argmax expected edge position
     directly, with validity gating and legality checks. No interval
     projection, no keep-if-inside, no endpoint projection.

Training (v2/v5 shared): position CE (dual-bin soft labels) on valid targets
only; balanced validity BCE; consistency L2 (weight 0.1) on orig-vs-perturbed
pairs. 50ep, seed 1234, AdamW 1e-4 -> cosine 1e-5, EMA .9999.

Result (43-even, EXPLORATORY): AP50 +1.52pp, AP75 +7.59pp, AP50:95 +4.11pp,
conf0.5 F1 +0.82pp vs frozen PERO-R. Label: METHOD_GAIN.

v1 post-mortem: the (W,H) coordinate swap reduced the positive set 16x;
  85.6:1 null:pos imbalance through a shared trunk degenerated the position
  branch. v2 fixed both; v5's expected-position decode recovered the
  improvement that v2's interval projection was discarding (mechanism:
  step 13 showed the boundary head's direct prediction reached IoU 0.493 on
  near-miss boxes vs 0.426 for the original, but the interval rule output
  only 0.436 — 23/34 direct predictions >= 0.5 vs only 2/34 after
  projection).
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEIM = os.path.join(_ROOT, "third_party/DEIM")
for _p in (_ROOT, _DEIM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nets.rmtq_fine.pero_cbp import (CBP_N_BINS, CBP_SEARCH, CBP_MIN_SIZE,
                                     CBP_INPUT, bin_centers, _percentile_bins,
                                     crop_rois_xyxy)

V2_N_BINS = 65


class _StripHead(nn.Module):
    def __init__(self, c_in, n_out, hidden=32):
        super().__init__()
        self.proj = nn.Linear(c_in, hidden)
        self.out = nn.Linear(hidden * 8, n_out)

    def forward(self, strip):
        N, S, C = strip.shape
        x = F.silu(self.proj(strip))
        if S > 8:
            x = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=S // 8).permute(0, 2, 1)
        return self.out(x.reshape(N, -1))


class PeroCBPv2(nn.Module):
    """params <= 0.30M. forward -> (position_logits (N,4,65), validity (N,4))."""

    def __init__(self):
        super().__init__()
        self.pix_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True))
        self.c2_stem = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True))
        self.q_proj = nn.Linear(256, 32)
        self.geom_proj = nn.Linear(4, 32)
        c_strip = 64 + 32 + 32 + 32
        self.pos_l = _StripHead(c_strip, V2_N_BINS)
        self.pos_r = _StripHead(c_strip, V2_N_BINS)
        self.pos_t = _StripHead(c_strip, V2_N_BINS)
        self.pos_b = _StripHead(c_strip, V2_N_BINS)
        self.val_l = _StripHead(c_strip, 1)
        self.val_r = _StripHead(c_strip, 1)
        self.val_t = _StripHead(c_strip, 1)
        self.val_b = _StripHead(c_strip, 1)
        self._zero_pixels = False
        self._zero_c2 = False
        self._edge_shift = None      # {edge: bins} sensor on the position head

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, pixels, c2p, query, boxes):
        N = pixels.shape[0]
        if self._zero_pixels:
            pixels = torch.zeros_like(pixels)
        if self._zero_c2:
            c2p = torch.zeros_like(c2p)
        f_pix = self.pix_stem(pixels)
        f_c2 = F.adaptive_avg_pool2d(self.c2_stem(c2p), 8)
        q = self.q_proj(query).unsqueeze(-1).unsqueeze(-1).expand(N, 32, 8, 8)
        geom = torch.stack([boxes[:, 0] / CBP_INPUT, boxes[:, 1] / CBP_INPUT,
                            torch.log(boxes[:, 2].clamp(min=1e-3)),
                            torch.log(boxes[:, 3].clamp(min=1e-3))], dim=-1)
        g = self.geom_proj(geom).unsqueeze(-1).unsqueeze(-1).expand(N, 32, 8, 8)
        fused = torch.cat([f_pix, f_c2, q, g], dim=1)
        strip_lr = fused.mean(dim=2).permute(0, 2, 1)
        strip_tb = fused.mean(dim=3).permute(0, 2, 1)
        pos = torch.stack([self.pos_l(strip_lr), self.pos_r(strip_lr),
                           self.pos_t(strip_tb), self.pos_b(strip_tb)], dim=1)
        val = torch.cat([self.val_l(strip_lr), self.val_r(strip_lr),
                         self.val_t(strip_tb), self.val_b(strip_tb)], dim=1).squeeze(-1)
        if self._edge_shift:
            pos = pos.clone()
            for e, k in self._edge_shift.items():
                if k:
                    p_ = pos[:, e, :]
                    sh = torch.zeros_like(p_)
                    if k > 0:
                        sh[:, k:] = p_[:, :-k]
                    else:
                        sh[:, :k] = p_[:, -k:]
                    pos[:, e, :] = sh
        return pos, val


def decode_v2(pos_logits, val_logits, b0_xyxy):
    """pos (N,4,65); val (N,4); b0_xyxy (N,4) double 640-space.
    Returns corrected (N,4) double, fallback (N,), moved (N,)."""
    N = b0_xyxy.shape[0]
    device = b0_xyxy.device
    p_valid = torch.sigmoid(val_logits.double())              # (N,4)
    fallback = (p_valid <= 0.5).any(dim=-1)
    p = torch.softmax(pos_logits.double(), dim=-1)            # (N,4,65)
    centers = bin_centers(b0_xyxy)
    lo = torch.stack([_percentile_bins(p[:, e], centers[:, e], 0.10)
                      for e in range(4)], dim=1)
    hi = torch.stack([_percentile_bins(p[:, e], centers[:, e], 0.90)
                      for e in range(4)], dim=1)
    b0e = torch.stack([b0_xyxy[:, 0], b0_xyxy[:, 2],
                       b0_xyxy[:, 1], b0_xyxy[:, 3]], dim=1)
    new_e = torch.where(b0e < lo, lo, torch.where(b0e > hi, hi, b0e))
    moved = (new_e != b0e).any(dim=-1)
    corr = torch.empty_like(b0_xyxy)
    corr[:, 0] = new_e[:, 0]; corr[:, 2] = new_e[:, 1]
    corr[:, 1] = new_e[:, 2]; corr[:, 3] = new_e[:, 3]
    bad = (corr[:, 2] - corr[:, 0] < CBP_MIN_SIZE) | (corr[:, 3] - corr[:, 1] < CBP_MIN_SIZE)
    bad = bad | (corr[:, 0] < 0) | (corr[:, 1] < 0) | (corr[:, 2] > CBP_INPUT) \
        | (corr[:, 3] > CBP_INPUT)
    bad = bad | ~torch.isfinite(corr).all(dim=-1)
    fallback = fallback | bad
    out = torch.where(fallback.unsqueeze(-1), b0_xyxy, corr)
    return out, fallback, moved & ~fallback


def expected_edges_v2(pos_logits, b0_xyxy):
    p = torch.softmax(pos_logits.double(), dim=-1)
    centers = bin_centers(b0_xyxy)
    return (p * centers).sum(-1)


# ════════════════════════════════════════════════════════════════════════════
# v5 decode: expected position + validity + legality (no interval projection)
# PAPER MAIN METHOD — sole change vs v2's decode; the network is identical.
# ════════════════════════════════════════════════════════════════════════════
def decode_v5(pos_logits, val_logits, b0_xyxy):
    """PAPER METHOD. pos (N,4,65); val (N,4); b0_xyxy (N,4) double 640-space.

    Rule (fixed, no free alpha/tau):
      1. per-edge validity: sigmoid(val) <= 0.5 on ANY edge -> whole-box
         fallback to b0 (protect boxes the head can't measure)
      2. new edge = expected position (soft-argmax of the 65-bin softmax)
      3. legality: positive size (>= 1px), in-bounds [0, 640], finite
         -> whole-box fallback if violated

    No central-80% interval, no keep-if-inside, no endpoint projection.
    Returns (corrected (N,4) double, fallback (N,), moved (N,)).
    """
    N = b0_xyxy.shape[0]
    device = b0_xyxy.device
    p_valid = torch.sigmoid(val_logits.double())
    fallback = (p_valid <= 0.5).any(dim=-1)
    p = torch.softmax(pos_logits.double(), dim=-1)
    centers = bin_centers(b0_xyxy)
    expected = (p * centers).sum(-1)               # (N,4) l,r,t,b
    corr = torch.empty_like(b0_xyxy)
    corr[:, 0] = expected[:, 0]; corr[:, 2] = expected[:, 1]
    corr[:, 1] = expected[:, 2]; corr[:, 3] = expected[:, 3]
    bad = (corr[:, 2] - corr[:, 0] < CBP_MIN_SIZE) | (corr[:, 3] - corr[:, 1] < CBP_MIN_SIZE)
    bad = bad | (corr[:, 0] < 0) | (corr[:, 1] < 0) | (corr[:, 2] > CBP_INPUT) \
        | (corr[:, 3] > CBP_INPUT)
    bad = bad | ~torch.isfinite(corr).all(dim=-1)
    fallback = fallback | bad
    out = torch.where(fallback.unsqueeze(-1), b0_xyxy, corr)
    moved = (out != b0_xyxy).any(dim=-1) & ~fallback
    return out, fallback, moved
