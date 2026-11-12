"""PERO-CBP — independent Pixel Boundary Correction heads for the frozen
PERO-R (PERO_CBP_1_OFFICIAL_IRDST50).

Problem being tested: PERO-R often FINDS the target but its final box misses
by a few pixels (near-miss FP/FN: cache diagnostics showed 208/289 suppressed
high-score FPs with best-IoU in [0.1, 0.5), median best-IoU 0.488 on images
with no TP at all). PERO-CBP re-measures the four box boundaries from local
pixels and corrects ONLY the box — scores stay bitwise PERO-R.

Inputs per candidate (all read-only w.r.t. PERO-R):
  - b0            : the final PERO-R box (cxcywh, model-input 640 space)
  - query         : the decoder's final query vector (256-d, read-only hook)
  - pixel patch   : crop centered at b0 with width/height x2.5, bilinear
                    32x32 sampled from the ACTUAL 640 input image
  - C2 patch      : the same crop ROIAligned 16x16 from the frozen stride-4
                    C2 feature (64ch)

Four INDEPENDENT boundary heads (left/right/top/bottom), each outputting a
65-bin position distribution + 1 null bin (66 logits). Search range per edge:
original boundary +/- 0.5 x corresponding box width/height.

Decoding rule (hard-constrained, no free alpha/tau):
  per edge: p = softmax(66); if argmax == null -> whole-box fallback to b0.
  position mass = p[:65] renormalized; central-80% interval = [q10, q90] of
  that distribution (in coordinates); if b0's boundary lies inside -> keep it
  (protect accurate boxes); else project to the NEAREST interval endpoint.
  Whole-box fallback to b0 if: any edge null-argmax, projected left >=
  projected right - MIN_SIZE (or top >= bottom - MIN_SIZE), projected box out
  of [0, 640], or non-finite values.

Eval-only probes (sensors): _zero_pixels / _zero_c2 (input-disconnect), and
_edge_shift {edge: bins} (shift one edge's position logits by whole bins —
RED/GREEN direction sensor).
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEIM = os.path.join(_ROOT, "third_party/DEIM")
for _p in (_ROOT, _DEIM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CBP_CROP_SCALE = 2.5          # width/height expansion of the crop
CBP_PATCH = 32                # pixel patch spatial size
CBP_C2_PATCH = 16             # C2 patch spatial size
CBP_N_BINS = 65               # position bins per edge
CBP_N_OUT = CBP_N_BINS + 1    # + null
CBP_SEARCH = 0.5              # +/- 0.5 x box width/height around b0 edge
CBP_INPUT = 640
CBP_MIN_SIZE = 1.0            # min projected box side (640-space px)
CBP_INTERVAL = 0.80           # central 80% interval


def crop_rois_xyxy(boxes_cxcywh):
    """b0 (...,4) cxcywh (640 space) -> xyxy of the 2.5x crop."""
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    w2 = w * CBP_CROP_SCALE
    h2 = h * CBP_CROP_SCALE
    return torch.stack([cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2], dim=-1)


def patch_pair(images, c2, boxes_cxcywh):
    """images (B,3,640,640); c2 (B,64,160,160); boxes (B,Q,4) cxcywh 640-space.
    Returns pixel patch (B,Q,3,32,32) and C2 patch (B,Q,64,16,16)."""
    B, Q = boxes_cxcywh.shape[:2]
    rois = crop_rois_xyxy(boxes_cxcywh)
    idx = torch.arange(B, device=images.device).view(B, 1).expand(B, Q).reshape(-1)
    flat = torch.cat([idx[:, None].float(), rois.reshape(B * Q, 4)], dim=-1)
    px = roi_align(images, flat, output_size=(CBP_PATCH, CBP_PATCH),
                   spatial_scale=1.0, aligned=True).reshape(B, Q, 3, CBP_PATCH, CBP_PATCH)
    flat_c2 = torch.cat([idx[:, None].float(),
                         rois.reshape(B * Q, 4) * (c2.shape[-1] / float(CBP_INPUT))],
                        dim=-1)
    c2p = roi_align(c2, flat_c2, output_size=(CBP_C2_PATCH, CBP_C2_PATCH),
                    spatial_scale=1.0, aligned=True).reshape(B, Q, 64, CBP_C2_PATCH,
                                                              CBP_C2_PATCH)
    return px, c2p


class _EdgeHead(nn.Module):
    """Takes an (N, S, C) strip (S spatial positions along the search axis)
    and outputs 66 logits."""

    def __init__(self, c_in, hidden=32):
        super().__init__()
        self.proj = nn.Linear(c_in, hidden)
        self.out = nn.Linear(hidden * 8, CBP_N_OUT)   # 8 = retained spatial size

    def forward(self, strip):
        N, S, C = strip.shape
        x = F.silu(self.proj(strip))                  # (N,S,h)
        x = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=S // 8) \
            .permute(0, 2, 1) if S > 8 else x         # (N,8,h)
        return self.out(x.reshape(N, -1))


class PeroCBP(nn.Module):
    """<= 0.30M params. forward: (pixels, c2p, query, boxes) -> (N,4,66)."""

    def __init__(self):
        super().__init__()
        self.pix_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True))                       # 32x32 -> 8x8x64
        self.c2_stem = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True))                       # 16x16 -> 16x16x32
        self.q_proj = nn.Linear(256, 32)
        self.geom_proj = nn.Linear(4, 32)
        c_strip = 64 + 32 + 32 + 32                     # pix + c2 + query + geom
        self.head_l = _EdgeHead(c_strip)
        self.head_r = _EdgeHead(c_strip)
        self.head_t = _EdgeHead(c_strip)
        self.head_b = _EdgeHead(c_strip)
        # eval-only probes
        self._zero_pixels = False
        self._zero_c2 = False
        self._edge_shift = None       # {0:l,1:r,2:t,3:b} -> int bins

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, pixels, c2p, query, boxes):
        """pixels (N,3,32,32); c2p (N,64,16,16); query (N,256);
        boxes (N,4) cxcywh 640-space. Returns (N,4,66)."""
        N = pixels.shape[0]
        if self._zero_pixels:
            pixels = torch.zeros_like(pixels)
        if self._zero_c2:
            c2p = torch.zeros_like(c2p)
        f_pix = self.pix_stem(pixels)                    # (N,64,8,8)
        f_c2 = self.c2_stem(c2p)                         # (N,32,16,16)
        # resample C2 map to 8x8 to align with the pixel map
        f_c2 = F.adaptive_avg_pool2d(f_c2, 8)
        q = self.q_proj(query).unsqueeze(-1).unsqueeze(-1).expand(N, 32, 8, 8)
        geom = torch.stack([boxes[:, 0] / CBP_INPUT, boxes[:, 1] / CBP_INPUT,
                            torch.log(boxes[:, 2].clamp(min=1e-3)),
                            torch.log(boxes[:, 3].clamp(min=1e-3))], dim=-1)
        g = self.geom_proj(geom).unsqueeze(-1).unsqueeze(-1).expand(N, 32, 8, 8)
        fused = torch.cat([f_pix, f_c2, q, g], dim=1)    # (N,160,8,8)
        # left/right read along x (mean over y); top/bottom along y (mean over x)
        strip_lr = fused.mean(dim=2).permute(0, 2, 1)    # (N,8,160)
        strip_tb = fused.mean(dim=3).permute(0, 2, 1)    # (N,8,160)
        logits = torch.stack([self.head_l(strip_lr), self.head_r(strip_lr),
                              self.head_t(strip_tb), self.head_b(strip_tb)], dim=1)
        if self._edge_shift:
            logits = logits.clone()
            for e, k in self._edge_shift.items():
                if k:
                    pos = logits[:, e, :CBP_N_BINS]
                    shifted = torch.zeros_like(pos)
                    if k > 0:
                        shifted[:, k:] = pos[:, :-k]
                    else:
                        shifted[:, :k] = pos[:, -k:]
                    logits[:, e, :CBP_N_BINS] = shifted
        return logits                                    # (N,4,66)


# ════════════════════════════════════════════════════════════════════════════
# decoding rule (double precision, no free alpha/tau)
# ════════════════════════════════════════════════════════════════════════════
def bin_centers(b0_xyxy):
    """b0 (N,4) xyxy double -> per-edge bin centers (N,4,65), edge order l,r,t,b.
    left search: [x1-0.5w, x1+0.5w]; right: [x2-0.5w, x2+0.5w]; etc."""
    x1, y1, x2, y2 = b0_xyxy.unbind(-1)
    w = (x2 - x1); h = (y2 - y1)
    t = torch.arange(CBP_N_BINS, dtype=torch.float64,
                     device=b0_xyxy.device).unsqueeze(0) / (CBP_N_BINS - 1.0)
    out = torch.empty(b0_xyxy.shape[0], 4, CBP_N_BINS, dtype=torch.float64,
                      device=b0_xyxy.device)
    out[:, 0] = (x1 - CBP_SEARCH * w).unsqueeze(1) + t * (2 * CBP_SEARCH * w).unsqueeze(1)
    out[:, 1] = (x2 - CBP_SEARCH * w).unsqueeze(1) + t * (2 * CBP_SEARCH * w).unsqueeze(1)
    out[:, 2] = (y1 - CBP_SEARCH * h).unsqueeze(1) + t * (2 * CBP_SEARCH * h).unsqueeze(1)
    out[:, 3] = (y2 - CBP_SEARCH * h).unsqueeze(1) + t * (2 * CBP_SEARCH * h).unsqueeze(1)
    return out


def _percentile_bins(p, centers, q):
    """p (N,65) probs (renormalized, no null); centers (N,65); q in (0,1).
    Linear-interpolated inverse CDF in coordinate space."""
    cdf = torch.cumsum(p, dim=-1)
    cdf = cdf / cdf[:, -1:].clamp(min=1e-12)
    target = torch.full_like(p[:, 0], q)
    idx = torch.searchsorted(cdf, target.unsqueeze(-1)).clamp(1, CBP_N_BINS - 1) \
        .squeeze(-1)                                     # (N,)
    n = torch.arange(p.shape[0], device=p.device)
    c0, c1 = cdf[n, idx - 1], cdf[n, idx]
    x0, x1 = centers[n, idx - 1], centers[n, idx]
    frac = ((target - c0) / (c1 - c0).clamp(min=1e-12)).clamp(0.0, 1.0)
    return x0 + frac * (x1 - x0)


def decode_boxes(logits, b0_xyxy):
    """logits (N,4,66) float; b0_xyxy (N,4) double (640 space).
    Returns corrected (N,4) double, fallback_mask (N,) bool (True -> b0 kept),
    moved_mask (N,) bool (True -> at least one edge changed)."""
    N = b0_xyxy.shape[0]
    device = b0_xyxy.device
    p = torch.softmax(logits.double(), dim=-1)           # (N,4,66)
    centers = bin_centers(b0_xyxy)                       # (N,4,65)
    fallback = torch.zeros(N, dtype=torch.bool, device=device)
    null_win = p.argmax(dim=-1) == CBP_N_BINS            # (N,4)
    fallback = fallback | null_win.any(dim=-1)
    pos = p[:, :, :CBP_N_BINS]
    pos = pos / pos.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    # central 80% interval [q10, q90] per edge
    lo = torch.stack([_percentile_bins(pos[:, e], centers[:, e], 0.10)
                      for e in range(4)], dim=1)
    hi = torch.stack([_percentile_bins(pos[:, e], centers[:, e], 0.90)
                      for e in range(4)], dim=1)
    b0e = torch.stack([b0_xyxy[:, 0], b0_xyxy[:, 2],
                       b0_xyxy[:, 1], b0_xyxy[:, 3]], dim=1)   # (N,4) l,r,t,b
    new_e = torch.where(b0e < lo, lo, torch.where(b0e > hi, hi, b0e))
    moved = (new_e != b0e).any(dim=-1)
    # assemble box
    corr = torch.empty_like(b0_xyxy)
    corr[:, 0] = new_e[:, 0]; corr[:, 2] = new_e[:, 1]
    corr[:, 1] = new_e[:, 2]; corr[:, 3] = new_e[:, 3]
    # conflict / out-of-bounds / illegal -> whole-box fallback
    bad = (corr[:, 2] - corr[:, 0] < CBP_MIN_SIZE) | (corr[:, 3] - corr[:, 1] < CBP_MIN_SIZE)
    bad = bad | (corr[:, 0] < 0) | (corr[:, 1] < 0) | (corr[:, 2] > CBP_INPUT) \
        | (corr[:, 3] > CBP_INPUT)
    bad = bad | ~torch.isfinite(corr).all(dim=-1)
    fallback = fallback | bad
    out = torch.where(fallback.unsqueeze(-1), b0_xyxy, corr)
    moved = moved & ~fallback
    return out, fallback, moved


def expected_edges(logits, b0_xyxy):
    """training-time expected edge coordinates (image space, null-excluded)."""
    p = torch.softmax(logits.double(), dim=-1)[:, :, :CBP_N_BINS]
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    centers = bin_centers(b0_xyxy)
    return (p * centers).sum(dim=-1)                     # (N,4) l,r,t,b


def edge_soft_targets(gt_xyxy, b0_xyxy):
    """dual-bin soft labels for the four GT edges in each b0's search window.
    gt_xyxy (N,4) double, b0_xyxy (N,4) double (640 space). Returns
    (targets (N,4,65) double, outside_mask (N,4) bool — GT edge outside the
    search window gets all mass on the nearest end bin)."""
    centers = bin_centers(b0_xyxy)                       # (N,4,65)
    lo = centers[:, :, 0]                                # (N,4)
    step = (centers[:, :, 1] - centers[:, :, 0]).clamp(min=1e-9)
    gt_e = torch.stack([gt_xyxy[:, 0], gt_xyxy[:, 2],
                        gt_xyxy[:, 1], gt_xyxy[:, 3]], dim=1)   # (N,4) l,r,t,b
    u = (gt_e - lo) / step                               # bin units (0..64)
    outside = (u < 0) | (u > CBP_N_BINS - 1)
    u = u.clamp(0.0, float(CBP_N_BINS - 1))
    i0 = u.floor().long().clamp(max=CBP_N_BINS - 2)
    w1 = (u - i0.float()).clamp(0.0, 1.0)
    N = gt_xyxy.shape[0]
    t = torch.zeros(N, 4, CBP_N_BINS, dtype=torch.float64, device=gt_xyxy.device)
    n = torch.arange(N, device=gt_xyxy.device).unsqueeze(1)
    e = torch.arange(4, device=gt_xyxy.device).unsqueeze(0)
    t[n, e, i0] = 1.0 - w1
    t[n, e, i0 + 1] += w1
    return t, outside
