"""PERO PlainReg head — the paper's main refinement head.

Same inputs / stems / training budget as PERO-CBP, but a single direct
xyxy regression instead of 65-bin edge distributions + validity gating:

    fused 8x8 map  ->  global average pool  ->  256 -> 128 MLP  ->  4 deltas
    edge_i = base_edge_i + delta_i * 0.5 * (box width or height)

The head is trained on top of the frozen PERO-R detector (matched queries +
geometric perturbations), and at deploy time its boxes are blended with the
detector's boxes by the fixed damping constants (lc, ls):

    centre' = c0 + lc * (c1 - c0)
    size'   = s0 + ls * (s1 - s0)

The constants are fixed per dataset (IRDST: lc=0.5, ls=0.75; they were selected
on a sequence-isolated dev split and are applied as deployment constants —
no per-image selector).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEIM = os.path.join(_ROOT, "third_party/DEIM")
for _p in (_ROOT, _DEIM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

INPUT_SIZE = 640
DAMP_LC = 0.5      # IRDST centre damping
DAMP_LS = 0.75     # IRDST size damping


class PlainRegHead(nn.Module):
    """Direct xyxy regression head (same inputs as PeroCBPv2)."""

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
        c_in = 64 + 32 + 32 + 32
        self.trunk = nn.Sequential(nn.Linear(c_in, 256), nn.SiLU(),
                                   nn.Linear(256, 128), nn.SiLU())
        self.out = nn.Linear(128, 4)     # dx1, dy1, dx2, dy2 (x 0.5 * box side)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, pixels, c2p, query, boxes):
        """pixels (N,3,32,32), c2p (N,64,16,16), query (N,256),
        boxes (N,4) cxcywh in 640-space -> refined xyxy in 640-space."""
        N = pixels.shape[0]
        f_pix = self.pix_stem(pixels)
        f_c2 = F.adaptive_avg_pool2d(self.c2_stem(c2p), 8)
        q = self.q_proj(query).unsqueeze(-1).unsqueeze(-1).expand(N, 32, 8, 8)
        geom = torch.stack([boxes[:, 0] / INPUT_SIZE, boxes[:, 1] / INPUT_SIZE,
                            torch.log(boxes[:, 2].clamp(min=1e-3)),
                            torch.log(boxes[:, 3].clamp(min=1e-3))], dim=-1)
        g = self.geom_proj(geom).unsqueeze(-1).unsqueeze(-1).expand(N, 32, 8, 8)
        fused = torch.cat([f_pix, f_c2, q, g], dim=1)
        feat = fused.mean(dim=(2, 3))
        delta = self.out(self.trunk(feat))
        cx, cy, w, h = boxes.unbind(-1)
        dx1 = delta[:, 0] * 0.5 * w
        dy1 = delta[:, 1] * 0.5 * h
        dx2 = delta[:, 2] * 0.5 * w
        dy2 = delta[:, 3] * 0.5 * h
        return torch.stack([cx - w / 2 + dx1, cy - h / 2 + dy1,
                            cx + w / 2 + dx2, cy + h / 2 + dy2], dim=-1)


def damp_boxes(raw_xyxy, ref_xyxy, lc=DAMP_LC, ls=DAMP_LS):
    """Fixed-constant blend of the base boxes (raw) and the refined boxes (ref).

    Accepts numpy arrays or tensors of shape (N, 4) in the same coordinate
    system and returns the same container type. lc interpolates centres,
    ls interpolates widths/heights.
    """
    is_np = isinstance(raw_xyxy, np.ndarray)
    raw = torch.as_tensor(np.asarray(raw_xyxy) if is_np else raw_xyxy,
                          dtype=torch.float64)
    ref = torch.as_tensor(np.asarray(ref_xyxy) if is_np else ref_xyxy,
                          dtype=torch.float64)
    cxr = (raw[..., 0] + raw[..., 2]) / 2
    cyr = (raw[..., 1] + raw[..., 3]) / 2
    cxf = (ref[..., 0] + ref[..., 2]) / 2
    cyf = (ref[..., 1] + ref[..., 3]) / 2
    wr = raw[..., 2] - raw[..., 0]
    hr = raw[..., 3] - raw[..., 1]
    wf = ref[..., 2] - ref[..., 0]
    hf = ref[..., 3] - ref[..., 1]
    cx = cxr + lc * (cxf - cxr)
    cy = cyr + lc * (cyf - cyr)
    w = wr + ls * (wf - wr)
    h = hr + ls * (hf - hr)
    out = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    out = torch.where(torch.isfinite(out), out, raw)
    return out.numpy() if is_np else out
