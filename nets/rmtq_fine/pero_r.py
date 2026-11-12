"""PERO-R — sub-cell relation readout in the LAST decoder layer's SECOND-round
detail read (PERO_R_OFFICIAL_50).

Leader-frozen architecture contract (task text 2026-09-26). Baseline = PERO_3
(nets/rmtq_fine/pero_jagg.py, read-only). The SOLE structural change is at
decoder layer 2 (0-indexed; 第3层, the LAST PERO layer), in the SECOND-round
phase-preserving detail read:

  Existing round-2 read (PERO_3, kept verbatim in structure): gather the 4
  sub-vectors d_p of the enclosing coarse cell at the round-2 head-averaged
  positions p2m, sub-attention w = softmax_p(<qd, d_p>/sqrt(Cd)), f2 = Σ_p w_p
  d_p, f2_agg = mean over points, O2 = v2_sem + det_out2(f2_agg).

  NEW (the only change): a small SHARED local relation readout INSIDE each
  group of 4 sub-cells, computed from
      - the sub-cell features d_i (Cd=32 each),
      - pairwise feature differences d_i − d_j,
      - the sub-cells' RELATIVE POSITIONS (their fine-pixel offsets vs the
        sampling point; pairwise differences pos_i − pos_j),
      - the corresponding per-point semantic feature s2_k (the round-2
        per-point semantic value at p2m, projected),
      - the query's FIRST-ROUND observation state ([v1_joint, u1], projected),
  producing one relation vector per ordered pair (i,j), i≠j (12 pairs):
      r_ij = MLP_r([d_i, d_j, d_i−d_j, pos_i−pos_j, proj_s(s2_k), proj_state(q-state)])
  aggregated per sub-cell (R_i = Σ_{j≠i} r_ij) and ENCODED BACK additively while
  preserving the original sub-cell content:
      d'_i = d_i + MLP_e([d_i, R_i])        (MLP_e ZERO-init -> at init d' == d
                                             bit-exact; bootstrap-compatible:
                                             PERO-R at init / with the PERO_3 EMA
                                             loaded == PERO_3 tensor-exact)
  The sub-attention / aggregation path is UNCHANGED and runs on d': w' =
  softmax(<qd, d'_p>), f2' = Σ w' d'_p -> det_out2 -> O2. NO new scoring head,
  NO direct edit of cls logits / box coords / FDR, NO change of sampling
  positions or count or budget, NO global scalar coefficient (per-sub-cell MLP
  residual only). Layer 1 (第2层) stays strictly PERO_1. PERO_3's jagg is kept
  (its round-2 inputs f2'/phi2' naturally reflect the relation-augmented read).

Instrumentation (eval-only probes; no effect on the deployed path):
  owner._rel_off (bool): skip the relation (d' = d) -> exact PERO_3 forward;
  owner._rel_pos_zero (bool): SABOTAGE — zeroes the pairwise position input in
    MLP_r (content-only relations are then permutation-equivariant; the shuffle
    sensor must go RED under this sabotage);
  owner._rel_shuffle (None | list perm): permutes the round-2 sub-cell CONTENTS
    among the four phase slots (positions kept) — shuffle sensor;
  owner._rel_state_perturb (callable|None): perturbs the first-round state
    input INSIDE the relation only (state-conditioning sensor);
  owner._stash_rel (bool) + owner._last_rel (dict): stashes p2m, d2, pos2,
    delta (Δ), w2, f2 for the sensors.
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

from nets.rmtq_fine.pero import PEROCore, PEROCrossAttn          # read-only reuse
from nets.rmtq_fine.pero_jagg import (JointAggregationCore,       # read-only reuse
                                      JAggCrossAttn)

# 12 ordered pairs (i, j), i != j, over the 4 phases
_PAIR_I = (0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3)
_PAIR_J = (1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2)


# ════════════════════════════════════════════════════════════════════════════
# RelationCore: per-group-of-4-sub-cell local relation readout (ALL new params)
# ════════════════════════════════════════════════════════════════════════════
class RelationCore(nn.Module):
    def __init__(self, d_model=256, d_channels=32, d_sem=32, d_state=64, d_rel=64,
                 chunk=256):
        super().__init__()
        self.d_model = int(d_model)
        self.d_channels = int(d_channels)
        self.d_rel = int(d_rel)
        self.chunk = int(chunk)
        self.proj_s = nn.Linear(d_model, d_sem)                    # per-point semantic
        self.proj_state = nn.Linear(d_model + 6, d_state)          # [v1_joint, u1]
        d_pair = 3 * d_channels + 2 + d_sem + d_state              # 194
        self.mlp_r = nn.Sequential(
            nn.Linear(d_pair, d_rel), nn.SiLU(), nn.Linear(d_rel, d_rel))
        self.mlp_e = nn.Linear(d_channels + d_rel, d_channels)     # ZERO-init
        nn.init.zeros_(self.mlp_e.weight)
        nn.init.zeros_(self.mlp_e.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _chunk(self, d, pos, s2, v1j, u1, pos_zero, state_perturb):
        """d (B,Qc,P,4,Cd); pos (B,Qc,P,4,2); s2 (B,Qc,P,d_model); v1j (B,Qc,d_model);
        u1 (B,Qc,6). Returns d2 = d + Δ and Δ."""
        B, Qc, P, _, Cd = d.shape
        dev = d.device
        dt = d.dtype
        pi = torch.tensor(_PAIR_I, device=dev)
        pj = torch.tensor(_PAIR_J, device=dev)
        di = d[:, :, :, pi, :]                                     # (B,Qc,P,12,Cd)
        dj = d[:, :, :, pj, :]
        posd = pos[:, :, :, pi, :] - pos[:, :, :, pj, :]           # (B,Qc,P,12,2)
        if pos_zero:                                               # sabotage probe
            posd = torch.zeros_like(posd)
        ps = self.proj_s(s2.to(dt))                                # (B,Qc,P,d_sem)
        st_in = torch.cat([v1j.to(dt), u1.to(dt)], dim=-1)
        if state_perturb is not None:
            st_in = state_perturb(st_in)
        st = self.proj_state(st_in)                                # (B,Qc,d_state)
        st_b = st[:, :, None, :].expand(B, Qc, P, st.shape[-1])
        r_in = torch.cat([di, dj, di - dj, posd.to(dt),
                          ps.unsqueeze(-2).expand(B, Qc, P, 12, ps.shape[-1]),
                          st_b.unsqueeze(-2).expand(B, Qc, P, 12, st.shape[-1])], dim=-1)
        r = self.mlp_r(r_in)                                       # (B,Qc,P,12,d_rel)
        # deterministic per-slot aggregation: pairs are ordered by i (3 j's each)
        R = r.view(B, Qc, P, 4, 3, self.d_rel).sum(dim=4)          # (B,Qc,P,4,d_rel)
        e_in = torch.cat([d, R], dim=-1)                           # (B,Qc,P,4,Cd+d_rel)
        delta = self.mlp_e(e_in)                                   # ZERO-init
        return d + delta, delta

    def forward(self, d, pos, s2, v1_joint, u1, pos_zero=False, state_perturb=None):
        """Chunked (+checkpoint in training) wrapper. d/pos/s2 (B,Q,P,...);
        v1_joint (B,Q,d_model); u1 (B,Q,6). Returns (d2, delta)."""
        Q = d.shape[1]
        if (self.chunk > 0 and Q > self.chunk and self.training
                and torch.is_grad_enabled()):
            ds, es = [], []
            for s in range(0, Q, self.chunk):
                e = s + self.chunk
                d_c, e_c = torch.utils.checkpoint.checkpoint(
                    self._chunk, d[:, s:e], pos[:, s:e], s2[:, s:e],
                    v1_joint[:, s:e], u1[:, s:e], bool(pos_zero), state_perturb,
                    use_reentrant=False)
                ds.append(d_c); es.append(e_c)
            return torch.cat(ds, 1), torch.cat(es, 1)
        return self._chunk(d, pos, s2, v1_joint, u1, pos_zero, state_perturb)


# ════════════════════════════════════════════════════════════════════════════
# Sub-cell gather: replicates PEROCore._detail_read_chunk's frozen gather
# formula (pero.py READ-ONLY), returning the 4 sub-vectors d AND their fine-pixel
# offsets vs the sampling point (the sub-cells' relative positions).
# ════════════════════════════════════════════════════════════════════════════
def subcell_gather_chunk(dd, p):
    """dd (B,Cd,Hc,Wc,4); p (B,Qc,P,2) normalized (x,y). Returns d (B,Qc,P,4,Cd),
    pos (B,Qc,P,4,2) with pos[...,0]=ex, pos[...,1]=ey (fine pixels, sub-cell
    offset vs the sampling point) — verbatim math of PEROCore._detail_read_chunk."""
    B, Qc, P, _ = p.shape
    Cd, Hc, Wc = dd.shape[1], dd.shape[2], dd.shape[3]
    fH, fW = 2.0 * Hc, 2.0 * Wc
    u = p.float() * torch.tensor([fW, fH], device=p.device, dtype=torch.float32) - 0.5
    cyf = torch.floor(u[..., 1] / 2).clamp(0, Hc - 1)
    cxf = torch.floor(u[..., 0] / 2).clamp(0, Wc - 1)
    cy, cx = cyf.long(), cxf.long()
    phase = torch.arange(4, device=p.device)
    dy = (phase // 2).view(1, 1, 1, 4).float()
    dx = (phase % 2).view(1, 1, 1, 4).float()
    idx = (cy * Wc + cx).unsqueeze(-1) * 4 + phase.view(1, 1, 1, 4)
    ddf = dd.reshape(B, Cd, -1).permute(0, 2, 1)
    bidx = torch.arange(B, device=p.device).view(B, 1, 1, 1)
    d = ddf[bidx, idx].float()                                     # (B,Qc,P,4,Cd)
    ey = (2 * cyf).unsqueeze(-1) + dy - u[..., 1].unsqueeze(-1)    # (B,Qc,P,4)
    ex = (2 * cxf).unsqueeze(-1) + dx - u[..., 0].unsqueeze(-1)
    pos = torch.stack([ex, ey], dim=-1)                            # (B,Qc,P,4,2)
    return d, pos


def subcell_gather(core, dd, p, chunk=256):
    Q = p.shape[1]
    if (chunk > 0 and Q > chunk and core.training and torch.is_grad_enabled()):
        ds, ps = [], []
        for s in range(0, Q, chunk):
            d_c, p_c = torch.utils.checkpoint.checkpoint(
                subcell_gather_chunk, dd, p[:, s:s + chunk], use_reentrant=False)
            ds.append(d_c); ps.append(p_c)
        return torch.cat(ds, 1), torch.cat(ps, 1)
    return subcell_gather_chunk(dd, p)


# ════════════════════════════════════════════════════════════════════════════
# RCrossAttn: replaces layers[2].cross_attn. Identical to PERO_3's
# JAggCrossAttn EXCEPT the round-2 detail read is relation-augmented (the sole
# structural change). Inactive -> DELEGATES to the original
# MSDeformableAttention (exact N2). rel zero-init -> exact PERO_3.
# ════════════════════════════════════════════════════════════════════════════
class RCrossAttn(JAggCrossAttn):
    def __init__(self, orig, jagg, rel, core, owner):
        # register rel as a PLAIN attribute (lives once in adapter.rel; avoids
        # double registration — same pattern as core/owner in pero.py)
        super().__init__(orig, jagg, core, owner)
        object.__setattr__(self, "rel", rel)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        if not self._active():
            return self.orig(query, reference_points, value, value_spatial_shapes)
        owner = self.owner
        core = self.core
        jagg = self.jagg
        rel = self.rel
        owner._pero_steps += 1
        bs, Q = query.shape[:2]
        Hn = self.orig.num_heads
        Pn = sum(self.orig.num_points_list)
        scale = reference_points[:, :, None, :, 2:]

        # ────────── ROUND 1 (verbatim PERO_3 / PERO_1) ──────────
        v1, p1, a = self._first_read(query, reference_points, value, value_spatial_shapes)
        dd = owner._detail_memory()
        qd = core.q_d(query)
        p1m = p1.mean(dim=2)
        f1, H1, ctr1, spr1 = core.detail_read(dd, p1m, qd)
        f1_agg = f1.mean(dim=2)
        v1_ev = owner._v1_perturb(v1) if getattr(owner, "_v1_perturb", None) is not None else v1
        v1_joint = v1_ev + core.det_out1(f1_agg.to(v1.dtype))
        u1 = self._evidence(f1_agg, v1_ev, H1, ctr1, spr1, a.float().mean(dim=2))
        if getattr(owner, "_u_zero", False):
            u1 = torch.zeros_like(u1)

        # x2 = x(1) + R * tanh(f_obs(q, O1, u1))   [evidence-conditioned; VERBATIM]
        mlp_in = torch.cat([query, v1_joint, u1.to(query.dtype)], dim=-1)
        dp2 = torch.tanh(core.mlp_obs(mlp_in)).reshape(bs, Q, Hn, Pn, 2)
        p2 = (p1 + dp2 * scale).clamp(0.0, 1.0)
        if getattr(owner, "_p2_perturb", None) is not None:
            p2 = owner._p2_perturb(p2)
        if owner._stash_p2:
            owner._last_p2 = p2.detach()
        p2m = p2.mean(dim=2)                                        # (B,Q,P,2)

        # ────────── ROUND 2 (relation-augmented detail read — the ONLY change) ──────────
        a2_pre = None
        s1 = self._per_point_semantic(value, value_spatial_shapes, p1m)
        s2 = self._per_point_semantic(value, value_spatial_shapes, p2m)
        # gather the 4 sub-cells at p2m (frozen formula replication)
        d2, pos2 = subcell_gather(core, dd, p2m)                    # (B,Q,P,4,Cd),(B,Q,P,4,2)
        if getattr(owner, "_rel_shuffle", None) is not None:        # shuffle sensor
            perm = torch.tensor(owner._rel_shuffle, device=d2.device)
            d2 = d2.index_select(3, perm)
        rel_off = getattr(owner, "_rel_off", False)
        if rel_off:
            d2r, delta = d2, torch.zeros_like(d2)
        else:
            pos_zero = getattr(owner, "_rel_pos_zero", False)
            st_pert = getattr(owner, "_rel_state_perturb", None)
            d2r, delta = rel(d2, pos2, s2, v1_joint, u1, pos_zero=pos_zero,
                             state_perturb=st_pert)
        # EXISTING readout path on the relation-augmented sub-cells
        logits2 = torch.einsum("bqc,bqpkc->bqpk", qd.float() / math.sqrt(d2r.shape[-1]), d2r)
        phi2 = F.softmax(logits2, dim=-1)                           # (B,Q,P,4)
        f2 = torch.einsum("bqpk,bqpkc->bqpc", phi2, d2r)            # (B,Q,P,Cd)
        if owner._stash_rel:
            owner._last_rel = {"p2m": p2m.detach(), "d2": d2.detach(),
                               "pos2": pos2.detach(), "delta": delta.detach(),
                               "phi2": phi2.detach(), "f2": f2.detach()}
        # round-2 semantic read at p2 (verbatim) + existing merge into query
        a2 = F.softmax(core.attn2(query).reshape(bs, Q, Hn, Pn), dim=-1)
        v2_sem = self.orig.ms_deformable_attn_core(
            value, value_spatial_shapes, p2, a2, self.orig.num_points_list)
        f2_agg = f2.mean(dim=2)
        O2 = v2_sem + core.det_out2(f2_agg.to(v2_sem.dtype))

        # ────────── JOINT AGGREGATION (verbatim PERO_3; inputs reflect the new read) ──────────
        jagg_term = None
        A = None; w = None
        if not getattr(owner, "_jagg_off", False):
            ref_xy = reference_points[:, :, 0, :2]
            scale_q = reference_points[:, :, 0, 2:]
            d1o = p1m - ref_xy[:, :, None, :]
            d2o = p2m - ref_xy[:, :, None, :]
            if getattr(owner, "_jagg_pos_zero", False):
                d1o = torch.zeros_like(d1o); d2o = torch.zeros_like(d2o)
                scale_q = torch.zeros_like(scale_q)
            phi1 = _phi1_round1(core, dd, p1m, qd)                   # round-1 phase weights
            qperturb = getattr(owner, "_q_perturb", None)
            A, w = jagg(query, f1, f2, s1, s2, d1o, d2o, scale_q, phi1, phi2,
                        qperturb=qperturb)
            jagg_term = jagg.jagg_out(A)
            if owner._stash_jagg:
                owner._last_jagg = {"f1": f1.detach(), "f2": f2.detach(),
                                    "s1": s1.detach(), "s2": s2.detach(),
                                    "d1": d1o.detach(), "d2": d2o.detach(),
                                    "phi1": phi1.detach(), "phi2": phi2.detach(),
                                    "w": w.detach(), "A": A.detach(),
                                    "jagg_term": jagg_term.detach()}

        # output: o = v1 + Proj([v1_joint, O2]) + jagg_out(A)  (PERO_3 formula)
        o = v1 + core.out_proj(torch.cat([v1_joint, O2], dim=-1))
        if jagg_term is not None:
            o = o + jagg_term
        return o


def _phi1_round1(core, dd, p1m, qd):
    """Round-1 phase weights via the frozen gather formula (PERO_3 computes
    phi1 the same way through pero_jagg.phase_gather; replicated here so the
    relation module file is self-contained; bit-exact vs pero.py's f path)."""
    from nets.rmtq_fine.pero_jagg import phase_gather
    w, _ = phase_gather(core, dd, p1m, qd)
    return w


# ════════════════════════════════════════════════════════════════════════════
# PERORAdapter: wraps the DEIM model.
# layer 1 -> PERO_1 PEROCrossAttn (UNCHANGED); layer 2 -> RCrossAttn.
# ════════════════════════════════════════════════════════════════════════════
class PERORAdapter(nn.Module):
    """New params: pero (292,960, shared L1/L2) + jagg (326,784, == PERO_3's) +
    rel (the relation readout, layer-2 round-2 ONLY)."""

    def __init__(self, model, c2_channels=64, d_channels=32, layer_ids=(1, 2),
                 chunk=256, d_attn=128):
        super().__init__()
        self.model = model
        self.pero = PEROCore(c2_channels=c2_channels, d_channels=d_channels,
                             chunk=chunk)
        n_levels = model.decoder.decoder.layers[2].cross_attn.num_levels
        self.jagg = JointAggregationCore(d_model=256, d_channels=d_channels,
                                         n_levels=n_levels, d_attn=d_attn, chunk=chunk)
        self.rel = RelationCore(d_model=256, d_channels=d_channels, chunk=chunk)
        self.layer_ids = tuple(layer_ids)
        self._enabled = True
        self._pero_active = False
        self._c2 = None
        self._dd = None
        self._pero_steps = 0
        # instrumentation (PERO_3 probes + relation sensors)
        self._stash_p2 = False
        self._last_p2 = None
        self._v1_perturb = None
        self._u_zero = False
        self._p2_perturb = None
        self._q_perturb = None
        self._jagg_off = False
        self._jagg_pos_zero = False
        self._stash_jagg = False
        self._last_jagg = None
        self._rel_off = False
        self._rel_pos_zero = False
        self._rel_shuffle = None
        self._rel_state_perturb = None
        self._stash_rel = False
        self._last_rel = None
        # non-modifying capture hook on the stride-4 stage
        self._c2_hook = model.backbone.stages[0].register_forward_hook(self._capture_c2)
        dec = model.decoder.decoder
        dec.layers[1].cross_attn = PEROCrossAttn(dec.layers[1].cross_attn, self.pero, self)
        dec.layers[2].cross_attn = RCrossAttn(dec.layers[2].cross_attn, self.jagg,
                                              self.rel, self.pero, self)

    def _capture_c2(self, module, inputs, output):
        if self._pero_active:
            self._c2 = output

    def _detail_memory(self):
        if self._dd is None:
            assert self._c2 is not None, "PERO active but C2 not captured"
            self._dd = self.pero.build_detail_memory(self._c2)
        return self._dd

    def forward(self, x, targets=None, route_off=False):
        prev = self._pero_active
        self._pero_active = self._enabled and not route_off
        try:
            self._c2 = None
            self._dd = None
            self._last_p2 = None
            self._last_jagg = None
            self._last_rel = None
            out = self.model(x, targets=targets)
            return out
        finally:
            self._pero_active = prev
            self._c2 = None
            self._dd = None

    def set_enabled(self, v):
        self._enabled = bool(v)

    def n_pero_params(self):
        return self.pero.n_params()

    def n_jagg_params(self):
        return self.jagg.n_params()

    def n_rel_params(self):
        return self.rel.n_params()
