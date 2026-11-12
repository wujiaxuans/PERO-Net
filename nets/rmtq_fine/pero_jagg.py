"""PERO-JAgg — dual-round evidence joint aggregation (PERO_3_JOINT_AGGREGATION_OFFICIAL50).

Leader-frozen architecture contract (task text 2026-09-24, "PERO双轮证据联合聚合").
Baseline = PERO_1 (nets/rmtq_fine/pero.py, read-only). Single structural change,
at the LAST PERO decoder layer ONLY (decoder layer 2, 0-indexed; = 第3层):

  In the last PERO layer, the two rounds' evidence is aggregated PER SAMPLING
  POINT, JOINTLY: for every sampling index k (paired across rounds by index),
  the aggregation input x_k retains
      - f1_k, f2_k   : round-1 / round-2 phase-preserving detail features
                       (verbatim from PEROCore.detail_read, head-avg positions)
      - s1_k, s2_k   : round-1 / round-2 per-point semantic values — the SAME
                       M3/M4/M5 memories interpolated at the SAME (head-averaged)
                       round positions p1m_k / p2m_k with unit weights. This is a
                       MATERIALIZATION of the existing rounds' reads (same
                       memories, same positions, no new sampling positions), NOT
                       a third/fourth read: the deformable core's attention sum
                       destroys per-point identity, so per-point aggregation
                       requires materializing it.
      - δ1_k, δ2_k   : the two rounds' position offsets (p1m_k − ref_xy,
                       p2m_k − ref_xy; their difference is the position change)
      - scale        : reference box [w, h]
      - ℓ_k          : multi-scale level identity (one-hot of the point's level)
      - φ1_k, φ2_k   : per-round sub-grid PHASE identity (the 4-phase
                       sub-attention weights; the gather formula is replicated
                       from the frozen PEROCore._detail_read_chunk — preflight
                       gate verifies einsum(φ, d) == detail_read's f)
  Aggregation weights are produced BY THE QUERY:
      w = softmax_k( <g_q(q), g_x(x_k)> / sqrt(d_attn) )
      A = Σ_k w_k · val(x_k)
  The joint result enters the ORIGINAL decoder main path additively, at the
  same insertion point as PERO_1's own aggregation term:
      o = v1 + out_proj([v1_joint, O2]) + jagg_out(A)      (jagg_out ZERO-init)
  o then feeds the ORIGINAL Gate residual + FFN + classification head + FDR
  (unchanged). NO new final scoring head, NO clutter classifier, NO extra loss,
  NO third/fourth-round read, NO new sampling positions.

Layer layout (PERO_1 semantics preserved exactly):
  layer 0 (第1层): ORIGINAL D-FINE cross-attention
  layer 1 (第2层): PERO_1 two-round forensics, VERBATIM (pero.py PEROCrossAttn
                   reused read-only; shares adapter.pero with layer 2)
  layer 2 (第3层): PERO_1 two-round forensics VERBATIM (incl. evidence-conditioned
                   second-round positions p2 = clamp(p1 + tanh(mlp_obs([q,
                   v1_joint, u]))·[w,h])) + the NEW joint aggregation term above.
                   Sole structural diff vs PERO_1.

Bootstrap / exact-recovery guarantees (by construction, preflight-verified):
  - route_off: wrappers DELEGATE to the saved original MSDeformableAttention
    -> bit-identical N2.
  - zero-init: jagg_out = 0 exactly -> o == PERO_1's output bit-exact (the
    PERO_1-EMA compat gate loads PERO_1's frozen ep50 EMA into this adapter and
    requires tensor-exact equality with the PERO_1 adapter's forward).
  - _jagg_off (eval-only): skips the jagg term -> exact PERO_1 forward at the
    same weights (same-checkpoint mechanism audit).
  - _jagg_pos_zero (eval-only): zeroes δ1/δ2/scale inputs inside the aggregation
    (secondary mechanism probe: is the position-change information used?).

Honest novelty statement (per task text): query-conditioned feature mixing is
NOT claimed as original — AdaMixer and deformable attention itself already
mix features under query-produced weights. The differentiator here is WHAT is
mixed: per-INDEX-PAIRED evidence from the two forensics rounds (round-1 read
vs the evidence-guided round-2 re-read at the same sampling index), including
the position change between the rounds, the box scale, the multi-scale level
identity and the sub-grid phase identity — an evidence-contrast aggregation,
not mere adaptive sampling.

Training: fresh COCO init + seed 1234 (NEVER continues from PERO_1/PERO_2
checkpoints; their EMAs are used ONLY read-only in the preflight compat gate).

Instrumentation (eval-only probes; no effect on the deployed path):
  owner._jagg_off (bool) / owner._jagg_pos_zero (bool) — mechanism probes;
  owner._stash_jagg (bool) + owner._last_jagg (dict) — stash jagg internals
  (f1/f2/s1/s2/d1/d2/phi1/phi2/w/A) for preflight pairing checks;
  owner._p2_perturb (callable|None) — perturbs layer-2 second-round positions
  after computation (pairing gate: round-1-side inputs must stay unchanged);
  owner._q_perturb (callable|None) — perturbs the query inside the jagg weight
  path only (weights-are-query-produced gate);
  owner._v1_perturb / _u_zero / _stash_p2 / _last_p2 — PERO_1 probes, kept at
  layer 2 (evidence-conditioning of p2 preserved).
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

from nets.rmtq_fine.pero import PEROCore, PEROCrossAttn  # noqa: E402  (read-only reuse)


# ════════════════════════════════════════════════════════════════════════════
# JointAggregationCore: per-point paired two-round aggregation (ALL new params)
# ════════════════════════════════════════════════════════════════════════════
class JointAggregationCore(nn.Module):
    """x_k = [f1|f2|s1|s2|δ1|δ2|scale|φ1|φ2|ℓ]  (d_in = 2Cd + 2d_model + 6 + 8 + n_levels)
    w = softmax_k(<g_q(q), g_x(x_k)>/sqrt(d_attn));  A = Σ_k w_k val(x_k)."""

    def __init__(self, d_model=256, d_channels=32, n_levels=3, d_attn=128, chunk=256):
        super().__init__()
        self.d_model = int(d_model)
        self.d_channels = int(d_channels)
        self.n_levels = int(n_levels)
        self.d_attn = int(d_attn)
        self.chunk = int(chunk)
        d_in = 2 * self.d_channels + 2 * self.d_model + 6 + 8 + self.n_levels
        self.d_in = d_in
        self.g_q = nn.Linear(d_model, d_attn)          # query side of the weights
        self.g_x = nn.Linear(d_in, d_attn)             # per-point side of the weights
        self.val = nn.Linear(d_in, d_model)            # per-point value projection
        self.jagg_out = nn.Linear(d_model, d_model)    # ZERO-init (bootstrap)
        nn.init.zeros_(self.jagg_out.weight)
        nn.init.zeros_(self.jagg_out.bias)

    def bind(self, num_points_list):
        """Registers the level one-hot buffer (P, n_levels) from num_points_list
        (e.g. [3,6,3] -> P=12, level(k) fixed by the sampling index)."""
        P = int(sum(num_points_list))
        lvl = torch.zeros(P, self.n_levels)
        s = 0
        for l, n in enumerate(num_points_list):
            lvl[s:s + n, l] = 1.0
            s += n
        self.register_buffer("level_onehot", lvl)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _agg_chunk(self, qc, f1c, f2c, s1c, s2c, d1c, d2c, scalec, phi1c, phi2c, lvl,
                   qperturb=None):
        B, Qc, P = f1c.shape[:3]
        dt = f1c.dtype
        scale_b = scalec[:, :, None, :].expand(B, Qc, P, 2).to(dt)
        lvl_b = lvl[None, None].expand(B, Qc, P, lvl.shape[-1]).to(dt)
        x = torch.cat([f1c.to(dt), f2c.to(dt), s1c.to(dt), s2c.to(dt),
                       d1c.to(dt), d2c.to(dt), scale_b, phi1c.to(dt), phi2c.to(dt),
                       lvl_b], dim=-1)                              # (B,Qc,P,d_in)
        q_eff = qperturb(qc) if qperturb is not None else qc
        gq = self.g_q(q_eff.to(dt))                                 # (B,Qc,d_attn)
        gx = self.g_x(x)                                            # (B,Qc,P,d_attn)
        att = torch.einsum("bqc,bqpc->bqp", gq / math.sqrt(self.d_attn), gx)
        w = F.softmax(att, dim=-1)                                  # (B,Qc,P)
        A = torch.einsum("bqp,bqpc->bqc", w.to(dt), self.val(x))    # (B,Qc,d_model)
        return A, w

    def forward(self, query, f1, f2, s1, s2, d1, d2, scale, phi1, phi2, qperturb=None):
        """query (B,Q,d_model); f1/f2 (B,Q,P,Cd); s1/s2 (B,Q,P,d_model); d1/d2 (B,Q,P,2);
        scale (B,Q,2); phi1/phi2 (B,Q,P,4). Returns A (B,Q,d_model), w (B,Q,P)."""
        Q = query.shape[1]
        lvl = self.level_onehot
        if (self.chunk > 0 and Q > self.chunk and self.training
                and torch.is_grad_enabled()):
            As, ws = [], []
            for s in range(0, Q, self.chunk):
                e = s + self.chunk
                A_c, w_c = torch.utils.checkpoint.checkpoint(
                    self._agg_chunk, query[:, s:e], f1[:, s:e], f2[:, s:e],
                    s1[:, s:e], s2[:, s:e], d1[:, s:e], d2[:, s:e], scale[:, s:e],
                    phi1[:, s:e], phi2[:, s:e], lvl,
                    qperturb, use_reentrant=False)
                As.append(A_c); ws.append(w_c)
            return torch.cat(As, 1), torch.cat(ws, 1)
        return self._agg_chunk(query, f1, f2, s1, s2, d1, d2, scale, phi1, phi2, lvl,
                               qperturb)


# ════════════════════════════════════════════════════════════════════════════
# Phase gather: replicates PEROCore._detail_read_chunk's gather + 4-phase
# sub-attention (pero.py is READ-ONLY, so the frozen formula is replicated
# here). Returns w (B,Q,P,4) and the gathered d (B,Q,P,4,Cd); preflight gate
# verifies einsum(w, d) == core.detail_read's f (bit-consistency proof).
# ════════════════════════════════════════════════════════════════════════════
def phase_gather_chunk(dd, p, qd):
    """dd (B,Cd,Hc,Wc,4); p (B,Qc,P,2) normalized (x,y); qd (B,Qc,Cd).
    Verbatim math of PEROCore._detail_read_chunk (gather + sub-attention)."""
    B, Qc, P, _ = p.shape
    Cd, Hc, Wc = dd.shape[1], dd.shape[2], dd.shape[3]
    fH, fW = 2.0 * Hc, 2.0 * Wc
    u = p.float() * torch.tensor([fW, fH], device=p.device, dtype=torch.float32) - 0.5
    cyf = torch.floor(u[..., 1] / 2).clamp(0, Hc - 1)
    cxf = torch.floor(u[..., 0] / 2).clamp(0, Wc - 1)
    cy, cx = cyf.long(), cxf.long()
    phase = torch.arange(4, device=p.device)
    idx = (cy * Wc + cx).unsqueeze(-1) * 4 + phase.view(1, 1, 1, 4)
    ddf = dd.reshape(B, Cd, -1).permute(0, 2, 1)
    bidx = torch.arange(B, device=p.device).view(B, 1, 1, 1)
    d = ddf[bidx, idx].float()                                    # (B,Qc,P,4,Cd)
    logits = torch.einsum("bqc,bqpkc->bqpk", qd.float() / math.sqrt(Cd), d)
    w = F.softmax(logits, dim=-1)                                 # (B,Qc,P,4)
    return w, d


def phase_gather(core, dd, p, qd, chunk=256):
    """Chunked (+checkpoint in training) wrapper around phase_gather_chunk."""
    Q = p.shape[1]
    if (chunk > 0 and Q > chunk and core.training and torch.is_grad_enabled()):
        ws, ds = [], []
        for s in range(0, Q, chunk):
            w_c, d_c = torch.utils.checkpoint.checkpoint(
                phase_gather_chunk, dd, p[:, s:s + chunk], qd[:, s:s + chunk],
                use_reentrant=False)
            ws.append(w_c); ds.append(d_c)
        return torch.cat(ws, 1), torch.cat(ds, 1)
    return phase_gather_chunk(dd, p, qd)


# ════════════════════════════════════════════════════════════════════════════
# JAggCrossAttn: replaces layers[2].cross_attn (decoder 第3层).
# Rounds 1-2 are op-for-op verbatim PERO_1 (same code path verified tensor-
# exact in PERO_2's frozen preflight g1); the joint aggregation term is the
# ONLY structural addition. Inactive -> DELEGATES to the original
# MSDeformableAttention (exact N2).
# ════════════════════════════════════════════════════════════════════════════
class JAggCrossAttn(nn.Module):
    def __init__(self, orig, jagg, core, owner):
        super().__init__()
        self.orig = orig                # original MSDeformableAttention (COCO init;
                                        # REGISTERED as a child -> state-dict continuity)
        object.__setattr__(self, "jagg", jagg)    # plain attr (lives once in adapter.jagg)
        object.__setattr__(self, "core", core)    # plain attr (lives once in adapter.pero)
        object.__setattr__(self, "owner", owner)  # plain attr (avoids module cycle)
        core.bind_attention_shapes(orig.num_heads, sum(orig.num_points_list))
        jagg.bind(orig.num_points_list)

    def _active(self):
        return bool(getattr(self.owner, "_pero_active", False)
                    and getattr(self.owner, "_c2", None) is not None)

    def _first_read(self, query, reference_points, value, value_spatial_shapes):
        """Replicates orig.forward exactly, also returning positions/weights
        (verbatim PERO_1 PEROCrossAttn._first_read)."""
        orig = self.orig
        bs, Len_q = query.shape[:2]
        sampling_offsets = orig.sampling_offsets(query).reshape(
            bs, Len_q, orig.num_heads, sum(orig.num_points_list), 2)
        attention_weights = orig.attention_weights(query).reshape(
            bs, Len_q, orig.num_heads, sum(orig.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)
        num_points_scale = orig.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
        offset = (sampling_offsets * num_points_scale
                  * reference_points[:, :, None, :, 2:] * orig.offset_scale)
        sampling_locations = reference_points[:, :, None, :, :2] + offset
        output = orig.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights,
            orig.num_points_list)
        return output, sampling_locations, attention_weights

    def _evidence(self, f_agg, sem, H_det, ctr_det, spr_det, a_m):
        """Evidence state from one read (SAME operator set as PERO_1 round 1;
        verbatim pero_triple._evidence)."""
        core = self.core
        Hs = -(a_m * (a_m + 1e-12).log()).sum(-1)
        Hd = H_det.mean(dim=2)
        det_n = F.normalize(core.disc_det(f_agg), dim=-1)
        sem_n = F.normalize(core.disc_sem(sem.float()), dim=-1)
        disc = (det_n - sem_n).norm(dim=-1)
        ctr = ctr_det.mean(dim=2)
        spr = spr_det.mean(dim=2)
        return torch.stack([Hd, Hs, disc, ctr[..., 0], ctr[..., 1], spr], dim=-1)

    def _per_point_semantic(self, value, value_spatial_shapes, p):
        """Materializes the per-point semantic values at the (head-averaged)
        round positions p (B,Q,P,2) -> s (B,Q,P,d_model). Single-level core
        calls with unit weights: SAME memories, SAME positions, NO new sampling
        positions (a materialization of the existing round read, not a new
        round)."""
        orig = self.orig
        outs = []
        s_idx = 0
        for l, n_l in enumerate(orig.num_points_list):
            pl = p[..., s_idx:s_idx + n_l, :]                       # (B,Q,n_l,2)
            B, Q, n_l_, _ = pl.shape
            loc = pl.reshape(B, Q * n_l_, 1, 1, 2).expand(B, Q * n_l_, orig.num_heads, 1, 2)
            loc = loc.contiguous()
            wt = torch.ones(B, Q * n_l_, orig.num_heads, 1, device=pl.device, dtype=loc.dtype)
            h, w = int(value_spatial_shapes[l][0]), int(value_spatial_shapes[l][1])
            s_l = orig.ms_deformable_attn_core([value[l]], [(h, w)], loc, wt, [1])
            outs.append(s_l.reshape(B, Q, n_l_, -1))                # (B,Q,n_l,d_model)
            s_idx += n_l
        return torch.cat(outs, dim=2)                               # (B,Q,P,d_model)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        if not self._active():
            return self.orig(query, reference_points, value, value_spatial_shapes)
        owner = self.owner
        core = self.core
        jagg = self.jagg
        owner._pero_steps += 1
        bs, Q = query.shape[:2]
        Hn = self.orig.num_heads
        Pn = sum(self.orig.num_points_list)
        scale = reference_points[:, :, None, :, 2:]                 # (B,Q,1,2) = [w,h]

        # ────────── ROUND 1 (verbatim PERO_1) ──────────
        v1, p1, a = self._first_read(query, reference_points, value, value_spatial_shapes)
        dd = owner._detail_memory()
        qd = core.q_d(query)
        p1m = p1.mean(dim=2)                                        # (B,Q,P,2)
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
        if getattr(owner, "_p2_perturb", None) is not None:         # pairing probe
            p2 = owner._p2_perturb(p2)
        if owner._stash_p2:
            owner._last_p2 = p2.detach()

        # ────────── ROUND 2 (verbatim PERO_1 second read) ──────────
        a2 = F.softmax(core.attn2(query).reshape(bs, Q, Hn, Pn), dim=-1)
        v2_sem = self.orig.ms_deformable_attn_core(
            value, value_spatial_shapes, p2, a2, self.orig.num_points_list)
        p2m = p2.mean(dim=2)                                        # (B,Q,P,2)
        f2, H2, ctr2, spr2 = core.detail_read(dd, p2m, qd)
        f2_agg = f2.mean(dim=2)
        O2 = v2_sem + core.det_out2(f2_agg.to(v2_sem.dtype))

        # ────────── JOINT AGGREGATION (the ONLY structural addition) ──────────
        jagg_term = None
        A = None; w = None
        if not getattr(owner, "_jagg_off", False):
            s1 = self._per_point_semantic(value, value_spatial_shapes, p1m)
            s2 = self._per_point_semantic(value, value_spatial_shapes, p2m)
            phi1, _ = phase_gather(core, dd, p1m, qd)               # (B,Q,P,4)
            phi2, _ = phase_gather(core, dd, p2m, qd)
            ref_xy = reference_points[:, :, 0, :2]                  # (B,Q,2)
            scale_q = reference_points[:, :, 0, 2:]                 # (B,Q,2)
            d1 = p1m - ref_xy[:, :, None, :]                        # round-1 offsets
            d2 = p2m - ref_xy[:, :, None, :]                        # round-2 offsets
            if getattr(owner, "_jagg_pos_zero", False):             # mechanism probe
                d1 = torch.zeros_like(d1); d2 = torch.zeros_like(d2)
                scale_q = torch.zeros_like(scale_q)
            qperturb = getattr(owner, "_q_perturb", None)
            A, w = jagg(query, f1, f2, s1, s2, d1, d2, scale_q, phi1, phi2,
                        qperturb=qperturb)
            jagg_term = jagg.jagg_out(A)
            if owner._stash_jagg:
                owner._last_jagg = {"f1": f1.detach(), "f2": f2.detach(),
                                    "s1": s1.detach(), "s2": s2.detach(),
                                    "d1": d1.detach(), "d2": d2.detach(),
                                    "phi1": phi1.detach(), "phi2": phi2.detach(),
                                    "w": w.detach(), "A": A.detach(),
                                    "jagg_term": jagg_term.detach()}

        # output: o = v1 + Proj([v1_joint, O2]) + jagg_out(A)  (jagg_out ZERO-init)
        o = v1 + core.out_proj(torch.cat([v1_joint, O2], dim=-1))
        if jagg_term is not None:
            o = o + jagg_term
        return o


# ════════════════════════════════════════════════════════════════════════════
# JointAggregationAdapter: wraps the DEIM model.
# layer 1 (decoder 第2层) -> PERO_1 PEROCrossAttn (two rounds, UNCHANGED)
# layer 2 (decoder 第3层) -> JAggCrossAttn (PERO_1 two rounds + joint aggregation)
# ════════════════════════════════════════════════════════════════════════════
class JointAggregationAdapter(nn.Module):
    """model = DEIM (backbone/encoder/decoder loop untouched). New params:
    pero (PERO_1's 292,960, shared by layers 1-2) + jagg (the joint aggregation
    core, used by layer 2 ONLY).

    forward(x, targets=None, route_off=False):
        route_off=False -> PERO-JAgg active (layer 1 two-round PERO_1;
            layer 2 two-round PERO_1 + joint aggregation)
        route_off=True  -> every wrapper delegates to the ORIGINAL
            MSDeformableAttention -> exact N2.
    """

    def __init__(self, model, c2_channels=64, d_channels=32, layer_ids=(1, 2),
                 chunk=256, d_attn=128):
        super().__init__()
        self.model = model
        self.pero = PEROCore(c2_channels=c2_channels, d_channels=d_channels,
                             chunk=chunk)
        n_levels = model.decoder.decoder.layers[2].cross_attn.num_levels
        self.jagg = JointAggregationCore(d_model=256, d_channels=d_channels,
                                         n_levels=n_levels, d_attn=d_attn, chunk=chunk)
        self.layer_ids = tuple(layer_ids)
        self._enabled = True
        self._pero_active = False
        self._c2 = None
        self._dd = None
        self._pero_steps = 0
        # instrumentation
        self._stash_p2 = False
        self._last_p2 = None
        self._v1_perturb = None       # PERO_1 probe (round-1 evidence)
        self._u_zero = False          # PERO_1 probe
        self._p2_perturb = None       # pairing probe (layer-2 second-round positions)
        self._q_perturb = None        # jagg weight path probe
        self._jagg_off = False        # mechanism ablation: exact PERO_1 forward
        self._jagg_pos_zero = False   # mechanism probe: zero δ1/δ2/scale inputs
        self._stash_jagg = False
        self._last_jagg = None
        # non-modifying capture hook on the stride-4 stage (stage1: 64ch @ stride 4)
        self._c2_hook = model.backbone.stages[0].register_forward_hook(self._capture_c2)
        dec = model.decoder.decoder
        # layer idx 1 (第2层): PERO_1 two-round forensics, VERBATIM reuse
        dec.layers[1].cross_attn = PEROCrossAttn(dec.layers[1].cross_attn, self.pero, self)
        # layer idx 2 (第3层): two-round + joint aggregation (the only change)
        dec.layers[2].cross_attn = JAggCrossAttn(dec.layers[2].cross_attn, self.jagg,
                                                 self.pero, self)

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
