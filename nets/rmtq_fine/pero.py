"""PERO — Phase-preserving Evidence Re-observation cross-attention (PERO_1_OFFICIAL_IRDST50).

Paper-candidate structure for the IRDST official 42/43 protocol. Sole structural
diff vs N2 (D-FINE-S, configs/rmtq_fine/deim_dfine_s_protocol_s.yml): the
cross-attention of decoder layers 1 and 2 (0-indexed; = design doc "第2、3层") is
replaced by PERO cross-attention — evidence-conditioned re-observation. NO
P2-upsample-concat, NO C2 extra detection head, NO Δscore / Δbox at the output,
NO query-only offset prediction, NO fixed clutter selector, NO global learnable
alpha, NO temporal/PCST/motion (single frame).

Architecture (frozen a-priori; preregistered before any training). Design doc:
results/IRDST/PERO_1_DESIGN/PERO_DETR_design.md (leader-frozen text).

  1. KEEP D-FINE-S backbone / HybridEncoder / M3-M5 semantic memories / 3-layer
     decoder / self-attn / FFN / Gate / FDR / LQE / GO-LSD / matcher / criterion /
     postprocessor untouched (read-only). Layer 0 cross-attn stays original.
  2. Detail memory (module 1): from the backbone stride-4 shallow feature C2
     (HGNetv2-B0 stage1 output, 64ch @ H/4 x W/4; captured with a non-modifying
     forward hook on backbone.stages[0], NEVER entering the HybridEncoder):
         D[y,x,p] = C2[:, :, 2y+dy(p), 2x+dx(p)],  p ∈ {(0,0),(0,1),(1,0),(1,1)}
         D = [B, 64, Hc, Wc, 4]   (Hc=H/4/2; at 640: [B,64,80,80,4])
         D_detail = Conv1x1(D)    (pointwise channel proj 64→32, weights SHARED
                                   across all positions AND all 4 phases — the
                                   four sub-cells keep separate identities and
                                   separate memory slots; nothing is averaged
                                   or folded into channels)
     D_detail never produces a heatmap and never directly edits a predicted box.
  3. First read (module 2): the ORIGINAL COCO-init MSDeformableAttention of the
     layer computes v_i^(1) and its sampling positions / attention weights
         p_ij^(1) = reference_point(b_i^l) + offset_ij(q_i^l, b_i^l)   (original)
     Detail is read at the SAME p_ij^(1) (head-averaged), phase-preserving:
     enclosing coarse cell (y,x) → gather the 4 sub-vectors → sub-attention
         w_p = softmax_p( <q_d, d_p> / sqrt(Cd) )    (q_d = Linear(q))
         f = Σ_p w_p d_p (first-read detail feature)
     Joint first observation: v1_joint = v^(1) + det_out1(f_agg).
  4. Evidence state (module 3):
         u_i = [detail attention entropy, semantic attention entropy,
                detail-semantic discrepancy (L2 distance of L2-normalized
                64-d projections of f_agg and v^(1)), detail support center
                (2D, weighted mean sub-cell offset vs the sampling point, fine
                pixels), detail support spread (weighted RMS offset)]
     u_i encodes ONLY "is the first-read evidence concentrated / consistent /
     sufficient" — it is never a target probability and never suppresses scores.
  5. Evidence-conditioned second sampling:
         Δp_ij = tanh(MLP_obs([q_i, v1_joint, u_i])) ⊙ [w_i, h_i]
         p_ij^(2) = clamp(p_ij^(1) + Δp_ij, 0, 1)
     The offset depends on the ACTUAL first-read evidence (v1_joint and u_i),
     NOT on the query alone (preflight gate enforces RED→GREEN on this).
  6. Second read (same memories): semantic via deformable core on M3/M4/M5 at
     p^(2) with new attention weights softmax(Linear(q)); detail via the SAME
     phase-preserving sub-grid read at p^(2). v^(2) = v2_sem + det_out2(f2_agg).
  7. Output:  o_i = v^(1) + Proj([v1_joint, v^(2)])
     Proj ZERO-init (weight AND bias; bootstrap-compatible formulation of the
     design's o_i = Proj([v^(1), v^(2)]): the v^(1) skip preserves the COCO-init
     original attention, so at init route_on ≡ N2 bit-exact and training
     bootstraps Proj first). o_i then enters the ORIGINAL decoder residual
     (Gate) and FFN; classification + FDR + LQE + GO-LSD unchanged.

One PEROCore (all new params) is SHARED by decoder layers 1 and 2. DN queries
are treated uniformly (no GT anywhere in the PERO path; positions/weights always
come from the model's own predictions). No GT exists at inference.

Exact-recovery guarantees (by construction, verified in preflight gates):
  - route_off : the PEROCrossAttn wrapper DELEGATES to the saved original
    MSDeformableAttention.forward -> bit-identical N2.
  - zero-init : Proj = 0 exactly -> o_i = v^(1) == original output bit-exact.
  - causal    : with Proj exposed, perturbing C2 / v^(1) / u changes outputs /
    second sampling positions; with Proj zero, perturbing C2 changes NOTHING.

Instrumentation (eval-only probes, no effect on the deployed path):
  owner._v1_perturb (callable|None): perturbs v^(1) inside the wrapper after the
    first read (tests evidence-conditioning of p^(2)); owner._u_zero (bool):
    zeroes the evidence state u; owner._last_p2: detached second sampling
    positions of the last wrapper call.
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


# ════════════════════════════════════════════════════════════════════════════
# PEROCore: detail memory + phase-preserving reads + evidence + re-observation
# (ALL new parameters; shared by decoder layers 1 and 2)
# ════════════════════════════════════════════════════════════════════════════
class PEROCore(nn.Module):
    def __init__(self, c2_channels=64, d_channels=32, d_model=256, disc_dim=64,
                 hidden=128, chunk=256):
        super().__init__()
        self.c2_channels = c2_channels
        self.d_channels = d_channels
        self.d_model = d_model
        self.chunk = int(chunk)

        # module 1: D_detail = Conv1x1(D)  (pointwise channel projection,
        # shared across positions AND the 4 phases; phase identity preserved)
        self.detail_proj = nn.Linear(c2_channels, d_channels)

        # phase-preserving sub-attention (q_d = Linear(q))
        self.q_d = nn.Linear(d_model, d_channels)
        # second-read semantic attention weights (replaces nothing; new head)
        self.attn2 = None            # built lazily (needs n_head/total_points)
        # evidence-state helpers
        self.disc_det = nn.Linear(d_channels, disc_dim)
        self.disc_sem = nn.Linear(d_model, disc_dim)
        # detail aggregation outputs
        self.det_out1 = nn.Linear(d_channels, d_model)
        self.det_out2 = nn.Linear(d_channels, d_model)
        # evidence-conditioned re-observation offsets (MLP_obs)
        self.mlp_obs = None          # built lazily
        # final projection: ZERO-init (bootstrap)
        self.out_proj = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # -- lazy heads (need runtime shapes of the wrapped attention) ----------
    def bind_attention_shapes(self, n_head, total_points):
        if self.attn2 is None:
            self.attn2 = nn.Linear(self.d_model, n_head * total_points)
            # MLP_obs: [q, v1_joint, u] (2*d_model + 6) -> 2 per (head, point)
            self.mlp_obs = nn.Sequential(
                nn.Linear(2 * self.d_model + 6, 128), nn.SiLU(),
                nn.Linear(128, n_head * total_points * 2),
            )
            # keep modules on the same device/dtype as the rest
            dev = self.out_proj.weight.device
            self.attn2.to(dev)
            self.mlp_obs.to(dev)

    # -- module 1: sub-grid detail memory ------------------------------------
    def build_detail_memory(self, c2):
        """c2 (B, C, H, W) stride-4 feature -> D_detail (B, Cd, Hc, Wc, 4).

        D[y,x,p] = C2[:, :, 2y+dy(p), 2x+dx(p)]; the 4 sub-cells are NEVER
        averaged and NEVER folded into channels — they remain 4 separate memory
        slots addressable by their fine spatial coordinates.
        """
        B, C, H, W = c2.shape
        Hc, Wc = H // 2, W // 2
        c2 = c2[:, :, :2 * Hc, :2 * Wc].float()                 # defensive (640 -> no-op)
        d = c2.view(B, C, Hc, 2, Wc, 2).permute(0, 1, 2, 4, 3, 5) \
              .reshape(B, C, Hc, Wc, 4)                          # (B,C,Hc,Wc,4)
        dd = self.detail_proj(d.permute(0, 2, 3, 4, 1))          # Conv1x1 over channels
        return dd.permute(0, 4, 1, 2, 3).contiguous()            # (B,Cd,Hc,Wc,4)

    # -- phase-preserving detail read (per query chunk) ----------------------
    def _detail_read_chunk(self, dd, p, qd):
        """dd (B,Cd,Hc,Wc,4); p (B,Qc,P,2) normalized (x,y); qd (B,Qc,Cd).

        Returns f (B,Qc,P,Cd), H (B,Qc,P) entropy, ctr (B,Qc,P,2) support
        center, spread (B,Qc,P) — all fp32.
        """
        B, Qc, P, _ = p.shape
        Cd, Hc, Wc = dd.shape[1], dd.shape[2], dd.shape[3]
        fH, fW = 2.0 * Hc, 2.0 * Wc                               # fine grid (stride 4)
        u = p.float() * torch.tensor([fW, fH], device=p.device, dtype=torch.float32) - 0.5
        cyf = torch.floor(u[..., 1] / 2).clamp(0, Hc - 1)
        cxf = torch.floor(u[..., 0] / 2).clamp(0, Wc - 1)
        cy, cx = cyf.long(), cxf.long()
        phase = torch.arange(4, device=p.device)                  # (dy,dx): 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
        dy = (phase // 2).view(1, 1, 1, 4).float()
        dx = (phase % 2).view(1, 1, 1, 4).float()
        # gather the 4 sub-vectors of the enclosing coarse cell
        idx = (cy * Wc + cx).unsqueeze(-1) * 4 + phase.view(1, 1, 1, 4)   # (B,Qc,P,4)
        ddf = dd.reshape(B, Cd, -1).permute(0, 2, 1)               # (B, Hc*Wc*4, Cd)
        bidx = torch.arange(B, device=p.device).view(B, 1, 1, 1)
        d = ddf[bidx, idx].float()                                 # (B,Qc,P,4,Cd)
        # sub-attention over the 4 phases (query-conditioned)
        logits = torch.einsum("bqc,bqpkc->bqpk",
                              qd.float() / math.sqrt(Cd), d)
        w = F.softmax(logits, dim=-1)                              # (B,Qc,P,4)
        f = torch.einsum("bqpk,bqpkc->bqpc", w, d)                 # (B,Qc,P,Cd)
        H = -(w * (w + 1e-12).log()).sum(-1)                       # (B,Qc,P)
        # support offsets vs the sampling point, fine pixels
        ey = (2 * cyf).unsqueeze(-1) + dy - u[..., 1].unsqueeze(-1)
        ex = (2 * cxf).unsqueeze(-1) + dx - u[..., 0].unsqueeze(-1)
        ctr_y = (w * ey).sum(-1, keepdim=True)
        ctr_x = (w * ex).sum(-1, keepdim=True)
        spread = ((w * ((ey - ctr_y) ** 2 + (ex - ctr_x) ** 2)).sum(-1) + 1e-12).sqrt()
        ctr = torch.stack([ctr_x.squeeze(-1), ctr_y.squeeze(-1)], dim=-1)   # (B,Qc,P,2)
        return f, H, ctr, spread

    def detail_read(self, dd, p, qd):
        """Chunked (training+grad) wrapper around _detail_read_chunk."""
        Q = p.shape[1]
        if (self.chunk > 0 and Q > self.chunk and self.training
                and torch.is_grad_enabled()):
            fs, Hs, cs, ss = [], [], [], []
            for s in range(0, Q, self.chunk):
                f_c, H_c, c_c, s_c = torch.utils.checkpoint.checkpoint(
                    self._detail_read_chunk, dd, p[:, s:s + self.chunk],
                    qd[:, s:s + self.chunk], use_reentrant=False)
                fs.append(f_c); Hs.append(H_c); cs.append(c_c); ss.append(s_c)
            return torch.cat(fs, 1), torch.cat(Hs, 1), torch.cat(cs, 1), torch.cat(ss, 1)
        return self._detail_read_chunk(dd, p, qd)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════════════════════
# PEROCrossAttn: replaces layers[1].cross_attn / layers[2].cross_attn.
# Inactive -> DELEGATES to the saved original MSDeformableAttention (exact N2).
# ════════════════════════════════════════════════════════════════════════════
class PEROCrossAttn(nn.Module):
    def __init__(self, orig, core, owner):
        super().__init__()
        self.orig = orig                # original MSDeformableAttention (COCO init;
                                        # REGISTERED as a child so its weights keep
                                        # state-dict continuity under .orig.*)
        # core / owner are kept as PLAIN attributes (NOT registered as child modules):
        #   - core  lives (once) in adapter.pero; registering it here as well would
        #     duplicate it in the module tree (double optimizer registration)
        #   - owner is the adapter itself; registering it would create a module cycle
        #     (adapter -> model -> wrapper -> owner -> adapter) -> infinite recursion
        #     in Module._apply / deepcopy
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "owner", owner)
        core.bind_attention_shapes(orig.num_heads, sum(orig.num_points_list))

    def _active(self):
        return bool(getattr(self.owner, "_pero_active", False)
                    and getattr(self.owner, "_c2", None) is not None)

    def _first_read(self, query, reference_points, value, value_spatial_shapes):
        """Replicates orig.forward exactly, also returning positions/weights."""
        orig = self.orig
        bs, Len_q = query.shape[:2]
        sampling_offsets = orig.sampling_offsets(query).reshape(
            bs, Len_q, orig.num_heads, sum(orig.num_points_list), 2)
        attention_weights = orig.attention_weights(query).reshape(
            bs, Len_q, orig.num_heads, sum(orig.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)
        # reference_points (B, Q, 1, 4) -> 4-dim branch of MSDeformableAttention
        num_points_scale = orig.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
        offset = (sampling_offsets * num_points_scale
                  * reference_points[:, :, None, :, 2:] * orig.offset_scale)
        sampling_locations = reference_points[:, :, None, :, :2] + offset
        output = orig.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights,
            orig.num_points_list)
        return output, sampling_locations, attention_weights

    def forward(self, query, reference_points, value, value_spatial_shapes):
        if not self._active():
            return self.orig(query, reference_points, value, value_spatial_shapes)
        owner = self.owner
        core = self.core
        owner._pero_steps += 1

        # ── first read (original attention; exact v^(1), positions, weights) ──
        v1, p1, a = self._first_read(query, reference_points, value, value_spatial_shapes)

        # ── detail memory (lazy, once per forward) ──
        dd = owner._detail_memory()

        # ── phase-preserving detail read at p^(1) + evidence state u ──
        qd = core.q_d(query)
        p1m = p1.mean(dim=2)                                   # (B,Q,P,2) head-avg
        f1, H1, ctr1, spr1 = core.detail_read(dd, p1m, qd)
        f1_agg = f1.mean(dim=2)                                # (B,Q,Cd)
        # instrumentation probe (eval-only): perturb the first-read evidence;
        # u / v1_joint / MLP_obs input all see the perturbed v1, while the o=v1
        # skip keeps the ORIGINAL v1 (isolates the re-observation pathway)
        v1_ev = owner._v1_perturb(v1) if getattr(owner, "_v1_perturb", None) is not None else v1
        v1_joint = v1_ev + core.det_out1(f1_agg.to(v1.dtype))  # joint first observation
        a_m = a.float().mean(dim=2)                            # (B,Q,P)
        Hs = -(a_m * (a_m + 1e-12).log()).sum(-1)              # semantic entropy (B,Q)
        Hd = H1.mean(dim=2)                                    # detail entropy (B,Q)
        det_n = F.normalize(core.disc_det(f1_agg), dim=-1)
        sem_n = F.normalize(core.disc_sem(v1_ev.float()), dim=-1)
        disc = (det_n - sem_n).norm(dim=-1)                    # discrepancy (B,Q)
        ctr = ctr1.mean(dim=2)                                 # (B,Q,2)
        spr = spr1.mean(dim=2)                                 # (B,Q)
        u = torch.stack([Hd, Hs, disc, ctr[..., 0], ctr[..., 1], spr], dim=-1)  # (B,Q,6)
        if getattr(owner, "_u_zero", False):
            u = torch.zeros_like(u)

        # ── evidence-conditioned second sampling positions ──
        mlp_in = torch.cat([query, v1_joint, u.to(query.dtype)], dim=-1)
        dp = torch.tanh(core.mlp_obs(mlp_in))                  # (B,Q,H*P*2)
        bs, Q = query.shape[:2]
        Hn = self.orig.num_heads
        Pn = sum(self.orig.num_points_list)
        dp = dp.reshape(bs, Q, Hn, Pn, 2)
        scale = reference_points[:, :, None, :, 2:]            # (B,Q,1,2) = [w,h]
        p2 = (p1 + dp * scale).clamp(0.0, 1.0)
        if owner._stash_p2:
            owner._last_p2 = p2.detach()

        # ── second read on the SAME memories ──
        a2 = F.softmax(core.attn2(query).reshape(bs, Q, Hn, Pn), dim=-1)
        v2_sem = self.orig.ms_deformable_attn_core(
            value, value_spatial_shapes, p2, a2, self.orig.num_points_list)
        p2m = p2.mean(dim=2)
        f2, _, _, _ = core.detail_read(dd, p2m, qd)
        v2 = v2_sem + core.det_out2(f2.mean(dim=2).to(v2_sem.dtype))

        # ── output: o = v^(1) + Proj([v1_joint, v^(2)]) (Proj zero-init) ──
        o = v1 + core.out_proj(torch.cat([v1_joint, v2], dim=-1))
        return o


# ════════════════════════════════════════════════════════════════════════════
# Adapter: wraps the DEIM model; non-modifying C2 hook + cross-attn wrapping
# ════════════════════════════════════════════════════════════════════════════
class PERODetectorAdapter(nn.Module):
    """model = DEIM (backbone/encoder/decoder loop untouched). New params: pero only.

    forward(x, targets=None, route_off=False):
        route_off=False -> PERO active: C2 hooked, decoder layers 1/2 use
            evidence-conditioned re-observation cross-attention.
        route_off=True  -> wrappers delegate to the ORIGINAL MSDeformableAttention
            -> exact N2.
    """

    def __init__(self, model, c2_channels=64, d_channels=32, layer_ids=(1, 2),
                 chunk=256):
        super().__init__()
        self.model = model
        self.pero = PEROCore(c2_channels=c2_channels, d_channels=d_channels,
                             chunk=chunk)
        self.layer_ids = tuple(layer_ids)
        self._enabled = True          # master switch
        self._pero_active = False     # per-forward effective flag
        self._c2 = None               # hooked stride-4 feature (current forward)
        self._dd = None               # lazy detail memory (current forward)
        self._pero_steps = 0          # instrumentation: wrappers actually applied
        self._stash_p2 = False        # instrumentation: stash second positions
        self._last_p2 = None
        self._v1_perturb = None       # eval-only probe
        self._u_zero = False          # eval-only probe
        # non-modifying capture hook on the stride-4 stage (stage1: 64ch @ stride 4)
        self._c2_hook = model.backbone.stages[0].register_forward_hook(self._capture_c2)
        # wrap cross-attention of decoder layers 1 and 2 (design: "第2、3层")
        dec = model.decoder.decoder
        for i in self.layer_ids:
            orig = dec.layers[i].cross_attn
            dec.layers[i].cross_attn = PEROCrossAttn(orig, self.pero, self)

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
