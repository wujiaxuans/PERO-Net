#!/usr/bin/env python3
"""Refinement-head training script (self-contained).

Trains one refinement head on top of the frozen PERO-R detector:

  --method plainreg  direct xyxy regression head (paper main model, + damping)
  --method cbp       PERO-CBP v5 (65-bin edge distributions + validity)

Usage:
    python train.py --method plainreg --data_root ./data/IRDST/images \
                    --peror_weights ./weights/peroR_ep50_ema.pth \
                    --output_dir ./outputs
"""
import argparse, json, math, os, sys, time, random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ─── path setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "third_party", "DEIM"))

from engine.core import YAMLConfig
from engine.data.transforms import _transforms as TT
from engine.optim.ema import ModelEMA
from nets.rmtq_fine.pero_r import PERORAdapter
from nets.rmtq_fine.pero_cbp_v2 import PeroCBPv2, expected_edges_v2
from nets.rmtq_fine.pero_cbp import edge_soft_targets, CBP_N_BINS
from nets.rmtq_fine.pero_plainreg import PlainRegHead

# ─── constants (fixed for the paper) ────────────────────────────────────────
INPUT_SIZE = 640
BATCH = 16
LR = 1e-4
LR_END = 1e-5
WD = 1e-4
WARMUP = 1000
EMA_DECAY = 0.9999
EPOCHS = 50
SEED = 1234
W_POS, W_VAL, W_CONS = 1.0, 1.0, 0.1
N_NULL_TOP, N_NULL_RND, N_PERT = 8, 8, 2
PERT_TRANS, PERT_SCALE = 0.35, (0.7, 1.3)
MATCH_MIN_IOU = 0.1
N_QUERIES = 300


def _patch_transforms():
    """Make DEIM's ConvertPILImage / ConvertBoxes work with torchvision>=0.17.

    Their v2 base class only dispatches through ``transform()``; DEIM (written
    for torchvision 0.15/0.16) implements ``_transform()`` only. torchvision
    already filters inputs by ``_transformed_types`` before calling
    ``transform()``, so aliasing is sufficient.
    """
    for cls in (TT.ConvertPILImage, TT.ConvertBoxes):
        cls.transform = cls._transform


def build_model(args, ann_file):
    """Build frozen PERO-R + capture hooks."""
    _patch_transforms()
    cfg_path = os.path.join(SCRIPT_DIR, "configs", "deim_dfine_s.yml")
    cfg = YAMLConfig(cfg_path)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = args.data_root
    cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = ann_file
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = BATCH
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 4
    model = cfg.model.to(args.device)
    adapter = PERORAdapter(model, c2_channels=64, d_channels=32,
                           layer_ids=(1, 2), chunk=256).to(args.device)
    # load frozen weights
    raw = torch.load(args.peror_weights, map_location="cpu")
    md = adapter.state_dict()
    loaded = {k for k in md if k in raw["model"] and md[k].shape == raw["model"][k].shape}
    assert len(loaded) == len(md), f"PERO-R load incomplete: {len(loaded)}/{len(md)}"
    md.update({k: raw["model"][k] for k in loaded})
    adapter.load_state_dict(md)
    adapter.eval()
    # capture hooks
    cap = {"c2": None, "query": None}
    cap["h1"] = model.backbone.stages[0].register_forward_hook(
        lambda m, i, o: cap.__setitem__("c2", o.detach()))
    dec = model.decoder.decoder
    cap["h3"] = dec.layers[dec.eval_idx].register_forward_hook(
        lambda m, i, o: cap.__setitem__("query", o.detach()))
    return cfg, adapter, cap


def forward_frozen(adapter, cap, samples, post, orig):
    """One no-grad forward through frozen PERO-R."""
    with torch.no_grad():
        out = adapter(samples, route_off=False)
        res = post(out, orig)
        B = samples.shape[0]
        logits = out["pred_logits"][..., 0].float()
        boxes_m = (out["pred_boxes"] * INPUT_SIZE).float()
        scores = torch.sigmoid(logits)
        _, idx = torch.topk(scores, scores.shape[1], dim=-1)
        post_boxes = torch.stack([r["boxes"] for r in res], 0).float()
        post_scores = torch.stack([r["scores"] for r in res], 0).float()
        queries = cap["query"].float()
    return {"logits": logits, "boxes_m": boxes_m, "idx": idx,
            "post_boxes": post_boxes, "post_scores": post_scores,
            "queries": queries}


def _cxcywh_to_xyxy(b):
    return np.array([b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2])


def _iou_mat(a, b):
    ix = np.maximum(0, np.minimum(a[:, None, 2], b[None, :, 2])
                    - np.maximum(a[:, None, 0], b[None, :, 0]))
    iy = np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3])
                    - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = ix * iy
    ua = ((a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]))[:, None] \
        + ((b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1]))[None, :] - inter
    return np.where(ua > 0, inter / ua, 0.0)


def orig_to_model(boxes, W, H):
    out = np.array(boxes, dtype=np.float64)
    out[:, 0] *= INPUT_SIZE / W; out[:, 2] *= INPUT_SIZE / W
    out[:, 1] *= INPUT_SIZE / H; out[:, 3] *= INPUT_SIZE / H
    return out


def build_samples(cb, targets, gt_by_img):
    """Hungarian matched + perturbations + nulls."""
    pos, null = [], []
    for b, t in enumerate(targets):
        iid = int(t["image_id"].item())
        gts = gt_by_img.get(iid, [])
        W = float(t["orig_size"][0]); H = float(t["orig_size"][1])
        boxes_m = cb["boxes_m"][b].cpu().numpy().astype(np.float64)
        scores = torch.sigmoid(cb["logits"][b]).cpu().numpy()
        if gts:
            gt_m = orig_to_model(np.array(gts, float), W, H)
            bm_xyxy = np.stack([boxes_m[:, 0]-boxes_m[:, 2]/2,
                                boxes_m[:, 1]-boxes_m[:, 3]/2,
                                boxes_m[:, 0]+boxes_m[:, 2]/2,
                                boxes_m[:, 1]+boxes_m[:, 3]/2], 1)
            iou_m = _iou_mat(bm_xyxy, gt_m)
            ri, ci = linear_sum_assignment(1.0 - iou_m)
            assign = {int(r): int(g) for r, g in zip(ri, ci) if iou_m[r, g] >= MATCH_MIN_IOU}
        else:
            assign = {}
        for qi, gi in assign.items():
            pos.append({"b": b, "qi": qi, "gt": gt_m[gi],
                        "b0": _cxcywh_to_xyxy(boxes_m[qi]), "key": (iid, qi)})
        unm = [q for q in range(N_QUERIES) if q not in assign]
        by_score = sorted(unm, key=lambda q: -scores[q])[:N_NULL_TOP]
        rng = np.random.RandomState((iid * 2654435761) % (2**31))
        pool = [q for q in unm if q not in set(by_score)]
        rnd = list(rng.choice(pool, size=min(N_NULL_RND, len(pool)),
                              replace=False)) if pool else []
        for q in list(by_score) + list(rnd):
            null.append({"b": b, "qi": q, "b0": _cxcywh_to_xyxy(boxes_m[q])})
    return pos, null


def make_pert(pos):
    pert = []
    for s in pos:
        iid, qi = s["key"]
        for k in range(N_PERT):
            rng = np.random.RandomState((iid*1000003 + qi*101 + k) % (2**31))
            dx = rng.uniform(-PERT_TRANS, PERT_TRANS)
            dy = rng.uniform(-PERT_TRANS, PERT_TRANS)
            sc = rng.uniform(*PERT_SCALE)
            b = s["b0"]
            w = (b[2]-b[0])*sc; h = (b[3]-b[1])*sc
            cx = (b[0]+b[2])/2 + dx*(b[2]-b[0])
            cy = (b[1]+b[3])/2 + dy*(b[3]-b[1])
            pert.append({"b": s["b"], "qi": s["qi"], "gt": s["gt"],
                         "b0": np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2]),
                         "key": (iid, qi), "k": k})
    return pert


def roi_crop(images, c2, samples, device):
    """Gather pixel + C2 patches for a flat sample list."""
    from torchvision.ops import roi_align
    if not samples:
        z = torch.zeros(0, 3, 32, 32, device=device)
        return z, z, torch.zeros(0, 4, device=device)
    idx = torch.tensor([s["b"] for s in samples], device=device).float()
    b0 = torch.tensor(np.stack([s["b0"] for s in samples]), device=device).float()
    # ROI = the predicted box itself (640-space), as in the reference method.
    rois = torch.cat([idx[:, None], b0], dim=-1)
    px = roi_align(images, rois, output_size=(32, 32), spatial_scale=1.0, aligned=True)
    scale = c2.shape[-1] / INPUT_SIZE
    c2p = roi_align(c2, rois*scale, output_size=(16, 16), spatial_scale=1.0, aligned=True)
    cxcywh = torch.stack([(b0[:, 0]+b0[:, 2])/2, (b0[:, 1]+b0[:, 3])/2,
                          b0[:, 2]-b0[:, 0], b0[:, 3]-b0[:, 1]], dim=-1)
    return px, c2p, cxcywh


def compute_loss(module, pos_s, pert_s, null_s, images, c2, queries, device):
    parts = {}
    all_s = pos_s + pert_s
    n_pos = len(pos_s)
    loss = torch.tensor(0.0, device=device)
    if all_s:
        px, c2p, boxes = roi_crop(images, c2, all_s, device)
        q = torch.stack([queries[s["b"], s["qi"]] for s in all_s])
        pos_logits, val = module(px, c2p, q, boxes)
        b0d = torch.tensor(np.stack([s["b0"] for s in all_s]),
                           dtype=torch.float64, device=device)
        gtd = torch.tensor(np.stack([s["gt"] for s in all_s]),
                           dtype=torch.float64, device=device)
        tgt, _ = edge_soft_targets(gtd, b0d)
        ce_pos = -(tgt * F.log_softmax(pos_logits, -1)).sum(-1).mean()
        val_pos = F.binary_cross_entropy_with_logits(val, torch.ones_like(val))
        parts["ce_pos"] = float(ce_pos.detach())
        # consistency
        ee = expected_edges_v2(pos_logits, b0d)
        keyidx = {}
        for j, s in enumerate(pert_s):
            keyidx.setdefault(s["key"], []).append(n_pos + j)
        diffs = []
        for i, s in enumerate(pos_s):
            for pj in keyidx.get(s["key"], []):
                diffs.append(((ee[i] - ee[pj]) ** 2).sum(-1))
        cons = torch.stack(diffs).mean() if diffs else torch.tensor(0.0, device=device)
        parts["cons"] = float(cons.detach()) if torch.is_tensor(cons) else 0.0
        loss = W_POS * ce_pos + 0.5 * W_VAL * val_pos + W_CONS * cons
    if null_s:
        pxn, c2n, boxesn = roi_crop(images, c2, null_s, device)
        qn = torch.stack([queries[s["b"], s["qi"]] for s in null_s])
        _, valn = module(pxn, c2n, qn, boxesn)
        val_null = F.binary_cross_entropy_with_logits(valn, torch.zeros_like(valn))
        parts["val_null"] = float(val_null.detach())
        loss = loss + 0.5 * W_VAL * val_null
    return loss, parts


def lr_at(step, total):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(total - WARMUP, 1)
    return LR_END + (LR - LR_END) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def lr_at_plainreg(ep, bi, ipe, total):
    """Reference schedule of the released PlainReg head.

    NOTE: the reference run ramps the warmup from the start of *every* epoch
    (``bi < WARMUP``) instead of once globally. Kept verbatim so the released
    checkpoint can be reproduced; do not "fix" it if you compare numbers.
    """
    if bi < WARMUP:
        return LR * min(1, (bi + 1) / WARMUP)
    t = (ep * ipe + bi - WARMUP) / max(total - WARMUP, 1)
    return LR_END + (LR - LR_END) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def compute_loss_plainreg(module, pos_s, pert_s, images, c2, queries, device):
    """L1 on xyxy, normalized per edge by the base box side (reference loss)."""
    all_s = pos_s + pert_s
    if not all_s:
        return torch.tensor(0.0, device=device), {}
    px, c2p, boxes = roi_crop(images, c2, all_s, device)
    q = torch.stack([queries[s["b"], s["qi"]] for s in all_s])
    pred = module(px, c2p, q, boxes)
    b0d = torch.tensor(np.stack([s["b0"] for s in all_s]),
                       dtype=torch.float64, device=device).float()
    gtd = torch.tensor(np.stack([s["gt"] for s in all_s]),
                       dtype=torch.float64, device=device).float()
    w_ = (b0d[:, 2] - b0d[:, 0]).clamp(min=1e-3).unsqueeze(1)
    h_ = (b0d[:, 3] - b0d[:, 1]).clamp(min=1e-3).unsqueeze(1)
    scale = torch.cat([w_, h_, w_, h_], dim=1)
    loss = F.l1_loss(pred / scale, gtd / scale)
    return loss, {"l1": float(loss.detach())}


def main():
    parser = argparse.ArgumentParser(description="Train a PERO refinement head")
    parser.add_argument("--method", choices=("plainreg", "cbp"), default="plainreg",
                        help="plainreg = paper main model, cbp = PERO-CBP v5")
    parser.add_argument("--data_root", required=True, help="IRDST images dir")
    parser.add_argument("--train_ann", default="splits/train_42odd_coco.json")
    parser.add_argument("--peror_weights", required=True, help="PERO-R ep50 EMA path")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.train_ann = os.path.join(SCRIPT_DIR, args.train_ann)
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    # load GT annotations
    tj = json.load(open(args.train_ann))
    gt_by_img = {}
    for a in tj["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_img.setdefault(a["image_id"], []).append((x, y, x + w, y + h))

    # build frozen model
    cfg, adapter, cap = build_model(args, args.train_ann)
    post = cfg.postprocessor
    loader = cfg.val_dataloader
    print(f"[train] ipe={len(loader)} epochs={args.epochs} "
          f"total={len(loader)*args.epochs}")

    # build head
    module = (PlainRegHead() if args.method == "plainreg" else PeroCBPv2()).to(args.device)
    print(f"[train] method={args.method} params={sum(p.numel() for p in module.parameters())}")
    opt = torch.optim.AdamW(module.parameters(), lr=LR, weight_decay=WD)
    ema = ModelEMA(module, decay=EMA_DECAY, warmups=1000)
    total_steps = len(loader) * args.epochs
    gstep = 0

    t0 = time.time()
    for ep in range(args.epochs):
        module.train()
        ep_loss = []
        for bi, (samples, targets) in enumerate(loader):
            samples = samples.to(args.device)
            orig = torch.stack([t["orig_size"] for t in targets], 0).to(args.device)
            with torch.no_grad():
                cb = forward_frozen(adapter, cap, samples, post, orig)
            pos_s, null_s = build_samples(cb, targets, gt_by_img)
            pert_s = make_pert(pos_s)
            if args.method == "plainreg":
                loss, _ = compute_loss_plainreg(module, pos_s, pert_s, samples,
                                                cap["c2"], cb["queries"], args.device)
                lr = lr_at_plainreg(ep, bi, len(loader), total_steps)
            else:
                loss, _ = compute_loss(module, pos_s, pert_s, null_s,
                                       samples, cap["c2"], cb["queries"], args.device)
                lr = lr_at(gstep, total_steps)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss ep{ep} it{bi}")
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.1)
            opt.step(); ema.update(module)
            gstep += 1
            ep_loss.append(float(loss))
            if (bi + 1) % 300 == 0:
                print(f"  ep{ep} it{bi+1}/{len(loader)} loss={float(loss):.4f} "
                      f"lr={lr:.6f} elapsed={(time.time()-t0)/60:.1f}min")
        print(f"== ep{ep} loss={np.mean(ep_loss):.4f} "
              f"elapsed={(time.time()-t0)/60:.1f}min")

    # save
    fname = ("plainreg_ep%02d_ema.pth" % args.epochs if args.method == "plainreg"
             else "perocbp_ep%02d_ema.pth" % args.epochs)
    ema_path = os.path.join(args.output_dir, fname)
    torch.save({"model": ema.module.state_dict()}, ema_path)
    print(f"[done] saved {ema_path} ({(time.time()-t0)/60:.0f}min)")


if __name__ == "__main__":
    main()
