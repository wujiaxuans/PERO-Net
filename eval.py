#!/usr/bin/env python3
"""Evaluation script (self-contained).

Runs the frozen PERO-R detector **once** over a COCO annotation file (default:
the 43-even IRDST split) and evaluates one or both refinement heads on top of
the identical candidate set:

  --plainreg_weights   PlainReg head (paper main model). With ``--lc/--ls`` the
                       refined boxes are blended with the base boxes by the
                       fixed damping constants; ``--lc 1 --ls 1`` is the
                       undamped (lambda = 1) variant.
  --cbp_weights        PERO-CBP v5 (edge distributions + validity gating).

Scores are never modified: every state keeps the detector's scores bitwise and
only box coordinates change.

Usage:
    python eval.py --data_root ./data/IRDST/images \
        --peror_weights    ./weights/peroR_ep50_ema.pth \
        --plainreg_weights ./weights/plainreg_ep50_ema.pth \
        --cbp_weights      ./weights/perocbp_v5_ep50_ema.pth
"""
import argparse
import io
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

# ─── path setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "third_party", "DEIM"))

from engine.core import YAMLConfig
from engine.data.transforms import _transforms as TT
from nets.rmtq_fine.pero_r import PERORAdapter
from nets.rmtq_fine.pero_cbp_v2 import PeroCBPv2, decode_v5
from nets.rmtq_fine.pero_plainreg import PlainRegHead, damp_boxes

INPUT_SIZE = 640
BATCH = 16
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
    raw = torch.load(args.peror_weights, map_location="cpu")
    md = adapter.state_dict()
    loaded = {k for k in md if k in raw["model"] and md[k].shape == raw["model"][k].shape}
    assert len(loaded) == len(md)
    md.update({k: raw["model"][k] for k in loaded})
    adapter.load_state_dict(md)
    adapter.eval()
    cap = {"c2": None, "query": None}
    cap["h1"] = model.backbone.stages[0].register_forward_hook(
        lambda m, i, o: cap.__setitem__("c2", o.detach()))
    dec = model.decoder.decoder
    cap["h3"] = dec.layers[dec.eval_idx].register_forward_hook(
        lambda m, i, o: cap.__setitem__("query", o.detach()))
    return cfg, adapter, cap


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def coco_eval(states, val_ann):
    """COCO AP (pycocotools) + fixed-conf 0.5 greedy P/R/F1 for every state."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(val_ann)
    gt_by_img = {}
    for a in gt.dataset["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_img.setdefault(a["image_id"], []).append((a["id"], (x, y, x+w, y+h)))
    results = {}
    for name, preds in states.items():
        dt = [{"image_id": int(i), "category_id": 0,
               "bbox": [p[0], p[1], p[2]-p[0], p[3]-p[1]], "score": p[4]}
              for i, ps in preds.items() for p in ps]
        cd = gt.loadRes(dt)
        ce = COCOeval(gt, cd, "bbox")
        with contextlib.redirect_stdout(io.StringIO()):
            ce.evaluate(); ce.accumulate(); ce.summarize()
        results[name] = {"AP50": float(ce.stats[1]), "AP75": float(ce.stats[2]),
                         "AP50:95": float(ce.stats[0])}
        tp = fp = fn = 0
        for iid, gts in gt_by_img.items():
            ps = sorted([p for p in preds.get(iid, []) if p[4] >= 0.5],
                        key=lambda p: -p[4])
            matched = [False] * len(gts)
            for p in ps:
                best, bi = 0.0, -1
                for gi, (aid, gb) in enumerate(gts):
                    if matched[gi]:
                        continue
                    v = _iou(p[:4], gb)
                    if v > best:
                        best, bi = v, gi
                if best >= 0.5:
                    matched[bi] = True; tp += 1
                else:
                    fp += 1
            fn += matched.count(False)
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        results[name]["conf05"] = {"TP": tp, "FP": fp, "FN": fn,
                                   "P": pr, "R": rc,
                                   "F1": 2*pr*rc/(pr+rc) if pr+rc else 0}
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PERO-R refinement heads")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--val_ann", default="splits/val_43even_coco.json")
    parser.add_argument("--peror_weights", required=True)
    parser.add_argument("--plainreg_weights", default=None,
                        help="PlainReg head (paper main model)")
    parser.add_argument("--cbp_weights", default=None, help="PERO-CBP v5 head")
    parser.add_argument("--lc", type=float, default=0.5,
                        help="centre damping for PlainReg (IRDST: 0.5)")
    parser.add_argument("--ls", type=float, default=0.75,
                        help="size damping for PlainReg (IRDST: 0.75)")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.val_ann = os.path.join(SCRIPT_DIR, args.val_ann)
    if args.plainreg_weights is None and args.cbp_weights is None:
        parser.error("give at least one of --plainreg_weights / --cbp_weights")

    cfg, adapter, cap = build_model(args, args.val_ann)
    post = cfg.postprocessor

    plain = None
    if args.plainreg_weights:
        plain = PlainRegHead().to(args.device)
        plain.load_state_dict(torch.load(args.plainreg_weights,
                                         map_location="cpu")["model"])
        plain.eval()
    cbp = None
    if args.cbp_weights:
        cbp = PeroCBPv2().to(args.device)
        cbp.load_state_dict(torch.load(args.cbp_weights, map_location="cpu")["model"])
        cbp.eval()

    loader = cfg.val_dataloader
    wh = {im["id"]: (im["width"], im["height"])
          for im in json.load(open(args.val_ann))["images"]}

    names = ["PERO-R"]
    if plain is not None:
        names += ["PlainReg (raw)"]
        if not (args.lc == 1.0 and args.ls == 1.0):
            names += ["PlainReg + damping"]
    if cbp is not None:
        names += ["PERO-CBP"]
    states = {n: {} for n in names}

    t0 = time.time()
    with torch.no_grad():
        for bi, (samples, targets) in enumerate(loader):
            samples = samples.to(args.device)
            orig = torch.stack([t["orig_size"] for t in targets], 0).to(args.device)
            out = adapter(samples, route_off=False)
            res = post(out, orig)
            B = samples.shape[0]
            b_all = (out["pred_boxes"] * INPUT_SIZE).float()
            b_xyxy = torch.stack([b_all[..., 0]-b_all[..., 2]/2,
                                  b_all[..., 1]-b_all[..., 3]/2,
                                  b_all[..., 0]+b_all[..., 2]/2,
                                  b_all[..., 1]+b_all[..., 3]/2], dim=-1).double()
            queries = cap["query"].float()
            # The post-processor returns the top-300 queries sorted by score:
            # reproduce that permutation so a head output is paired with the
            # score / fallback flag of the same query.
            _, top_idx = torch.topk(torch.sigmoid(out["pred_logits"]).flatten(1),
                                    N_QUERIES, dim=-1)
            top_idx = top_idx // out["pred_logits"].shape[-1]

            from torchvision.ops import roi_align
            ref = None
            if plain is not None:
                fp_ = []
                for s0 in range(0, B * N_QUERIES, 1024):
                    s1 = min(s0 + 1024, B * N_QUERIES)
                    bx = b_all.reshape(B * N_QUERIES, 4)[s0:s1]
                    bidx = (torch.arange(s0, s1, device=args.device) // N_QUERIES).float()
                    rois = torch.cat([bidx[:, None],
                                      b_xyxy.reshape(B * N_QUERIES, 4)[s0:s1].float()], -1)
                    px = roi_align(samples, rois, output_size=(32, 32),
                                   spatial_scale=1.0, aligned=True)
                    c2p = roi_align(cap["c2"], rois * (cap["c2"].shape[-1] / INPUT_SIZE),
                                    output_size=(16, 16), spatial_scale=1.0, aligned=True)
                    qsel = queries.reshape(B * N_QUERIES, 256)[s0:s1]
                    fp_.append(plain(px, c2p, qsel, bx))
                ref = torch.cat(fp_, 0).reshape(B, N_QUERIES, 4).double()

            corr = fb = None
            if cbp is not None:
                fp_ = []; fv_ = []
                for s0 in range(0, B * N_QUERIES, 1024):
                    s1 = min(s0 + 1024, B * N_QUERIES)
                    bx = b_all.reshape(B * N_QUERIES, 4)[s0:s1]
                    bidx = (torch.arange(s0, s1, device=args.device) // N_QUERIES).float()
                    rois = torch.cat([bidx[:, None],
                                      b_xyxy.reshape(B * N_QUERIES, 4)[s0:s1].float()], -1)
                    px = roi_align(samples, rois, output_size=(32, 32),
                                   spatial_scale=1.0, aligned=True)
                    c2p = roi_align(cap["c2"], rois * (cap["c2"].shape[-1] / INPUT_SIZE),
                                    output_size=(16, 16), spatial_scale=1.0, aligned=True)
                    qsel = queries.reshape(B * N_QUERIES, 256)[s0:s1]
                    p_, v_ = cbp(px, c2p, qsel, bx)
                    fp_.append(p_); fv_.append(v_)
                pos_logits = torch.cat(fp_, 0).reshape(B * N_QUERIES, 4, 65)
                val_logits = torch.cat(fv_, 0).reshape(B * N_QUERIES, 4)
                corr, fb, _ = decode_v5(pos_logits, val_logits,
                                        b_xyxy.reshape(B * N_QUERIES, 4))
                corr = corr.reshape(B, N_QUERIES, 4)
                fb = fb.reshape(B, N_QUERIES)

            for b, t in enumerate(targets):
                iid = int(t["image_id"].item())
                W, H = wh[iid]
                pb = res[b]["boxes"].cpu().numpy().astype(np.float64)
                ps = res[b]["scores"].cpu().numpy()
                ti = top_idx[b].cpu().numpy()
                states["PERO-R"][iid] = [(float(pb[k][0]), float(pb[k][1]),
                                          float(pb[k][2]), float(pb[k][3]),
                                          float(ps[k])) for k in range(N_QUERIES)]
                if plain is not None:
                    r64 = ref[b].cpu().numpy()[ti]
                    r_orig = np.stack([r64[:, 0]*(W/INPUT_SIZE), r64[:, 1]*(H/INPUT_SIZE),
                                       r64[:, 2]*(W/INPUT_SIZE), r64[:, 3]*(H/INPUT_SIZE)], 1)
                    states["PlainReg (raw)"][iid] = [
                        (float(r_orig[k, 0]), float(r_orig[k, 1]),
                         float(r_orig[k, 2]), float(r_orig[k, 3]), float(ps[k]))
                        for k in range(N_QUERIES)]
                    if "PlainReg + damping" in states:
                        dmp = damp_boxes(pb[:, :4], r_orig, args.lc, args.ls)
                        states["PlainReg + damping"][iid] = [
                            (float(dmp[k, 0]), float(dmp[k, 1]), float(dmp[k, 2]),
                             float(dmp[k, 3]), float(ps[k])) for k in range(N_QUERIES)]
                if cbp is not None:
                    cb64 = corr[b].cpu().numpy()[ti]
                    fbn = fb[b].cpu().numpy()[ti]
                    out_l = []
                    for k in range(N_QUERIES):
                        if fbn[k]:
                            nb = pb[k]
                        else:
                            nb = (cb64[k, 0]*(W/INPUT_SIZE), cb64[k, 1]*(H/INPUT_SIZE),
                                  cb64[k, 2]*(W/INPUT_SIZE), cb64[k, 3]*(H/INPUT_SIZE))
                        out_l.append((float(nb[0]), float(nb[1]), float(nb[2]),
                                      float(nb[3]), float(ps[k])))
                    states["PERO-CBP"][iid] = out_l
            if (bi + 1) % 300 == 0:
                print(f"  [{bi+1}/{len(loader)}] ({(time.time()-t0)/60:.1f}min)")

    print("\n=== Results ===")
    results = coco_eval(states, args.val_ann)
    base = results["PERO-R"]
    print(f"  {'state':24s} {'AP50':>7s} {'AP75':>7s} {'AP50:95':>8s} "
          f"{'dAP50':>7s} {'dAP75':>7s} {'F1@0.5':>7s}")
    for name in names:
        m = results[name]
        c = m["conf05"]
        print(f"  {name:24s} {m['AP50']:7.4f} {m['AP75']:7.4f} {m['AP50:95']:8.4f} "
              f"{(m['AP50']-base['AP50'])*100:+7.2f} "
              f"{(m['AP75']-base['AP75'])*100:+7.2f} {c['F1']:7.4f}")
    print("  conf0.5 TP/FP/FN: " + " | ".join(
        f"{n} {results[n]['conf05']['TP']}/{results[n]['conf05']['FP']}/"
        f"{results[n]['conf05']['FN']}" for n in names))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
