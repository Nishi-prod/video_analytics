"""Person detection evaluation: AP@IoU, precision / recall / F1 at an operating threshold,
and recall broken down by person size (small far-away people are the main risk at 640x360).
Ground-truth boxes with conf == 0 are 'ignore': detections matching them are not counted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tracking import box_iou, frame_slices

SIZE_BINS = [("small <50px", 0, 50), ("medium 50-100px", 50, 100), ("large >100px", 100, 1e9)]


def _match_frame(gb, gign, pb, ps, thr):
    """Greedy by score (COCO/VOC style). Returns per-pred label: 1 TP, 0 FP, -1 ignore; and matched gt mask."""
    order = np.argsort(-ps)
    lab = np.zeros(len(pb), int)
    taken = np.zeros(len(gb), bool)
    if len(gb) == 0:
        return lab, taken, order
    iou = box_iou(pb[order], gb)
    for k in range(len(order)):
        cand = np.where(~taken & (iou[k] >= thr))[0]
        if len(cand) == 0:
            continue
        # prefer real gt over ignore gt
        real = cand[~gign[cand]]
        j = real[np.argmax(iou[k, real])] if len(real) else cand[np.argmax(iou[k, cand])]
        taken[j] = True
        lab[order[k]] = -1 if gign[j] else 1
    return lab, taken, order


def average_precision(scores, labels, n_gt) -> float:
    keep = labels >= 0
    s, l = scores[keep], labels[keep]
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-s)
    tp = np.cumsum(l[o] == 1)
    fp = np.cumsum(l[o] == 0)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.r_[0, rec, 1]
    mpre = np.r_[1, prec, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_detection(gt: pd.DataFrame, pred: pd.DataFrame, iou_thr: float = 0.5, score_thr: float = 0.5,
                       min_vis: float = 0.0) -> dict:
    gt = gt.copy()
    if "conf" not in gt:
        gt["conf"] = 1.0
    if min_vis > 0 and "vis" in gt:
        gt.loc[gt["vis"] < min_vis, "conf"] = 0
    if "conf" not in pred:
        pred = pred.assign(conf=1.0)
    gt = gt.sort_values("frame", kind="stable")
    pred = pred.sort_values("frame", kind="stable")
    gs, ps_ = frame_slices(gt, None), frame_slices(pred, None)
    GB, GC = gt[["x", "y", "w", "h"]].to_numpy(float), gt["conf"].to_numpy(float)
    PB, PS = pred[["x", "y", "w", "h"]].to_numpy(float), pred["conf"].to_numpy(float)
    all_s, all_l = [], []
    size_hit = {b[0]: [0, 0] for b in SIZE_BINS}
    tp = fp = 0
    n_gt = int((gt["conf"] > 0).sum())
    for f in set(gs) | set(ps_):
        a, b = gs.get(f, (0, 0))
        c, d = ps_.get(f, (0, 0))
        gb, gign = GB[a:b], GC[a:b] == 0
        pb, ps = PB[c:d], PS[c:d]
        lab, _, _ = _match_frame(gb, gign, pb, ps, iou_thr)
        all_s.append(ps)
        all_l.append(lab)
        sel = ps >= score_thr
        lab_op, taken_op, _ = _match_frame(gb, gign, pb[sel], ps[sel], iou_thr)
        tp += int((lab_op == 1).sum())
        fp += int((lab_op == 0).sum())
        for name, lo, hi in SIZE_BINS:
            m = (~gign) & (gb[:, 3] >= lo) & (gb[:, 3] < hi)
            size_hit[name][0] += int((taken_op & m).sum())
            size_hit[name][1] += int(m.sum())
    s, l = (np.concatenate(all_s), np.concatenate(all_l)) if all_s else (np.zeros(0), np.zeros(0, int))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    return {
        f"AP{int(iou_thr*100)}": average_precision(s, l, n_gt),
        "precision": prec, "recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-9),
        "score_thr": score_thr, "n_gt": n_gt, "TP": tp, "FP": fp, "FN": n_gt - tp,
        "recall_by_size": {k: (v[0] / v[1] if v[1] else None) for k, v in size_hit.items()},
        "n_gt_by_size": {k: v[1] for k, v in size_hit.items()},
    }
