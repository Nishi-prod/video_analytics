"""Tracking evaluation: HOTA, CLEAR (MOTA, IDSW, MT/ML) and Identity (IDF1) via the official
TrackEval metric code, fed directly from MOT-format data frames.

MOTChallenge-style preprocessing: ground-truth boxes with conf == 0 (fully occluded / ignore)
are removed, together with any predicted box that matches one of them at IoU >= 0.5, so a
tracker is neither rewarded nor punished for them.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

with contextlib.redirect_stdout(io.StringIO()):
    from trackeval.metrics import CLEAR, HOTA, Identity


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4) b (M,4) in x,y,w,h -> IoU (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = ix * iy
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def frame_slices(df: pd.DataFrame, frames: np.ndarray) -> dict:
    """frame -> (start, end) row range of a frame-sorted data frame (fast per-frame access)."""
    fr = df["frame"].to_numpy()
    if len(fr) == 0:
        return {}
    b = np.flatnonzero(np.diff(fr)) + 1
    st, en = np.r_[0, b], np.r_[b, len(fr)]
    return dict(zip(fr[st].tolist(), zip(st.tolist(), en.tolist())))


def build_sequence(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Convert MOT data frames into TrackEval's per-sequence data dict."""
    gt = gt[gt["cls"] == 1] if "cls" in gt else gt
    gt = gt.sort_values("frame", kind="stable")
    pred = pred.sort_values("frame", kind="stable")
    frames = sorted(set(gt["frame"]).union(pred["frame"]))
    gs, ps = frame_slices(gt, None), frame_slices(pred, None)
    GB, GI = gt[["x", "y", "w", "h"]].to_numpy(float), gt["id"].to_numpy()
    GC = gt["conf"].to_numpy(float) if "conf" in gt else np.ones(len(gt))
    PB, PI = pred[["x", "y", "w", "h"]].to_numpy(float), pred["id"].to_numpy()
    gt_ids, tr_ids, sims = [], [], []
    for f in frames:
        a, b = gs.get(f, (0, 0))
        c, d = ps.get(f, (0, 0))
        gb, gi, ign = GB[a:b], GI[a:b], GC[a:b] == 0
        pb, pi = PB[c:d], PI[c:d]
        keep_p = np.ones(len(pb), bool)
        if ign.any() and len(pb):
            iou = box_iou(gb, pb)
            r, cc = linear_sum_assignment(-iou)
            for i, j in zip(r, cc):
                if ign[i] and iou[i, j] >= 0.5:
                    keep_p[j] = False
        gb, gi = gb[~ign], gi[~ign]
        pb, pi = pb[keep_p], pi[keep_p]
        gt_ids.append(gi)
        tr_ids.append(pi)
        sims.append(box_iou(gb, pb))
    ug = np.unique(np.concatenate(gt_ids)) if gt_ids else np.array([])
    ut = np.unique(np.concatenate(tr_ids)) if tr_ids else np.array([])
    return {
        "num_timesteps": len(frames),
        "num_gt_ids": len(ug), "num_tracker_ids": len(ut),
        "num_gt_dets": int(sum(len(x) for x in gt_ids)), "num_tracker_dets": int(sum(len(x) for x in tr_ids)),
        "gt_ids": [np.searchsorted(ug, x).astype(int) for x in gt_ids],
        "tracker_ids": [np.searchsorted(ut, x).astype(int) for x in tr_ids],
        "similarity_scores": sims,
    }


KEYS = {"HOTA": ["HOTA", "DetA", "AssA", "LocA"], "CLEAR": ["MOTA", "MOTP", "IDSW", "Frag", "MT", "ML", "CLR_FP", "CLR_FN"],
        "Identity": ["IDF1", "IDP", "IDR"]}


def _summ(res: dict) -> dict:
    out = {}
    for fam, keys in KEYS.items():
        r = res[fam]
        for k in keys:
            v = r[k]
            out[k] = float(np.mean(v)) if np.ndim(v) else float(v)
    return out


def evaluate_tracking(pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
    """pairs: {sequence_name: (gt_df, pred_df)} -> per-sequence and combined metrics."""
    per_seq = {}
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = {"HOTA": HOTA(), "CLEAR": CLEAR(), "Identity": Identity()}
        raw = {m: {} for m in metrics}
        for name, (gt, pred) in pairs.items():
            data = build_sequence(gt, pred)
            res = {m: metrics[m].eval_sequence(data) for m in metrics}
            for m in metrics:
                raw[m][name] = res[m]
            per_seq[name] = _summ(res)
        combined = {m: metrics[m].combine_sequences(raw[m]) for m in metrics} if pairs else None
    return {"per_sequence": per_seq, "combined": _summ(combined) if combined else {}}
