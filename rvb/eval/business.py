"""Business-level evaluation: are C Spire's numbers right?

Compares ground-truth events with predicted events (both produced by rvb.zones with the
same rules). Staff are excluded from customer metrics (GT role vs predicted role).

Metrics
* footfall      - store entries per hour: MAE, MAPE, total % error, bias
* peak hours    - top-k busiest hours per day: overlap@k and Spearman rank correlation
* zone traffic  - visits per zone (people passing + stopping): % error
* engagement    - stop-vs-pass: visit-level precision/recall/F1 on matched visits, and the
                  engagement-rate error per zone (the "interest vs bypass" number)
* dwell         - mean dwell of engaged visits per zone (abs error, seconds) and matched-visit MAE
* occupancy     - people inside each zone per minute: MAE
* transitions   - zone->zone flows within each camera: L1 distance of normalised matrices
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from ..zones import events_frame


def _customers(ev: pd.DataFrame) -> pd.DataFrame:
    return ev[ev["role"] != "staff"]


def _hour_counts(ev: pd.DataFrame) -> pd.Series:
    e = ev[ev["type"] == "STORE_ENTER"]
    return (e["t"] // 3600).astype(int).value_counts().sort_index()


def footfall_metrics(gt: pd.DataFrame, pr: pd.DataFrame, hours: list[int] | None = None, top_k: int = 3) -> dict:
    g, p = _hour_counts(gt), _hour_counts(pr)
    idx = sorted(set(hours or []) | set(g.index) | set(p.index))
    g, p = g.reindex(idx, fill_value=0), p.reindex(idx, fill_value=0)
    err = (p - g).abs()
    nz = g > 0
    res = {
        "gt_total": int(g.sum()), "pred_total": int(p.sum()),
        "total_pct_error": float(100 * (p.sum() - g.sum()) / max(g.sum(), 1)),
        "hourly_MAE": float(err.mean()) if len(err) else None,
        "hourly_MAPE": float(100 * (err[nz] / g[nz]).mean()) if nz.any() else None,
        "hourly_bias": float((p - g).mean()) if len(err) else None,
        "per_hour": {str(h): {"gt": int(g[h]), "pred": int(p[h])} for h in idx},
    }
    # peak hours per day
    days = sorted({h // 24 for h in idx})
    overlaps, rhos = [], []
    for d in days:
        hh = [h for h in idx if h // 24 == d]
        if len(hh) < 2:
            continue
        k = min(top_k, len(hh))
        tg = set(g[hh].sort_values(ascending=False).index[:k])
        tp = set(p[hh].sort_values(ascending=False).index[:k])
        overlaps.append(len(tg & tp) / k)
        if g[hh].nunique() > 1 and p[hh].nunique() > 1:
            rhos.append(spearmanr(g[hh], p[hh]).statistic)
    res["peak_overlap_at_k"] = float(np.mean(overlaps)) if overlaps else None
    res["peak_spearman"] = float(np.mean(rhos)) if rhos else None
    res["peak_k"] = top_k
    return res


def _tiou(a0, a1, b0, b1):
    inter = np.clip(np.minimum(a1[:, None], b1[None]) - np.maximum(a0[:, None], b0[None]), 0, None)
    union = (a1 - a0)[:, None] + (b1 - b0)[None] - inter
    return inter / np.maximum(union, 1e-9)


def match_visits(gv: pd.DataFrame, pv: pd.DataFrame, min_tiou: float = 0.3):
    """Identity-free one-to-one matching of visits in the same zone by temporal IoU."""
    pairs = []
    for z in set(gv["zone"]) | set(pv["zone"]):
        g, p = gv[gv["zone"] == z], pv[pv["zone"] == z]
        if g.empty or p.empty:
            continue
        for day in set((g["t"] // 86400).astype(int)) | set((p["t"] // 86400).astype(int)):
            gd, pd_ = g[(g["t"] // 86400) == day], p[(p["t"] // 86400) == day]
            if gd.empty or pd_.empty:
                continue
            iou = _tiou(gd["t"].to_numpy(), gd["t_end"].to_numpy(), pd_["t"].to_numpy(), pd_["t_end"].to_numpy())
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= min_tiou:
                    pairs.append((gd.index[i], pd_.index[j], float(iou[i, j])))
    return pairs


def zone_metrics(gt: pd.DataFrame, pr: pd.DataFrame, zones: list[str], min_tiou: float = 0.3) -> dict:
    gv = gt[gt["type"] == "ZONE_VISIT"].copy()
    pv = pr[pr["type"] == "ZONE_VISIT"].copy()
    pairs = match_visits(gv, pv, min_tiou)
    mg = gv.loc[[a for a, _, _ in pairs]].reset_index(drop=True)
    mp = pv.loc[[b for _, b, _ in pairs]].reset_index(drop=True)
    out = {"matched_visits": len(pairs), "gt_visits": len(gv), "pred_visits": len(pv),
           "visit_recall": len(pairs) / max(len(gv), 1), "visit_precision": len(pairs) / max(len(pv), 1)}
    # stop-vs-pass on matched visits (engaged = positive class)
    ge, pe = mg["engaged"].astype(bool).to_numpy(), mp["engaged"].astype(bool).to_numpy()
    tp, fp, fn = int((ge & pe).sum()), int((~ge & pe).sum()), int((ge & ~pe).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out["stop_vs_pass"] = {"precision": prec, "recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-9),
                           "accuracy": float((ge == pe).mean()) if len(ge) else None,
                           "TP": tp, "FP": fp, "FN": fn, "TN": int((~ge & ~pe).sum())}
    out["matched_dwell_MAE_s"] = float((mg["dwell"] - mp["dwell"]).abs().mean()) if len(mg) else None
    per = {}
    for z in zones:
        g, p = gv[gv["zone"] == z], pv[pv["zone"] == z]
        g_eng, p_eng = g[g["engaged"].astype(bool)], p[p["engaged"].astype(bool)]
        rate_g = len(g_eng) / len(g) if len(g) else None
        rate_p = len(p_eng) / len(p) if len(p) else None
        per[z] = {
            "gt_visits": len(g), "pred_visits": len(p),
            "visits_pct_error": 100 * (len(p) - len(g)) / len(g) if len(g) else None,
            "gt_engaged": len(g_eng), "pred_engaged": len(p_eng),
            "engaged_pct_error": 100 * (len(p_eng) - len(g_eng)) / len(g_eng) if len(g_eng) else None,
            "gt_engagement_rate": rate_g, "pred_engagement_rate": rate_p,
            "engagement_rate_abs_error": abs(rate_p - rate_g) if rate_g is not None and rate_p is not None else None,
            "gt_mean_dwell_s": float(g_eng["dwell"].mean()) if len(g_eng) else None,
            "pred_mean_dwell_s": float(p_eng["dwell"].mean()) if len(p_eng) else None,
        }
        if per[z]["gt_mean_dwell_s"] is not None and per[z]["pred_mean_dwell_s"] is not None:
            per[z]["mean_dwell_abs_error_s"] = abs(per[z]["pred_mean_dwell_s"] - per[z]["gt_mean_dwell_s"])
        per[z]["occupancy_MAE_per_min"] = occupancy_mae(g, p)
    out["per_zone"] = per
    vals = [v for v in per.values() if v.get("mean_dwell_abs_error_s") is not None]
    out["mean_dwell_abs_error_s"] = float(np.mean([v["mean_dwell_abs_error_s"] for v in vals])) if vals else None
    vals = [v["engagement_rate_abs_error"] for v in per.values() if v["engagement_rate_abs_error"] is not None]
    out["engagement_rate_abs_error"] = float(np.mean(vals)) if vals else None
    vals = [v["occupancy_MAE_per_min"] for v in per.values() if v["occupancy_MAE_per_min"] is not None]
    out["occupancy_MAE_per_min"] = float(np.mean(vals)) if vals else None
    return out


def _occupancy_series(v: pd.DataFrame, t0: float, t1: float) -> np.ndarray:
    n = int(np.ceil((t1 - t0) / 60))
    occ = np.zeros(n)
    for a, b in zip(v["t"].to_numpy(), v["t_end"].to_numpy()):
        for m in range(int((a - t0) // 60), int((b - t0) // 60) + 1):
            lo, hi = t0 + 60 * m, t0 + 60 * (m + 1)
            occ[m] += max(0.0, min(b, hi) - max(a, lo)) / 60
    return occ


def occupancy_mae(g: pd.DataFrame, p: pd.DataFrame) -> float | None:
    if g.empty and p.empty:
        return None
    t0 = min(g["t"].min() if len(g) else np.inf, p["t"].min() if len(p) else np.inf)
    t1 = max(g["t_end"].max() if len(g) else -np.inf, p["t_end"].max() if len(p) else -np.inf)
    t0 = np.floor(t0 / 60) * 60
    og, op = _occupancy_series(g, t0, t1), _occupancy_series(p, t0, t1)
    active = (og > 0) | (op > 0)
    return float(np.abs(og - op)[active].mean()) if active.any() else 0.0


def transition_metrics(gt: pd.DataFrame, pr: pd.DataFrame, zones_by_camera: dict[str, list[str]]) -> dict:
    out = {}
    for cam, zs in zones_by_camera.items():
        if len(zs) < 2:
            continue
        idx = {z: i for i, z in enumerate(zs)}

        def mat(ev, cam_filter):
            v = ev[(ev["type"] == "ZONE_VISIT") & ev["zone"].isin(zs)]
            if cam_filter:
                v = v[v["source"] == cam]
            M = np.zeros((len(zs), len(zs)))
            for _, s in v.sort_values("t").groupby("track_id"):
                seq = [z for i, z in enumerate(s["zone"]) if i == 0 or z != s["zone"].iloc[i - 1]]
                for a, b in zip(seq, seq[1:]):
                    M[idx[a], idx[b]] += 1
            return M

        G, P = mat(gt, False), mat(pr, True)
        Gn, Pn = G / max(G.sum(), 1), P / max(P.sum(), 1)
        out[cam] = {"zones": zs, "gt_counts": G.astype(int).tolist(), "pred_counts": P.astype(int).tolist(),
                    "L1_normalised": float(np.abs(Gn - Pn).sum())}
    return out


def evaluate_business(gt_events: list[dict], pred_events: list[dict], zones: list[str],
                      zones_by_camera: dict[str, list[str]] | None = None, hours: list[int] | None = None,
                      min_tiou: float = 0.3) -> dict:
    gt, pr = events_frame(gt_events), events_frame(pred_events)
    staff = {"gt_staff_tracks": int(gt.loc[gt["role"] == "staff", "track_id"].nunique()),
             "pred_staff_tracks": int(pr.loc[pr["role"] == "staff", "track_id"].nunique())}
    gt, pr = _customers(gt), _customers(pr)
    res = {"footfall": footfall_metrics(gt, pr, hours), "zones": zone_metrics(gt, pr, zones, min_tiou), "staff": staff}
    if zones_by_camera:
        res["transitions"] = transition_metrics(gt, pr, zones_by_camera)
    return res
