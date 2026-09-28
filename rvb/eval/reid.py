"""Person Re-ID evaluation (Market-1501 / MSMT17 protocol): CMC Rank-k and mAP.

Gallery images with the same identity AND the same camera as the query are excluded
(they are trivially easy); identities -1 are junk/distractors and ignored.
Input features are any embedding (N, D); cosine distance by default.
"""
from __future__ import annotations

import numpy as np


def distance(q: np.ndarray, g: np.ndarray, metric: str = "cosine") -> np.ndarray:
    if metric == "cosine":
        qn = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
        gn = g / np.linalg.norm(g, axis=1, keepdims=True).clip(1e-12)
        return 1 - qn @ gn.T
    d = (q ** 2).sum(1)[:, None] + (g ** 2).sum(1)[None] - 2 * q @ g.T
    return np.sqrt(np.clip(d, 0, None))


def evaluate_reid(qf, q_pids, q_cams, gf, g_pids, g_cams, ranks=(1, 5, 10), metric="cosine") -> dict:
    q_pids, q_cams, g_pids, g_cams = map(np.asarray, (q_pids, q_cams, g_pids, g_cams))
    dist = distance(np.asarray(qf, float), np.asarray(gf, float), metric)
    max_rank = max(ranks)
    cmc_all, aps = [], []
    for i in range(len(q_pids)):
        order = np.argsort(dist[i])
        remove = ((g_pids[order] == q_pids[i]) & (g_cams[order] == q_cams[i])) | (g_pids[order] == -1)
        match = (g_pids[order] == q_pids[i])[~remove]
        if not match.any():
            continue                                  # query identity not in gallery
        cmc = match.cumsum().clip(0, 1)
        cmc_all.append(cmc[:max_rank] if len(cmc) >= max_rank else np.pad(cmc, (0, max_rank - len(cmc)), constant_values=1))
        hits = np.where(match)[0]
        precision_at_hits = np.arange(1, len(hits) + 1) / (hits + 1)
        aps.append(precision_at_hits.mean())
    if not aps:
        return {"n_valid_queries": 0}
    cmc = np.mean(cmc_all, axis=0)
    out = {f"rank{r}": float(cmc[r - 1]) for r in ranks}
    out.update({"mAP": float(np.mean(aps)), "n_valid_queries": len(aps)})
    return out


def evaluate_reid_npz(path: str, metric: str = "cosine") -> dict:
    """npz with arrays: q_feat, q_pid, q_cam, g_feat, g_pid, g_cam (e.g. exported from torchreid)."""
    d = np.load(path)
    return evaluate_reid(d["q_feat"], d["q_pid"], d["q_cam"], d["g_feat"], d["g_pid"], d["g_cam"], metric=metric)
