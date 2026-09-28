"""Tracker error model: ground-truth MOT boxes -> realistic 'predicted' tracks.

Used to (1) test the eval framework end to end before any model exists and (2) measure how
detection/tracking errors propagate into business metrics (footfall, dwell, stop-vs-pass).

Error types simulated per camera:
* missed detections - more likely for small (far) and occluded people
* box jitter        - proportional to box height
* fragmentation     - after a run of misses the track restarts with a NEW id
* identity switches - when two people overlap, their ids can swap
* false positives   - short spurious tracks
* staff/customer    - per-track role classification errors
* min track length  - tracks shorter than N frames are discarded (like a real tracker)

`reid_recovery` in [0, 1] removes that share of fragmentations, i.e. what a Re-ID module
would repair - handy for "is Re-ID worth it?" experiments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class NoiseConfig:
    miss_base: float = 0.05
    miss_small: float = 0.20      # extra miss prob when box height < small_px
    small_px: float = 45.0
    miss_occluded: float = 0.45   # extra miss prob scaled by (1 - visibility)
    jitter: float = 0.05          # std of box noise as a fraction of box height
    frag_gap: int = 12            # frames unseen (missed or hidden) before a track may break
    p_frag: float = 0.25          # prob that a break starts a new id
    p_switch: float = 0.10        # prob of id swap per new close encounter (IoU > 0.3)
    fp_rate: float = 0.002        # expected new false-positive tracks per frame (0.002 @15fps ~ 1 every 33 s)
    fp_len: tuple = (3, 25)
    min_track_len: int = 5
    role_acc: float = 0.95
    reid_recovery: float = 0.0

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "perfect": NoiseConfig(0, 0, 45, 0, 0, 10**9, 0, 0, 0, (3, 3), 1, 1.0, 0.0),
    "good": NoiseConfig(0.02, 0.08, 45, 0.30, 0.03, 15, 0.08, 0.03, 0.0007, (3, 15), 5, 0.98),
    "medium": NoiseConfig(),
    "poor": NoiseConfig(0.10, 0.35, 45, 0.70, 0.08, 8, 0.50, 0.30, 0.006, (3, 45), 5, 0.85),
}


def get_preset(name: str, **overrides) -> NoiseConfig:
    return replace(PRESETS[name], **overrides)


def _iou_matrix(b):
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix = np.clip(np.minimum(x2[:, None], x2) - np.maximum(x1[:, None], x1), 0, None)
    iy = np.clip(np.minimum(y2[:, None], y2) - np.maximum(y1[:, None], y1), 0, None)
    inter = ix * iy
    area = b[:, 2] * b[:, 3]
    return inter / (area[:, None] + area - inter + 1e-9)


def corrupt(gt: pd.DataFrame, cfg: NoiseConfig, width: int = 640, height: int = 360, seed: int = 0) -> pd.DataFrame:
    """gt: MOT df (frame,id,x,y,w,h,vis[,role]). Returns predicted MOT df with conf and role."""
    rng = np.random.default_rng(seed)
    gt = gt[gt.get("conf", 1) > 0] if "conf" in gt else gt
    gt = gt.sort_values(["frame", "id"]).reset_index(drop=True)
    roles = gt.groupby("id")["role"].first().to_dict() if "role" in gt else {}
    next_pid = [int(gt["id"].max() if len(gt) else 0) + 100000]

    def new_id():
        next_pid[0] += 1
        return next_pid[0]

    assign: dict[int, int] = {}       # gt id -> current predicted id
    last_hit: dict[int, int] = {}     # last frame each gt id was detected (gaps = misses OR fully hidden frames)
    close_pairs: set[tuple[int, int]] = set()
    out_rows = []
    frames = gt["frame"].to_numpy()
    bounds = np.flatnonzero(np.diff(frames)) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(frames)]
    X = gt[["x", "y", "w", "h"]].to_numpy(float)
    IDS = gt["id"].to_numpy()
    VIS = gt["vis"].to_numpy(float) if "vis" in gt else np.ones(len(gt))
    fps_tracks = []   # (pred_id, start_frame, n, box)
    for s, e in zip(starts, ends):
        f = int(frames[s])
        ids, boxes, vis = IDS[s:e], X[s:e], VIS[s:e]
        # --- identity switches on new close encounters
        if len(ids) > 1 and cfg.p_switch > 0:
            iou = _iou_matrix(boxes)
            ii, jj = np.where(np.triu(iou, 1) > 0.3)
            now = set()
            for i, j in zip(ii, jj):
                pair = (int(min(ids[i], ids[j])), int(max(ids[i], ids[j])))
                now.add(pair)
                if pair not in close_pairs and rng.random() < cfg.p_switch and pair[0] in assign and pair[1] in assign:
                    assign[pair[0]], assign[pair[1]] = assign[pair[1]], assign[pair[0]]
            close_pairs = now
        # --- detections
        p_miss = cfg.miss_base + cfg.miss_small * (boxes[:, 3] < cfg.small_px) + cfg.miss_occluded * (1 - vis)
        hit = rng.random(len(ids)) >= np.clip(p_miss, 0, 0.98)
        for k, gid in enumerate(ids):
            gid = int(gid)
            if not hit[k]:
                continue
            if gid not in assign:
                assign[gid] = new_id()
            elif f - last_hit.get(gid, f) - 1 >= cfg.frag_gap and rng.random() < cfg.p_frag * (1 - cfg.reid_recovery):
                assign[gid] = new_id()
            last_hit[gid] = f
            b = boxes[k] + rng.normal(0, cfg.jitter, 4) * boxes[k, 3] * np.array([1, 1, 0.5, 1])
            out_rows.append((f, assign[gid], *b, float(np.clip(rng.beta(8, 2), 0.3, 1)), gid))
        # --- false positives
        for _ in range(rng.poisson(cfg.fp_rate)):
            h = rng.uniform(30, 140)
            box = np.array([rng.uniform(0, width - h * 0.41), rng.uniform(0, height - h), h * 0.41, h])
            fps_tracks.append((new_id(), f, int(rng.integers(cfg.fp_len[0], cfg.fp_len[1] + 1)), box))
    for pid, f0, n, box in fps_tracks:
        drift = rng.normal(0, 1.0, (n, 2)).cumsum(0)
        for k in range(n):
            out_rows.append((f0 + k, pid, box[0] + drift[k, 0], box[1] + drift[k, 1], box[2], box[3],
                             float(np.clip(rng.beta(3, 4), 0.3, 1)), -1))
    pred = pd.DataFrame(out_rows, columns=["frame", "id", "x", "y", "w", "h", "conf", "gt_id"])
    if pred.empty:
        return pred.assign(cls=1, vis=1.0, role="customer")
    # tracker min length
    lens = pred.groupby("id")["frame"].transform("size")
    pred = pred[lens >= cfg.min_track_len].copy()
    # per-track role from the majority gt id, with classification errors
    maj = pred[pred.gt_id >= 0].groupby("id")["gt_id"].agg(lambda s: s.mode().iat[0])
    role = {}
    for pid in pred["id"].unique():
        true = roles.get(int(maj.get(pid, -1)), "customer")
        flip = rng.random() > cfg.role_acc
        role[pid] = ("customer" if true == "staff" else "staff") if flip else true
    pred["role"] = pred["id"].map(role)
    pred["cls"], pred["vis"] = 1, 1.0
    return pred.sort_values(["frame", "id"]).reset_index(drop=True)   # gt_id kept for diagnostics (not written to MOT)
