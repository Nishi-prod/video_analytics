"""Zone / event engine: tracks -> store entries/exits and zone visits.

This is the *reference* implementation of the business rules. The same code turns
ground-truth tracks into ground-truth events and predicted tracks into predicted events,
so the business evaluation measures only the effect of perception errors.

Rules (all configurable in RuleConfig):
* STORE_ENTER / STORE_EXIT - a track crosses the entrance line (within its span), direction
  from the line's inside normal. Two opposite crossings within `debounce_s` cancel (jitter).
* ZONE_VISIT - consecutive time inside a zone polygon. Gaps shorter than `exit_grace_s`
  (brief occlusion, box jitter) are bridged; visits shorter than `min_visit_s` are dropped.
* engaged ("stopped") - visit dwell >= zone.stop_seconds; otherwise a pass-by (bypass).
* visit stitching - a tracker often breaks one person into several ids (occlusion at a busy
  table). A visit that starts within `stitch_gap_s` of another visit ending in the same zone,
  within `stitch_dist` of where it ended, is merged into it. Without this, fragments inflate
  zone traffic and deflate the engagement rate (see the sensitivity report).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .layout import Entrance, Zone


@dataclass
class RuleConfig:
    min_visit_s: float = 1.0
    exit_grace_s: float = 1.5
    max_track_gap_s: float = 2.0   # a missing stretch longer than this splits a visit
    debounce_s: float = 1.5
    min_track_s: float = 1.0       # ignore tracks shorter than this (spurious detections)
    stitch: bool = False           # opt-in: merge fragments across ids (test with the sensitivity run)
    stitch_gap_s: float = 3.0
    stitch_dist: float = 1.0       # metres in world space; set ~60 px for image-plane zones


@dataclass
class Event:
    type: str
    track_id: int
    t: float
    t_end: float | None = None
    zone: str | None = None
    dwell: float | None = None
    engaged: bool | None = None
    role: str = "customer"
    source: str = ""

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [i, j] (inclusive) where mask is True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def zone_visits(t: np.ndarray, xy: np.ndarray, zone: Zone, cfg: RuleConfig, dt: float) -> list[tuple[float, float]]:
    inside = zone.contains(xy)
    runs = [(t[i], t[j] + dt) for i, j in _runs(inside)]
    # split runs across large gaps in the track itself
    split = []
    for a, b in runs:
        idx = np.where((t >= a) & (t < b))[0]
        gaps = np.where(np.diff(t[idx]) > cfg.max_track_gap_s)[0]
        s = a
        for g in gaps:
            split.append((s, t[idx[g]] + dt))
            s = t[idx[g + 1]]
        split.append((s, b))
    merged: list[list[float]] = []
    for a, b in split:
        if merged and a - merged[-1][1] <= cfg.exit_grace_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= cfg.min_visit_s]


def entrance_crossings(t: np.ndarray, xy: np.ndarray, ent: Entrance, cfg: RuleConfig) -> list[tuple[float, str]]:
    side = ent.side(xy)
    near = ent.within_span(xy)
    out = []
    for k in np.where((side[1:] != side[:-1]) & (side[1:] != 0) & (side[:-1] != 0))[0]:
        if not (near[k] or near[k + 1]) or t[k + 1] - t[k] > cfg.max_track_gap_s:
            continue
        out.append((float(t[k + 1]), "STORE_ENTER" if side[k + 1] > 0 else "STORE_EXIT"))
    # debounce: opposite crossings in quick succession cancel out
    clean: list[tuple[float, str]] = []
    for ev in out:
        if clean and ev[0] - clean[-1][0] < cfg.debounce_s and ev[1] != clean[-1][1]:
            clean.pop()
        else:
            clean.append(ev)
    return clean


def _stitch(visits: list[dict], cfg: RuleConfig) -> list[dict]:
    """Merge visit fragments of (probably) the same person across track ids, per zone."""
    out = []
    for z in {v["zone"] for v in visits}:
        vs = sorted((v for v in visits if v["zone"] == z), key=lambda v: v["a"])
        open_: list[dict] = []
        for v in vs:
            best, best_d = None, cfg.stitch_dist
            for u in open_:
                gap = v["a"] - u["b"]
                if u["tid"] != v["tid"] and 0 <= gap <= cfg.stitch_gap_s and not u.get("continued"):
                    d = float(np.linalg.norm(v["xy_a"] - u["xy_b"]))
                    if d <= best_d:
                        best, best_d = u, d
            if best is not None:
                best["b"], best["xy_b"] = max(best["b"], v["b"]), v["xy_b"]
                best["stitched"] = best.get("stitched", 0) + 1
                continue
            open_.append(v)
            out.append(v)
    return out


def derive_events(tracks: pd.DataFrame, zones: list[Zone], entrance: Entrance | None = None,
                  fps: float = 15.0, cfg: RuleConfig | None = None, source: str = "") -> list[dict]:
    """tracks: columns t,id,x,y[,role] (one coordinate frame - world metres or one camera's pixels)."""
    cfg = cfg or RuleConfig()
    dt = 1.0 / fps
    events: list[dict] = []
    visits: list[dict] = []
    if tracks.empty:
        return events
    if "role" not in tracks:
        tracks = tracks.assign(role="customer")
    for tid, g in tracks.sort_values(["id", "t"]).groupby("id", sort=False):
        t = g["t"].to_numpy(float)
        if t[-1] - t[0] < cfg.min_track_s:
            continue
        xy = g[["x", "y"]].to_numpy(float)
        role = g["role"].mode().iat[0]
        if entrance is not None:
            for tc, kind in entrance_crossings(t, xy, entrance, cfg):
                events.append(Event(kind, int(tid), tc, role=role, source=source).to_dict())
        for z in zones:
            for a, b in zone_visits(t, xy, z, cfg, dt):
                ia, ib = np.searchsorted(t, a), min(np.searchsorted(t, b - dt), len(t) - 1)
                visits.append({"tid": int(tid), "zone": z.id, "a": float(a), "b": float(b), "xy_a": xy[ia], "xy_b": xy[ib],
                               "role": role, "stop": z.stop_seconds})
    if cfg.stitch:
        visits = _stitch(visits, cfg)
    for v in visits:
        d = v["b"] - v["a"]
        e = Event("ZONE_VISIT", v["tid"], v["a"], v["b"], v["zone"], float(d), bool(d >= v["stop"]), v["role"], source).to_dict()
        if v.get("stitched"):
            e["stitched"] = v["stitched"]
        events.append(e)
    events.sort(key=lambda e: (e["t"], e["track_id"]))
    return events


def events_frame(events: list[dict]) -> pd.DataFrame:
    cols = ["type", "track_id", "t", "t_end", "zone", "dwell", "engaged", "role", "source"]
    df = pd.DataFrame(events)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df[cols]
