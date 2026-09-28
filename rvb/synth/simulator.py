"""Agent-based shopper simulator -> ground-truth footfall / zone data.

What it produces (per session = one store day or a slice of it):
* world tracks (t, frame, id, x, y, role, group) at the camera frame rate - exact positions
* intents: what each shopper *meant* to do (zone, engaged or passed) - checks the zone rules
* per-camera MOT ground truth (boxes, visibility incl. occlusion and truncation)

Behaviour is driven by a scenario file (arrival rates per hour, group sizes, zone visit and
stop probabilities, dwell distributions, staff). Numbers in the demo scenarios are
illustrative - calibrate them against a labelled sample of C Spire footage.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..layout import Layout


@dataclass
class Session:
    name: str
    day_index: int
    day_type: str
    start_hour: float
    end_hour: float


def load_scenario(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def sessions_from(scn: dict) -> list[Session]:
    return [Session(s.get("name", f"day{i}"), i, s["day_type"], s.get("start_hour", scn["open_hour"]),
                    s.get("end_hour", scn["close_hour"])) for i, s in enumerate(scn["sessions"])]


# ------------------------------------------------------------------ motion primitives
def _walk(p0, p1, speed, fps, rng, sway=0.04):
    d = np.linalg.norm(p1 - p0)
    n = max(1, int(math.ceil(d / speed * fps)))
    s = np.linspace(0, 1, n + 1)[1:]
    pts = p0 + (p1 - p0) * s[:, None]
    if d > 0.2:  # smooth lateral sway (Ornstein-Uhlenbeck), zero at both ends
        normal = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / d
        ou = np.zeros(n)
        for k in range(1, n):
            ou[k] = 0.9 * ou[k - 1] + rng.normal(0, sway)
        ou *= np.sin(np.pi * s)
        pts = pts + ou[:, None] * normal
    return pts


def _linger(p, seconds, fps, rng, radius=0.25):
    n = max(1, int(round(seconds * fps)))
    steps = rng.normal(0, 0.012, size=(n, 2))
    pts = np.empty((n, 2))
    cur = p.copy()
    for k in range(n):
        cur = cur + steps[k] - 0.02 * (cur - p)            # random walk pulled back to p
        if np.linalg.norm(cur - p) > radius:
            cur = p + (cur - p) * radius / np.linalg.norm(cur - p)
        pts[k] = cur
    return pts


def build_path(waypoints: list[tuple[np.ndarray, float, float]], fps: float, rng) -> np.ndarray:
    """waypoints: (point, dwell_seconds, walking speed to reach it)."""
    pts = [waypoints[0][0][None]]
    cur = waypoints[0][0]
    for p, dwell, speed in waypoints[1:]:
        pts.append(_walk(cur, p, speed, fps, rng))
        cur = p
        if dwell > 0:
            pts.append(_linger(p, dwell, fps, rng))
    return np.vstack(pts)


# ------------------------------------------------------------------ simulation
def _dwell(rng, spec, lo, hi):
    median, sigma = spec
    return float(np.clip(rng.lognormal(math.log(median), sigma), lo, hi))


def simulate(layout: Layout, scn: dict, seed: int | None = None):
    rng = np.random.default_rng(scn.get("seed", 0) if seed is None else seed)
    fps = layout.fps
    ent = layout.entrance
    door_in = (ent.a + ent.b) / 2 + ent.inside_normal * 0.8
    counter = np.asarray(scn.get("counter_point", layout.staff_home if layout.staff_home is not None else door_in), float)
    zones = {z.id: z for z in layout.zones}
    rows, intents = [], []
    next_id = 1
    for ses in sessions_from(scn):
        day0 = ses.day_index * 86400.0
        t_open, t_close = day0 + ses.start_hour * 3600, day0 + ses.end_hour * 3600
        profile = scn["hourly_arrivals"][ses.day_type]
        # ---- customers (arrive in groups)
        gp = np.asarray(scn.get("group_size_probs", [1.0]), float)
        mean_group = (np.arange(1, len(gp) + 1) * gp).sum() / gp.sum()
        h = ses.start_hour
        while h < ses.end_hour:
            h_end = min(math.floor(h) + 1, ses.end_hour)
            rate = profile[int(h) - scn["open_hour"]] * (h_end - h)       # people in this slice
            for _ in range(rng.poisson(rate / mean_group)):
                t0 = day0 + rng.uniform(h, h_end) * 3600
                size = 1 + rng.choice(len(gp), p=gp / gp.sum())
                gid = next_id
                plan = _plan_visit(rng, scn, zones)
                for m in range(size):
                    pid = next_id
                    next_id += 1
                    speed = float(np.clip(rng.normal(*scn.get("walk_speed", [1.1, 0.15])), 0.6, 1.6))
                    jit = lambda p: p + rng.normal(0, 0.35, 2)          # noqa: E731 (members spread out)
                    spawn = ent.outside_spawn + rng.normal(0, 0.6, 2)
                    wps = [(spawn, 0.0, speed), (door_in + rng.normal(0, 0.3, 2), 0.0, speed)]
                    for zid, engaged, dwell, target in plan:
                        if zid == "__counter__":
                            wps.append((jit(counter), dwell * rng.uniform(0.9, 1.1), speed))
                            continue
                        p = zones[zid].sample_point(rng) if engaged else target + rng.normal(0, 0.2, 2)
                        dw = dwell * rng.uniform(0.85, 1.15) if dwell > 0 else 0.0
                        sp = speed if engaged or dwell == 0 else speed * 0.6
                        wps.append((p, dw, sp))
                        intents.append({"person_id": pid, "group_id": gid, "session": ses.name, "zone": zid,
                                        "engaged": bool(engaged), "planned_dwell": round(dw, 2)})
                    wps += [(door_in + rng.normal(0, 0.3, 2), 0.0, speed), (ent.outside_spawn + rng.normal(0, 0.6, 2), 0.0, speed)]
                    xy = build_path(wps, fps, rng)
                    k0 = int(round((t0 + m * rng.uniform(0.4, 1.2)) * fps))
                    frames = k0 + np.arange(len(xy))
                    keep = frames / fps < t_close + 1800  # shoppers inside at closing are allowed to finish
                    rows.append(pd.DataFrame({"k": frames[keep], "id": pid, "x": xy[keep, 0], "y": xy[keep, 1],
                                              "role": "customer", "group": gid, "session": ses.name}))
            h = h_end
        # ---- staff (inside the whole session, never cross the entrance)
        st = scn.get("staff", {"count": 0})
        for _ in range(st.get("count", 0)):
            pid = next_id
            next_id += 1
            wps, t = [(counter + rng.normal(0, 0.5, 2), 0.0, 1.0)], t_open
            while t < t_close:
                if rng.random() < st.get("counter_prob", 0.5):
                    p, dw = counter + rng.normal(0, 0.6, 2), _dwell(rng, st.get("counter_dwell", [240, 0.6]), 20, 1800)
                else:
                    z = zones[rng.choice(list(zones))]
                    p, dw = z.sample_point(rng), _dwell(rng, st.get("zone_dwell", [60, 0.7]), 5, 900)
                wps.append((p, dw, 1.0))
                t += dw + 8
            xy = build_path(wps, fps, rng)
            k0 = int(round(t_open * fps))
            n = min(len(xy), int((t_close - t_open) * fps))
            rows.append(pd.DataFrame({"k": k0 + np.arange(n), "id": pid, "x": xy[:n, 0], "y": xy[:n, 1],
                                      "role": "staff", "group": pid, "session": ses.name}))
    tracks = pd.concat(rows, ignore_index=True)
    tracks["t"] = tracks["k"] / fps
    ses_start = {s.name: int(round((s.day_index * 86400 + s.start_hour * 3600) * fps)) for s in sessions_from(scn)}
    tracks["frame"] = tracks["k"] - tracks["session"].map(ses_start) + 1
    tracks = tracks[tracks["frame"] >= 1].drop(columns="k")
    return tracks[["session", "t", "frame", "id", "x", "y", "role", "group"]].reset_index(drop=True), intents


def _plan_visit(rng, scn, zones):
    """Ordered list of (zone_id, engaged, dwell_s, pass_point) for one shopping group."""
    plan = []
    for zid in zones:
        if rng.random() < scn["zone_visit_prob"].get(zid, 0.0):
            z = zones[zid]
            u = rng.random()
            if u < scn["engage_prob"].get(zid, 0.5):
                dwell = _dwell(rng, scn["dwell_s"].get(zid, [40, 0.7]), z.stop_seconds + 1.5, 900)
                plan.append((zid, True, dwell, z.centroid))
            elif u < scn["engage_prob"].get(zid, 0.5) + scn.get("glance_prob", 0.1):
                plan.append((zid, False, float(rng.uniform(0.5, 2.5)), z.sample_point(rng)))   # near-threshold slow-down
            else:
                plan.append((zid, False, 0.0, z.sample_point(rng)))
    order = rng.permutation(len(plan)) if rng.random() < 0.3 else np.argsort([zones[p[0]].centroid[0] for p in plan])
    if rng.random() < 0.5:
        order = order[::-1]
    plan = [plan[i] for i in order]
    if rng.random() < scn.get("counter_prob", 0.3):
        plan.insert(int(rng.integers(0, len(plan) + 1)), ("__counter__", True, _dwell(rng, scn.get("counter_dwell", [150, 0.6]), 20, 1200), None))
    return plan


# ------------------------------------------------------------------ cameras
def project_to_cameras(tracks: pd.DataFrame, layout: Layout, seed: int = 0) -> dict[str, dict[str, pd.DataFrame]]:
    """World tracks -> {camera: {session: MOT ground truth df}} with occlusion-aware visibility."""
    rng = np.random.default_rng(seed)
    heights = {pid: float(np.clip(rng.normal(1.70, 0.08), 1.45, 1.95)) for pid in tracks["id"].unique()}
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for cam in layout.cameras:
        out[cam.id] = {}
        for ses, g in tracks.groupby("session"):
            n = len(g)
            hh = g["id"].map(heights).to_numpy()
            foot = np.column_stack([g["x"], g["y"], np.zeros(n)])
            head = np.column_stack([g["x"], g["y"], hh])
            uvf, zf = cam.project(foot)
            uvh, _ = cam.project(head)
            bh = uvf[:, 1] - uvh[:, 1]
            bw = 0.41 * bh
            x1, y1 = uvf[:, 0] - bw / 2, uvh[:, 1]
            ok = (zf > 0.5) & (bh > 4) & (x1 + bw > 0) & (x1 < cam.width) & (y1 + bh > 0) & (y1 < cam.height)
            df = pd.DataFrame({"frame": g["frame"].to_numpy()[ok], "id": g["id"].to_numpy()[ok],
                               "x": x1[ok], "y": y1[ok], "w": bw[ok], "h": bh[ok], "depth": zf[ok],
                               "role": g["role"].to_numpy()[ok]})
            # truncation by the image border
            cx1, cy1 = df["x"].clip(0, cam.width), df["y"].clip(0, cam.height)
            cx2, cy2 = (df["x"] + df["w"]).clip(0, cam.width), (df["y"] + df["h"]).clip(0, cam.height)
            trunc = ((cx2 - cx1) * (cy2 - cy1) / (df["w"] * df["h"])).to_numpy()
            occ = _occlusion(df)
            df["vis"] = np.clip(trunc * (1 - occ), 0, 1)
            df["conf"] = (df["vis"] >= 0.1).astype(float)   # fully hidden boxes -> ignore in eval
            df["cls"] = 1
            out[cam.id][ses] = df.sort_values(["frame", "id"]).reset_index(drop=True)
    return out


def _occlusion(df: pd.DataFrame) -> np.ndarray:
    occ = np.zeros(len(df))
    d = df.sort_values("frame")
    idx = d.index.to_numpy()
    fr = d["frame"].to_numpy()
    X1, Y1, W, H, Z = (d[c].to_numpy() for c in ("x", "y", "w", "h", "depth"))
    bounds = np.flatnonzero(np.diff(fr)) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(fr)]):
        if e - s < 2:
            continue
        x1, y1, x2, y2, z = X1[s:e], Y1[s:e], X1[s:e] + W[s:e], Y1[s:e] + H[s:e], Z[s:e]
        ix = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
        iy = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
        inter = ix * iy
        nearer = z[None, :] < z[:, None]          # j is in front of i
        area = (x2 - x1) * (y2 - y1)
        occ[s:e] = np.clip((inter * nearer).sum(1) / area, 0, 1)
    out = np.zeros(len(df))
    out[df.index.get_indexer(idx)] = occ
    return out
