"""Glue between per-camera MOT files and the zone engine, plus directory conventions.

Directory layout (same for synthetic GT, real GT and model outputs):

  <dir>/sessions.json                      {"<session>": {"start_t": seconds, "fps": 15}}
  <dir>/<camera>/<session>.txt             MOT file (frame,id,x,y,w,h,conf,cls,vis[,role])
  <dir>/<camera>/<session>.roles.csv       optional: id,role (staff/customer) per track

Timestamps: t = start_t + (frame - 1) / fps. For C Spire footage use the NTP start time of
each recording (seconds since a fixed epoch); hour buckets are then floor(t / 3600).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import mot_to_footpoints, read_mot, write_mot
from .layout import Entrance, Layout, Zone
from .zones import RuleConfig, derive_events


def camera_view(layout: Layout, cam_id: str, space: str = "world"):
    """Zones (+ entrance) handled by one camera, in world metres or in that camera's pixels."""
    cam = layout.camera(cam_id)
    zones = layout.zones_for_camera(cam_id)
    ent = layout.entrance if layout.entrance.camera == cam_id else None
    if space == "image":
        raw = {z["id"]: z for z in layout.raw["zones"]}
        zones = [Zone(z.id, np.asarray(raw[z.id]["image_polygon"], float) if "image_polygon" in raw[z.id]
                      else cam.ground_to_pixel(z.polygon), z.camera, z.stop_seconds, z.label) for z in zones]
        if ent is not None:
            e = layout.raw["entrance"]
            a, b = (np.asarray(e["image_line"], float) if "image_line" in e else cam.ground_to_pixel(np.array([ent.a, ent.b])))
            inside_px = cam.ground_to_pixel(((ent.a + ent.b) / 2 + ent.inside_normal)[None])[0]
            n = np.array([-(b - a)[1], (b - a)[0]])
            n = n if (inside_px - a) @ n > 0 else -n
            ent = Entrance(a, b, n / np.linalg.norm(n), cam_id)
    return zones, ent


def mot_to_tracks(mot: pd.DataFrame, layout: Layout, cam_id: str, start_t: float, fps: float, space: str = "world",
                  roles: dict | None = None) -> pd.DataFrame:
    if "role" not in mot:
        mot = mot.assign(role=mot["id"].map(roles or {}).fillna("customer"))
    tr = mot_to_footpoints(mot, fps)
    tr["t"] = tr["t"] + start_t
    if space == "world":
        xy = layout.camera(cam_id).pixel_to_ground(tr[["x", "y"]].to_numpy(float))
        tr["x"], tr["y"] = xy[:, 0], xy[:, 1]
    return tr


def load_dir(d: str | Path):
    d = Path(d)
    sessions = json.loads((d / "sessions.json").read_text())
    files = {}
    for cam_dir in sorted(p for p in d.iterdir() if p.is_dir()):
        for f in sorted(cam_dir.glob("*.txt")):
            files[(cam_dir.name, f.stem)] = f
    return sessions, files


def read_roles(mot_path: Path) -> dict:
    rp = mot_path.with_suffix(".roles.csv")
    if rp.exists():
        r = pd.read_csv(rp)
        return dict(zip(r["id"], r["role"]))
    return {}


def events_from_dir(d: str | Path, layout: Layout, space: str = "world", cfg: RuleConfig | None = None) -> list[dict]:
    sessions, files = load_dir(d)
    events = []
    for (cam, ses), path in files.items():
        if ses not in sessions:
            continue
        mot = read_mot(path)
        tr = mot_to_tracks(mot, layout, cam, sessions[ses]["start_t"], sessions[ses].get("fps", layout.fps), space, read_roles(path))
        zones, ent = camera_view(layout, cam, space)
        for e in derive_events(tr, zones, ent, sessions[ses].get("fps", layout.fps), cfg, source=cam):
            e["session"] = ses
            events.append(e)
    events.sort(key=lambda e: e["t"])
    return events


def write_mot_dir(per_cam: dict[str, dict[str, pd.DataFrame]], out: str | Path, sessions: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sessions.json").write_text(json.dumps(sessions, indent=1))
    for cam, by_ses in per_cam.items():
        for ses, df in by_ses.items():
            write_mot(df, out / cam / f"{ses}.txt")
            if "role" in df:
                roles = df.groupby("id")["role"].agg(lambda s: s.mode().iat[0]).reset_index()
                roles.to_csv(out / cam / f"{ses}.roles.csv", index=False)
