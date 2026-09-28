#!/usr/bin/env python3
"""Convert an NVIDIA PhysicalAI-SmartSpaces (2025/2026 edition) scene to rvb formats.

Why: SmartSpaces gives *world* positions for every person, so we can draw virtual zones on
its top-down map and derive exact footfall / dwell / stop-vs-pass ground truth on realistic
multi-camera video - a "derived ground truth" benchmark that is CC BY 4.0 (commercial OK).

Input  <scene>/ground_truth.json  {frame_id: [{object_type, object_id, 3d_location,
                                    2d_bounding_box_visible: {camera_id: [x1,y1,x2,y2]}}]}
Output <dst>/gt/<camera>/<scene>.txt       per-camera MOT (optionally rescaled, e.g. 1080p -> 640x360)
       <dst>/gt/sessions.json
       <dst>/world_tracks.csv              t,frame,id,x,y,role (people only)

Then: write a layout JSON with zone polygons (and an entrance line) in the scene's world
coordinates - see configs/cspire_store.json - and derive exact GT events:
  python -m rvb derive --layout my_scene_layout.json --tracks <dst>/world_tracks_<scene>.csv --out gt_events.jsonl

Check the first scene by hand: field names follow the dataset README (Sept 2026); adjust
PERSON_TYPES if a new edition renames classes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PERSON_TYPES = {"person", "Person", "human", "Human"}


def convert(scene: Path, dst: Path, src_fps: float, fps: float, scale: float, cameras: list[str] | None):
    gt = json.loads((scene / "ground_truth.json").read_text())
    step = max(1, int(round(src_fps / fps)))
    boxes, world = [], []
    for fid_s, objs in gt.items():
        fid = int(fid_s)
        if fid % step:
            continue
        f = fid // step + 1
        for o in objs:
            if o.get("object_type") not in PERSON_TYPES:
                continue
            pid = int(o["object_id"])
            loc = o.get("3d_location")
            if loc:
                world.append((f, pid, float(loc[0]), float(loc[1])))
            for cam, b in (o.get("2d_bounding_box_visible") or {}).items():
                if cameras and cam not in cameras:
                    continue
                x1, y1, x2, y2 = (v * scale for v in b)
                boxes.append((cam, f, pid, x1, y1, x2 - x1, y2 - y1))
    name = scene.name
    B = pd.DataFrame(boxes, columns=["cam", "frame", "id", "x", "y", "w", "h"])
    for cam, g in B.groupby("cam"):
        p = dst / "gt" / cam / f"{name}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        g.assign(conf=1, cls=1, vis=1.0)[["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]] \
            .sort_values(["frame", "id"]).to_csv(p, header=False, index=False, float_format="%.2f")
    sp = dst / "gt" / "sessions.json"
    ses = json.loads(sp.read_text()) if sp.exists() else {}
    ses[name] = {"start_t": 0.0, "fps": fps}
    sp.write_text(json.dumps(ses, indent=1))
    W = pd.DataFrame(world, columns=["frame", "id", "x", "y"]).sort_values(["id", "frame"])
    W.insert(0, "t", (W["frame"] - 1) / fps)
    W["role"] = "customer"
    W.to_csv(dst / f"world_tracks_{name}.csv", index=False, float_format="%.3f")
    return {"scene": name, "cameras": int(B["cam"].nunique()), "people": int(W["id"].nunique()), "frames": int(W["frame"].max() or 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, nargs="+", help="scene folder(s) containing ground_truth.json")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--src-fps", type=float, default=30)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--scale", type=float, default=1 / 3, help="box scale, 1/3 maps 1920x1080 -> 640x360")
    ap.add_argument("--cameras", nargs="*")
    a = ap.parse_args()
    for s in a.scene:
        print(convert(Path(s), Path(a.dst), a.src_fps, a.fps, a.scale, a.cameras))


if __name__ == "__main__":
    main()
