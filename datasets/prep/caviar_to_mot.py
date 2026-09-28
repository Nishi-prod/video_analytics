#!/usr/bin/env python3
"""Convert CAVIAR shopping-centre ground truth (XML) to MOT format plus behaviour labels.

CAVIAR is the only public real-video set here with per-person *behaviour* labels in a
shop setting (walking past a shop front, browsing, entering/leaving a shop). We use it to
sanity-check stop-vs-pass logic on real footage.

Outputs per sequence (in --dst/<sequence>/):
  gt/gt.txt              MOT boxes at the source 25 fps (or --fps, e.g. 15)
  labels.csv             frame,id,movement,role,context,situation (per box)
  behaviour.csv          one row per person: browsed (any 'browsing'), entered_shop, frames, seconds

  python datasets/prep/caviar_to_mot.py --src data/caviar --dst data/caviar_mot [--fps 15]
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

SRC_FPS = 25.0


def _txt(obj, tag):
    el = obj.find(f".//{tag}")
    return (el.text or "").strip() if el is not None else ""


def parse_caviar_xml(path: Path) -> pd.DataFrame:
    root = ET.parse(path).getroot()
    rows = []
    for fr in root.iter("frame"):
        f = int(fr.get("number")) + 1                      # CAVIAR frames start at 0
        for obj in fr.iter("object"):
            box = obj.find("box")
            if box is None:
                continue
            h, w = float(box.get("h")), float(box.get("w"))
            xc, yc = float(box.get("xc")), float(box.get("yc"))
            rows.append({"frame": f, "id": int(obj.get("id")) + 1, "x": xc - w / 2, "y": yc - h / 2, "w": w, "h": h,
                         "appearance": _txt(obj, "appearance"), "movement": _txt(obj, "movement"),
                         "role": _txt(obj, "role"), "context": _txt(obj, "context"), "situation": _txt(obj, "situation")})
    return pd.DataFrame(rows)


def convert(seq_dir: Path, out: Path, fps: float | None) -> dict | None:
    xmls = sorted(seq_dir.glob("*.xml"))
    if not xmls:
        return None
    df = parse_caviar_xml(xmls[0])
    if df.empty:
        return None
    if fps and fps < SRC_FPS:
        n = df["frame"].max()
        keep = np.unique(np.round(np.arange(0, n / SRC_FPS, 1 / fps) * SRC_FPS).astype(int) + 1)
        remap = {int(k): i + 1 for i, k in enumerate(keep)}
        df = df[df["frame"].isin(remap)].copy()
        df["frame"] = df["frame"].map(remap)
    eff_fps = fps or SRC_FPS
    out.mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(exist_ok=True)
    mot = df.assign(conf=np.where(df["appearance"].str.contains("occluded"), 0, 1), cls=1, vis=1.0)
    mot[["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]].to_csv(out / "gt" / "gt.txt", header=False, index=False, float_format="%.1f")
    df[["frame", "id", "movement", "role", "context", "situation"]].to_csv(out / "labels.csv", index=False)
    txt = (df["context"] + " " + df["situation"] + " " + df["role"]).str.lower()
    beh = df.assign(browse=txt.str.contains("brows"), enter=txt.str.contains("enter"),
                    still=df["movement"].str.lower().isin(["inactive", "active"]))
    per = beh.groupby("id").agg(frames=("frame", "size"), browsed=("browse", "any"), entered_shop=("enter", "any"),
                                still_frames=("still", "sum")).reset_index()
    per["seconds"] = per["frames"] / eff_fps
    per["still_seconds"] = per["still_frames"] / eff_fps
    per.to_csv(out / "behaviour.csv", index=False)
    return {"seq": seq_dir.name, "people": len(per), "browsed": int(per["browsed"].sum()), "entered": int(per["entered_shop"].sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--fps", type=float, help="resample, e.g. 15 to match C Spire")
    a = ap.parse_args()
    for seq in sorted(p for p in Path(a.src).iterdir() if p.is_dir()):
        r = convert(seq, Path(a.dst) / seq.name, a.fps)
        if r:
            print(r)


if __name__ == "__main__":
    main()
