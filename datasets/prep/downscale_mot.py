#!/usr/bin/env python3
"""Downscale MOTChallenge-style sequences to C Spire's camera format (default 640x360 @ 15 fps).

Every public benchmark is scored at the resolution and frame rate the POC will actually see.
Handles MOT17 / MOT20 folders (img1/, gt/gt.txt, det/det.txt, seqinfo.ini):
  * keeps the frames closest to the target frame rate and renumbers them 1..N
  * resizes images to fit inside the target size (aspect ratio kept, no padding)
  * scales gt/det boxes by the same factor and writes a new seqinfo.ini

  python datasets/prep/downscale_mot.py --src data/mot17/MOT17/train --dst data/mot17_640x360_15fps/train
  python datasets/prep/downscale_mot.py --src data/mot17/MOT17/train --dst ... --only FRCNN   # one detector copy
"""
from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import numpy as np
import pandas as pd


def read_seqinfo(p: Path) -> dict:
    c = configparser.ConfigParser()
    c.read(p)
    s = c["Sequence"]
    return {k: s[k] for k in s}


def pick_frames(n: int, src_fps: float, dst_fps: float) -> np.ndarray:
    """1-based source frame numbers to keep."""
    if dst_fps >= src_fps:
        return np.arange(1, n + 1)
    t = np.arange(0, n / src_fps, 1 / dst_fps)
    return np.unique(np.clip(np.round(t * src_fps).astype(int) + 1, 1, n))


def convert_txt(src: Path, dst: Path, keep: np.ndarray, scale: float) -> int:
    if not src.exists():
        return 0
    df = pd.read_csv(src, header=None)
    remap = {int(f): i + 1 for i, f in enumerate(keep)}
    df = df[df[0].isin(remap)].copy()
    df[0] = df[0].map(remap)
    df[[2, 3, 4, 5]] = df[[2, 3, 4, 5]] * scale
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, header=False, index=False, float_format="%.2f")
    return len(df)


def convert_sequence(seq: Path, out: Path, width: int, height: int, fps: float, copy_images: bool = True) -> dict:
    import cv2
    info = read_seqinfo(seq / "seqinfo.ini")
    n, sfps = int(info["seqlength"]), float(info["framerate"])
    iw, ih = int(info["imwidth"]), int(info["imheight"])
    scale = min(width / iw, height / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    keep = pick_frames(n, sfps, fps)
    ext = info.get("imext", ".jpg")
    if copy_images:
        (out / "img1").mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(keep, 1):
            img = cv2.imread(str(seq / info.get("imdir", "img1") / f"{f:06d}{ext}"))
            cv2.imwrite(str(out / "img1" / f"{i:06d}.jpg"), cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
    n_gt = convert_txt(seq / "gt" / "gt.txt", out / "gt" / "gt.txt", keep, scale)
    convert_txt(seq / "det" / "det.txt", out / "det" / "det.txt", keep, scale)
    real_fps = sfps * (len(keep) - 1) / max(keep[-1] - keep[0], 1) if len(keep) > 1 else fps
    (out / "seqinfo.ini").write_text(
        "[Sequence]\n" f"name={out.name}\nimDir=img1\nframeRate={round(real_fps, 3)}\nseqLength={len(keep)}\n"
        f"imWidth={nw}\nimHeight={nh}\nimExt=.jpg\n" f"; downscaled from {iw}x{ih}@{sfps} by rvb (scale {scale:.4f})\n")
    return {"seq": seq.name, "src": f"{iw}x{ih}@{sfps:g}", "dst": f"{nw}x{nh}@{real_fps:.2f}", "frames": len(keep), "gt_rows": n_gt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder holding sequence folders (e.g. MOT17/train)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--only", help="substring filter, e.g. FRCNN (MOT17 has 3 identical copies per sequence)")
    ap.add_argument("--labels-only", action="store_true", help="skip image resizing (boxes/frames only)")
    a = ap.parse_args()
    for seq in sorted(p for p in Path(a.src).iterdir() if (p / "seqinfo.ini").exists()):
        if a.only and a.only not in seq.name:
            continue
        print(convert_sequence(seq, Path(a.dst) / seq.name, a.width, a.height, a.fps, not a.labels_only))


if __name__ == "__main__":
    main()
