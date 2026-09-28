"""File formats used across the benchmark.

* MOT format (MOTChallenge): frame,id,x,y,w,h,conf,class,visibility  (frame is 1-based)
* Tracks (world or image plane, used by the zone engine): CSV t,frame,id,x,y[,role]
* Events: JSON lines, one event per line (see rvb.zones.Event)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MOT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]


def read_mot(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :9]
    n = df.shape[1]
    df.columns = MOT_COLS[:n]
    for c, default in (("conf", 1.0), ("cls", 1), ("vis", 1.0)):
        if c not in df:
            df[c] = default
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    return df


def write_mot(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c, default in (("conf", 1.0), ("cls", 1), ("vis", 1.0)):
        if c not in out:
            out[c] = default
    out = out[MOT_COLS].sort_values(["frame", "id"])
    out.to_csv(path, header=False, index=False, float_format="%.2f")


def read_tracks(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "role" not in df:
        df["role"] = "customer"
    return df


def write_tracks(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.3f")


def mot_to_footpoints(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Image-plane track points = bottom-centre of each box (where the person stands)."""
    return pd.DataFrame({
        "t": (df["frame"].to_numpy() - 1) / fps,
        "frame": df["frame"].to_numpy(),
        "id": df["id"].to_numpy(),
        "x": df["x"].to_numpy() + df["w"].to_numpy() / 2,
        "y": df["y"].to_numpy() + df["h"].to_numpy(),
        "role": df["role"].to_numpy() if "role" in df else "customer",
    })


def write_events(events, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e if isinstance(e, dict) else e.to_dict()) + "\n")


def read_events(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o
