"""Evidence / VLM evaluation for insights.

The POC is scored on "evidence provided per insight (video clip, timestamp): 100%".
Each insight the system reports is a JSON line like:

  {"id": "I3", "text": "Samsung table: 38% of passers-by stop, avg 41 s",
   "zone": "samsung_table", "metric": "mean_dwell_s", "value": 41.0,
   "evidence": [{"camera": "cam_samsung", "t_start": 58210.0, "t_end": 58262.0, "clip": "clips/I3_1.mp4",
                 "caption": "Two shoppers stop at the Samsung table and pick up phones"}]}

Automatic checks (no model needed):
* coverage   - share of insights with >= 1 evidence item that has a clip and a timestamp
* grounding  - share of evidence windows that overlap a real GT event in the claimed zone
               (entrance events for footfall claims)
* numeric    - share of numeric claims within tolerance of the value computed from GT events
Captions are judged by people with the rubric in rubric_template(); scores are merged in
via merge_human_scores().
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ..zones import events_frame

SUPPORTED_METRICS = ("mean_dwell_s", "engagement_rate", "visits", "footfall_total")


def gt_value(ev: pd.DataFrame, metric: str, zone: str | None, t0=None, t1=None) -> float | None:
    e = ev[ev["role"] != "staff"]
    if t0 is not None:
        e = e[(e["t"] >= t0) & (e["t"] < t1)]
    if metric == "footfall_total":
        return float((e["type"] == "STORE_ENTER").sum())
    v = e[(e["type"] == "ZONE_VISIT") & (e["zone"] == zone)]
    if v.empty:
        return None
    if metric == "visits":
        return float(len(v))
    if metric == "engagement_rate":
        return float(v["engaged"].astype(bool).mean())
    if metric == "mean_dwell_s":
        eng = v[v["engaged"].astype(bool)]
        return float(eng["dwell"].mean()) if len(eng) else None
    return None


def evaluate_evidence(insights: list[dict], gt_events: list[dict], rel_tol: float = 0.15) -> dict:
    ev = events_frame(gt_events)
    covered, grounded, n_ev, numeric_ok, numeric_n = 0, 0, 0, 0, 0
    rows = []
    for ins in insights:
        items = ins.get("evidence", [])
        has = any(it.get("clip") and it.get("t_start") is not None for it in items)
        covered += has
        zone = ins.get("zone")
        g_hits = 0
        for it in items:
            if it.get("t_start") is None:
                continue
            n_ev += 1
            a, b = float(it["t_start"]), float(it.get("t_end", it["t_start"]))
            if zone:
                cand = ev[(ev["type"] == "ZONE_VISIT") & (ev["zone"] == zone)]
                hit = ((cand["t"] <= b) & (cand["t_end"] >= a)).any()
            else:
                cand = ev[ev["type"].isin(["STORE_ENTER", "STORE_EXIT"])]
                hit = ((cand["t"] >= a - 2) & (cand["t"] <= b + 2)).any()
            g_hits += bool(hit)
        grounded += g_hits
        num = None
        if ins.get("metric") in SUPPORTED_METRICS and ins.get("value") is not None:
            ref = gt_value(ev, ins["metric"], zone, ins.get("t0"), ins.get("t1"))
            if ref is not None:
                numeric_n += 1
                num = abs(ins["value"] - ref) <= rel_tol * max(abs(ref), 1e-9)
                numeric_ok += bool(num)
        rows.append({"id": ins.get("id"), "has_evidence": has, "grounded_items": g_hits,
                     "evidence_items": len(items), "numeric_ok": num})
    n = max(len(insights), 1)
    return {"n_insights": len(insights), "evidence_coverage": covered / n,
            "evidence_grounding": grounded / n_ev if n_ev else None,
            "numeric_accuracy": numeric_ok / numeric_n if numeric_n else None,
            "numeric_tolerance": rel_tol, "per_insight": rows}


RUBRIC = [
    ("faithful", "Caption describes only what is visible in the clip (no invented actions/products)", "0-2"),
    ("relevant", "Clip actually supports the insight it is attached to", "0-2"),
    ("specific", "Mentions the zone/display and behaviour (stopped, browsed, walked past)", "0-2"),
    ("privacy", "No identifying description of a person (face, name, ethnicity, etc.)", "pass/fail"),
]


def rubric_template(insights: list[dict], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["insight_id", "clip", "caption"] + [r[0] for r in RUBRIC] + ["rater", "notes"])
        for ins in insights:
            for it in ins.get("evidence", []):
                w.writerow([ins.get("id"), it.get("clip"), it.get("caption", "")] + [""] * (len(RUBRIC) + 2))


def merge_human_scores(path: str | Path) -> dict:
    df = pd.read_csv(path)
    out = {}
    for name, _, scale in RUBRIC:
        col = df[name].dropna()
        if scale == "pass/fail":
            out[name + "_pass_rate"] = float(col.astype(str).str.lower().isin(["pass", "1", "true", "yes"]).mean()) if len(col) else None
        else:
            out[name + "_mean_0to2"] = float(pd.to_numeric(col, errors="coerce").mean()) if len(col) else None
    out["n_rated"] = int(len(df))
    return out
