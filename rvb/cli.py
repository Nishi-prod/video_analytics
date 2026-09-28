"""Command line entry point:  python -m rvb <command> ...

  synth        simulate shoppers -> synthetic ground truth (world tracks, per-camera MOT, events)
  corrupt      ground-truth MOT -> simulated tracker output (error presets)
  events       MOT directory -> footfall / zone events (the reference zone engine)
  derive       world-coordinate tracks -> ground-truth events (derived GT for SmartSpaces/MMPTRACK)
  eval         score predictions against ground truth at every pipeline level
  sensitivity  sweep tracker quality and Re-ID recovery -> how errors reach business metrics
  reid         score Re-ID embeddings (npz) with the Market-1501 protocol
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_events, read_mot, to_jsonable, write_events, write_tracks
from .layout import load_layout
from .pipeline import events_from_dir, load_dir, read_roles, write_mot_dir
from .report import to_markdown


def cmd_synth(a):
    from .synth.simulator import load_scenario, project_to_cameras, sessions_from, simulate
    from .zones import derive_events, events_frame
    L, S = load_layout(a.layout), load_scenario(a.scenario)
    out = Path(a.out)
    t0 = time.time()
    tracks, intents = simulate(L, S, a.seed)
    sessions = {s.name: {"start_t": s.day_index * 86400 + s.start_hour * 3600, "fps": L.fps, "day_type": s.day_type}
                for s in sessions_from(S)}
    per_cam = project_to_cameras(tracks, L, seed=a.seed or 0)
    write_mot_dir(per_cam, out / "gt", sessions)
    write_tracks(tracks, out / "world_tracks.csv")
    write_events(intents, out / "intents.jsonl")
    world_ev = derive_events(tracks, L.zones, L.entrance, L.fps, source="world")
    write_events(world_ev, out / "gt_events_world.jsonl")
    gt_ev = events_from_dir(out / "gt", L)
    write_events(gt_ev, out / "gt_events.jsonl")
    we, ge = events_frame(world_ev), events_frame(gt_ev)
    summary = {
        "layout": L.name, "scenario": a.scenario, "sessions": sessions,
        "people": int(tracks.loc[tracks.role == "customer", "id"].nunique()),
        "staff": int(tracks.loc[tracks.role == "staff", "id"].nunique()),
        "store_entries_world": int(((we.type == "STORE_ENTER") & (we.role != "staff")).sum()),
        "store_entries_cameras": int(((ge.type == "STORE_ENTER") & (ge.role != "staff")).sum()),
        "zone_visits_world": int(((we.type == "ZONE_VISIT") & (we.role != "staff")).sum()),
        "zone_visits_cameras": int(((ge.type == "ZONE_VISIT") & (ge.role != "staff")).sum()),
        "mot_boxes": {c: {s: len(d) for s, d in v.items()} for c, v in per_cam.items()},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=1))
    if not a.no_preview:
        from .viz import preview
        preview(L, tracks, per_cam, out / "preview.png")
    print(json.dumps(summary, indent=1))


def cmd_corrupt(a):
    from .synth.noise import get_preset, corrupt
    L = load_layout(a.layout)
    sessions, files = load_dir(a.gt)
    over = json.loads(a.override) if a.override else {}
    if a.reid_recovery is not None:
        over["reid_recovery"] = a.reid_recovery
    cfg = get_preset(a.preset, **over)
    per_cam: dict = {}
    for i, ((cam, ses), path) in enumerate(files.items()):
        gt = read_mot(path)
        gt["role"] = gt["id"].map(read_roles(path)).fillna("customer")
        c = L.camera(cam)
        per_cam.setdefault(cam, {})[ses] = corrupt(gt, cfg, c.width, c.height, seed=(a.seed or 0) + i)
    write_mot_dir(per_cam, a.out, sessions)
    (Path(a.out) / "noise_config.json").write_text(json.dumps({"preset": a.preset, **cfg.to_dict()}, indent=1))
    print(f"wrote simulated tracker output ({a.preset}) to {a.out}")


def cmd_events(a):
    L = load_layout(a.layout)
    ev = events_from_dir(a.mot_dir, L, a.space)
    write_events(ev, a.out)
    print(f"{len(ev)} events -> {a.out}")


def cmd_derive(a):
    """World-coordinate tracks (e.g. SmartSpaces / MMPTRACK footpoints) + layout zones -> GT events."""
    from .io import read_tracks
    from .zones import derive_events
    L = load_layout(a.layout)
    tr = read_tracks(a.tracks)
    ev = derive_events(tr, L.zones, L.entrance if not a.no_entrance else None, a.fps or L.fps, source="world")
    write_events(ev, a.out)
    print(f"{len(ev)} events -> {a.out}")


def run_eval(L, gt_dir, pred_dir, stages, frame_stride=1, gt_events_path=None, insights=None, space="world", track_all=False):
    from .eval.business import evaluate_business
    from .eval.detection import evaluate_detection
    from .eval.tracking import evaluate_tracking
    res = {}
    g_sessions, g_files = load_dir(gt_dir)
    p_sessions, p_files = load_dir(pred_dir)
    common = [k for k in g_files if k in p_files]
    if "det" in stages or "track" in stages:
        pairs = {}
        for cam, ses in common:
            g, p = read_mot(g_files[(cam, ses)]), read_mot(p_files[(cam, ses)])
            g["role"] = g["id"].map(read_roles(g_files[(cam, ses)])).fillna("customer")
            if frame_stride > 1:
                g, p = g[(g.frame - 1) % frame_stride == 0], p[(p.frame - 1) % frame_stride == 0]
            pairs[f"{cam}/{ses}"] = (g, p)
        if "det" in stages:
            res["detection"] = {k: evaluate_detection(g, p) for k, (g, p) in pairs.items()}
        if "track" in stages:
            # staff stand around for hours; their very long tracks would dominate IDF1/HOTA, so tracking is
            # scored on customers (staff boxes become 'ignore'). Use --track-all to also score everyone.
            cust = {k: (g.assign(conf=np.where(g["role"] == "staff", 0, g["conf"])), p) for k, (g, p) in pairs.items()}
            res["tracking"] = evaluate_tracking(cust)
            res["tracking"]["scope"] = "customers (staff boxes ignored)"
            if track_all:
                res["tracking_all_people"] = evaluate_tracking(pairs)
    if "biz" in stages:
        gt_ev = read_events(gt_events_path) if gt_events_path else events_from_dir(gt_dir, L, space)
        pr_ev = events_from_dir(pred_dir, L, space)
        hours = None
        zbc = {c.id: [z.id for z in L.zones_for_camera(c.id)] for c in L.cameras}
        res["business"] = evaluate_business(gt_ev, pr_ev, [z.id for z in L.zones], zbc, hours)
        if insights:
            from .eval.vlm import evaluate_evidence
            res["vlm"] = evaluate_evidence(read_events(insights), gt_ev)
    return res


def cmd_eval(a):
    L = load_layout(a.layout)
    stages = a.stages.split(",")
    res = run_eval(L, a.gt, a.pred, stages, a.frame_stride, a.gt_events, a.insights, a.space, a.track_all)
    if "reid" in stages and a.reid_npz:
        from .eval.reid import evaluate_reid_npz
        res["reid"] = evaluate_reid_npz(a.reid_npz)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(to_jsonable(res), indent=1))
    md = to_markdown(res, a.title)
    (out / "report.md").write_text(md)
    print(md)


def cmd_sensitivity(a):
    """Run the tracker error model at several quality levels -> business error table + chart."""
    import shutil
    import tempfile
    from .synth.noise import corrupt, get_preset
    L = load_layout(a.layout)
    sessions, files = load_dir(a.gt)
    gt_ev = read_events(Path(a.gt).parent / "gt_events.jsonl") if (Path(a.gt).parent / "gt_events.jsonl").exists() else None
    runs = [(p, 0.0) for p in a.presets.split(",")] + [(a.reid_preset, r) for r in (0.5, 0.9)]
    rows = []
    tmp = Path(tempfile.mkdtemp())
    for preset, rr in runs:
        cfg = get_preset(preset, reid_recovery=rr)
        per_cam = {}
        for i, ((cam, ses), path) in enumerate(files.items()):
            gt = read_mot(path)
            gt["role"] = gt["id"].map(read_roles(path)).fillna("customer")
            per_cam.setdefault(cam, {})[ses] = corrupt(gt, cfg, seed=i)
        pdir = tmp / f"{preset}_{rr}"
        write_mot_dir(per_cam, pdir, sessions)
        res = run_eval(L, a.gt, pdir, ["track", "biz"], a.frame_stride, Path(a.gt).parent / "gt_events.jsonl" if gt_ev else None)
        tc, b = res["tracking"]["combined"], res["business"]
        rows.append({"preset": preset, "reid_recovery": rr, "HOTA": tc["HOTA"], "IDF1": tc["IDF1"], "MOTA": tc["MOTA"],
                     "IDSW": tc["IDSW"], "footfall_total_pct_err": b["footfall"]["total_pct_error"],
                     "footfall_hourly_MAPE": b["footfall"]["hourly_MAPE"], "peak_overlap": b["footfall"]["peak_overlap_at_k"],
                     "stop_vs_pass_F1": b["zones"]["stop_vs_pass"]["F1"],
                     "engagement_rate_abs_err": b["zones"]["engagement_rate_abs_error"],
                     "visits_pct_err_mean": float(np.mean([abs(v["visits_pct_error"]) for v in b["zones"]["per_zone"].values() if v["visits_pct_error"] is not None])),
                     "mean_dwell_abs_err_s": b["zones"]["mean_dwell_abs_error_s"]})
        print(rows[-1])
    shutil.rmtree(tmp, ignore_errors=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sensitivity.csv", index=False)
    from .viz import sensitivity_chart
    sensitivity_chart(df, out / "sensitivity.png")
    print(df.round(3).to_string(index=False))


def cmd_reid(a):
    from .eval.reid import evaluate_reid_npz
    print(json.dumps(evaluate_reid_npz(a.npz, a.metric), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="rvb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--layout", required=True)
    s.add_argument("--scenario", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int)
    s.add_argument("--no-preview", action="store_true")
    c = sub.add_parser("corrupt")
    c.add_argument("--layout", required=True)
    c.add_argument("--gt", required=True, help="MOT directory (e.g. synth_out/gt)")
    c.add_argument("--preset", default="medium", choices=["perfect", "good", "medium", "poor"])
    c.add_argument("--reid-recovery", type=float)
    c.add_argument("--override", help='JSON of NoiseConfig fields, e.g. \'{"p_switch":0.3}\'')
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    e = sub.add_parser("events")
    e.add_argument("--layout", required=True)
    e.add_argument("--mot-dir", required=True)
    e.add_argument("--space", default="world", choices=["world", "image"])
    e.add_argument("--out", required=True)
    dv = sub.add_parser("derive")
    dv.add_argument("--layout", required=True)
    dv.add_argument("--tracks", required=True, help="CSV t,id,x,y[,role] in the layout's world coordinates")
    dv.add_argument("--fps", type=float)
    dv.add_argument("--no-entrance", action="store_true")
    dv.add_argument("--out", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--layout", required=True)
    ev.add_argument("--gt", required=True)
    ev.add_argument("--pred", required=True)
    ev.add_argument("--stages", default="det,track,biz")
    ev.add_argument("--gt-events", help="use these GT events instead of deriving them from --gt")
    ev.add_argument("--insights", help="insights.jsonl for the evidence/VLM check")
    ev.add_argument("--reid-npz")
    ev.add_argument("--frame-stride", type=int, default=1, help="evaluate det/track every Nth frame (5 = 3 fps)")
    ev.add_argument("--space", default="world", choices=["world", "image"])
    ev.add_argument("--track-all", action="store_true", help="also score tracking on everyone incl. staff")
    ev.add_argument("--title", default="Evaluation report")
    ev.add_argument("--out", required=True)
    se = sub.add_parser("sensitivity")
    se.add_argument("--layout", required=True)
    se.add_argument("--gt", required=True)
    se.add_argument("--presets", default="perfect,good,medium,poor")
    se.add_argument("--reid-preset", default="medium")
    se.add_argument("--frame-stride", type=int, default=5)
    se.add_argument("--out", required=True)
    r = sub.add_parser("reid")
    r.add_argument("npz")
    r.add_argument("--metric", default="cosine")
    a = p.parse_args(argv)
    {"synth": cmd_synth, "corrupt": cmd_corrupt, "events": cmd_events, "derive": cmd_derive, "eval": cmd_eval,
     "sensitivity": cmd_sensitivity, "reid": cmd_reid}[a.cmd](a)


if __name__ == "__main__":
    main()
