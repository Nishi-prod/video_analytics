"""Fast unit tests: python -m pytest -q tests"""
import numpy as np
import pandas as pd
import pytest

from rvb.eval.business import evaluate_business
from rvb.eval.detection import evaluate_detection
from rvb.eval.reid import evaluate_reid
from rvb.eval.tracking import evaluate_tracking
from rvb.eval.vlm import evaluate_evidence
from rvb.layout import Camera, Entrance, Zone
from rvb.synth.noise import corrupt, get_preset
from rvb.zones import RuleConfig, derive_events

FPS = 15


def mot(rows):
    return pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"]).assign(conf=1.0, cls=1, vis=1.0)


def walker(pid, start_frame, n, x0, y0, dx=0.0, dy=0.0):
    return [(start_frame + k, pid, x0 + dx * k, y0 + dy * k, 40, 100) for k in range(n)]


# ---------------------------------------------------------------- tracking
def test_tracking_perfect_is_one():
    gt = mot(walker(1, 1, 50, 10, 10, dx=2) + walker(2, 1, 50, 300, 10, dx=-2))
    r = evaluate_tracking({"s": (gt, gt.copy())})["combined"]
    assert r["HOTA"] == pytest.approx(1.0) and r["IDF1"] == pytest.approx(1.0) and r["IDSW"] == 0


def test_tracking_id_switch_detected():
    gt = mot(walker(1, 1, 40, 10, 10, dx=1) + walker(2, 1, 40, 300, 10, dx=-1))
    pred = gt.copy()
    pred.loc[(pred.frame > 20) & (pred.id == 1), "id"] = 99       # fragment id 1 halfway
    r = evaluate_tracking({"s": (gt, pred)})["combined"]
    assert r["IDSW"] >= 1 and r["IDF1"] < 1.0 and r["MOTA"] < 1.0


def test_tracking_ignores_conf0_gt():
    gt = mot(walker(1, 1, 30, 10, 10) + walker(2, 1, 30, 300, 10))
    gt.loc[gt.id == 2, "conf"] = 0
    pred = gt.copy()                           # predicting the ignored person must not count as FP
    r = evaluate_tracking({"s": (gt, pred)})["combined"]
    assert r["MOTA"] == pytest.approx(1.0) and r["CLR_FP"] == 0


# ---------------------------------------------------------------- detection
def test_detection_counts():
    gt = mot(walker(1, 1, 10, 10, 10) + walker(2, 1, 10, 300, 10))
    pred = mot(walker(1, 1, 10, 12, 11) + walker(7, 1, 10, 500, 200)).assign(conf=0.9)
    r = evaluate_detection(gt, pred)
    assert r["TP"] == 10 and r["FP"] == 10 and r["FN"] == 10
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------- zone engine
def square_zone(stop=5.0):
    return Zone("z", np.array([[0, 0], [2, 0], [2, 2], [0, 2.0]]), stop_seconds=stop)


def track_through(tid, t0, pts_and_secs):
    rows, t = [], t0
    for (x, y), secs in pts_and_secs:
        for _ in range(int(secs * FPS)):
            rows.append((t, tid, x, y))
            t += 1 / FPS
    return pd.DataFrame(rows, columns=["t", "id", "x", "y"])


def test_stop_vs_pass_threshold():
    stay = track_through(1, 0, [((-1, 1), 1), ((1, 1), 8), ((3, 1), 1)])
    pas = track_through(2, 0, [((-1, 1), 1), ((1, 1), 2), ((3, 1), 1)])
    ev = derive_events(pd.concat([stay, pas]), [square_zone()], None, FPS)
    v = {e["track_id"]: e for e in ev if e["type"] == "ZONE_VISIT"}
    assert v[1]["engaged"] and v[1]["dwell"] == pytest.approx(8, abs=0.1)
    assert not v[2]["engaged"]


def test_exit_grace_bridges_short_gap():
    tr = track_through(1, 0, [((1, 1), 3), ((3, 1), 1.0), ((1, 1), 3)])   # steps out 1 s
    ev = [e for e in derive_events(tr, [square_zone()], None, FPS) if e["type"] == "ZONE_VISIT"]
    assert len(ev) == 1 and ev[0]["dwell"] == pytest.approx(7, abs=0.2)


def test_entrance_direction_and_debounce():
    ent = Entrance(np.array([0, 0.0]), np.array([2, 0.0]), np.array([0, 1.0]))
    walk_in = track_through(1, 0, [((1, -1), 1), ((1, 1), 1)])
    jitter = track_through(2, 0, [((1, -0.05), 0.2), ((1, 0.05), 0.2), ((1, -0.05), 0.2)])
    ev = derive_events(pd.concat([walk_in, jitter]), [], ent, FPS)
    kinds = [(e["track_id"], e["type"]) for e in ev]
    assert (1, "STORE_ENTER") in kinds and not any(t == 2 for t, _ in kinds)


# ---------------------------------------------------------------- business eval
def test_business_perfect_prediction():
    tr = pd.concat([track_through(i, 3600 * (10 + i % 3) + i, [((1, -1), 1), ((1, 1), 6 + i), ((1, -1), 1)]) for i in range(12)])
    ent = Entrance(np.array([0, 0.0]), np.array([2, 0.0]), np.array([0, 1.0]))
    ev = derive_events(tr, [square_zone()], ent, FPS)
    r = evaluate_business(ev, ev, ["z"])
    assert r["footfall"]["total_pct_error"] == 0 and r["footfall"]["hourly_MAE"] == 0
    assert r["zones"]["stop_vs_pass"]["F1"] == pytest.approx(1.0)
    assert r["zones"]["per_zone"]["z"]["mean_dwell_abs_error_s"] == pytest.approx(0)


# ---------------------------------------------------------------- re-id
def test_reid_protocol():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(20, 16))
    g = np.repeat(centers, 3, 0) + rng.normal(0, 0.05, (60, 16))
    gp, gc = np.repeat(np.arange(20), 3), np.tile([1, 2, 3], 20)
    q = centers + rng.normal(0, 0.05, (20, 16))
    r = evaluate_reid(q, np.arange(20), np.ones(20, int), g, gp, gc)
    assert r["rank1"] == pytest.approx(1.0) and r["mAP"] > 0.99 and r["n_valid_queries"] == 20


# ---------------------------------------------------------------- camera geometry
def test_camera_ground_roundtrip():
    cam = Camera("c", np.array([0, 0, 3.0]), np.array([3, 5, 0.0]), 400)
    xy = np.array([[2.0, 4.0], [4.0, 7.0]])
    assert np.allclose(cam.pixel_to_ground(cam.ground_to_pixel(xy)), xy, atol=1e-6)


# ---------------------------------------------------------------- noise model
def test_noise_perfect_is_identity_and_poor_degrades():
    gt = mot(walker(1, 1, 60, 10, 10, dx=2) + walker(2, 1, 60, 300, 10, dx=-2)).assign(role="customer")
    perfect = corrupt(gt, get_preset("perfect"))
    r = evaluate_tracking({"s": (gt, perfect)})["combined"]
    assert r["HOTA"] == pytest.approx(1.0)
    poor = corrupt(gt, get_preset("poor"), seed=1)
    assert evaluate_tracking({"s": (gt, poor)})["combined"]["HOTA"] < 0.95


# ---------------------------------------------------------------- evidence
def test_evidence_checks():
    tr = track_through(1, 100, [((1, 1), 10)])
    ev = derive_events(tr, [square_zone()], None, FPS)
    ins = [{"id": "I1", "zone": "z", "metric": "mean_dwell_s", "value": 10.2,
            "evidence": [{"camera": "c", "t_start": 101, "t_end": 105, "clip": "c.mp4"}]},
           {"id": "I2", "zone": "z", "metric": "visits", "value": 5, "evidence": []}]
    r = evaluate_evidence(ins, ev)
    assert r["evidence_coverage"] == 0.5 and r["evidence_grounding"] == 1.0 and r["numeric_accuracy"] == 0.5
