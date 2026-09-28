# retail-video-bench

Benchmark tooling for the C Spire retail video analytics POC (Cimba AI). It covers the three assigned tasks:

| Task | Priority | What's here |
|---|---|---|
| Download the open-source test datasets | P0 | `datasets/download.py` + `datasets/manifest.json`, prep scripts in `datasets/prep/` |
| Synthetic footfall and zone data for benchmarking | P1 | `rvb/synth/` shopper simulator, projection to cameras, tracker error model |
| Eval framework for each part of the pipeline | P1 | `rvb/eval/`: detection, tracking, Re-ID, business metrics, evidence/VLM, plus CLI reports |

Everything runs on CPU in Python 3.10+. Install with `pip install -r requirements.txt`.

---

## 1. Datasets (P0)

Run on the team server, not a laptop: MOT17 alone is 5.5 GB and SmartSpaces videos are around 70 GB.

```bash
export RVB_DATA=/data/rvb
python datasets/download.py list                          # sizes, licences, how each is fetched
python datasets/download.py get p0                         # MOT17, MOT20, CAVIAR, SmartSpaces (annotations), + access steps for MMPTRACK
python datasets/download.py get smartspaces --with-videos  # add the MP4s (depth maps are always excluded, ~3 TB)
python datasets/download.py get market1501 personpath22
python datasets/download.py verify                         # checks expected files; DOWNLOADS.json holds sha256 of every file
```

| id | Used for | Size | Licence | Commercial? | How |
|---|---|---|---|---|---|
| mmptrack | tracking, Re-ID, retail (640x320 @ 15fps) | ~50 GB | Research, signed agreement | no | **request now**, takes days |
| mot17 / mot20 | detection, tracking | 5.5 / 5 GB | CC BY-NC-SA 3.0 | no | direct |
| caviar | stop-vs-pass on real shop-front video | 0.4 GB | Research, acknowledge CAVIAR | no | direct |
| smartspaces | multi-camera tracking, derived zone GT | 70 GB (no depth) | CC BY 4.0 | **yes** | Hugging Face |
| personpath22 | tracking robustness | ~60 GB | CC BY-NC 4.0 | no | upstream script (needs aws cli + ffmpeg) |
| market1501 | Re-ID | 0.15 GB | Research | no | direct / Google Drive mirror |
| msmt17 | Re-ID (harder) | 3 GB | Research, signed agreement | no | request |
| meva, merl_shopping, retailaction | activities / VLM tests | large | mixed | see manifest | S3 / index / HF |

**Preparing the data to match C Spire's cameras:**

```bash
# every public benchmark is scored at 640x360 @ 15 fps
python datasets/prep/downscale_mot.py --src $RVB_DATA/mot17/MOT17/train --dst $RVB_DATA/mot17_cspire/train --only FRCNN
python datasets/prep/downscale_mot.py --src $RVB_DATA/mot20/MOT20/train --dst $RVB_DATA/mot20_cspire/train
python datasets/prep/caviar_to_mot.py --src $RVB_DATA/caviar --dst $RVB_DATA/caviar_mot --fps 15
python datasets/prep/smartspaces_to_mot.py --scene $RVB_DATA/smartspaces/MTMC_Tracking_2025/val/* --dst $RVB_DATA/smartspaces_mot
```

---

## 2. Synthetic footfall and zone data (P1)

No public dataset labels footfall, dwell per display or stop-vs-pass in a phone store, so we generate ground truth three ways. Each is cheaper than the last is realistic:

| Tier | What | Realism | Cost | Status |
|---|---|---|---|---|
| **A. Trajectory simulation** | Agent-based shoppers in a store layout, projected into 3 virtual 640x360 cameras; tracker errors injected | Behaviour realistic, no pixels | Minutes on CPU | **Built here** |
| **B. Derived GT on synthetic video** | SmartSpaces world positions + our zone polygons, giving exact events on real rendered video | Real pixels, synthetic people | Download + one command | **Built here** (`prep/smartspaces_to_mot.py` + `rvb derive`) |
| **C. Rendered C Spire twin** | Same scenarios rendered in NVIDIA Omniverse / Isaac Sim IRA with a C Spire-like layout and 3 cameras | Closest to target | Days of setup, GPU | Plan only: scenario + layout JSON are the inputs |

Tier A answers "if our tracker is this good, how wrong will the business numbers be?" before any model exists. Tier B checks the full pixel-to-answer pipeline. The final check is always the labelled C Spire golden set.

**Run it:**

```bash
python -m rvb synth --layout configs/cspire_store.json --scenario configs/scenario_week.json --out out/synth_week
# out/synth_week/
#   gt/<camera>/<day>.txt       MOT ground truth (boxes, occlusion-aware visibility; conf=0 = fully hidden)
#   gt/<camera>/<day>.roles.csv staff / customer per track
#   world_tracks.csv            exact positions in metres
#   intents.jsonl               what each shopper meant to do (engage / glance / pass per zone)
#   gt_events.jsonl             ground-truth footfall + zone events, from the cameras
#   gt_events_world.jsonl       the same, from exact world positions
#   preview.png                 floor plan + a frame from each camera
```

`configs/scenario_*.json` controls hourly arrival profiles (weekday vs weekend), group sizes, zone visit / stop / glance probabilities, dwell distributions, counter visits and staff. **The demo numbers are illustrative; calibrate them from C Spire's week of footage** (hourly entrance tallies and a few hand-timed stops are enough).

`configs/cspire_store.json` is an illustrative layout (entrance plus iPhone and Samsung tables and walls). Replace it with C Spire's zone map. Zones can also be given directly in pixels (`image_polygon`, `image_line`) when there is no floor plan.

**Simulated tracker output:**

```bash
python -m rvb corrupt --layout configs/cspire_store.json --gt out/synth_week/gt --preset medium --out out/pred_medium
# presets: perfect | good | medium | poor ; --reid-recovery 0.9 simulates Re-ID repairing 90% of track breaks
# --override '{"p_switch":0.3,"miss_small":0.4}' to explore one error type
```

---

## 3. Eval framework (P1)

One command scores a prediction folder against ground truth at every level. The folder layout is the same for synthetic data, public benchmarks, the C Spire golden set and real model output: one MOT file per camera and session.

```bash
python -m rvb eval --layout configs/cspire_store.json --gt out/synth_week/gt --pred out/pred_medium \
    --gt-events out/synth_week/gt_events.jsonl --frame-stride 5 --out out/report_medium
# -> report.md (scorecard) + report.json (every number)
python -m rvb sensitivity --layout configs/cspire_store.json --gt out/synth_week/gt --out out/sensitivity
python -m rvb reid features.npz          # q_feat,q_pid,q_cam,g_feat,g_pid,g_cam (e.g. exported from torchreid)
```

| Level | Module | Metrics | Ground truth from |
|---|---|---|---|
| 1 Detection | `rvb/eval/detection.py` | AP50, precision / recall / F1 at a threshold, **recall by person size** (<50 px is the risk at 640x360) | MOT17Det, MMPTRACK, synthetic, golden set |
| 2 Tracking | `rvb/eval/tracking.py` | HOTA, DetA, AssA, MOTA, IDF1, IDSW, MT/ML (official **TrackEval** code); customers only, staff ignored | MOT17/20 (downscaled), MMPTRACK, PersonPath22, SmartSpaces |
| 2 Re-ID | `rvb/eval/reid.py` | Rank-1/5/10, mAP (Market-1501 protocol, same-camera matches removed) | Market-1501, MSMT17 |
| 3 Business | `rvb/eval/business.py` | Footfall total % error, hourly MAE/MAPE; **peak hours** overlap@3 and Spearman; zone visits; **stop-vs-pass** P/R/F1; engagement-rate error; mean dwell error; occupancy MAE per minute; zone-to-zone flow error | Synthetic, SmartSpaces-derived, CAVIAR, golden set |
| Evidence / VLM | `rvb/eval/vlm.py` | Evidence coverage (target 100%), grounding (clip overlaps a real event), numeric accuracy ±15%, human rubric (faithful / relevant / specific / privacy) | Golden set |

**How business metrics are computed:** `rvb/zones.py` is the single reference zone engine. It turns tracks into `STORE_ENTER`/`STORE_EXIT` crossings and `ZONE_VISIT`s with dwell and an engaged flag (dwell ≥ `stop_seconds`, default 5 s). The **same rules** are applied to ground-truth and predicted tracks, so the business score isolates perception errors. Rule constants (`RuleConfig`) are documented in the file. Agree the stop threshold with C Spire at kickoff.

**Plugging in the real pipeline:** write your tracker's output as `<pred>/<camera>/<session>.txt` (MOT format), plus `<session>.roles.csv` if you classify staff, and `sessions.json` with each recording's NTP start time. Then run `rvb eval`. For Ultralytics YOLO + ByteTrack/BoT-SORT, dump `frame,id,x,y,w,h,conf` per box.

---

## Demo results (synthetic, 4 store-hours, 125 shoppers + 4 staff, one seed)

Produced by `examples/run_demo.sh`. **These illustrate the framework; they are not model results.** Outputs are in `examples/` (`demo_report_medium.md`, `sensitivity.csv`, `sensitivity.png`, `synthetic_store_preview.png`).

| Simulated tracker | IDF1 | Footfall error | Peak hours right | Zone visit count error | Stop-vs-pass F1 | Mean dwell error |
|---|---|---|---|---|---|---|
| perfect | 1.00 | -2.4% | 3/3 | 2% | 1.00 | 2.1 s |
| good | 0.72 | -4.0% | 3/3 | 8% | 0.98 | 1.5 s |
| medium | 0.59 | -8.9% | 3/3 | 18% | 0.95 | 5.0 s |
| poor | 0.31 | -21.0% | 3/3 | 97% | 0.93 | 9.3 s |
| medium + Re-ID fixes 90% of breaks | 0.63 | -6.5% | 3/3 | 15% | 0.95 | 7.5 s |

What this shows:

- **Peak hours are robust.** Every tracker ranked the busiest hours correctly, so "when is the store busiest" is the safest answer.
- **Footfall degrades steadily with tracker quality**, from -2% to -21%. Trackers under-count because people are missed while crossing the entrance line.
- **Zone traffic is the most fragile number.** Track breaks at busy tables split one visit into several, inflating visits by 18% (medium) to 97% (poor). Stop-vs-pass on matched visits stays accurate.
- **Simulated Re-ID helps modestly** (IDF1 0.59 → 0.63, footfall -8.9% → -6.5%, visit error 18% → 15%). One seed is noisy (dwell error went up), so repeat with several seeds before drawing conclusions.
- **Even a perfect tracker is ~2% off on footfall**, because people fully hidden at the moment they cross the line cannot be recovered from boxes. That is the floor for this camera placement.

`configs/scenario_week.json` (7 days x 10 h) has not been run here. Expect roughly 20x the demo's time and memory.

## Tests

```bash
python -m pytest -q tests      # 12 unit tests, ~3 s
```

## Layout of the repo

```
datasets/  manifest.json, download.py, prep/{downscale_mot,caviar_to_mot,smartspaces_to_mot}.py
configs/   cspire_store.json (layout + cameras), scenario_demo.json, scenario_week.json
rvb/       layout.py, zones.py, pipeline.py, io.py, report.py, viz.py, cli.py
rvb/synth/ simulator.py (shoppers -> world tracks -> per-camera MOT GT), noise.py (tracker error model)
rvb/eval/  detection.py, tracking.py, reid.py, business.py, vlm.py
tests/     unit tests
examples/  run_demo.sh + demo outputs
```
