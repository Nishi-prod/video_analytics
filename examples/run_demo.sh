#!/usr/bin/env bash
# End-to-end demo on synthetic data (~10 min on a laptop CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-out/demo}
python -m rvb synth --layout configs/cspire_store.json --scenario configs/scenario_demo.json --out $OUT/synth
python -m rvb corrupt --layout configs/cspire_store.json --gt $OUT/synth/gt --preset medium --out $OUT/pred_medium
python -m rvb eval --layout configs/cspire_store.json --gt $OUT/synth/gt --pred $OUT/pred_medium \
  --gt-events $OUT/synth/gt_events.jsonl --frame-stride 5 --title "Demo - simulated 'medium' tracker" --out $OUT/report_medium
python -m rvb sensitivity --layout configs/cspire_store.json --gt $OUT/synth/gt --out $OUT/sensitivity
