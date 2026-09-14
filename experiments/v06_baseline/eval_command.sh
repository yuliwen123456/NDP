#!/usr/bin/env bash
set -euo pipefail
# Re-evaluate EXISTING predictions; no inference or training. Require a new metrics file.
if [[ "${1:-}" != "--execute" ]]; then echo 'Record only. Pass --execute and NEW_METRICS to deliberately evaluate.'; exit 0; fi
: "${NEW_METRICS:?Choose a new JSON output path}"
[[ ! -e "$NEW_METRICS" ]] || exit 2
exec /root/autodl-tmp/NDP/envs/ndp/bin/python /root/autodl-tmp/NDP/code/stu_dataset/compute_point_level_ood.py --data-dir /root/autodl-tmp/NDP/data/test --pred-dir /root/autodl-tmp/NDP/outputs/compare-bn-ema-b2-20260914-005611/epoch10-inference/prediction --output "$NEW_METRICS"
