#!/usr/bin/env bash
set -euo pipefail
# Historical command, with an explicitly NEW output required. Never relaunch the old runner.
if [[ "${1:-}" != "--execute" ]]; then echo 'Record only. Pass --execute and NEW_OUTPUT to deliberately retrain.'; exit 0; fi
: "${NEW_OUTPUT:?Choose a new absolute directory under /root/autodl-tmp/NDP/outputs}"
[[ "$NEW_OUTPUT" == /root/autodl-tmp/NDP/outputs/* && ! -e "$NEW_OUTPUT" ]] || exit 2
source /root/autodl-tmp/NDP/activate_ndp.sh
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
exec python -B /root/autodl-tmp/NDP/outputs/sixth-bn-ema-20260914/train.py formal "general.save_dir=$NEW_OUTPUT/training" "hydra.run.dir=$NEW_OUTPUT/hydra"
