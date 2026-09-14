#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/NDP/activate_ndp.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo 'Use one GPU only.' >&2; exit 2; }
DATA_ROOT="${DATA_ROOT:-$NDP_ROOT/data}"
NDP_CKPT="${NDP_CKPT:-$NDP_ROOT/checkpoints/NDP-OOD/ap74.ckpt}"
[[ -f "$NDP_CKPT" ]] || { echo "Checkpoint missing: $NDP_CKPT" >&2; exit 2; }
[[ -d "$DATA_ROOT/test" ]] || { echo "Missing mapped STU test directory: $DATA_ROOT/test. Run prepare_after_stu_upload.sh first." >&2; exit 2; }
[[ -n "$(find -L "$DATA_ROOT/test" -type f -name '*.bin' -print -quit)" ]] || { echo 'No LiDAR .bin files in mapped test directory.' >&2; exit 2; }
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
OUT="${OUTPUT_DIR:-$NDP_ROOT/outputs/inference-$RUN_ID}"
[[ "$OUT" == "$NDP_ROOT/outputs/"* && ! -e "$OUT" ]] || { echo 'OUTPUT_DIR must be a new directory under NDP/outputs (upstream rewrites ckpt_path for existing directories).' >&2; exit 2; }
cd "$NDP_ROOT/code/ndp"
exec python main_panoptic.py model=mask4former3d data/datasets=semantic_kitti_206 "general.ckpt_path=$NDP_CKPT" general.mode=test "data.base_path=$DATA_ROOT" "general.save_dir=$OUT" "hydra.run.dir=$NDP_ROOT/outputs/hydra-$RUN_ID" "$@"
