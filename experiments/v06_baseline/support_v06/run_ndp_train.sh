#!/usr/bin/env bash
# Official NDP does NOT support multi-GPU training. No DDP modifications.
# Authors used a single A100; for a 24GB 3090, explicitly choose batch size
# and learning rate after reviewing conf/data/kitti3d.yaml and conf/optimizer/adamw.yaml.
# Example additional overrides: data.batch_size=... optimizer.lr=...
set -euo pipefail
source /root/autodl-tmp/NDP/activate_ndp.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo 'Official code supports one GPU only.' >&2; exit 2; }
DATA_ROOT="${DATA_ROOT:-$NDP_ROOT/data}"
STU_CKPT="${STU_CKPT:-$NDP_ROOT/checkpoints/STU/59p6pq_ens1.ckpt}"
BATCH_SIZE="${NDP_BATCH_SIZE:-1}"
TEST_BATCH_SIZE="${NDP_TEST_BATCH_SIZE:-1}"
[[ -d "$DATA_ROOT/original" && -f "$DATA_ROOT/train_database.yaml" && -f "$DATA_ROOT/train_instances_database.yaml" ]] || { echo 'STU mapping and authorized preprocessing are required first.' >&2; exit 2; }
[[ -f "$STU_CKPT" ]] || { echo "Missing STU pretrained checkpoint: $STU_CKPT" >&2; exit 2; }
cd "$NDP_ROOT/code/ndp"
# Upstream hardcodes this filename. Never replace an existing file/link.
if [[ ! -e 59p6pq_ens1.ckpt && ! -L 59p6pq_ens1.ckpt ]]; then ln -s "$STU_CKPT" 59p6pq_ens1.ckpt; fi
[[ "$(readlink -f 59p6pq_ens1.ckpt)" == "$(readlink -f "$STU_CKPT")" ]] || { echo 'Existing checkpoint path differs; preserved, refusing to run.' >&2; exit 2; }
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
OUT="${OUTPUT_DIR:-$NDP_ROOT/outputs/train-$RUN_ID}"
[[ "$OUT" == "$NDP_ROOT/outputs/"* && ! -e "$OUT" ]] || { echo 'OUTPUT_DIR must be new and under NDP/outputs.' >&2; exit 2; }
exec python main_panoptic.py model=mask4former3d data/datasets=semantic_kitti_206 general.mode=train "data.base_path=$DATA_ROOT" "general.save_dir=$OUT" "data.batch_size=$BATCH_SIZE" "data.test_batch_size=$TEST_BATCH_SIZE" "hydra.run.dir=$NDP_ROOT/outputs/hydra-$RUN_ID" "$@"
