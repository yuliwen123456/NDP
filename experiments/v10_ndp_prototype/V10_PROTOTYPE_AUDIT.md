# V10 Prototype audit — 2026-09-15

Audit completed before implementation. Latest user request explicitly authorizes smoke → full ID prototype construction → full frozen evaluation. No training is authorized or needed. The older conditional-combination V10 proposal stays separate and unchanged.

## Frozen baseline and inputs

- Baseline snapshot: `/root/autodl-tmp/NDP/research_versions/experiments/v06_baseline`; this version also imports its byte-identical `source_v06` copy in the new independent worktree.
- Config: `config_v06_baseline.yaml`; historical training save_dir is superseded by the recorded Round6 runtime override, not treated as checkpoint identity.
- Runtime: `/root/autodl-tmp/NDP/outputs/sixth-bn-ema-20260914/train.py`, frozen copy `runtime_v06/train.py`. BN/loss correction affects training only; strict frozen state_dict loading preserves the resulting model.
- Checkpoint: `/root/autodl-tmp/NDP/outputs/compare-bn-ema-b2-20260914-005611/training/epoch=9.ckpt`.
- Original predictions: same run, `epoch10-inference/prediction`.
- Original metrics: AP 74.55320680581926, AUROC 99.3599571729236, FPR95 1.4472667591263995. AP is sklearn average precision, not trapezoidal PR integration.
- Original full evaluation command:

```bash
/root/autodl-tmp/NDP/envs/ndp/bin/python /root/autodl-tmp/NDP/code/stu_dataset/compute_point_level_ood.py --data-dir /root/autodl-tmp/NDP/data/test --pred-dir /root/autodl-tmp/NDP/outputs/compare-bn-ema-b2-20260914-005611/epoch10-inference/prediction --output <new-output.json>
```

## Feature, scores, labels

`source_v06/main_panoptic.py` imports `trainer/pq_trainer.py:PanopticSegmentation` (not the legacy trainer). `models/mask4former.py` constructs Sparse UNet and `point_features_head`. Forward around lines 130/245 produces sparse `point_features.F` **[N_voxel,128]** immediately before the Projector. V10 uses a read-only forward hook here. Original full-point feature correspondence is **[N_original,128]** via the unchanged VoxelizeCollate inverse_map; sparse coordinate order must match exactly before applying it.

Original Projector outputs [N_voxel,38]; `id_logits` is its first 19 columns. The unchanged semantic mask branch outputs query `pred_logits` and `pred_masks`. Original commented semantic export in `pq_trainer.py:test_step` uses `sigmoid(pred_masks) @ softmax(pred_logits)[...,:-1]`, then argmax. V10 may export this without changing model outputs.

Original NDP score in `models/mask4former.py` around lines 245–267:

`S_ndp = (logsumexp(logits_all_38) - logsumexp(logits_first_19)) * class_weight(logits_all_38) + training_bias`.

The Neural Distribution Prior block is original NDP; V10 does not replace it. Raw `pixel_ood` is retained exactly and checked against saved Round6 predictions on three predetermined evaluation frames.

GT is uint32 raw panoptic label [N_original]; semantic is low 16 bits. Apply the original `conf/mask4former3d.yaml:learning_map`. Mapped 1..19 become model class IDs 0..18. Ignore/unlabeled/invalid/void mappings and raw OOD=2 are excluded from fitting. Check labels and inverse_map have identical point count. C=19, D=128.

## ID fitting policy, audited class coverage

Only `/root/autodl-tmp/NDP/data/train_database.yaml`: all 898 scene206 frames. No evaluation labels, predictions or threshold statistics enter fitting. Original pose, intensity, time=0, centered distance feature and 0.05m voxelization are reused. This is deterministic offline inference on the raw training frames, without stochastic training augmentation or synthetic instance population; no original training config, augmentation, dataset or source is edited. The same deterministic view is used for both prototype and normalization passes.

Full raw GT audit class counts, IDs 0..18:

`[2973370,1192,8084,334836,49750,49780,0,3410,18226563,0,1645694,87798,45646604,7203308,23838604,3712108,1578424,614584,355890]`.

Excluded points: 11372657, including 14029 raw OOD points. **Class6 bicyclist and class9 parking have zero training examples.** Keep [19,128] storage with explicit validity mask: absent rows remain zero, never participate in nearest-prototype search. No fabricated class mean or evaluation-data fallback. This limits ID coverage to 17 classes and must appear in results.

Prototype P_c is the mean of raw f_i as specified by the explicit mean formula, then unit-normalized. Query features are unit-normalized for cosine. Full original points contribute (including multiple points mapping to one voxel), with their own GT, not a possibly different representative voxel label. Accumulate in float64. Similarity is clamped only to the mathematical [-1,1] range for numerical roundoff.

Normalization is fixed ID-training population mean/std for both raw NDP and cosine distance, computed in a second forward pass after prototypes are fixed. No shared-scale assumption: raw NDP includes learned bias, whereas cosine distance is [0,2]. Alpha defaults to 0.5 in config. Scores use float64 fusion to avoid unnecessary rank ties. No alpha sweep or test tuning.

## Evaluation and mIoU qualification

Same `/root/autodl-tmp/NDP/data/test`: 8659 public STU validation frames (folder named test does not make it the hidden challenge test set). Import original `compute_point_level_ood.py:PointOODMetricsCalculator` unchanged. It ignores raw0, treats raw2 as OOD, retains distance 2.5–50m, skips frames with fewer than 5 retained OOD points, uses sklearn AP/AUC and first TPR strictly greater than .95. Three independent calls reuse these exact filters/formulas on A raw NDP, B cosine distance, C fused scores. NPZ is a lossless storage adapter, not a metric change.

Round6 did not save mIoU; validation_step is disabled and semantic export is commented. `source_v06/models/metrics/panoptic_eval.py:PanopticEval.addBatchSemIoU/getSemIoU` is the existing implementation. Supplementary mIoU is computed using its original 19-class mean and original semantic export, raw GT learning_map 1..19, other labels ignored as0. This is a newly calculated diagnostic on available STU labels, not a historical Round6 measurement or official STU semantic benchmark. All three methods share the identical semantic predictions and thus identical diagnostic mIoU; OOD fusion does not reject or relabel semantic predictions. OOD statistics/nearest-class distributions use the exact eligible point/frame population returned by the original OOD calculator.

## Isolation, resources, gates

Excluded implementations: `/root/autodl-tmp/NDP/research_version_worktrees/v08/experiments/v08_dual_scale_implementation01` and corresponding `v09/experiments/v09_class_conditional_implementation01`; no imports/calls to those or their shared protocol. No neighborhood operation or conditional calibration is introduced.

2026-09-15 GPU audit: torch sees two NVIDIA GeForce RTX3090, both initially 0% and 11MiB. Use GPU0 only, FP32 model inference, evaluation batch1, same as Round6; GPU1 left available. Record actual peak allocated/reserved memory. Prototype accumulation and z-score do not train anything.

New branch `exp/v10-ndp-prototype` starts at public main 5942dc5eaa740d12a82c0099ab1373bb9996a549. Only the new experiment directory is committed/pushed; private unapproved V08/V09 additions are not ancestors. Original sources/checkpoints and all historical versions stay unchanged, verified by integrity scripts and external hashes.

Smoke gate: bounded ID fit, serialization, all score paths, finite/range/shape/alignment checks, original-score equality, and actual metric invocation on three predetermined eval frames. Only after PASS build all 898-frame prototypes and ID normalization, then evaluate all8659 frames. New output directories are exclusive and never overwrite previous runs. Finite pipeline only; periodic monitoring stays disabled.
