# V08 proposal: DSDC spatial-context / learned-prior consistency

Version v08; parent V06; date 2026-09-14; branch feature/v08-dual-scale.
Git commit: resolve the immutable proposal tag `proposal-v08` with `git rev-parse proposal-v08`; actual SHA is appended to main VERSION_HISTORY.md after commit.
Conclusion: NEEDS_MORE_TESTING. No algorithm implementation, training or evaluation has run.

Read main analysis/research_audit_v01.md for the full hypothesis, formula, duplication analysis, risks and acceptance protocol. Module planned only: experiments/v08_dual_scale/modules/dual_scale_distribution_v08.py.

All files here are NEW: README.md, config_v08_dual_scale.json, metrics.json, train_command.sh, eval_command.sh, manifest_v08.json. Future implementation needs additional versioned files outside this frozen proposal directory (for example experiments/v08_dual_scale_implementation01/); do not overwrite this proposal to implement it. This preserves the proposal and later implementation independently.

First version is training-free on the frozen V06 checkpoint. Training GPU count0; planned inference GPU1. Parent training setup: physical2, accumulation4, nominal effective8, tail2, FP32, AdamW2e-4, OneCycle1130 optimizer steps, seed2025,10epochs, unchanged V06 BN and loss policies.

Checkpoint: /root/autodl-tmp/NDP/outputs/compare-bn-ema-b2-20260914-005611/training/epoch=9.ckpt. No checkpoint copied. AP/FPR95/AUROC and deltas are unknown (null).

Commands currently refuse execution with exit2: `bash train_command.sh` / `bash eval_command.sh`. They are explicit proposal placeholders, not executable experiments. A future approved implementation must provide NEW entrypoints, new output directories and exact inference/evaluation argv. No STU OOD validation/test labels may fit hyperparameters. Identity settings must reproduce the frozen baseline pointwise.

When results exist, create a new timestamped result via scripts/record_result_v01.py and append history. Keep this proposal unchanged, including metrics.json. V10 is not authorized unless V08/V09 both succeed independently; never select the best metric from different checkpoints and label it a combined model.
