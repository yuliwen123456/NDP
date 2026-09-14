# V06 immutable baseline

Version: V06. Parent: original STU initialization, not V05 continuation.
Date: 2026-09-14. Git branch: main (baseline snapshot commit).
Git tag: baseline-v06-ap74.5532-fpr1.4473-auroc99.3600.
Git commit: resolve this immutable tag with `git rev-parse baseline-v06-ap74.5532-fpr1.4473-auroc99.3600`; the actual SHA is recorded in the later append-only VERSION_HISTORY.md entry. A commit cannot embed its own hash.

Conclusion: KEEP. Main research baseline, best observed self-trained AP.

AP 74.55320680581926; FPR95 1.4472667591263995; AUROC 99.3599571729236.
STU public validation, 8659 frames, point-level OOD. Not hidden test or object-level metrics.

GPU: 1 RTX3090, GPU0. FP32. Physical batch2, accumulation4, nominal effective batch8.
AdamW peak LR2e-4, OneCycle1130 updates, 10 epochs, seed2025, 449 micro-batches/epoch.
Mixed-reduction compensation and partial-group loss compensation enabled.
62 backbone BNs: 16 originally .02, 46 originally .1; m=1-(1-original_m)**(1/r), actual group length r; full4 and tail1.
OOD score: EE * learned NDP weight + training_bias. It is not plain entropy or plain energy.

Checkpoint (do not copy to Git): /root/autodl-tmp/NDP/outputs/compare-bn-ema-b2-20260914-005611/training/epoch=9.ckpt
Initialization: /root/autodl-tmp/NDP/checkpoints/STU/59p6pq_ens1.ckpt
Historical entry/log: /root/autodl-tmp/NDP/outputs/sixth-bn-ema-20260914/train.py and controller.log
Full run: /root/autodl-tmp/NDP/outputs/compare-bn-ema-b2-20260914-005611
Original code: /root/autodl-tmp/NDP/code/ndp, commit f11dfbe4181db03a6a036d46b52a308f45d0e255.

New snapshot files: config_v06_baseline.yaml, metrics.json, freeze_evidence_v06.json, train_command.sh, eval_command.sh, runtime_v06/, source_v06/, support_v06/, README.md, manifest_v06.json.
source_v06 and runtime_v06 preserve historical bytes and filenames within an independent version directory. Do not edit them to develop a new version. Runtime scripts contain historical absolute paths; the documented commands use the preserved server checkout. Recovering on another host requires restoring these paths or a NEW versioned portability entrypoint, never editing this snapshot.
The evaluation command is recorded in eval_command.sh; historical inference and evaluation argv are retained in runtime_v06/plan.json. Scripts default to record-only and require explicit new output paths.

Original source is Apache-2.0; its LICENSE and README are retained in source_v06. Support evaluation provenance: kumuji/stu_dataset, commit 8f0f09c (full SHA recorded in subsequent audit). No dataset, checkpoint, credentials, or full prediction files are tracked.

Rollback/reproduce: inspect the baseline tag in a NEW git worktree, then verify manifest_v06.json and external file hashes. Do not reset the active workspace, delete later versions, or overwrite outputs. Code rollback does not recreate absent datasets, dependencies or checkpoints.
