# Append-only version history

## V06 baseline — 2026-09-14

AP 74.55320680581926 / FPR95 1.4472667591263995 / AUROC 99.3599571729236.
Commit ca76a699cd17d85a90f71aef9c950912af5ad332; branch main; tag baseline-v06-ap74.5532-fpr1.4473-auroc99.3600.
Status KEEP / BASELINE. Original checkpoint stays in /root/autodl-tmp/NDP/outputs/compare-bn-ema-b2-20260914-005611/training/epoch=9.ckpt.

## V07 retained comparison — 2026-09-14

AP 60.53965299396991 / FPR95 0.7721332000154598 / AUROC 99.53671289839652.
Delta to V06 AP -14.01355381184935 / FPR95 -0.6751335591109397 / AUROC +0.17675572547293.
Original loss reduction + V06 BN policy. Status DROP as replacement for main AP baseline; retained for diagnostic comparison. Historical source/checkpoints remain unchanged and external hashes are recorded in V06 freeze evidence. No retroactive claim that this historical run originally had its own Git commit.

## V08 proposal — 2026-09-14

Commit 17acc5261a8f62c727648ccf24649eb1f6766248; branch feature/v08-dual-scale; tag proposal-v08. AP/FPR95/AUROC and deltas: unknown. Stage PROPOSAL_ONLY; status NEEDS_MORE_TESTING. No training or algorithm implementation.

## V09 proposal — 2026-09-14

Commit eaf27537653f5a69e786a27f8cb9bb43634abdad; branch feature/v09-class-conditional; tag proposal-v09. AP/FPR95/AUROC and deltas: unknown. Stage PROPOSAL_ONLY; status NEEDS_MORE_TESTING. No training or algorithm implementation.

## V10 proposal — 2026-09-14

Commit 04ef5d168659e28b2c003017f712c144894ee790; branch feature/v10-conditional-combination; tag proposal-v10. AP/FPR95/AUROC and deltas: unknown. Stage PROPOSAL_ONLY; status NEEDS_MORE_TESTING. No training or algorithm implementation.

## Implementation path clarification — 2026-09-14

Proposal directories are themselves frozen. Paths under their modules/ mentioned in the initial design are conceptual insertion locations, not permission to add into a frozen snapshot. Approved implementations must instead use NEW sibling directories experiments/v08_dual_scale_implementation01/ and experiments/v09_class_conditional_implementation01/, with new configs and entrypoints. V10 remains conditional. Do not overwrite the proposal metrics.

Public GitHub publication pending resolution of automatic approval rejection. Local commits/tags do not imply successful push.
