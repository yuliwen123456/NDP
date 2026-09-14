# Append-only version history

## V06 baseline — 2026-09-14

AP 74.55320680581926 / FPR95 1.4472667591263995 / AUROC 99.3599571729236.
Commit ca76a699cd17d85a90f71aef9c950912af5ad332; branch main; tag baseline-v06-ap74.5532-fpr1.4473-auroc99.3600.
Status KEEP / BASELINE. Original checkpoint stays in /root/autodl-tmp/NDP/outputs/compare-bn-ema-b2-20260914-005611/training/epoch=9.ckpt.

## V07 retained comparison — 2026-09-14

AP 60.53965299396991 / FPR95 0.7721332000154598 / AUROC 99.53671289839652.
Delta to V06 AP -14.01355381184935 / FPR95 -0.6751335591109397 / AUROC +0.17675572547293.
Original loss reduction + V06 BN policy. Status DROP as replacement for main AP baseline; retained for diagnostic comparison. Historical source/checkpoints remain unchanged and external hashes are recorded in V06 freeze evidence. No retroactive claim that this historical run originally had its own Git commit.
