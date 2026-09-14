# NDP reproduction research versions

Protected main baseline: V06, AP74.5532 / FPR951.4473 / AUROC99.3600 (STU public validation point-level).

Read [research audit](analysis/research_audit_v01.md), [version history](VERSION_HISTORY.md) and [baseline](experiments/v06_baseline/README.md). New proposals live on independent feature branches; no V08/V09/V10 algorithm or training has run.

Run `python3 -B scripts/check_version_integrity.py`; on the original server add `--external` to check source and checkpoint hashes. Enable local protection with `git config core.hooksPath .githooks` and executable `.githooks/pre-commit`. These are cooperative Git checks, not root-proof storage protection.

New completed metrics are recorded with `scripts/record_result_v01.py --version vXX --metrics <new result JSON> --conclusion KEEP|DROP|NEEDS_MORE_TESTING`; include AP/FPR95/AUROC, checkpoint, split and evaluation_command. This creates new results and appends history, then requires a scoped commit. Never overwrite historical proposal files.

Recover by `git worktree add <NEW_EMPTY_PATH> <tag>`; do not reset or delete existing research. Checkpoints/datasets remain on the original server, excluded from Git. Historical source paths may require restoration or a new portability entrypoint.

Upstream NDP code license is retained in experiments/v06_baseline/source_v06/LICENSE. Evaluation provenance is kumuji/stu_dataset. Repository publication is pending explicit resolution of an automatic approval rejection; no public push is claimed.
