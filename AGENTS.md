# Immutable NDP research versions

V06 is the protected main baseline. Never overwrite any historical experiment source, configuration, scoring, loss, evaluation, predictions or checkpoints. Create a new numbered file and independent version directory for every change. Keep unsuccessful versions.

Run `python3 -B scripts/check_version_integrity.py` before each commit; use `--external` on the original server to also check preserved historical source and checkpoints. Enable the included pre-commit hook in every new clone: `git config core.hooksPath .githooks`. Hooks are local enforcement, not tamper-proof access control.

VERSION_HISTORY.md is append-only. Version proposals remain immutable after commit; later results go in a new timestamped results file, with deltas to V06, and an appended history entry. Do not edit the proposal metrics.json to fill in results.

Use separate version branches, scoped commits and tags. Restore a version into a NEW worktree instead of resetting the current workspace. Never stage dataset or checkpoint files. Do not change the original upstream checkout remote. The research remote is https://github.com/yuliwen123456/NDP.git. Public publication currently needs resolution of an automatic approval rejection; do not bypass it via another transport.

V08/V09 are proposals only. No algorithm implementation or training until the user approves the module design. V10 is conditional on independent successful ablations. Never fit thresholds, score fusion, or gates against STU OOD validation/test labels. Diagnostic reports are not calibration datasets. Keep periodic monitoring paused unless explicitly requested again.
