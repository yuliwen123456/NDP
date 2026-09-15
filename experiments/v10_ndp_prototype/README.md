# V10: frozen Round6 NDP + Prototype

Read `V10_PROTOTYPE_AUDIT.md` first. Only Prototype is new. The original raw NDP score, semantic branch, training source/config/checkpoint and STU metrics remain unchanged. Missing training classes are explicitly masked. No fitting uses evaluation labels.

```bash
source /root/autodl-tmp/NDP/activate_ndp.sh
python -B /root/autodl-tmp/NDP/research_version_worktrees/v10_prototype/experiments/v10_ndp_prototype/pipeline.py --output /root/autodl-tmp/NDP/outputs/v10_ndp_prototype_<unique_timestamp>
```

The output path must not exist. `--smoke-only` runs bounded tests only. Without it, the finite pipeline runs unit tests, three-frame smoke, original metric smoke, all898 training frames to form prototypes, a second ID-only pass for fixed z-score, all8659 evaluation frames, and three metric reports. Failure stops subsequent phases. No recurring monitoring or training is installed.

`config.json` controls alpha (default0.5), epsilon and data/checkpoint identity. No evaluation-based alpha search. All runtime outputs are outside Git. In `full/`: prototypes.pt, training_manifest.json, normalization.json, prediction/<scene>/<frame>.npz, inference_result.json, three score metrics, semantic_metrics.json, score_statistics.json, nearest_prototype_statistics.json, three histograms and V10_RESULTS.md. Three score arrays and nearest class/similarity/prototype distance are preserved for every point. The semantic prediction is shared across methods.

AP is the Round6 sklearn average_precision_score implementation. mIoU is newly recomputed using the existing PanopticEval and commented original semantic decoder on available STU labels; it is qualified as supplementary, since no historical Round6 mIoU or official STU semantic benchmark was present. No original evaluation definition is silently changed.

Feature means use all original ID points (raw GT aligned by the unchanged voxel inverse map), float64 accumulation, then unit normalization. Prototype scoring uses L2 query features. Two zero-count classes in the complete available training split remain invalid; they never win nearest-prototype search. This limitation is recorded rather than filling from OOD/validation data.

Offline fitting uses a deterministic inference view of raw training scans with original pose/features/voxelization. Stochastic training augmentation and synthetic instance population are not run during frozen inference; no original augmentation configuration or training behavior is edited.

Branch: exp/v10-ndp-prototype. Old conditional-combination V10 remains separate. No V08/V09 implementation imports are present. CPU threads8, batch1, one RTX3090 GPU0, FP32 model. No dependency installation required in the existing NDP environment.

Plotting runs as an isolated subprocess using the existing base Python `/root/miniconda3/bin/python` (matplotlib3.9.0), receiving only computed histogram bins. The NDP environment and its NumPy/PyTorch dependencies are unchanged. First smoke attempt stopped on missing matplotlib before full fitting; that failed run and its draft archive are retained.
