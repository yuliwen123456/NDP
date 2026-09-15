# V10 NDP + Prototype results

| Experiment | OOD Score | AUPRC (AP) | AUROC | FPR95 | mIoU* |
|---|---|---:|---:|---:|---:|
| Round6 Baseline | Original NDP | 74.553207 | 99.359957 | 1.447267 | 3.437569 |
| Diagnostic | Prototype Only | 0.665150 | 95.041522 | 18.084882 | 3.437569 |
| V10 | NDP + Prototype | 74.158468 | 99.427259 | 1.240967 | 3.437569 |

Delta V10−Round6 (percentage points): {"AP": -0.3947383645868001, "AUROC": 0.06730212851158512, "FPR95": -0.206299553331458, "mIoU": 0.0}

*mIoU is newly recomputed with the unchanged semantic branch and original PanopticEval on available STU labels. Round6 historical mIoU was not recorded. It is a supplementary diagnostic, not an official STU semantic benchmark. All methods retain identical semantic predictions.

C=19, D=128. ID train scene206/all898 frames. Classes6 bicyclist and9 parking have no training examples and are masked; no fabricated prototypes. Raw feature means then L2; cosine distance; ID-training z-score; alpha=0.5.
OOD used for prototype/normalization: NO. Retraining: NO. V08: DISABLED. V09: DISABLED.
Same Round6 checkpoint and STU public validation split; original metric filters and formulas. No alpha selection on evaluation results.
GPU0 RTX3090 only; batch1 FP32 frozen inference. Actual memory and runtime: inference_result.json.
Scores and nearest prototype attributes: prediction/<scene>/<frame>.npz. Normalization: normalization.json. Prototype bank: prototypes.pt.
Class coverage limitation: 17/19 observed classes in a single training scene; semantic-label limitations apply to mIoU.
