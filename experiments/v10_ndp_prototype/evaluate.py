"""Lossless NPZ adapter to the unchanged Round6 STU metric implementation."""
import argparse
import hashlib
import importlib.util
import json
import sys
import subprocess
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'v06_baseline'
NAMES = ['car','bicycle','motorcycle','truck','other-vehicle','person','bicyclist',
         'motorcyclist','road','parking','sidewalk','other-ground','building','fence',
         'vegetation','trunk','terrain','pole','traffic-sign']


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def describe(values, ood):
    percentiles = [10, 50] if ood else [90, 95]
    result = {'count': len(values), 'mean': float(values.mean(dtype=np.float64)),
              'std': float(values.std(dtype=np.float64)), 'median': float(np.median(values))}
    for p, v in zip(percentiles, np.percentile(values, percentiles)):
        result['P'+str(p)] = float(v)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args(); cfg = json.loads(a.config.read_text()); root = a.output.resolve()
    folder = root/('smoke' if a.smoke else 'full')
    stu = Path(cfg['stu_root']); sys.path.insert(0, str(stu))
    official = module('round6_point_metric', stu/'compute_point_level_ood.py')
    # Assert the actual original metric is identical to the frozen Round6 snapshot.
    metric_sha = hashlib.sha256((stu/'compute_point_level_ood.py').read_bytes()).hexdigest()
    assert metric_sha == hashlib.sha256((BASE/'support_v06/eval.py').read_bytes()).hexdigest()
    sem_mod = module('round6_semantic_metric', BASE/'source_v06/models/metrics/panoptic_eval.py')
    semantic = sem_mod.PanopticEval(n_classes=19)
    import yaml
    mapping = yaml.safe_load((BASE/'source_v06/conf/mask4former3d.yaml').read_text())['learning_map']
    lut = np.zeros(65536, np.int32)
    for k, v in mapping.items():
        if 1 <= int(v) <= 19:
            lut[int(k)] = int(v)
    files = sorted((folder/'prediction').glob('*/*.npz'))
    required = cfg['smoke_frames'] if a.smoke else cfg['expected_evaluation_frames']
    if len(files) != required:
        raise ValueError('Missing or extra prediction files')
    stats = {}; metrics = {}; nearest_count = np.zeros(19, np.int64)
    start = time.time()
    for key, hist in [('S_ndp','ndp'), ('S_proto','prototype'), ('S_final','final')]:
        calculator = official.PointOODMetricsCalculator()
        for i, path in enumerate(files):
            data_path = Path(cfg['evaluation_data'])/path.parent.name
            points, _ = official.load_point_cloud(data_path/'velodyne'/(path.stem+'.bin'))
            labels, _ = official.load_labels(data_path/'labels'/(path.stem+'.label'))
            with np.load(path) as values:
                scores = values[key]
                before = len(calculator.all_scores)
                calculator.update(points, scores, labels)
                if key == 'S_ndp':
                    pred = values['semantic_prediction'].astype(np.int64)+1
                    semantic.addBatchSemIoU(pred, lut[labels.astype(np.int64)])
                    if len(calculator.all_scores) > before:
                        # Use a second call to the same original filter for aligned nearest classes.
                        probe = official.PointOODMetricsCalculator()
                        probe.update(points, values['nearest_proto_class'], labels)
                        kept = probe.all_scores[0][probe.all_labels[0] == 1]
                        nearest_count += np.bincount(kept.astype(np.int64), minlength=19)
            if i < 3 or (i+1) % 500 == 0:
                print('METRIC_LOAD', key, i+1, '/', len(files), 'elapsed', round(time.time()-start, 1), flush=True)
        print('METRIC_COMPUTE', key, 'eligible_frames', len(calculator.all_scores), flush=True)
        result = calculator.compute_metrics()
        if not result:
            raise ValueError('No eligible OOD evaluation frames')
        metrics[key] = {k: float(v) for k, v in result.items()}
        save(folder/(key+'_metrics.json'), metrics[key])
        print('METRIC_RESULT', key, json.dumps(metrics[key]), flush=True)
        scores = np.concatenate(calculator.all_scores); targets = np.concatenate(calculator.all_labels)
        calculator.all_scores.clear(); calculator.all_labels.clear()
        parts = {'ID': scores[targets == 0], 'OOD': scores[targets == 1]}
        del scores, targets
        stats[key] = {name: describe(v, name == 'OOD') for name, v in parts.items()}
        low = min(float(v.min()) for v in parts.values()); high = max(float(v.max()) for v in parts.values())
        bins = np.linspace(low, high, 151)
        histogram = {'score': key, 'series': {}}
        for name, v in parts.items():
            counts, edges = np.histogram(v, bins=bins, density=True)
            histogram['series'][name] = {'density': counts.tolist(), 'edges': edges.tolist()}
        hist_file = folder/(hist+'_histogram_bins.json')
        save(hist_file, histogram)
        subprocess.run([cfg['plotting_python'], '-B', str(HERE/'plot_histogram.py'), str(hist_file),
                        str(folder/(hist+'_score_histogram.png'))], check=True)
        del parts, calculator
    mean_iou, class_iou = semantic.getSemIoU()
    miou = float(mean_iou*100)
    semantic_result = {'mIoU': miou, 'per_class_IoU_percent': (class_iou[1:]*100).tolist(),
                      'class_names': NAMES, 'shared_identical_predictions_for_all_methods': True,
                      'historical_round6_mIoU': None,
                      'qualification': 'New supplementary measurement on available STU labels using original PanopticEval. Not a historical Round6 result or official STU semantic benchmark. 19-class mean; raw mapped non-ID labels ignored; no OOD score based semantic rejection.',
                      'confusion_matrix': semantic.px_iou_conf_matrix.tolist()}
    save(folder/'semantic_metrics.json', semantic_result)
    save(folder/'score_statistics.json', {'population': 'Original STU OOD calculator eligible frames/points', 'scores': stats})
    total = int(nearest_count.sum())
    save(folder/'nearest_prototype_statistics.json', {'ood_points': total, 'classes': [
        {'class_id': i, 'name': name, 'count': int(nearest_count[i]), 'percent': float(nearest_count[i]/total*100)} for i, name in enumerate(NAMES)]})
    historical = json.loads(Path(cfg['baseline_metrics']).read_text())
    if not a.smoke:
        for key in ['AP','AUROC','FPR95']:
            if not np.isclose(metrics['S_ndp'][key], historical[key], rtol=0, atol=1e-8):
                raise ValueError('Full Round6 metric mismatch: '+key)
    delta = {k: metrics['S_final'][k]-metrics['S_ndp'][k] for k in ['AP','AUROC','FPR95']}
    delta['mIoU'] = 0.
    save(folder/'evaluation_result.json', {'metrics': metrics, 'supplementary_mIoU': miou, 'delta': delta,
              'frames': len(files), 'metric_sha256': metric_sha, 'seconds': time.time()-start,
              'historical_round6_metrics_reproduced': not a.smoke, 'alpha': cfg['prototype']['alpha']})
    lines = ['# V10 NDP + Prototype results', '',
             '| Experiment | OOD Score | AUPRC (AP) | AUROC | FPR95 | mIoU* |',
             '|---|---|---:|---:|---:|---:|']
    for name, score, key in [('Round6 Baseline','Original NDP','S_ndp'), ('Diagnostic','Prototype Only','S_proto'), ('V10','NDP + Prototype','S_final')]:
        m = metrics[key]
        lines.append(f"| {name} | {score} | {m['AP']:.6f} | {m['AUROC']:.6f} | {m['FPR95']:.6f} | {miou:.6f} |")
    lines += ['', 'Delta V10−Round6 (percentage points): '+json.dumps(delta), '',
              '*mIoU is newly recomputed with the unchanged semantic branch and original PanopticEval on available STU labels. Round6 historical mIoU was not recorded. It is a supplementary diagnostic, not an official STU semantic benchmark. All methods retain identical semantic predictions.', '',
              'C=19, D=128. ID train scene206/all898 frames. Classes6 bicyclist and9 parking have no training examples and are masked; no fabricated prototypes. Raw feature means then L2; cosine distance; ID-training z-score; alpha='+str(cfg['prototype']['alpha'])+'.',
              'OOD used for prototype/normalization: NO. Retraining: NO. V08: DISABLED. V09: DISABLED.',
              'Same Round6 checkpoint and STU public validation split; original metric filters and formulas. No alpha selection on evaluation results.',
              'GPU0 RTX3090 only; batch1 FP32 frozen inference. Actual memory and runtime: inference_result.json.',
              'Scores and nearest prototype attributes: prediction/<scene>/<frame>.npz. Normalization: normalization.json. Prototype bank: prototypes.pt.',
              'Class coverage limitation: 17/19 observed classes in a single training scene; semantic-label limitations apply to mIoU.', '']
    if a.smoke:
        lines.insert(2, '**SMOKE ONLY — not full experiment results.**')
        save(folder/'smoke_result.json', {'passed': True, 'metric_pipeline_passed': True, 'frames': len(files)})
        print('V10 NDP + Prototype smoke test PASSED', flush=True)
    with (folder/'V10_RESULTS.md').open('x') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
