"""One finite smoke→fit→infer→evaluate run. No retraining or scheduler."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=HERE/'config.json')
    p.add_argument('--smoke-only', action='store_true')
    a = p.parse_args(); root = a.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    cfg = a.config.resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1')
    jobs = [('unit', [sys.executable, '-B', str(HERE/'test_prototype.py')]),
            ('smoke-forward', [sys.executable, '-B', str(HERE/'run.py'), 'smoke']),
            ('smoke-metrics', [sys.executable, '-B', str(HERE/'evaluate.py'), '--smoke'])]
    if not a.smoke_only:
        jobs += [('prototype-fit', [sys.executable, '-B', str(HERE/'run.py'), 'fit']),
                 ('full-inference', [sys.executable, '-B', str(HERE/'run.py'), 'infer']),
                 ('full-evaluation', [sys.executable, '-B', str(HERE/'evaluate.py')])]
    for phase, command in jobs:
        if phase != 'unit':
            command += ['--config', str(cfg), '--output', str(root)]
        event = {'phase': phase, 'command': command, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        print(json.dumps(event), flush=True)
        with (root/(phase+'.log')).open('x') as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, cwd=HERE, env=env)
        event.update(exit_code=code, finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
        if code:
            return code
    with (root/'pipeline_complete.json').open('x') as f:
        json.dump({'completed': True, 'smoke_only': a.smoke_only, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
