"""Render precomputed bins using existing base-environment matplotlib only."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = json.loads(Path(sys.argv[1]).read_text())
destination = Path(sys.argv[2])
if destination.exists():
    raise FileExistsError(destination)
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, series in data['series'].items():
    ax.stairs(series['density'], series['edges'], label=name, linewidth=1.5)
ax.set(title=data['score']+' — unchanged STU eligible points',
       xlabel='OOD score (higher = anomalous)', ylabel='Density')
ax.legend(); fig.tight_layout(); fig.savefig(destination, dpi=180); plt.close(fig)
