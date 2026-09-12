"""Redraw archived sensor means using one shared color scale; no model fitting."""

import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np

HERE = Path(__file__).resolve().parent


def import_means(workspace):
    source = workspace / 'benchmark/visual/plot_watch_recall_topomaps.py'
    tree = ast.parse(source.read_text())
    metadata = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ['get_electrodes', 'get_positions']:
            returned = next(n for n in node.body if isinstance(n, ast.Return))
            metadata[node.name] = ast.literal_eval(returned.value)
    names, positions = metadata['get_electrodes'], metadata['get_positions']
    means = {}
    for condition in ['watch', 'recall']:
        payload = json.loads((workspace / f'benchmark/results_buffer/result_human_{condition}_raw.json').read_text())
        values = np.asarray(payload[condition]['human']['conformer']['raw'], dtype=float)
        assert values.shape == (60, 62) and np.isfinite(values).all()
        means[condition] = values.mean(axis=0)
    with (HERE / 'sensor_means.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['channel', 'x', 'y', 'perception', 'recall'])
        for i, name in enumerate(names):
            writer.writerow([name, *positions[name], means['watch'][i], means['recall'][i]])


def plot():
    with (HERE / 'sensor_means.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    positions = np.array([[float(row['x']), float(row['y'])] for row in rows])
    values = [np.array([float(row[condition]) for row in rows]) for condition in ['perception', 'recall']]
    bounds = (min(v.min() for v in values), max(v.max() for v in values))
    radius = np.linalg.norm(positions, axis=1).max()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), layout='constrained')
    for ax, scores, label in zip(axes, values, ['Perception', 'Recall']):
        im, _ = mne.viz.plot_topomap(scores, positions, axes=ax, show=False,
                                    cmap='RdBu_r', contours=6, sensors=True,
                                    sphere=(0, 0, 0, radius), vlim=bounds)
        ax.set_title(label)
    fig.colorbar(im, ax=axes, shrink=0.78, label='Validation accuracy (%)')
    fig.suptitle('Human meta-feature: sensor-level validation scores')
    fig.savefig(HERE.parent / 'images/topomap_human.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('Shared color scale:', bounds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-workspace', type=Path)
    args = parser.parse_args()
    if args.import_workspace:
        import_means(args.import_workspace)
    plot()
