"""Summarize archived validation scores; never imports or trains a model."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
TASKS = ['20-c', '4-c', 'fast_slow', 'color', 'number', 'face', 'human']
CONTENT_MODELS = {
    'raw': ['shallownet', 'deepnet', 'eegnet', 'conformer', 'tsconv', 'glfnet', 'svm', 'GCN_LSTM'],
    'psd': ['mlpnet', 'glfnet_mlp'],
    'de': ['mlpnet', 'glfnet_mlp'],
}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_scores(workspace):
    names = sorted(p.name for p in (workspace / 'data/watch_cleaned').glob('*.npy'))
    assert len(names) == 60
    for folder in ['recall_cleaned', 'watch_PSD_DE', 'recall_PSD_DE']:
        other = sorted(p.name.replace('_PSD_DE', '') for p in (workspace / 'data' / folder).glob('*.npy'))
        assert names == other, folder
    participants = sorted({name.split('_')[0] for name in names})
    assert len(participants) == 20
    mapping, counts = [], defaultdict(int)
    for name in names:
        participant = name.split('_')[0]
        counts[participant] += 1
        mapping.append((f'sub-{participants.index(participant)+1:03d}', f'recording-{counts[participant]:02d}'))
    assert set(counts.values()) == {3}
    rows = []

    def add(values, source, key, task, model, feature, condition):
        values = np.asarray(values, dtype=float).reshape(-1)
        assert values.shape == (60,) and np.isfinite(values).all()
        assert ((values >= 0) & (values <= 100)).all()
        for index, (value, (participant, recording)) in enumerate(zip(values, mapping)):
            rows.append(dict(subject_id=participant, recording_id=recording, source_index=index,
                             task=task, model=model, feature=feature, condition=condition,
                             score_type='validation' if model == 'svm' else 'best_validation',
                             accuracy_percent=float(value), source_file=source, source_key=key))

    source = 'benchmark/benchmark_results.json'
    payload = json.loads((workspace / source).read_text())
    for condition in ['watch', 'recall']:
        for task in TASKS:
            for feature, models in CONTENT_MODELS.items():
                for model in models:
                    add(payload[condition][task][model][feature], source,
                        '/'.join([condition, task, model, feature]), task, model, feature, condition)
    for condition, source in [('full', 'biclassification/benchmark_results.json'),
                              ('without_alpha', 'biclassification/benchmark_results_alpha_removed.json')]:
        payload = json.loads((workspace / source).read_text())
        for feature, models in payload.items():
            for model, values in models.items():
                assert np.asarray(values).shape == (1, 60)
                add(values, source, f'{feature}/{model}', 'state', model, feature, condition)
    write_csv(HERE / 'session_scores.csv', rows)


def holm(rows):
    order = np.argsort([row['p_uncorrected'] for row in rows])
    adjusted = np.maximum.accumulate([
        (len(rows)-rank)*rows[index]['p_uncorrected'] for rank, index in enumerate(order)
    ])
    for index, value in zip(order, adjusted):
        rows[index]['p_holm'] = float(min(1, value))
        rows[index]['family_size'] = len(rows)


def summarize():
    with (HERE / 'session_scores.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ['task', 'model', 'feature', 'condition', 'subject_id'])
        assert row['recording_id'] not in grouped[key]
        grouped[key][row['recording_id']] = float(row['accuracy_percent'])
    subject_rows, population = [], defaultdict(dict)
    for key, sessions in sorted(grouped.items()):
        assert len(sessions) == 3
        value = float(np.mean(list(sessions.values())))
        task, model, feature, condition, subject = key
        subject_rows.append(dict(task=task, model=model, feature=feature, condition=condition,
                                 subject_id=subject, sessions=3, accuracy_percent=value))
        population[key[:4]][subject] = value
    summaries = []
    for key, subjects in sorted(population.items()):
        assert len(subjects) == 20
        values = np.array(list(subjects.values()))
        summaries.append(dict(task=key[0], model=key[1], feature=key[2], condition=key[3],
                              n_subjects=20, mean=float(values.mean()), sd=float(values.std(ddof=1))))

    comparisons = []
    combinations = sorted({key[:3] for key in population})
    for task, model, feature in combinations:
        first, second = ('full', 'without_alpha') if task == 'state' else ('watch', 'recall')
        left, right = population[(task, model, feature, first)], population[(task, model, feature, second)]
        assert left.keys() == right.keys()
        difference = np.array([left[s]-right[s] for s in sorted(left)])
        assert difference.std(ddof=1) > 0, 'Degenerate paired difference; review before testing.'
        result = stats.ttest_1samp(difference, 0)
        mean, se = difference.mean(), stats.sem(difference)
        margin = stats.t.ppf(0.975, len(difference)-1) * se
        comparisons.append(dict(task=task, model=model, feature=feature,
                                contrast=f'{first}-minus-{second}', n_subjects=20,
                                difference_pp=float(mean), ci95_low=float(mean-margin),
                                ci95_high=float(mean+margin), t=float(result.statistic), df=19,
                                p_uncorrected=float(result.pvalue), p_holm=None,
                                family='state_alpha' if task == 'state' else 'content', family_size=None))
    for family in ['content', 'state_alpha']:
        subset = [row for row in comparisons if row['family'] == family]
        holm(subset)
    write_csv(HERE / 'subject_scores.csv', subject_rows)
    write_csv(HERE / 'summary.csv', summaries)
    write_csv(HERE / 'paired_comparisons.csv', comparisons)
    report = dict(n_subjects=20, sessions_per_subject=3, session_rows=len(rows),
                  content_comparisons=sum(row['family']=='content' for row in comparisons),
                  state_comparisons=sum(row['family']=='state_alpha' for row in comparisons),
                  significant_content=[row for row in comparisons if row['family']=='content' and row['p_holm']<0.05],
                  significant_alpha=[row for row in comparisons if row['family']=='state_alpha' and row['p_holm']<0.05])
    (HERE / 'analysis_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return summaries, report


def plot_content(summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lookup = {(r['task'], r['model'], r['feature'], r['condition']): r for r in summaries}
    models = [('deepnet', 'DeepNet'), ('conformer', 'Conformer'), ('GCN_LSTM', 'GCN-LSTM')]
    colors = {'watch': '#567bbb', 'recall': '#d4875b'}
    labels = ['20-c', '4-c', 'F/S', 'Color', 'Num.', 'Face', 'Human']
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    x = np.arange(len(TASKS))
    for ax, (model, label) in zip(axes, models):
        for condition, offset, caption in [('watch', -0.19, 'Perception'), ('recall', 0.19, 'Recall')]:
            values = [lookup[(task, model, 'raw', condition)] for task in TASKS]
            ax.bar(x+offset, [v['mean'] for v in values], width=0.37,
                   yerr=[v['sd'] for v in values], capsize=2, color=colors[condition], label=caption)
        ax.set_title(label)
        ax.set_xticks(x, labels)
        ax.set_xlabel('Task')
        ax.set_ylim(0, 75)
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Best validation accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE.parent / 'images/PerceptionVsRecall.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-workspace', type=Path,
                        help='Import archived JSON and sorted recording filenames once; writes anonymous scores.')
    args = parser.parse_args()
    if args.import_workspace:
        import_scores(args.import_workspace)
    summaries, report = summarize()
    plot_content(summaries)
    print(json.dumps(report, indent=2))
