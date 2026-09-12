"""Summarize archived validation scores; never imports or trains a model."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

TASKS = ["20-c", "4-c", "fast_slow", "color", "number", "face", "human"]
CONTENT_MODELS = {
    "raw": ["shallownet", "deepnet", "eegnet", "conformer", "tsconv", "glfnet", "svm", "GCN_LSTM"],
    "psd": ["mlpnet", "glfnet_mlp"],
    "de": ["mlpnet", "glfnet_mlp"],
}


def write_csv(path: Path, rows: list[dict]):
    """Write ordered rows using their field names as the CSV header."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm(rows: list[dict]):
    """Add Holm-adjusted p-values in place for one declared comparison family."""
    order = np.argsort([row["p_uncorrected"] for row in rows])
    adjusted = np.maximum.accumulate(
        [(len(rows) - rank) * rows[index]["p_uncorrected"] for rank, index in enumerate(order)]
    )
    for index, value in zip(order, adjusted):
        rows[index]["p_holm"] = float(min(1, value))
        rows[index]["family_size"] = len(rows)


def summarize(input_dir: Path, output_dir: Path):
    """Aggregate archived sessions and write exploratory participant statistics.

    Args:
        input_dir: Directory containing the frozen session_scores.csv.
        output_dir: Destination for summaries; source scores are never overwritten.

    Returns:
        Summary rows and the full exploratory comparison report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with (input_dir / "session_scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ["task", "model", "feature", "condition", "subject_id"])
        assert row["recording_id"] not in grouped[key]
        grouped[key][row["recording_id"]] = float(row["accuracy_percent"])
    subject_rows, population = [], defaultdict(dict)
    for key, sessions in sorted(grouped.items()):
        assert len(sessions) == 3
        value = float(np.mean(list(sessions.values())))
        task, model, feature, condition, subject = key
        subject_rows.append(
            dict(
                task=task,
                model=model,
                feature=feature,
                condition=condition,
                subject_id=subject,
                sessions=3,
                accuracy_percent=value,
            )
        )
        population[key[:4]][subject] = value
    summaries = []
    for key, subjects in sorted(population.items()):
        assert len(subjects) == 20
        values = np.array(list(subjects.values()))
        summaries.append(
            dict(
                task=key[0],
                model=key[1],
                feature=key[2],
                condition=key[3],
                n_subjects=20,
                mean=float(values.mean()),
                sd=float(values.std(ddof=1)),
            )
        )

    comparisons = []
    combinations = sorted({key[:3] for key in population})
    for task, model, feature in combinations:
        first, second = ("full", "without_alpha") if task == "state" else ("watch", "recall")
        left, right = (
            population[(task, model, feature, first)],
            population[(task, model, feature, second)],
        )
        assert left.keys() == right.keys()
        difference = np.array([left[s] - right[s] for s in sorted(left)])
        assert difference.std(ddof=1) > 0, "Degenerate paired difference; review before testing."
        result = stats.ttest_1samp(difference, 0)
        mean, se = difference.mean(), stats.sem(difference)
        margin = stats.t.ppf(0.975, len(difference) - 1) * se
        comparisons.append(
            dict(
                task=task,
                model=model,
                feature=feature,
                contrast=f"{first}-minus-{second}",
                n_subjects=20,
                difference_pp=float(mean),
                ci95_low=float(mean - margin),
                ci95_high=float(mean + margin),
                t=float(result.statistic),
                df=19,
                p_uncorrected=float(result.pvalue),
                p_holm=None,
                family="state_alpha" if task == "state" else "content",
                family_size=None,
            )
        )
    for family in ["content", "state_alpha"]:
        subset = [row for row in comparisons if row["family"] == family]
        holm(subset)
    write_csv(output_dir / "subject_scores.csv", subject_rows)
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "paired_comparisons.csv", comparisons)
    report = dict(
        n_subjects=20,
        sessions_per_subject=3,
        session_rows=len(rows),
        content_comparisons=sum(row["family"] == "content" for row in comparisons),
        state_comparisons=sum(row["family"] == "state_alpha" for row in comparisons),
        significant_content=[
            row for row in comparisons if row["family"] == "content" and row["p_holm"] < 0.05
        ],
        significant_alpha=[
            row for row in comparisons if row["family"] == "state_alpha" and row["p_holm"] < 0.05
        ],
    )
    (output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return summaries, report


def plot_content(summaries: list[dict], output_path: Path):
    """Plot the archived content means with participant-level SD error bars.

    Args:
        summaries: Rows returned by summarize.
        output_path: Destination PNG file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(r["task"], r["model"], r["feature"], r["condition"]): r for r in summaries}
    models = [("deepnet", "DeepNet"), ("conformer", "Conformer"), ("GCN_LSTM", "GCN-LSTM")]
    colors = {"watch": "#567bbb", "recall": "#d4875b"}
    labels = ["20-c", "4-c", "F/S", "Color", "Num.", "Face", "Human"]
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    x = np.arange(len(TASKS))
    for ax, (model, label) in zip(axes, models):
        for condition, offset, caption in [
            ("watch", -0.19, "Perception"),
            ("recall", 0.19, "Recall"),
        ]:
            values = [lookup[(task, model, "raw", condition)] for task in TASKS]
            ax.bar(
                x + offset,
                [v["mean"] for v in values],
                width=0.37,
                yerr=[v["sd"] for v in values],
                capsize=2,
                color=colors[condition],
                label=caption,
            )
        ax.set_title(label)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Task")
        ax.set_ylim(0, 75)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Best validation accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
