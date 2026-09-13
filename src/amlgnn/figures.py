"""Ring recall by model variant, gradient boosting and the GNN side by side.

Same visual language as `laundering-ring-detection`'s own
`detection_by_variant.png`, since this figure sits next to that one, not in
a different register: identity carried by fill colour, split by panel,
bootstrap intervals drawn on every bar because the point of the chart is
whether a gap is bigger than the noise around it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d7d2"


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


ORDER = ["txn_without_format", "graph_without_format", "gnn_without_format"]
LABELS = ["transaction\nwithout format", "graph feature\nwithout format",
          "GNN\nwithout format"]
FILL = {"txn_without_format": "#eb6834", "graph_without_format": "#2a78d6",
        "gnn_without_format": "#3fa34d"}


def gnn_vs_baseline_figure(evaluation: pd.DataFrame, budget: float,
                           path: Path | None = None) -> Path:
    """Ring recall for the three honest (without payment_format) variants,
    one panel per split. Only the without-format variants, since those are
    the ones the source project says to believe."""
    frame = evaluation[evaluation["alert_budget"] == budget]
    splits = sorted(frame["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(9, 4.2), sharey=True)
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        rows = frame[frame["split"] == split].set_index("variant")
        values = [rows.loc[v, "ring_recall_at_least_1"] for v in ORDER]
        lo = [rows.loc[v, "ring_recall_at_least_1_lo"] for v in ORDER]
        hi = [rows.loc[v, "ring_recall_at_least_1_hi"] for v in ORDER]
        errors = [[v - l for v, l in zip(values, lo)],
                  [h - v for v, h in zip(values, hi)]]
        positions = range(len(ORDER))
        ax.bar(positions, values, color=[FILL[v] for v in ORDER], width=0.6)
        ax.errorbar(positions, values, yerr=errors, fmt="none",
                    ecolor=TEXT_SECONDARY, elinewidth=1.1, capsize=4)
        for x, value in zip(positions, values):
            ax.text(x, value + 0.015, f"{value:.3f}", ha="center",
                    fontsize=8, color=TEXT_PRIMARY)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_title(split, fontsize=10, color=TEXT_PRIMARY, pad=8)
        _style(ax)

    axes[0].set_ylabel("Ring recall, at least one transaction flagged",
                       fontsize=9, color=TEXT_SECONDARY)
    fig.suptitle(
        f"Does a graph neural network beat hand engineered graph features? "
        f"(alert budget {budget:.1%})",
        fontsize=11, color=TEXT_PRIMARY, y=0.99,
    )
    fig.tight_layout()
    path = path or FIGURES / "gnn_vs_baseline.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path
