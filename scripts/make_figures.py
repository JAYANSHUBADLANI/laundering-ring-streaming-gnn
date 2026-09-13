"""Render the GNN vs. baseline figure from the committed evaluation results.

Reads the same CSVs `README.md` cites rather than recomputing anything, so
the figure and the numbers in the text can never quietly drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from amlgnn.config import RESULTS
from amlgnn.figures import gnn_vs_baseline_figure


def main() -> int:
    hi = pd.read_csv(RESULTS / "gnn_evaluation_20260912T214134Z.csv")
    hi = hi[hi["variant"].isin(
        ["txn_without_format", "graph_without_format", "gnn_without_format"])]

    li_gbm = pd.read_csv(RESULTS / "cycle_LI-Small_20260912T235012Z_gbm_evaluation.csv")
    li_gnn = pd.read_csv(
        RESULTS / "cycle_LI-Small_20260913T001852Z_gnn_without_format_evaluation.csv")
    li = pd.concat([
        li_gbm[li_gbm["variant"].isin(["txn_without_format", "graph_without_format"])],
        li_gnn[li_gnn["variant"] == "gnn_without_format"],
    ], ignore_index=True)

    combined = pd.concat([hi, li], ignore_index=True)
    path = gnn_vs_baseline_figure(combined, budget=0.001)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
