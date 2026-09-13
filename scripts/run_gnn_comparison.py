"""CLI entry point over `amlgnn.pipeline.run_cycle`.

Fits the GNN on this project's landed (streamed) transactions, retrains the
existing gradient boosting baselines on the same landed data, and scores
every variant against the identical evaluation rules. See
`amlgnn/pipeline.py` for why the baselines are refit here rather than read
from the source repository's committed numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from amlgnn import stream_store
from amlgnn.pipeline import run_cycle


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["HI-Small", "LI-Small"])
    ap.add_argument("--conv", default="sage", choices=["sage", "gcn"])
    ap.add_argument("--variants", nargs="+",
                    default=["gbm", "gnn_with_format", "gnn_without_format"],
                    choices=["gbm", "gnn_with_format", "gnn_without_format"],
                    help="restrict to specific variants, e.g. to retry just one "
                         "GNN variant without refitting the others")
    args = ap.parse_args()

    con = stream_store.connect()
    results = []
    for split in args.splits:
        landed = stream_store.landed_count(con, split)
        print(f"[{split}] landed transactions available: {landed:,}")
        result = run_cycle(con, split, conv_type=args.conv, variants=tuple(args.variants))
        print(f"[{split}] wrote {result.eval_path}")
        results.append(result.evaluation)
    con.close()

    combined = pd.concat(results, ignore_index=True)
    show = ["split", "variant", "alert_budget", "alerts", "txn_recall",
            "ring_recall_at_least_1", "ring_recall_at_least_2", "rings_evaluable"]
    primary = combined[combined["alert_budget"] == 0.001]
    print("\n" + primary[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
