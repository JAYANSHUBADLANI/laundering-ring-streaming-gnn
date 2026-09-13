"""Question 5: run the orchestrated pipeline through three scheduled cycles
with new training data arriving between them, and see whether retraining on
the accumulated stream actually moves ring recall, in either direction, or
plateaus once the first cycle already saw most of the signal.

The fixed evaluation window (the same test/holdout split
`laundering-ring-detection` uses) is streamed once, whole, before the first
cycle, since every cycle has to be scored against the same holdout for the
recall numbers to be comparable across cycles. The training window is then
streamed in three disjoint time-ordered thirds, one per cycle, each landed
through the real Kafka broker with its own producer and consumer run before
that cycle's retrain, so this is the actual orchestrated pipeline running
three times, not the same numbers computed once and split into a table.

This resets and reuses the split's landed table and graph checkpoint rather
than the ones the full-scale streaming equivalence check and the phase 4
comparison run left behind, so the accumulation here starts from zero and
the earlier runs' results stay intact in their own `results/` files.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from amlgnn import stream_store
from amlgnn.config import KAFKA_BOOTSTRAP, RESULTS
from amlgnn.consumer import checkpoint_path, drain_topic
from amlgnn.pipeline import run_cycle

REPO_ROOT = Path(__file__).resolve().parents[1]


def reset_split(split: str) -> None:
    con = stream_store.connect()
    table = stream_store.table_for(split)
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.close()
    ckpt = checkpoint_path(split)
    if ckpt.exists():
        ckpt.unlink()
    print(f"[{split}] reset: dropped landed table, cleared graph checkpoint")


def produce(split: str, segment: str, topic: str, from_q: float = 0.0, to_q: float = 1.0) -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "produce_stream.py"),
        "--split", split, "--segment", segment, "--topic", topic,
        "--from-quantile", str(from_q), "--to-quantile", str(to_q),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="HI-Small", choices=["HI-Small", "LI-Small"])
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--conv", default="sage", choices=["sage", "gcn"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    split = args.split

    reset_split(split)

    test_topic = f"mc-test-{split.lower()}"
    produce(split, "test", test_topic)
    result = drain_topic(split, topic=test_topic, bootstrap=KAFKA_BOOTSTRAP,
                         group=f"mc-test-{split.lower()}")
    print(f"[{split}] test window landed: {result.landed_this_run} transactions")

    summary_rows = []
    stamp_prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for cycle in range(1, args.cycles + 1):
        from_q = (cycle - 1) / args.cycles
        to_q = cycle / args.cycles
        topic = f"mc-train-{split.lower()}-cycle{cycle}"
        print(f"\n=== cycle {cycle}/{args.cycles}: training data quantile "
              f"[{from_q:.2f}, {to_q:.2f}) ===")
        produce(split, "train", topic, from_q=from_q, to_q=to_q)
        drain = drain_topic(split, topic=topic, bootstrap=KAFKA_BOOTSTRAP,
                            group=f"mc-cycle{cycle}-{split.lower()}")
        print(f"[{split}] cycle {cycle} landed {drain.landed_this_run} new training "
              f"transactions ({drain.transactions_seen_total} total seen)")

        con = stream_store.connect()
        cycle_result = run_cycle(con, split, conv_type=args.conv, epochs=args.epochs,
                                 stamp=f"{stamp_prefix}_cycle{cycle}")
        con.close()

        primary = cycle_result.evaluation[cycle_result.evaluation["alert_budget"] == 0.001]
        for _, row in primary.iterrows():
            summary_rows.append({
                "cycle": cycle,
                "train_transactions_seen": drain.transactions_seen_total,
                "variant": row["variant"],
                "ring_recall_at_least_1": row["ring_recall_at_least_1"],
                "ring_recall_at_least_1_lo": row["ring_recall_at_least_1_lo"],
                "ring_recall_at_least_1_hi": row["ring_recall_at_least_1_hi"],
                "txn_recall": row["txn_recall"],
            })
            print(f"  {row['variant']:24s} ring_recall_at_least_1={row['ring_recall_at_least_1']:.4f} "
                  f"[{row['ring_recall_at_least_1_lo']:.4f}, {row['ring_recall_at_least_1_hi']:.4f}]")

    summary = pd.DataFrame(summary_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"multi_cycle_summary_{split}_{stamp_prefix}.csv"
    summary.to_csv(out, index=False)
    print(f"\nwrote {out}")

    pivot = summary.pivot(index="variant", columns="cycle", values="ring_recall_at_least_1")
    print("\nring_recall_at_least_1 by cycle:\n")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
