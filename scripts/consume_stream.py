"""CLI entry point over `amlgnn.consumer.drain_topic`.

See that module's docstring: this exists so a manual run and the
orchestration op that does the same job share one implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlgnn.config import KAFKA_BOOTSTRAP, KAFKA_TOPIC
from amlgnn.consumer import drain_topic


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=["HI-Small", "LI-Small"])
    ap.add_argument("--topic", default=KAFKA_TOPIC)
    ap.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    ap.add_argument("--group", default=None)
    ap.add_argument("--idle-timeout-ms", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=20_000)
    args = ap.parse_args()

    result = drain_topic(
        split=args.split, topic=args.topic, bootstrap=args.bootstrap,
        group=args.group, idle_timeout_ms=args.idle_timeout_ms,
        batch_size=args.batch_size,
    )
    print(f"{'ended cleanly on sentinel' if result.ended_on_sentinel else 'stopped on idle timeout, no sentinel seen'}")
    print(f"consumed {result.landed_this_run} transactions this run in {result.seconds:.1f}s")
    print(f"graph state now: {result.node_count} accounts, {result.edge_count} edges, "
          f"{result.transactions_seen_total} transactions total")
    return 0 if result.ended_on_sentinel else 1


if __name__ == "__main__":
    raise SystemExit(main())
