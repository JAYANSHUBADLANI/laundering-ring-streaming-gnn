"""Does the graph the streaming consumer builds match the one the batch
query would have built over the identical transactions?

This is checked directly rather than assumed. `batch_reference` runs one SQL
query over the source database and dedupes account pairs in pandas, the
ordinary batch path. The incremental side is whatever
`amlgnn.incremental_graph.IncrementalGraphState` has accumulated after
draining the real Kafka topic that `scripts/produce_stream.py` and
`scripts/consume_stream.py` were run against, a completely separate code
path that touches the broker and rebuilds the graph message by message. If
either had a bug, an off by one in the quantile boundary, an account id
string versus integer mismatch, a dropped or double counted message, the two
account and edge sets would not match, and this reports exactly where.

Run `produce_stream.py` and `consume_stream.py` for the split under test
before this script; it only compares, it does not stream anything itself.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlgnn.config import RESULTS
from amlgnn.graph_build import batch_structure
from amlgnn.incremental_graph import IncrementalGraphState
from amlgnn.periods_source import load_periods
from amlgnn.source_repo import connect_source, import_aml


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=["HI-Small", "LI-Small"])
    ap.add_argument("--segment", required=True, choices=["train", "test"])
    args = ap.parse_args()

    aml = import_aml()
    periods = load_periods(args.split)
    table = aml.periods.table_for(args.split)
    clause = periods.train_clause if args.segment == "train" else periods.test_clause

    print(f"batch reference query: {table} WHERE {clause}")
    t0 = time.perf_counter()
    con = connect_source(read_only=True)
    frame = con.execute(
        f"SELECT from_account, to_account FROM {table} WHERE {clause}"
    ).df()
    con.close()
    batch = batch_structure(frame)
    print(f"batch reference: {len(batch.accounts):,} accounts, {len(batch.edges):,} edges, "
          f"{len(frame):,} transactions, built in {time.perf_counter()-t0:.1f}s")

    checkpoint = Path(__file__).resolve().parents[1] / "data" / "interim" / "stream_state" / f"{args.split}_graph_state.json"
    incremental_state = IncrementalGraphState.load(checkpoint)
    incremental = incremental_state.structure()
    print(f"incremental (streamed) state: {len(incremental.accounts):,} accounts, "
          f"{len(incremental.edges):,} edges, {incremental_state.transactions_seen:,} transactions seen")

    match = batch == incremental
    diff = {} if match else batch.diff(incremental)

    result = {
        "split": args.split,
        "segment": args.segment,
        "batch_transactions": int(len(frame)),
        "batch_accounts": len(batch.accounts),
        "batch_edges": len(batch.edges),
        "streamed_transactions": incremental_state.transactions_seen,
        "streamed_accounts": len(incremental.accounts),
        "streamed_edges": len(incremental.edges),
        "accounts_match": batch.accounts == incremental.accounts,
        "edges_match": batch.edges == incremental.edges,
        "exact_match": match,
        "accounts_only_in_batch": len(diff.get("accounts_only_in_self", [])) if diff else 0,
        "accounts_only_in_streamed": len(diff.get("accounts_only_in_other", [])) if diff else 0,
        "edges_only_in_batch": len(diff.get("edges_only_in_self", [])) if diff else 0,
        "edges_only_in_streamed": len(diff.get("edges_only_in_other", [])) if diff else 0,
    }
    print(json.dumps(result, indent=2))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"stream_equivalence_{args.split}_{args.segment}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")

    if not match:
        print("MISMATCH: the streamed graph does not equal the batch graph.")
        return 1
    print("MATCH: the streamed graph is exactly the batch graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
