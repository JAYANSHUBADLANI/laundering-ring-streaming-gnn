"""Replay one split's transactions onto Kafka in timestamp order, at a
configurable rate, as if they were arriving live.

This is a simulation of arrival order from a static, already-collected
dataset. It is not a live feed, and nothing here talks to a real bank or a
managed broker: the topic lives in the local Kafka container started by
`infra/kafka/docker-compose.yml`.

Two segments exist. `test` streams the fixed evaluation window once, whole,
since every cycle scores against the same holdout. `train` streams a
quantile slice of the training window's time range, so that three separate
invocations with disjoint quantile ranges simulate three batches of new data
arriving between scheduled retraining cycles, with nothing sent twice and
nothing skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kafka import KafkaProducer

from amlgnn.config import KAFKA_BOOTSTRAP, KAFKA_TOPIC
from amlgnn.periods_source import load_periods
from amlgnn.source_repo import connect_source, import_aml


def build_clause(periods, segment: str, from_quantile: float, to_quantile: float) -> tuple[str, str]:
    aml = import_aml()
    table = aml.periods.table_for(periods.split)
    if segment == "test":
        return table, periods.test_clause
    if segment != "train":
        raise ValueError(f"unknown segment {segment!r}")

    con = connect_source(read_only=True)
    t_start, t_end = con.execute(
        f"SELECT min(ts), max(ts) FROM {table} WHERE {periods.train_clause}"
    ).fetchone()
    con.close()
    import datetime as dt

    span = (t_end - t_start).total_seconds()
    t_from = t_start + dt.timedelta(seconds=span * from_quantile)
    t_to = t_start + dt.timedelta(seconds=span * to_quantile) if to_quantile < 1.0 else periods.cut
    clause = f"{periods.train_clause} AND ts >= TIMESTAMP '{t_from}' AND ts < TIMESTAMP '{t_to}'"
    return table, clause


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=["HI-Small", "LI-Small"])
    ap.add_argument("--segment", required=True, choices=["train", "test"])
    ap.add_argument("--from-quantile", type=float, default=0.0)
    ap.add_argument("--to-quantile", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=0.0, help="messages/sec, 0 = unthrottled")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topic", default=KAFKA_TOPIC)
    ap.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    args = ap.parse_args()

    periods = load_periods(args.split)
    table, clause = build_clause(periods, args.segment, args.from_quantile, args.to_quantile)

    con = connect_source(read_only=True)
    query = f"""
        SELECT txn_id, ts, from_bank, from_account, to_bank, to_account,
               amount_received, currency_received, amount_paid, currency_paid,
               payment_format, is_laundering
        FROM {table}
        WHERE {clause}
        ORDER BY ts
    """
    if args.limit:
        query += f" LIMIT {args.limit}"
    frame = con.execute(query).df()
    con.close()
    print(f"replaying {len(frame)} transactions from {args.split}/{args.segment} "
          f"[{args.from_quantile}, {args.to_quantile}) onto '{args.topic}'")

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )

    delay = (1.0 / args.rate) if args.rate > 0 else 0.0
    t0 = time.perf_counter()
    for row in frame.itertuples(index=False):
        message = {
            "txn_id": int(row.txn_id),
            "ts": row.ts.isoformat(),
            "from_bank": int(row.from_bank),
            "from_account": str(row.from_account),
            "to_bank": int(row.to_bank),
            "to_account": str(row.to_account),
            "amount_received": float(row.amount_received),
            "currency_received": str(row.currency_received),
            "amount_paid": float(row.amount_paid),
            "currency_paid": str(row.currency_paid),
            "payment_format": str(row.payment_format),
            "is_laundering": int(row.is_laundering),
        }
        producer.send(args.topic, key=str(row.from_account), value=message)
        if delay:
            time.sleep(delay)

    producer.send(args.topic, key="__control__", value={
        "type": "END_OF_STREAM", "split": args.split, "segment": args.segment,
        "from_quantile": args.from_quantile, "to_quantile": args.to_quantile,
        "count": len(frame),
    })
    producer.flush()
    elapsed = time.perf_counter() - t0
    print(f"sent {len(frame)} transactions + sentinel in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
