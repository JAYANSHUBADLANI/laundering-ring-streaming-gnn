"""The consuming side of the replay: drain whatever is new on the Kafka
topic, landing it and folding it into the incremental graph state.

This is the code both `scripts/consume_stream.py` and the orchestration
op that pulls new data call, so a Dagster run and a manual run behave
identically rather than being two implementations that happen to agree.
"""

from __future__ import annotations

import json as jsonlib
import time
from dataclasses import dataclass
from pathlib import Path

from kafka import KafkaConsumer

from . import stream_store
from .config import KAFKA_BOOTSTRAP, KAFKA_TOPIC, STREAM_STATE_DIR
from .incremental_graph import IncrementalGraphState


def checkpoint_path(split: str) -> Path:
    return STREAM_STATE_DIR / f"{split}_graph_state.json"


@dataclass
class DrainResult:
    landed_this_run: int
    ended_on_sentinel: bool
    node_count: int
    edge_count: int
    transactions_seen_total: int
    seconds: float


def drain_topic(split: str, topic: str = KAFKA_TOPIC, bootstrap: str = KAFKA_BOOTSTRAP,
                group: str | None = None, idle_timeout_ms: int = 30_000,
                batch_size: int = 20_000) -> DrainResult:
    group = group or f"amlgnn-consumer-{split}"

    con = stream_store.connect()
    stream_store.ensure_schema(con, split)
    stream_store.import_patterns(con, split)

    state = IncrementalGraphState.load(checkpoint_path(split))

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=idle_timeout_ms,
        value_deserializer=lambda v: jsonlib.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k is not None else None,
    )

    buffer: list[dict] = []
    landed = 0
    t0 = time.perf_counter()
    ended = False

    def flush():
        nonlocal buffer
        if buffer:
            stream_store.insert_batch(con, split, buffer)
            buffer = []

    try:
        for message in consumer:
            value = message.value
            if message.key == "__control__" and value.get("type") == "END_OF_STREAM":
                ended = True
                consumer.commit()
                break

            state.add_transaction(value["from_account"], value["to_account"])
            buffer.append({
                "txn_id": value["txn_id"], "ts": value["ts"],
                "from_bank": value["from_bank"], "from_account": value["from_account"],
                "to_bank": value["to_bank"], "to_account": value["to_account"],
                "amount_received": value["amount_received"],
                "currency_received": value["currency_received"],
                "amount_paid": value["amount_paid"],
                "currency_paid": value["currency_paid"],
                "payment_format": value["payment_format"],
                "is_laundering": value["is_laundering"],
            })
            landed += 1
            if len(buffer) >= batch_size:
                flush()
                consumer.commit()
    finally:
        flush()
        consumer.commit()
        consumer.close()
        state.save(checkpoint_path(split))
        con.close()

    return DrainResult(
        landed_this_run=landed, ended_on_sentinel=ended,
        node_count=state.node_count(), edge_count=state.edge_count(),
        transactions_seen_total=state.transactions_seen,
        seconds=time.perf_counter() - t0,
    )
