"""Where consumed transactions land: this project's own DuckDB file, built up
incrementally by the Kafka consumer instead of loaded from a CSV in one shot.

The table it creates, `trans_<split>`, has the exact column set and name the
source project's own `trans_<split>` table has, so `aml.graphfeat.build`,
`aml.baseline.load_frame` and `aml.evaluate.ring_membership` all run against
it completely unmodified: none of the existing repository's feature or
evaluation code needed to change to accept a table that arrived a row at a
time instead of a bulk load.

Ring labels are not re-derived here. The pattern parser, the composite key
join, and the rings it names are the label side of the existing project and
this one reuses that table by copying it once rather than reimplementing the
parser against a stream that was never going to carry ring identifiers
transaction by transaction in the first place.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import DATA_INTERIM
from .source_repo import source_duckdb_path

STREAM_DB_PATH = DATA_INTERIM / "stream.duckdb"

TRANS_COLUMNS = (
    "txn_id", "ts", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "currency_received", "amount_paid", "currency_paid",
    "payment_format", "is_laundering",
)


def table_for(split: str) -> str:
    return f"trans_{split.replace('-', '_').lower()}"


def patterns_table_for(split: str) -> str:
    return f"patterns_{split.replace('-', '_').lower()}"


def connect(path: Path = STREAM_DB_PATH) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def ensure_schema(con: duckdb.DuckDBPyConnection, split: str) -> None:
    table = table_for(split)
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            txn_id BIGINT,
            ts TIMESTAMP,
            from_bank BIGINT,
            from_account VARCHAR,
            to_bank BIGINT,
            to_account VARCHAR,
            amount_received DOUBLE,
            currency_received VARCHAR,
            amount_paid DOUBLE,
            currency_paid VARCHAR,
            payment_format VARCHAR,
            is_laundering TINYINT
        )
        """
    )


def import_patterns(con: duckdb.DuckDBPyConnection, split: str) -> int:
    """Copy the verified ring pattern table in from the source database, once.

    Returns the row count copied, or the existing row count if it was already
    present, so callers can log whether this actually did anything.
    """
    table = patterns_table_for(split)
    existing = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0]
    if existing:
        return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    source_path = source_duckdb_path()
    con.execute(f"ATTACH '{source_path.as_posix()}' AS source_db (READ_ONLY)")
    try:
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM source_db.{table}")
    finally:
        con.execute("DETACH source_db")
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def insert_batch(con: duckdb.DuckDBPyConnection, split: str, rows: list[dict]) -> None:
    if not rows:
        return
    table = table_for(split)
    frame = pd.DataFrame(rows, columns=TRANS_COLUMNS)
    frame["ts"] = pd.to_datetime(frame["ts"])
    con.register("batch_frame", frame)
    con.execute(f"INSERT INTO {table} SELECT * FROM batch_frame")
    con.unregister("batch_frame")


def compact(con: duckdb.DuckDBPyConnection, split: str) -> None:
    """Rewrite the landed table as one physically ordered segment.

    Streaming lands data through thousands of small appended batches, one
    per consumer flush, rather than the source project's single bulk load
    from a CSV. A table built that way turned out not to give a stable scan
    order across separate connections: two otherwise identical retraining
    runs on identical data produced different gradient boosting numbers,
    traced to that unordered scan feeding the fit in a different row order
    each time. Sorting and rewriting into one segment before training is
    fitted removes the row order as a variable, the same way a periodic
    compaction job would in a real warehouse fed by a stream.
    """
    table = table_for(split)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM {table} ORDER BY ts, txn_id")


def landed_count(con: duckdb.DuckDBPyConnection, split: str) -> int:
    table = table_for(split)
    existing = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0]
    if not existing:
        return 0
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
