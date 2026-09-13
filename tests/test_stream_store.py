import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlgnn import stream_store


def _row(txn_id, ts, from_account, to_account, is_laundering=0):
    return {
        "txn_id": txn_id, "ts": ts, "from_bank": 1, "from_account": from_account,
        "to_bank": 1, "to_account": to_account, "amount_received": 10.0,
        "currency_received": "USD", "amount_paid": 10.0, "currency_paid": "USD",
        "payment_format": "ACH", "is_laundering": is_laundering,
    }


def test_insert_and_count(tmp_path):
    con = stream_store.connect(tmp_path / "test.duckdb")
    stream_store.ensure_schema(con, "HI-Small")
    assert stream_store.landed_count(con, "HI-Small") == 0

    rows = [
        _row(1, "2022-09-01 10:00:00", "A", "B"),
        _row(2, "2022-09-01 10:05:00", "B", "C"),
    ]
    stream_store.insert_batch(con, "HI-Small", rows)
    assert stream_store.landed_count(con, "HI-Small") == 2
    con.close()


def test_compact_preserves_row_count_and_content(tmp_path):
    con = stream_store.connect(tmp_path / "test.duckdb")
    stream_store.ensure_schema(con, "HI-Small")
    rows = [
        _row(3, "2022-09-01 10:10:00", "C", "D"),
        _row(1, "2022-09-01 10:00:00", "A", "B"),
        _row(2, "2022-09-01 10:05:00", "B", "C"),
    ]
    stream_store.insert_batch(con, "HI-Small", rows)
    stream_store.compact(con, "HI-Small")

    assert stream_store.landed_count(con, "HI-Small") == 3
    ordered_ids = con.execute(
        "SELECT txn_id FROM trans_hi_small ORDER BY txn_id"
    ).df()["txn_id"].tolist()
    assert ordered_ids == [1, 2, 3]

    # compacting is idempotent and does not duplicate or drop rows
    stream_store.compact(con, "HI-Small")
    assert stream_store.landed_count(con, "HI-Small") == 3
    con.close()


def test_landed_count_zero_when_table_absent(tmp_path):
    con = stream_store.connect(tmp_path / "test.duckdb")
    assert stream_store.landed_count(con, "HI-Small") == 0
    con.close()
