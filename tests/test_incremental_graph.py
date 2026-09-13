import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlgnn.graph_build import batch_structure
from amlgnn.incremental_graph import IncrementalGraphState
import pandas as pd


def test_incremental_matches_batch_on_synthetic_transactions():
    transactions = [
        ("A", "B"), ("B", "C"), ("A", "B"), ("C", "A"), ("D", "D"), ("B", "A"),
    ]
    state = IncrementalGraphState()
    for frm, to in transactions:
        state.add_transaction(frm, to)

    frame = pd.DataFrame(transactions, columns=["from_account", "to_account"])
    batch = batch_structure(frame)

    assert state.structure() == batch
    assert state.transactions_seen == len(transactions)


def test_checkpoint_round_trip(tmp_path):
    state = IncrementalGraphState()
    for frm, to in [("A", "B"), ("B", "C")]:
        state.add_transaction(frm, to)

    path = tmp_path / "checkpoint.json"
    state.save(path)

    restored = IncrementalGraphState.load(path)
    assert restored.structure() == state.structure()
    assert restored.transactions_seen == state.transactions_seen


def test_load_missing_checkpoint_is_empty(tmp_path):
    state = IncrementalGraphState.load(tmp_path / "does_not_exist.json")
    assert state.node_count() == 0
    assert state.edge_count() == 0
    assert state.transactions_seen == 0


def test_resuming_from_checkpoint_accumulates(tmp_path):
    path = tmp_path / "checkpoint.json"
    first = IncrementalGraphState()
    first.add_transaction("A", "B")
    first.save(path)

    second = IncrementalGraphState.load(path)
    second.add_transaction("B", "C")

    assert second.structure().accounts == frozenset({"A", "B", "C"})
    assert second.structure().edges == frozenset({("A", "B"), ("B", "C")})
    assert second.transactions_seen == 2
