import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlgnn.graph_build import batch_structure


def test_self_loops_excluded_from_edges():
    frame = pd.DataFrame({
        "from_account": ["A", "B", "C"],
        "to_account": ["B", "B", "C"],  # B->B and C->C are self loops
    })
    structure = batch_structure(frame)
    assert structure.accounts == frozenset({"A", "B", "C"})
    assert structure.edges == frozenset({("A", "B")})


def test_duplicate_edges_deduplicate():
    frame = pd.DataFrame({
        "from_account": ["A", "A", "A"],
        "to_account": ["B", "B", "B"],
    })
    structure = batch_structure(frame)
    assert structure.edges == frozenset({("A", "B")})


def test_direction_matters():
    frame = pd.DataFrame({
        "from_account": ["A", "B"],
        "to_account": ["B", "A"],
    })
    structure = batch_structure(frame)
    assert structure.edges == frozenset({("A", "B"), ("B", "A")})


def test_empty_frame():
    frame = pd.DataFrame({"from_account": [], "to_account": []})
    structure = batch_structure(frame)
    assert structure.accounts == frozenset()
    assert structure.edges == frozenset()
