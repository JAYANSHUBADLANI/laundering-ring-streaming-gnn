import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlgnn.model import UNSEEN_INDEX, Vocabulary, numeric_features


def test_vocabulary_assigns_distinct_indices_and_reserves_zero():
    vocab = Vocabulary(pd.Series(["ACH", "Wire", "ACH", "Cash"]))
    assert vocab.size == 4  # 3 uniques + reserved unseen slot
    encoded = vocab.encode(pd.Series(["ACH", "Wire"]))
    assert encoded[0] != UNSEEN_INDEX
    assert encoded[0] != encoded[1]


def test_vocabulary_maps_unseen_category_to_reserved_index():
    vocab = Vocabulary(pd.Series(["ACH", "Wire"]))
    encoded = vocab.encode(pd.Series(["Bitcoin"]))  # never seen at fit time
    assert encoded[0] == UNSEEN_INDEX


def test_numeric_features_shape_and_self_loop_flag():
    frame = pd.DataFrame({
        "amount_paid": [100.0, 250.0],
        "amount_received": [100.0, 250.0],
        "currency_paid": ["USD", "USD"],
        "currency_received": ["USD", "EUR"],
        "from_bank": [1, 1],
        "to_bank": [1, 2],
        "from_account": ["A", "A"],
        "to_account": ["A", "B"],
        "ts": pd.to_datetime(["2022-09-01 10:00", "2022-09-02 11:00"]),
    })
    features = numeric_features(frame)
    assert features.shape == (2, 10)
    self_loop_col = 5
    assert features[0, self_loop_col] == 1.0  # A -> A
    assert features[1, self_loop_col] == 0.0  # A -> B
    is_fx_col = 3
    assert features[0, is_fx_col] == 0.0  # USD -> USD
    assert features[1, is_fx_col] == 1.0  # USD -> EUR


def test_numeric_features_round_amount_flags():
    frame = pd.DataFrame({
        "amount_paid": [100.0, 137.42],
        "amount_received": [100.0, 137.42],
        "currency_paid": ["USD", "USD"],
        "currency_received": ["USD", "USD"],
        "from_bank": [1, 1],
        "to_bank": [1, 1],
        "from_account": ["A", "A"],
        "to_account": ["B", "B"],
        "ts": pd.to_datetime(["2022-09-01 10:00", "2022-09-01 10:00"]),
    })
    features = numeric_features(frame)
    round_100_col = 8
    assert features[0, round_100_col] == 1.0
    assert features[1, round_100_col] == 0.0
