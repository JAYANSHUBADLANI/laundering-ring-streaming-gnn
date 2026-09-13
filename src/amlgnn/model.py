"""Phase 4: a graph neural network trained end to end on the account graph,
scored the same way `laundering-ring-detection`'s graph feature model is.

The claim under test is that message passing learns what the hand engineered
graph features (degree, reciprocity, component size, and the rest) were
built to approximate, without anyone having to design them. It is checked by
producing a `Scored` object with the exact same shape
`aml.baseline.fit_and_score` does, so `aml.evaluate.evaluate` and
`aml.evaluate.per_typology` run against it completely unmodified: the GNN's
numbers land in the same table as the transaction and graph feature
baselines, not a separate one with a note explaining why it cannot be
compared directly.

Leakage is avoided the same way the existing graph feature model avoids it.
The node vocabulary and the message passing structure are both built from
the training window only. An account never seen in training does not get a
zero vector, since "no history" is different information from "history of
nothing": it gets a single shared, learned "unseen account" embedding, the
neural equivalent of the null the existing project's gradient booster
receives for the same account.

Class imbalance is handled differently than the gradient booster, which
relies on small leaves to isolate a label that is one in a thousand. A
weighted binary cross entropy loss plays the equivalent role here, up-
weighting the rare positive class rather than restructuring how leaves split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

UNSEEN_INDEX = 0
UNK_CATEGORY = "__unk__"


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class Scored:
    """Matches `aml.baseline.Scored` field for field so the existing
    evaluation code accepts either without knowing which model produced it."""

    split: str
    variant: str
    txn_id: np.ndarray
    score: np.ndarray
    label: np.ndarray
    train_rows: int
    test_rows: int


class Vocabulary:
    """A training-only category-to-index map with a reserved unseen slot."""

    def __init__(self, values: pd.Series) -> None:
        uniques = sorted(values.dropna().unique().tolist())
        self.index = {v: i + 1 for i, v in enumerate(uniques)}  # 0 reserved
        self.size = len(uniques) + 1

    def encode(self, values: pd.Series) -> np.ndarray:
        return values.map(self.index).fillna(0).to_numpy(dtype=np.int64)


def numeric_features(frame: pd.DataFrame) -> np.ndarray:
    log_amt_paid = np.log1p(frame["amount_paid"].to_numpy(dtype=np.float64))
    log_amt_recv = np.log1p(frame["amount_received"].to_numpy(dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            frame["amount_paid"].to_numpy(dtype=np.float64) > 0,
            frame["amount_received"].to_numpy(dtype=np.float64)
            / frame["amount_paid"].to_numpy(dtype=np.float64),
            0.0,
        )
    is_fx = (frame["currency_paid"] != frame["currency_received"]).to_numpy(dtype=np.float32)
    same_bank = (frame["from_bank"] == frame["to_bank"]).to_numpy(dtype=np.float32)
    self_loop = (frame["from_account"] == frame["to_account"]).to_numpy(dtype=np.float32)
    ts = pd.to_datetime(frame["ts"])
    hour = (ts.dt.hour.to_numpy(dtype=np.float32)) / 23.0
    dow = (ts.dt.dayofweek.to_numpy(dtype=np.float32)) / 6.0
    amount_paid = frame["amount_paid"].to_numpy(dtype=np.float64)
    round_100 = np.isclose(amount_paid % 100, 0).astype(np.float32)
    round_1000 = np.isclose(amount_paid % 1000, 0).astype(np.float32)
    return np.stack(
        [log_amt_paid, log_amt_recv, ratio, is_fx, same_bank, self_loop, hour, dow,
         round_100, round_1000],
        axis=1,
    ).astype(np.float32)


NUMERIC_DIM = 10
CURRENCY_EMB_DIM = 4
FORMAT_EMB_DIM = 4


class TxnGNN(nn.Module):
    """Node embeddings refined by message passing, then an edge classifier
    that reads both endpoints' embeddings plus the transaction's own
    attributes. `conv_type` is "sage" or "gcn"."""

    def __init__(self, n_nodes: int, n_currencies: int, n_formats: int,
                 emb_dim: int, hidden_dim: int, use_format: bool,
                 conv_type: str = "sage") -> None:
        super().__init__()
        self.use_format = use_format
        self.node_emb = nn.Embedding(n_nodes, emb_dim)
        self.currency_emb = nn.Embedding(n_currencies, CURRENCY_EMB_DIM)
        if use_format:
            self.format_emb = nn.Embedding(n_formats, FORMAT_EMB_DIM)

        conv_cls = SAGEConv if conv_type == "sage" else GCNConv
        self.conv1 = conv_cls(emb_dim, hidden_dim)
        self.conv2 = conv_cls(hidden_dim, hidden_dim)

        extra_dim = NUMERIC_DIM + CURRENCY_EMB_DIM + (FORMAT_EMB_DIM if use_format else 0)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(self.node_emb.weight, edge_index))
        return self.conv2(h, edge_index)

    def score(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
              numeric: torch.Tensor, currency: torch.Tensor,
              fmt: torch.Tensor | None) -> torch.Tensor:
        parts = [h[src], h[dst], numeric, self.currency_emb(currency)]
        if self.use_format:
            parts.append(self.format_emb(fmt))
        rep = torch.cat(parts, dim=1)
        return self.head(rep).squeeze(-1)
