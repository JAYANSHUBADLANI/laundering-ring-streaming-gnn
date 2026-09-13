"""Fit the GNN on one split's landed transactions and score it the way
`aml.evaluate` expects: a `Scored` object per variant, train windowed only,
tested on the fixed holdout.

GraphSAGE is the default convolution. Both it and a GCN round tripped
cleanly in this project's environment check; GraphSAGE was picked because it
learns an aggregation function over a node's neighbourhood rather than
assuming the fixed, degree-normalised propagation a GCN does, closer to what
an account graph with wildly uneven degree actually needs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .model import (UNSEEN_INDEX, Scored, TxnGNN, Vocabulary, numeric_features,
                    pick_device)

TRANS_SELECT = """
    SELECT txn_id, ts, from_bank, from_account, to_bank, to_account,
           amount_received, currency_received, amount_paid, currency_paid,
           payment_format, is_laundering
    FROM {table}
    WHERE {clause}
"""


def load_frame(con: duckdb.DuckDBPyConnection, table: str, clause: str) -> pd.DataFrame:
    return con.execute(TRANS_SELECT.format(table=table, clause=clause)).df()


def build_node_index(train_frame: pd.DataFrame) -> dict[str, int]:
    """Training-window accounts only, index 0 reserved for anything unseen."""
    accounts = pd.unique(
        pd.concat([train_frame["from_account"], train_frame["to_account"]]).astype(str)
    )
    return {a: i + 1 for i, a in enumerate(accounts)}


def build_structure_edges(train_frame: pd.DataFrame, node_index: dict[str, int]) -> torch.Tensor:
    """Deduplicated, both-directions account-pair edges from the training
    window, the same shape of structure `aml.graphfeat.structural_features`
    computes for the gradient boosting baseline."""
    frm = train_frame["from_account"].astype(str)
    to = train_frame["to_account"].astype(str)
    pairs = pd.DataFrame({"a": frm, "b": to})
    pairs = pairs[pairs["a"] != pairs["b"]].drop_duplicates()
    src = pairs["a"].map(node_index).to_numpy(dtype=np.int64)
    dst = pairs["b"].map(node_index).to_numpy(dtype=np.int64)
    both = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    return torch.tensor(both, dtype=torch.long)


def encode_accounts(series: pd.Series, node_index: dict[str, int]) -> np.ndarray:
    return series.astype(str).map(node_index).fillna(UNSEEN_INDEX).to_numpy(dtype=np.int64)


@dataclass
class FitResult:
    scored: Scored
    train_seconds: float
    inference_seconds: float
    epoch_seconds: list[float]


def fit_and_score(con: duckdb.DuckDBPyConnection, table: str, split: str,
                  train_clause: str, test_clause: str, use_format: bool,
                  epochs: int, emb_dim: int, hidden_dim: int, lr: float,
                  conv_type: str = "sage", seed: int = 0) -> FitResult:
    torch.manual_seed(seed)
    device = pick_device()

    train = load_frame(con, table, train_clause)
    test = load_frame(con, table, test_clause)

    node_index = build_node_index(train)
    n_nodes = len(node_index) + 1  # + the reserved unseen slot
    edge_index = build_structure_edges(train, node_index).to(device)

    currency_vocab = Vocabulary(train["currency_paid"])
    format_vocab = Vocabulary(train["payment_format"]) if use_format else None

    def tensors(frame: pd.DataFrame):
        src = torch.tensor(encode_accounts(frame["from_account"], node_index), device=device)
        dst = torch.tensor(encode_accounts(frame["to_account"], node_index), device=device)
        numeric = torch.tensor(numeric_features(frame), device=device)
        currency = torch.tensor(currency_vocab.encode(frame["currency_paid"]), device=device)
        fmt = (torch.tensor(format_vocab.encode(frame["payment_format"]), device=device)
               if use_format else None)
        labels = torch.tensor(frame["is_laundering"].to_numpy(dtype=np.float32), device=device)
        return src, dst, numeric, currency, fmt, labels

    train_src, train_dst, train_numeric, train_currency, train_fmt, train_labels = tensors(train)
    test_src, test_dst, test_numeric, test_currency, test_fmt, test_labels = tensors(test)

    model = TxnGNN(
        n_nodes=n_nodes, n_currencies=currency_vocab.size,
        n_formats=(format_vocab.size if use_format else 1),
        emb_dim=emb_dim, hidden_dim=hidden_dim, use_format=use_format,
        conv_type=conv_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_pos = float(train_labels.sum().item())
    n_neg = float(train_labels.numel()) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)

    epoch_seconds = []
    t_train_start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        h = model.encode(edge_index)
        logits = model.score(h, train_src, train_dst, train_numeric, train_currency, train_fmt)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        epoch_seconds.append(time.perf_counter() - t0)
    train_seconds = time.perf_counter() - t_train_start

    model.eval()
    t_infer_start = time.perf_counter()
    with torch.no_grad():
        h_eval = model.encode(edge_index)
        test_logits = model.score(h_eval, test_src, test_dst, test_numeric, test_currency, test_fmt)
        test_scores = torch.sigmoid(test_logits)
        if device.type == "mps":
            torch.mps.synchronize()
    inference_seconds = time.perf_counter() - t_infer_start

    variant = "gnn_with_format" if use_format else "gnn_without_format"
    scored = Scored(
        split=split, variant=variant,
        txn_id=test["txn_id"].to_numpy(),
        score=test_scores.cpu().numpy(),
        label=test["is_laundering"].to_numpy(),
        train_rows=len(train), test_rows=len(test),
    )

    # Two variants are fit back to back in the same process (with and
    # without payment_format). The second variant of a pair was
    # consistently several times slower to train and score than the first
    # in early runs on this machine's shared 16 GB unified memory; freeing
    # this run's GPU tensors explicitly, rather than leaving it to eventual
    # garbage collection, is the fix tried for that.
    del model, optimizer, edge_index
    del train_src, train_dst, train_numeric, train_currency, train_fmt, train_labels
    del test_src, test_dst, test_numeric, test_currency, test_fmt, test_labels, h_eval, test_logits
    if device.type == "mps":
        torch.mps.empty_cache()

    return FitResult(scored=scored, train_seconds=train_seconds,
                      inference_seconds=inference_seconds, epoch_seconds=epoch_seconds)
