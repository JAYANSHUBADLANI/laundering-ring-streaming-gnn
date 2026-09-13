"""Paths and settings local to this project.

Evaluation rules, split definitions and the account level graph feature code
are not repeated here: they are read from `laundering-ring-detection` through
`source_repo.py`, so a ring counts as caught the same way in both projects and
the numbers sit in the same table rather than next to it with an asterisk.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_INTERIM = REPO_ROOT / "data" / "interim"
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

STREAM_STATE_DIR = DATA_INTERIM / "stream_state"

ROOT_SEED = 20260913

KAFKA_BOOTSTRAP = "localhost:19092"
KAFKA_TOPIC = "aml-transactions"

# GNN architecture and training, fixed here rather than tuned after seeing a
# result for the same reason the original project fixed its evaluation rules
# before fitting anything: it removes the temptation to pick numbers that
# flatter the newer method.
GNN_EMBEDDING_DIM = 32
GNN_HIDDEN_DIM = 64
GNN_EPOCHS = 30
GNN_LEARNING_RATE = 1e-3
