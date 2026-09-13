"""Graph state maintained one transaction at a time, as the Kafka consumer
sees them, rather than rebuilt from scratch on every message.

This is checked against `amlgnn.graph_build.batch_structure` on the same
transaction set in `tests/test_streaming_equivalence.py` and in
`scripts/verify_stream_equivalence.py`. Nothing here is allowed to assume the
comparison will pass; it is the thing being tested.
"""

from __future__ import annotations

import json
from pathlib import Path

from .graph_build import GraphStructure


class IncrementalGraphState:
    """Accounts and account-pair edges, updated one transaction at a time."""

    def __init__(self) -> None:
        self._accounts: set[str] = set()
        self._edges: set[tuple[str, str]] = set()
        self.transactions_seen = 0

    def add_transaction(self, from_account: str, to_account: str) -> None:
        from_account, to_account = str(from_account), str(to_account)
        self._accounts.add(from_account)
        self._accounts.add(to_account)
        if from_account != to_account:
            self._edges.add((from_account, to_account))
        self.transactions_seen += 1

    def structure(self) -> GraphStructure:
        return GraphStructure(accounts=frozenset(self._accounts), edges=frozenset(self._edges))

    def node_count(self) -> int:
        return len(self._accounts)

    def edge_count(self) -> int:
        return len(self._edges)

    def save(self, path: Path) -> None:
        """Checkpoint state to disk so a separate process (an orchestration
        op, or a restarted consumer) can pick up where this one left off."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "transactions_seen": self.transactions_seen,
            "accounts": sorted(self._accounts),
            "edges": sorted(self._edges),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "IncrementalGraphState":
        state = cls()
        if not path.exists():
            return state
        payload = json.loads(path.read_text())
        state._accounts = set(payload["accounts"])
        state._edges = {tuple(e) for e in payload["edges"]}
        state.transactions_seen = payload["transactions_seen"]
        return state
