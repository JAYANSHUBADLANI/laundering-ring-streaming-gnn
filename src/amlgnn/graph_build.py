"""The account level graph structure, built the same way regardless of how
the transactions arrived.

This is the one function both the batch path and the streaming path are
checked against: `batch_structure` runs once over a complete frame, and
`amlgnn.incremental_graph.IncrementalGraphState` builds the same node and
edge sets one message at a time. They are two independent code paths on
purpose, so a message dropped, double counted, or misparsed on the streaming
side shows up as a set that does not match, rather than being hidden by
both sides sharing one implementation.

A self loop, an account transacting with itself, is not a structural edge:
`laundering-ring-detection`'s own graph features exclude it for the same
reason, since it says nothing about which accounts are connected to which
others.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GraphStructure:
    """Accounts and directed account pairs seen so far."""

    accounts: frozenset
    edges: frozenset  # of (from_account, to_account) tuples, from_account != to_account

    def diff(self, other: "GraphStructure") -> dict:
        """Where two structures disagree, for a readable failure report."""
        return {
            "accounts_only_in_self": self.accounts - other.accounts,
            "accounts_only_in_other": other.accounts - self.accounts,
            "edges_only_in_self": self.edges - other.edges,
            "edges_only_in_other": other.edges - self.edges,
        }


def batch_structure(transactions: pd.DataFrame) -> GraphStructure:
    """Build the graph structure from a complete transaction frame at once.

    `transactions` needs `from_account` and `to_account` columns. Anything
    else is ignored, since structure here means only who transacted with whom.
    """
    frm = transactions["from_account"].astype(str)
    to = transactions["to_account"].astype(str)
    accounts = frozenset(pd.concat([frm, to]).unique())
    pairs = frozenset(
        (a, b) for a, b in zip(frm, to) if a != b
    )
    return GraphStructure(accounts=accounts, edges=pairs)
