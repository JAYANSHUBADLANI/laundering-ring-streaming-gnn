"""The train/test/tail boundary for a split, read from the source project's
own committed result rather than recomputed here.

`laundering-ring-detection` derives the dense period boundary and the
temporal cut from a rule over the full file (see its `src/aml/periods.py`).
That derivation needs the complete split to run correctly. This project
often has only part of a split landed at any one point, mid replay, so the
cut cannot be rederived from what has arrived so far without silently
drifting as more data lands. The cut is a fixed evaluation boundary, not a
per-cycle computation, so it is read once from the source repo's
`results/periods.csv` and reused exactly.
"""

from __future__ import annotations

import pandas as pd

from .source_repo import import_aml, source_repo_path


def load_periods(split: str):
    """A `Periods` instance for `split`, built from the committed table."""
    aml = import_aml()
    from aml.periods import Periods

    table = pd.read_csv(source_repo_path() / "results" / "periods.csv")
    row = table.loc[table["split"] == split]
    if row.empty:
        raise ValueError(f"No committed periods row for split {split!r}")
    row = row.iloc[0]
    return Periods(
        split=split,
        dense_end=pd.Timestamp(row["dense_end"]),
        cut=pd.Timestamp(row["temporal_cut"]),
        last_ts=pd.Timestamp(row["last_ts"]),
    )
