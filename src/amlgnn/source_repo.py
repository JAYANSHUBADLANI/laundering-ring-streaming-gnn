"""Where the original ring detection project lives, and how to reach it.

This project does not re-fetch the IBM AML data or re-derive the account
level graph features and gradient boosting baseline: it reads them from
`laundering-ring-detection`, cloned as a sibling directory by default. The
path is never hardcoded as an absolute path in a tracked file; it is derived
from this repository's own location, or overridden with `AML_SOURCE_REPO`
for a different layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_source_repo() -> Path:
    return REPO_ROOT.parent / "laundering-ring-detection"


def source_repo_path() -> Path:
    import os

    override = os.environ.get("AML_SOURCE_REPO")
    path = Path(override).expanduser().resolve() if override else _default_source_repo()
    if not path.is_dir():
        raise FileNotFoundError(
            f"laundering-ring-detection not found at {path}. Clone it as a sibling "
            "of this repository, or set AML_SOURCE_REPO to point at it."
        )
    return path


def import_aml():
    """Put the source repo's `src` on sys.path so `aml.*` submodules import.

    Imported lazily rather than at module load time so that importing
    `amlgnn` does not itself require the sibling repo to exist yet.
    """
    src = source_repo_path() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import aml.config
    import aml.periods
    import aml.db
    import aml.patterns
    import aml.graphfeat
    import aml.baseline
    import aml.evaluate
    return aml


def source_duckdb_path() -> Path:
    aml = import_aml()
    return aml.config.DUCKDB_PATH


def connect_source(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Read-only connection to the source repo's existing DuckDB database.

    Both splits are already loaded there by its own `scripts/load_duckdb.py`.
    This project never re-runs that load or writes back into that database.
    """
    path = source_duckdb_path()
    return duckdb.connect(str(path), read_only=read_only)
