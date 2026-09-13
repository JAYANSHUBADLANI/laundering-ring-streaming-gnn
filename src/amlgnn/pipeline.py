"""One retrain-and-score cycle: the same function a manual run and a
Dagster job both call.

Reuses `laundering-ring-detection`'s own graph feature and gradient boosting
code untouched; adds the GNN variants; scores every variant with the
existing project's own evaluation rules; writes timestamped results so a
later cycle does not overwrite what a reviewer might want to compare
against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import duckdb
import pandas as pd

from . import gnn_pipeline, stream_store
from .config import GNN_EMBEDDING_DIM, GNN_EPOCHS, GNN_HIDDEN_DIM, GNN_LEARNING_RATE, RESULTS
from .periods_source import load_periods
from .source_repo import import_aml


@dataclass
class CycleResult:
    split: str
    stamp: str
    evaluation: pd.DataFrame
    typology: pd.DataFrame
    cost: pd.DataFrame
    eval_path: str = ""
    typology_path: str = ""
    cost_path: str = ""


def run_cycle(con: duckdb.DuckDBPyConnection, split: str, conv_type: str = "sage",
             epochs: int = GNN_EPOCHS, write: bool = True, stamp: str | None = None,
             variants: tuple[str, ...] = ("gbm", "gnn_with_format", "gnn_without_format"),
             ) -> CycleResult:
    """`variants` restricts which model families this call fits, so a variant
    that fails or is too slow to finish on a given machine does not cost the
    ones that already succeeded: each variant's rows are written to disk the
    moment it completes, not held in memory until the whole cycle is done."""
    aml = import_aml()
    periods = load_periods(split)
    table = stream_store.table_for(split)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"[{split}] compacting landed table...", flush=True)
    stream_store.compact(con, split)

    total_rings = con.execute(
        f"SELECT count(DISTINCT ring_id) FROM {stream_store.patterns_table_for(split)}"
    ).fetchone()[0]
    rings = aml.evaluate.ring_membership(con, split, periods)

    RESULTS.mkdir(parents=True, exist_ok=True)
    eval_rows, typology_rows, cost_rows = [], [], []

    def write_variant(tag: str, eval_df: pd.DataFrame, typ_df: pd.DataFrame | None,
                      cost: list[dict]) -> None:
        """Append this variant's rows to disk immediately, so a later crash
        in a different variant does not lose what already succeeded."""
        eval_rows.append(eval_df)
        if typ_df is not None:
            typology_rows.append(typ_df)
        cost_rows.extend(cost)
        eval_df.to_csv(RESULTS / f"cycle_{split}_{stamp}_{tag}_evaluation.csv", index=False)
        if typ_df is not None:
            typ_df.to_csv(RESULTS / f"cycle_{split}_{stamp}_{tag}_by_typology.csv", index=False)
        pd.DataFrame(cost).to_csv(RESULTS / f"cycle_{split}_{stamp}_{tag}_cost.csv", index=False)

    if "gbm" in variants:
        print(f"[{split}] rebuilding graph features...", flush=True)
        t0 = time.perf_counter()
        graph_features = aml.graphfeat.build(con, split, periods)
        graph_seconds = time.perf_counter() - t0
        print(f"[{split}] graph features: {len(graph_features):,} accounts in {graph_seconds:.1f}s", flush=True)

        print(f"[{split}] retraining gradient boosting baselines...", flush=True)
        t0 = time.perf_counter()
        gbm_scored = aml.baseline.run_split(con, split, periods, graph_features)
        gbm_seconds = time.perf_counter() - t0
        print(f"[{split}] gradient boosting: 4 variants in {gbm_seconds:.1f}s", flush=True)

        gbm_eval = pd.concat([aml.evaluate.evaluate(s, rings, total_rings) for s in gbm_scored],
                             ignore_index=True)
        gbm_typ = pd.concat(
            [aml.evaluate.per_typology(s, rings, aml.config.PRIMARY_ALERT_BUDGET) for s in gbm_scored],
            ignore_index=True)
        write_variant("gbm", gbm_eval, gbm_typ, [
            {"split": split, "model": "graph_features", "stage": "rebuild", "seconds": graph_seconds},
            {"split": split, "model": "gradient_boosting", "stage": "retrain_all_4_variants",
             "seconds": gbm_seconds},
        ])

    for variant_tag, use_format in (("gnn_with_format", True), ("gnn_without_format", False)):
        if variant_tag not in variants:
            continue
        print(f"[{split}] fitting {variant_tag} ({conv_type}, {epochs} epochs)...", flush=True)
        result = gnn_pipeline.fit_and_score(
            con, table, split, periods.train_clause, periods.test_clause,
            use_format=use_format, epochs=epochs, emb_dim=GNN_EMBEDDING_DIM,
            hidden_dim=GNN_HIDDEN_DIM, lr=GNN_LEARNING_RATE, conv_type=conv_type,
        )
        print(f"[{split}] {variant_tag}: train {result.train_seconds:.1f}s "
              f"(mean epoch {sum(result.epoch_seconds) / len(result.epoch_seconds):.2f}s), "
              f"inference {result.inference_seconds:.2f}s", flush=True)
        gnn_eval = aml.evaluate.evaluate(result.scored, rings, total_rings)
        gnn_typ = aml.evaluate.per_typology(result.scored, rings, aml.config.PRIMARY_ALERT_BUDGET)
        write_variant(variant_tag, gnn_eval, gnn_typ, [
            {"split": split, "model": variant_tag, "stage": "train", "seconds": result.train_seconds},
            {"split": split, "model": variant_tag, "stage": "inference",
             "seconds": result.inference_seconds},
        ])

    evaluation = pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame()
    typology = pd.concat(typology_rows, ignore_index=True) if typology_rows else pd.DataFrame()
    cost = pd.DataFrame(cost_rows)

    result = CycleResult(split=split, stamp=stamp, evaluation=evaluation,
                         typology=typology, cost=cost)
    if write and not evaluation.empty:
        result.eval_path = str(RESULTS / f"cycle_{split}_{stamp}_evaluation.csv")
        result.typology_path = str(RESULTS / f"cycle_{split}_{stamp}_by_typology.csv")
        result.cost_path = str(RESULTS / f"cycle_{split}_{stamp}_cost.csv")
        evaluation.to_csv(result.eval_path, index=False)
        typology.to_csv(result.typology_path, index=False)
        cost.to_csv(result.cost_path, index=False)
    return result
