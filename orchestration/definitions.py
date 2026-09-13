"""The orchestrated pipeline: pull new data, refresh the graph and its
features, retrain both models, score them, write timestamped results.

Dagster was picked over Airflow after actually trying to run both locally,
not by convention: Airflow's scheduler and triggerer processes crash looped
with repeated SIGSEGV on this machine even with the standard macOS
fork-safety workaround, so no DAG could actually be scheduled through it,
while Dagster's webserver and daemon started cleanly. See
`docs/environment_notes.md`.

This runs a real, if local and unmanaged, orchestrator: `dagster dev -f
orchestration/definitions.py` stands up a working scheduler, a sensor and a
web UI. The op sequence below is the one thing genuinely shared between a
manual invocation and a scheduled or sensor triggered run, since both call
straight into `amlgnn.consumer.drain_topic` and `amlgnn.pipeline.run_cycle`.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import dagster as dg
from dagster import OpExecutionContext, SensorEvaluationContext

from amlgnn import stream_store
from amlgnn.config import KAFKA_BOOTSTRAP, KAFKA_TOPIC
from amlgnn.consumer import checkpoint_path, drain_topic
from amlgnn.incremental_graph import IncrementalGraphState
from amlgnn.pipeline import run_cycle

SPLIT_CONFIG = dg.Field(dg.String, default_value="HI-Small", description="HI-Small or LI-Small")


@dg.op(config_schema={"split": SPLIT_CONFIG, "topic": dg.Field(dg.String, default_value=KAFKA_TOPIC),
                      "idle_timeout_ms": dg.Field(dg.Int, default_value=10_000)})
def pull_latest_consumed_transactions(context: OpExecutionContext) -> str:
    """Drain whatever is new on the Kafka topic since the last run."""
    split = context.op_config["split"]
    result = drain_topic(
        split=split, topic=context.op_config["topic"], bootstrap=KAFKA_BOOTSTRAP,
        idle_timeout_ms=context.op_config["idle_timeout_ms"],
    )
    context.log.info(
        f"[{split}] landed {result.landed_this_run} new transactions this run "
        f"({result.transactions_seen_total} total); graph now {result.node_count} "
        f"accounts, {result.edge_count} edges"
    )
    return split


@dg.op
def update_graph(context: OpExecutionContext, split: str) -> str:
    """Compact the landed table into one physically ordered segment.

    Streaming lands data through many small appended batches. Leaving it
    fragmented turned out to make gradient boosting's fit order-sensitive and
    not reproducible run to run on identical data; compacting removes that
    before anything is refit. See `amlgnn.stream_store.compact`.
    """
    con = stream_store.connect()
    try:
        stream_store.compact(con, split)
        landed = stream_store.landed_count(con, split)
    finally:
        con.close()
    context.log.info(f"[{split}] compacted, {landed} transactions in the landed table")
    return split


@dg.op(config_schema={"conv": dg.Field(dg.String, default_value="sage"),
                      "epochs": dg.Field(dg.Int, default_value=30)})
def refresh_features_and_retrain(context: OpExecutionContext, split: str) -> dict:
    """Graph feature refresh, gradient boosting retrain, GNN retrain and
    scoring, all in one op: `amlgnn.pipeline.run_cycle` already does exactly
    the sequence the task list calls for, and splitting it into four Dagster
    ops with DataFrames passed between them would need one more concept
    (an IO manager for pandas frames) for no benefit a single local run gets
    from it.
    """
    con = stream_store.connect()
    try:
        result = run_cycle(con, split, conv_type=context.op_config["conv"],
                           epochs=context.op_config["epochs"])
    finally:
        con.close()
    context.log.info(f"[{split}] cycle {result.stamp} written to {result.eval_path}")
    primary = result.evaluation[result.evaluation["alert_budget"] == 0.001]
    for _, row in primary.iterrows():
        context.log.info(
            f"[{split}] {row['variant']:24s} ring_recall_at_least_1={row['ring_recall_at_least_1']:.4f}"
        )
    return {"split": split, "stamp": result.stamp, "eval_path": result.eval_path}


@dg.job
def retrain_pipeline_job():
    split = pull_latest_consumed_transactions()
    split = update_graph(split)
    refresh_features_and_retrain(split)


@dg.sensor(job=retrain_pipeline_job, minimum_interval_seconds=15)
def new_data_sensor(context: SensorEvaluationContext):
    """Fire a run when either split's checkpoint shows more transactions
    consumed than the last time this sensor fired.

    A cursor keyed by split holds the transaction count as of the last run
    request, so a producer replay that lands between sensor ticks is picked
    up on the next tick without firing twice for the same data.
    """
    import json

    cursor = json.loads(context.cursor) if context.cursor else {}
    requests = []
    for split in ("HI-Small", "LI-Small"):
        state = IncrementalGraphState.load(checkpoint_path(split))
        seen_before = cursor.get(split, 0)
        if state.transactions_seen > seen_before:
            requests.append(dg.RunRequest(
                run_key=f"{split}-{state.transactions_seen}",
                run_config={
                    "ops": {
                        "pull_latest_consumed_transactions": {"config": {"split": split}},
                    }
                },
            ))
            cursor[split] = state.transactions_seen
    context.update_cursor(json.dumps(cursor))
    return requests


defs = dg.Definitions(jobs=[retrain_pipeline_job], sensors=[new_data_sensor])
