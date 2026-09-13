# Environment checks, run before anything else was built

Machine: Apple M5, 10 CPU cores, 16 GB RAM, 136 GB free disk, macOS, arm64.

## Dataset

Already present locally from `laundering-ring-detection`'s own fetch:
`data/raw/{HI,LI}-Small_Trans.csv`, the matching `_accounts.csv` and
`_Patterns.txt` files, and a built `data/interim/aml.duckdb`. Nothing was
re-downloaded.

## Docker

`docker --version` and `docker compose version` reported 29.7.2 and v5.3.1,
but the daemon was not running. Started Docker Desktop, waited for
`docker info` to succeed, then ran `docker run --rm hello-world` end to end
to confirm a container actually executes, not just that the binaries exist.

## PyTorch Geometric vs DGL

`pip install torch torch_geometric` installed cleanly on this machine:
torch 2.14.0, torch_geometric 2.8.0.post1, with MPS (Apple GPU) available. A
minimal graph (`Data(x=..., edge_index=...)`) round tripped through both
`SAGEConv` and `GCNConv` with no errors. DGL was not tried, since PyTorch
Geometric worked without needing a fallback.

## Orchestrator: Dagster vs Airflow

Both installed. `dagster dev` started a working webserver and daemon in
about three seconds with no errors; `curl` against it returned HTTP 200.

`airflow standalone` (3.0.3, with the constraints file for this Python
version) came up too, the webserver answered on port 8080, but its scheduler
and triggerer processes crash looped continuously: gunicorn workers received
SIGSEGV, respawned, and crashed again, dozens of times over 30 seconds. This
is the well known macOS fork-safety failure mode for multi-threaded Python
processes (the log carries the matching `DeprecationWarning: this process is
multi-threaded, use of fork() may lead to deadlocks in the child`). The usual
workaround, `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` plus `no_proxy='*'`,
was tried and did not fix it; the crash loop continued at the same rate.

Airflow never reached a state where a DAG could actually be scheduled on
this machine. Dagster did, cleanly, on the first try. That is the entire
reason for the choice: not familiarity, not popularity, an orchestrator that
was tried directly and ran, against one that was tried directly and did not.

## Compute and runtime

One full batch GraphSAGE training step (forward, loss, backward, optimizer
step) over HI-Small's entire training window, 3,554,066 transactions and
513,284 accounts, took 1.94s cold and settled to about 1.2s per epoch on the
Apple GPU (MPS backend) after that. A scoring pass over the 1,523,171
transaction test window took under a second. A full retrain cycle on
HI-Small, graph feature rebuild plus a gradient boosting refit of all four
existing baseline variants plus a 30 epoch GNN fit and score, lands in
roughly one to two minutes. See `results/cycle_*_cost.csv` for the measured
breakdown from an actual run, not an estimate.

**LI-Small does not scale the way HI-Small's numbers would predict, and this
was checked rather than assumed away.** LI-Small's graph is only about 37
percent larger than HI-Small's on every dimension that should matter for
GraphSAGE (703,504 accounts against 513,284, 802,623 structural edges against
588,606, 4,846,510 training transactions against 3,554,066), and its account
degree distribution is nearly identical (mean 19.62 against 19.72, 99.9th
percentile 191 against 195, max degree 223,590 against 169,756, so no single
outlier hub explains it). Despite that, one epoch on LI-Small settled to
about 55 seconds, not the roughly 1.6 seconds a 37 percent size increase
would predict from HI-Small's number: a full 30 epoch fit of one
`payment_format` variant took 1651 to 1658 seconds (about 27.5 minutes),
reproduced twice with the two runs agreeing to within half a percent
(1658.4s and 1651.8s), under two different levels of system memory pressure
(swap 94 percent full on one run, 76 percent full on the other, checked with
`vm_stat` and `sysctl vm.swapusage` while the process was live). That
consistency across different memory conditions argues against ordinary
system contention as the sole explanation, though this machine was also
running several other concurrent processes the whole time, so
contention was not zero either. The most likely remaining explanation is a
size-dependent performance cliff in PyTorch's MPS (Apple GPU) backend for
GraphSAGE's scatter/gather aggregation, somewhere between HI-Small's edge
count and LI-Small's; this was not fully bisected or root-caused within this
project's time budget, and is reported as an open question rather than
quietly worked around. It does not affect correctness: LI-Small's GNN still
trains and produces valid scores, just far slower than HI-Small's numbers
would suggest, which is why the three-cycle retraining experiment
(`results/multi_cycle_summary_*.csv`) was run on HI-Small rather than
LI-Small, where a few-minutes-per-cycle cost had already been established
directly rather than assumed.

## A reproducibility issue this surfaced, not anticipated going in

Refitting the existing gradient boosting baseline on data landed through
this project's own streaming path, rather than `laundering-ring-detection`'s
original single bulk CSV load, gave a different ring recall than the
committed number the first time it was tried, and a different number again
on a second run of the identical script against the identical data. Traced
to two separate causes:

1. **A scikit-learn version drift between the two projects' virtual
   environments** (1.9.1 here against 1.7.2 in the source repo, from an
   unpinned `scikit-learn>=1.5.0` picking up whatever was current).
   `HistGradientBoostingClassifier` is not guaranteed bit stable across
   minor versions even with the same `random_state`. Fixed by pinning this
   project's `requirements.txt` to the exact versions the source repo's
   virtual environment actually has installed (not just what its own
   `requirements.txt` says, which had itself drifted from what it is
   actually running).
2. **Table fragmentation from streaming.** The source repo's `trans_*`
   table is one bulk `CREATE TABLE ... AS SELECT` from a CSV. This project's
   landed table is built from thousands of small `INSERT` batches, one per
   consumer flush, which left it in around 1,400 storage fragments. A plain
   `SELECT ... WHERE clause` with no `ORDER BY` does not read those
   fragments back in the same order on every connection, and
   `HistGradientBoostingClassifier`'s fit turned out to be sensitive enough
   to that row order to move ring recall by a wide margin between two runs
   on identical data. `amlgnn.stream_store.compact` rewrites the table
   sorted by timestamp before every retrain cycle to remove this as a
   variable. After compacting, two consecutive runs on identical data give
   identical ring recall (checked directly, not assumed). The retrained
   baseline still does not reproduce the source repo's committed number to
   the fourth decimal place, because it is now a different, but internally
   consistent, physical row order than the source's original CSV-derived
   one; it does not drift between runs anymore, which is the property this
   project actually needed.
