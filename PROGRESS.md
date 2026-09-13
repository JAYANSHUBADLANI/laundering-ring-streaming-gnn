# Progress log

## Status

Environment checks, the streaming layer, the streaming-vs-batch correctness
proof, the orchestrated pipeline, and the GNN vs. gradient boosting
comparison are all done and reported in `README.md`. This log carries the
detail that does not belong in the README: what was tried, what broke, and
why each fix was the right one rather than a workaround.

## Environment checks

Full detail in `docs/environment_notes.md`. The two findings worth
repeating here because they shaped what got built: Airflow's scheduler and
triggerer crash loop with SIGSEGV on this machine (Apple Silicon, macOS),
even with the standard fork-safety workaround, so Dagster was the only
orchestrator that actually ran a DAG here; and PyTorch Geometric installed
and round tripped cleanly on the first try, so DGL and a hand rolled message
passing layer were never needed.

## Deciding where this lives

New phases inside `laundering-ring-detection`, or a separate repository that
imports its data and code, was an open decision until the environment
checks were in. Went with a separate repository:
`laundering-ring-detection` is a complete, tightly scoped project with 25
tests and a fixed, small dependency list; this project adds Kafka, Docker,
Dagster and PyTorch, none of which have anything to do with the original's
data quality and evaluation story. Bolting all of that onto a finished
project would make it harder to read, not easier. This project imports the
original's `aml` package (config, periods, graph features, the gradient
boosting baseline, the evaluation rules) directly, on the assumption that
both repositories are cloned as siblings; `amlgnn.source_repo` resolves that
path from an environment variable or the sibling default, never as a
hardcoded absolute path.

## Building the streaming layer

Kafka in Docker, one broker, KRaft mode (`apache/kafka:3.8.0`), no
Zookeeper. Confirmed with a plain produce/consume round trip before writing
anything domain specific.

The producer replays a split's transactions from the source repository's own
DuckDB, in timestamp order, keyed by `from_account`. The consumer drains the
topic and folds each transaction into `IncrementalGraphState`, a plain
accounts set and account-pair edges set updated one message at a time, then
lands the transaction into this project's own DuckDB table
(`amlgnn.stream_store`), built with the exact column set and name the source
project's `trans_<split>` table has. That last choice is what let
`aml.graphfeat.build`, `aml.baseline.load_frame` and
`aml.evaluate.ring_membership` all run against the streamed data completely
unmodified: none of the existing repository's feature or evaluation code
needed to change to accept a table that arrived a row at a time. Ring labels
are not re-derived from the stream; the verified `patterns_<split>` table is
copied in once from the source database, since ring identity is the label
side of this project, not something a transaction stream was ever going to
carry.

**Checked the streaming graph against the batch graph directly, at full
scale, not a sample.** `amlgnn.graph_build.batch_structure` builds the
account graph with one SQL query and a pandas dedupe. The real Kafka broker
replayed HI-Small's full training window (3,554,066 transactions) and
separately LI-Small's (4,846,510), each consumed message by message into
`IncrementalGraphState`. Both matched the batch reference exactly: same
513,284 accounts and 588,606 edges for HI-Small, same 703,504 accounts and
802,623 edges for LI-Small, zero missing on either side. Producer throughput
was about 26,000 messages/second and consumer throughput about 38,000/second
on this machine, so the full HI-Small training window replayed and drained
in under three minutes combined.

## A reproducibility bug, found by refitting the baseline rather than
## trusting the streamed copy

The plan was always to refit `laundering-ring-detection`'s own gradient
boosting baseline on this project's landed data, specifically so that a
divergence from the committed numbers would surface a bug in the streaming
path rather than being silently assumed away. It did surface something, but
not what was expected.

**First symptom.** `graph_without_format` on HI-Small, retrained on
streamed data, gave a ring recall of 0.16, against 0.3761 committed. Ruled
out in order: transaction counts matched (5,077,237 landed, matching the
dense period total exactly); the account and edge structure had already
been proven identical by the streaming equivalence check; `round_100` /
`round_1000` counts matched exactly between the two databases despite one
storing amounts as `DOUBLE` and the other as `DECIMAL(20,2)`, so that was not
it either.

**Actual cause, first half.** This project's `requirements.txt` pinned
`scikit-learn>=1.5.0`, which resolved to 1.9.1; the source repository's
actual installed virtual environment (not just its own `requirements.txt`,
which has itself drifted from what it has installed) is on 1.7.2.
Reproduced the exact committed numbers by running the source repository's
own fit-and-score logic in its own virtual environment twice in a row,
bit-identical both times, which ruled out any inherent non-determinism in
`HistGradientBoostingClassifier` on this machine. Pinned this project to the
exact versions the source repository's environment actually has: duckdb
1.5.5, pandas 2.3.3, numpy 2.3.5, scikit-learn 1.7.2.

**Actual cause, second half.** Even after the pin, two back to back runs of
this project's own retrain script, on identical landed data, gave two
different numbers (0.1606, then 0.1927). That ruled out a one-time
environment mismatch and pointed at something in how the data was being
read. The landed table is built from thousands of small `INSERT` batches,
one per consumer flush (`storage_info` showed roughly 1,442 fragments),
against the source table's single bulk `CREATE TABLE ... AS SELECT` from a
CSV. A plain `SELECT ... WHERE clause`, with no `ORDER BY`, does not
guarantee the same physical scan order across separate connections when a
table is that fragmented, and `HistGradientBoostingClassifier`'s fit turned
out to be sensitive enough to row order to move ring recall by a wide margin
between otherwise identical runs.

**Fix.** `amlgnn.stream_store.compact` rewrites the landed table, sorted by
timestamp then transaction id, into one physical segment before every
retrain cycle: a compaction step, the kind any warehouse fed by a stream
would run periodically, not a change to any reused evaluation or feature
code. Checked directly rather than assumed: two consecutive retrains on
identical, compacted data now give identical ring recall. The retrained
baseline still does not match the source repository's committed number to
the fourth decimal place, because compacting sorts by timestamp, which is a
different, if equally valid, physical order than the original CSV's; it no
longer drifts between runs, which is the property this project actually
needed from it.

## Building the GNN

GraphSAGE over a GCN: the environment check found both installed and ran
cleanly, so the choice came down to which convolution suits the graph,
GraphSAGE's learned neighbourhood aggregation over a GCN's fixed
degree-normalised propagation, given how uneven account degree is in this
data (the source repository's own typology table already shows hub shaped
rings, scatter-gather and fan-out, sitting next to chain shaped ones,
cycle and random, with very different degree profiles).

Two layers, trained end to end for edge (transaction) classification: a
learned embedding per account, refined by message passing over the
training-window account graph, then a small MLP head scoring each
transaction from its two endpoints' embeddings plus its own attributes.
Leakage handling mirrors the existing graph feature model on purpose: the
node vocabulary and the message passing structure are both built from the
training window alone, and an account never seen there gets one shared,
learned "unseen" embedding rather than a zero vector, the neural equivalent
of the null the gradient booster gets for the same account. Every model is
fitted twice, with `payment_format` and without, for the identical reason
the existing project fits its own models twice: the generator's format
signature has to be checked for, not assumed absent.

One epoch over HI-Small's full training window (3,554,066 transactions,
513,284 accounts) settles to about 1.2 seconds on this machine's GPU after a
roughly 2 second cold first epoch. A scoring pass over the 1,523,171
transaction test window takes under a second. Full cost breakdown, measured
per cycle, is in `results/cycle_*_cost.csv`.

## LI-Small's GNN: two runs that trained successfully and still lost the result

Getting a usable LI-Small GNN number took three attempts, not because the
model was wrong but because two runs never survived to write one down. The
first was killed by hand after 30 minutes of `gnn_with_format` training with
no output, on the wrong assumption that it had hung; a `sample` of the
process while it was still alive showed it waiting on ordinary worker thread
synchronisation, not deadlocked, and killing it lost a training run that
would have finished. The second died on its own partway through the second
`payment_format` variant, with `vm.swapusage` showing swap over 90 percent
full at the time, consistent with an out-of-memory kill, though this
machine was also running several other memory-heavy processes throughout so
contention was never fully isolated as the only cause. Both times, the
whole cycle's results, including the gradient boosting refit and whichever
GNN variant had already finished, were lost with it, because `run_cycle`
only wrote to disk once at the very end.

Fixed by writing each model family's rows to `results/` the moment that
family finishes, not after the whole cycle (`amlgnn.pipeline.run_cycle`'s
`write_variant` and its new `variants` parameter, which also let the
`with_format` and `without_format` GNN variants be retried independently
once the fix was in, instead of re-paying for gradient boosting or the
variant that had already succeeded). The third attempt, split into
`--variants gbm gnn_with_format` and then `--variants gnn_without_format`
separately, completed both: `gnn_with_format` in 1617.9s (having previously
trained in 1658.4s and 1651.8s across the two runs that did not survive to
write anything, a useful confirmation that the timing was real and
reproducible even though those runs' recall numbers never got the chance to
be), `gnn_without_format` in 1818.8s. Both scored exactly 0.0 ring recall,
0.0 to 0.0 interval, on every one of LI-Small's 68 evaluable rings.

## The final numbers

HI-Small: the graph feature model (0.3165) beats the GNN (0.0229) by an
order of magnitude, non-overlapping intervals. LI-Small: the graph feature
model (0.1618) against a GNN that caught nothing at all, either
`payment_format` variant, 0.0 to 0.0. The claim this project set out to
test does not survive on either split. Full numbers, the per-typology
breakdown, and the reasoning for why the GNN's failure looks structural
rather than a tuning problem are in `README.md`.

The three-cycle retraining experiment ran on HI-Small, not LI-Small, once
LI-Small's per-epoch cost turned out to be an open, unexplained, roughly
45-times outlier rather than the couple of minutes HI-Small's own numbers
would have predicted: three repeated cycles at that cost were not worth
paying for under this project's time budget, and running them would not by
itself have explained the anomaly either. On HI-Small, three cycles showed
retraining on more streamed data does move ring recall, substantially and
without plateauing, for the graph feature model (0.055 to 0.138 to 0.317
across the three cycles); the GNN moved too, but inside a band close enough
to zero throughout that the movement reads as fitting noise around a
non-working model rather than a trend.
