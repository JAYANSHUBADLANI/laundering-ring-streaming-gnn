"""Execute the Dagster job once in process, without a live webserver.

`dagster dev -f orchestration/definitions.py` is the way to see the job, the
sensor and run history in the browser. This script runs the identical job
definition through Dagster's own execution engine for a single, scriptable
invocation, the way a CI check or this project's own multi cycle experiment
driver calls it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))

from definitions import retrain_pipeline_job


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="HI-Small", choices=["HI-Small", "LI-Small"])
    ap.add_argument("--idle-timeout-ms", type=int, default=10_000)
    args = ap.parse_args()

    result = retrain_pipeline_job.execute_in_process(run_config={
        "ops": {
            "pull_latest_consumed_transactions": {
                "config": {"split": args.split, "idle_timeout_ms": args.idle_timeout_ms}
            },
        }
    })
    print(f"success: {result.success}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
