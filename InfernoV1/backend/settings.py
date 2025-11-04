from pathlib import Path
from typing import Dict


RESULTS_DIR = Path("./inferno_runs").resolve()
RUNS_DIR = RESULTS_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# concurrency limits per pool (MVP; tune as needed)
ACTIVE_LIMIT: Dict[str, int] = {
"A100-80GB": 4,
"H100": 4,
"H200": 4,
"B200": 4,
}


SCHEDULER_POLL_SECONDS = 2.0


DB_PATH = RESULTS_DIR / "index.sqlite"