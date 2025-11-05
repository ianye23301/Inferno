import os, json, sqlite3, threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from settings import DB_PATH, RUNS_DIR
from models import RunState, JobSpec


_LOCK = threading.Lock()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
run_id TEXT PRIMARY KEY,
job_name TEXT,
gpu_pool TEXT,
state TEXT,
modal_job_id TEXT,
created_at TEXT,
updated_at TEXT,
config_json TEXT,
spec_json TEXT,
metrics_path TEXT,
logs_path TEXT
);
"""

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


# initialize schema
with _conn() as c:
    c.execute(_SCHEMA)



def _run_folder(run_id: str) -> Path:
    p = RUNS_DIR / run_id
    p.mkdir(parents=True, exist_ok=True)
    (p/"artifacts").mkdir(exist_ok=True)
    return p




def create_run(run_id: str, spec: JobSpec, config: Dict[str, Any]) -> None:
    now = datetime.utcnow().isoformat()+"Z"
    state: RunState = "PENDING"
    folder = _run_folder(run_id)
    # write spec & config mirrors for human debugging
    (folder/"spec.json").write_text(spec.model_dump_json(indent=2))
    (folder/"config.json").write_text(json.dumps(config, indent=2))
    (folder/"state.json").write_text(json.dumps({"state": state, "history": [{"at": now, "to": state}]} , indent=2))
    logs_path = str((folder/"logs.txt").resolve())
    with _conn() as c, _LOCK:
        c.execute(
        """
        INSERT INTO runs(run_id, job_name, gpu_pool, state, modal_job_id, created_at, updated_at,
        config_json, spec_json, metrics_path, logs_path)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (run_id, spec.job_name, spec.gpu_pool, state, None, now, now,
        json.dumps(config), spec.model_dump_json(), None, logs_path)
        )




def transition(run_id: str, new_state: RunState) -> None:
    now = datetime.utcnow().isoformat()+"Z"
    folder = _run_folder(run_id)
    # update json mirror
    state_json = json.loads((folder/"state.json").read_text())
    state_json["state"] = new_state
    state_json["history"].append({"at": now, "to": new_state})
    (folder/"state.json").write_text(json.dumps(state_json, indent=2))
    # update db
    with _conn() as c, _LOCK:
        c.execute("UPDATE runs SET state=?, updated_at=? WHERE run_id=?", (new_state, now, run_id))




def attach_modal_id(run_id: str, modal_job_id: str) -> None:
    with _conn() as c, _LOCK:
        c.execute("UPDATE runs SET modal_job_id=? WHERE run_id=?", (modal_job_id, run_id))




def save_metrics_path(run_id: str, metrics_path: str) -> None:
    with _conn() as c, _LOCK:   
        c.execute("UPDATE runs SET metrics_path=? WHERE run_id=?", (metrics_path, run_id))




def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        cur = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))




def list_runs_by_job(job_name: str) -> List[Dict[str, Any]]:
    with _conn() as c:
        cur = c.execute("SELECT * FROM runs WHERE job_name=? ORDER BY created_at DESC", (job_name,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]




def next_scheduled_for_pool(gpu_pool: str) -> Optional[str]:
    with _conn() as c, _LOCK:
        cur = c.execute(
        "SELECT run_id FROM runs WHERE state='SCHEDULED' AND gpu_pool=? ORDER BY created_at ASC LIMIT 1",
        (gpu_pool,)
        )
        row = cur.fetchone()
        return row[0] if row else None




def count_active_for_pool(gpu_pool: str) -> int:
    with _conn() as c:
        cur = c.execute(
        "SELECT COUNT(1) FROM runs WHERE gpu_pool=? AND state IN ('PROVISIONING','RUNNING','COLLECTING')",
        (gpu_pool,)
        )
        return int(cur.fetchone()[0])




def set_scheduled(run_id: str) -> None:
    transition(run_id, "SCHEDULED")


# --- add at bottom of registry.py ---

def get_spec_and_config(run_id: str) -> tuple[JobSpec, Dict[str, Any]]:
    """Return (spec: JobSpec, config: dict) for a run."""
    with _conn() as c:
        cur = c.execute("SELECT spec_json, config_json FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if not row:
            raise KeyError(f"run not found: {run_id}")
        spec_json, cfg_json = row
    # Prefer DB copies; fall back to files if you want:
    # folder = _run_folder(run_id)
    # if not spec_json: spec_json = (folder/"spec.json").read_text()
    # if not cfg_json:  cfg_json  = (folder/"config.json").read_text()
    spec = JobSpec.model_validate_json(spec_json)
    cfg  = json.loads(cfg_json) if isinstance(cfg_json, str) else cfg_json
    return spec, cfg


def set_paths(run_id: str, paths: Dict[str, str]) -> None:
    """
    Update artifact path fields for a run (metrics/logs). Safe to call multiple times.
    Accepts keys: 'metrics', 'logs'. Ignores unknown keys.
    """
    metrics = paths.get("metrics")
    logs    = paths.get("logs")
    with _conn() as c, _LOCK:
        if metrics and logs:
            c.execute("UPDATE runs SET metrics_path=?, logs_path=? WHERE run_id=?", (metrics, logs, run_id))
        elif metrics:
            c.execute("UPDATE runs SET metrics_path=? WHERE run_id=?", (metrics, run_id))
        elif logs:
            c.execute("UPDATE runs SET logs_path=? WHERE run_id=?", (logs, run_id))



def iter_job_metrics(job_name: str):
    """Yield (row, metrics_dict) for runs in this job that have metrics.json."""
    rows = list_runs_by_job(job_name)
    for r in rows:
        mp = r.get("metrics_path")
        if not mp:
            # fall back to standard location if not yet set
            mp = str((RUNS_DIR / r["run_id"] / "metrics.json").resolve())
        if mp and os.path.exists(mp):
            try:
                yield r, json.loads(Path(mp).read_text())
            except Exception:
                continue

def write_job_best(job_name: str, metric: str, best_payload: dict) -> str:
    job_dir = JOBS_DIR / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    out = job_dir / f"best_{metric}.json"
    out.write_text(json.dumps(best_payload, indent=2))
    return str(out)