import threading, time, json
from typing import Dict, Any
from settings import ACTIVE_LIMIT, SCHEDULER_POLL_SECONDS
import registry
from modal_driver import ModalDriver
from logger import log_event
from pathlib import Path

_driver = ModalDriver()
_stop_flag = False

RESULTS_DIR = Path("./inferno_runs").resolve()
RUNS_DIR = RESULTS_DIR / "runs"


def _run_once_for_pool(pool: str) -> None:
    if registry.count_active_for_pool(pool) >= ACTIVE_LIMIT.get(pool, 1):
        return
    run_id = registry.next_scheduled_for_pool(pool)
    if not run_id:
        return

    print(f"Transitioning to PROVISIONING for run: {run_id}")
    registry.transition(run_id, "PROVISIONING")
    log_event("provisioning_started", run_id=run_id, pool=pool)

    # Launch (MVP CLI bookkeeping)
    spec, cfg = registry.get_spec_and_config(run_id)  # add this if missing
    modal_id = _driver.launch({"run_id": run_id})
    registry.attach_modal_id(run_id, modal_id)
    registry.transition(run_id, "RUNNING")

    # Execute and collect synchronously (MVP)
    try:
        metrics_path = _driver.execute_locally_and_collect(run_id, spec.model_dump(), cfg)
        # record artifact paths for UI
        folder = RUNS_DIR / run_id
        registry.set_paths(run_id, {
            "metrics": str(metrics_path),
            "logs": str(folder / "logs.txt"),
            "state": str(folder / "state.json"),
        })
        registry.transition(run_id, "COMPLETED")
        # scheduler.py after marking COMPLETED
        try:
            import requests
            # warm best-by-throughput cache
            requests.get(f"http://localhost:8000/jobs/{spec.job_name}/best?metric=throughput_tok_s&mode=max", timeout=1.5)
        except Exception:
            pass  # best will compute on demand anyway
    except Exception as e:
        log_event("run_failed", run_id=run_id, error=str(e))
        registry.transition(run_id, "FAILED")
        


def scheduler_loop():
    global _stop_flag
    pools = list(ACTIVE_LIMIT.keys())
    log_event("scheduler_started", pools=pools)
    while not _stop_flag:
        for p in pools:
            try:
                _run_once_for_pool(p)
            except Exception as e:
                log_event("scheduler_pool_error", pool=p, error=str(e))
                time.sleep(SCHEDULER_POLL_SECONDS)
    log_event("scheduler_stopped")




def start_background_scheduler() -> threading.Thread:
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    return t



def stop_scheduler():
    global _stop_flag
    _stop_flag = True

