import threading, time, json
from typing import Dict, Any
from settings import ACTIVE_LIMIT, SCHEDULER_POLL_SECONDS
import registry
from modal_driver import ModalDriver
from logger import log_event


_driver = ModalDriver()
_stop_flag = False




def _run_once_for_pool(pool: str) -> None:
# Backpressure
    if registry.count_active_for_pool(pool) >= ACTIVE_LIMIT.get(pool, 1):
        return
    # print(f"Running once for pool: {pool}")
    run_id = registry.next_scheduled_for_pool(pool)
    if not run_id:
        return
    print(f"Transitioning to PROVISIONING for run: {run_id}")
    registry.transition(run_id, "PROVISIONING")
    log_event("provisioning_started", run_id=run_id, pool=pool)
    modal_id = _driver.launch({"run_id": run_id})
    registry.attach_modal_id(run_id, modal_id)
    registry.transition(run_id, "RUNNING")
    # For MVP stub, execute locally and collect immediately
    row = registry.get_run(run_id)
    spec = json.loads(row["spec_json"]) if row else {}
    config = json.loads(row["config_json"]) if row else {}
    metrics_path = _driver.execute_locally_and_collect(run_id, spec, config)
    registry.transition(run_id, "COLLECTING")
    registry.save_metrics_path(run_id, metrics_path)
    registry.transition(run_id, "COMPLETED")
    log_event("run_completed", run_id=run_id, metrics_path=metrics_path)




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

