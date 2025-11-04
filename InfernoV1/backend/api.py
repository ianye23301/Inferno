from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, List
from uuid import uuid4
from pathlib import Path
import json, asyncio
from models import JobSpec
import sweeper, registry
from scheduler import start_background_scheduler
from settings import RUNS_DIR


app = FastAPI(title="Inferno Control Plane (MVP)")
_background = start_background_scheduler()




@app.post("/jobs")
def create_job(spec: JobSpec):
# expand sweep → create runs
    points = sweeper.grid_points(spec.sweep) if spec.strategy == "grid" else sweeper.random_points(spec.sweep, spec.random_trials)
    run_ids: List[str] = []
    for cfg in points:
        # basic constraint: TP <= num_gpus
        tp = int(cfg.get("tensor_parallel", 1))
        if tp > int(spec.num_gpus):
            continue
        run_id = f"{uuid4().hex[:8]}_{spec.model.split('/')[-1]}_{spec.gpu_pool}"
        registry.create_run(run_id, spec, cfg)
        registry.set_scheduled(run_id)
        run_ids.append(run_id)
    return {"job_name": spec.job_name, "count": len(run_ids), "run_ids": run_ids}




@app.get("/runs/{run_id}")
def get_run(run_id: str):
    row = registry.get_run(run_id)
    if not row:
        raise HTTPException(404, "run not found")
    # include state.json contents
    folder = RUNS_DIR / run_id
    state = json.loads((folder/"state.json").read_text())
    return {"row": row, "state": state}




@app.get("/jobs/{job_name}")
def get_job(job_name: str):
    rows = registry.list_runs_by_job(job_name)
    if not rows:
        raise HTTPException(404, "job not found")
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    return {"job_name": job_name, "counts": counts, "runs": rows}




@app.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str):
    folder = RUNS_DIR / run_id
    mp = folder/"metrics.json"
    if not mp.exists():
        raise HTTPException(404, "metrics not available")
    return JSONResponse(json.loads(mp.read_text()))




@app.get("/runs/{run_id}/logs")
def get_logs(run_id: str, follow: bool = False):
    logfile = RUNS_DIR / run_id / "logs.txt"
    if not logfile.exists():
        raise HTTPException(404, "logs not available")

    if not follow:
        return StreamingResponse(open(logfile, "rb"), media_type="text/plain")

    async def streamer():
        with open(logfile, "r") as f:
            f.seek(0, 2) # go to end
            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.25)
                    continue
                yield line
    return StreamingResponse(streamer(), media_type="text/plain")

