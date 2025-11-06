from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, List, Tuple
from uuid import uuid4
from pathlib import Path
import json, asyncio
from models import JobSpec
import sweeper, registry
from scheduler import start_background_scheduler
from settings import RUNS_DIR, JOBS_DIR
import math
from datetime import datetime

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



def _choose_best(job_name: str, metric: str, mode: str = "max") -> Tuple[dict, dict]:
    """
    Returns (run_row, metrics) for the best run in a job according to:
      - primary: metric (+max or +min)
      - tie-breakers: higher accuracy, then lower ttft_s
    """
    best_row, best_m = None, None

    for row, m in registry.iter_job_metrics(job_name):
        # Skip incomplete metric rows
        if metric not in m or m[metric] is None:
            continue
        val = m[metric]
        # normalize for comparison
        def better(cur, best):
            if best is None:
                return True
            # primary
            if mode == "max":
                if cur[metric] != best[metric]:
                    return cur[metric] > best[metric]
            else:
                if cur[metric] != best[metric]:
                    return cur[metric] < best[metric]
            # tie-breakers: accuracy desc, ttft asc
            ca, ba = (cur.get("accuracy"), best.get("accuracy"))
            if (ca is not None) or (ba is not None):
                if (ca or -1) != (ba or -1):
                    return (ca or -1) > (ba or -1)
            ctt, btt = (cur.get("ttft_s"), best.get("ttft_s"))
            if (ctt is not None) and (btt is not None) and ctt != btt:
                return ctt < btt
            return False

        if better(m, best_m):
            best_row, best_m = row, m

    if not best_row:
        raise HTTPException(404, f"No metrics found for job '{job_name}' with metric '{metric}'")
    return best_row, best_m


@app.get("/jobs/{job_name}/best")
def get_job_best(
    job_name: str, 
    metric: str = "throughput_tok_s", 
    mode: str = "max",
    return_all: bool = False,
    filter_passed: bool = False
):
    """
    Get best run or all runs for a job.
    
    Args:
        metric: Metric to optimize for
        mode: 'max' or 'min'
        return_all: If True, return all runs sorted by metric instead of just the best
        filter_passed: If True, only consider runs where accuracy > 0 (code passed eval)
    """
    if not return_all:
        # Original behavior: return only the best
        # Apply filter if requested
        candidates = []
        for row, m in registry.iter_job_metrics(job_name):
            if metric not in m or m[metric] is None:
                continue
            if filter_passed and (m.get("accuracy") is None or m.get("accuracy") <= 0):
                continue
            candidates.append((row, m))
        
        if not candidates:
            raise HTTPException(404, f"No runs found for job '{job_name}' with metric '{metric}'" + 
                              (" that passed evaluation" if filter_passed else ""))
        
        # Find best among candidates
        best_row, best_m = None, None
        for row, m in candidates:
            def better(cur, best):
                if best is None:
                    return True
                if mode == "max":
                    if cur[metric] != best[metric]:
                        return cur[metric] > best[metric]
                else:
                    if cur[metric] != best[metric]:
                        return cur[metric] < best[metric]
                # Tie-breakers
                ca, ba = (cur.get("accuracy"), best.get("accuracy"))
                if (ca is not None) or (ba is not None):
                    if (ca or -1) != (ba or -1):
                        return (ca or -1) > (ba or -1)
                ctt, btt = (cur.get("ttft_avg_s") or cur.get("ttft_s"), 
                           best.get("ttft_avg_s") or best.get("ttft_s"))
                if (ctt is not None) and (btt is not None) and ctt != btt:
                    return ctt < btt
                return False

            if better(m, best_m):
                best_row, best_m = row, m

        payload = {
            "job_name": job_name,
            "metric": metric,
            "mode": mode,
            "filter_passed": filter_passed,
            "run_id": best_row["run_id"],
            "gpu_pool": best_row["gpu_pool"],
            "config": best_m.get("config", {}),
            "model": best_m.get("model"),
            "metrics": {
                "throughput_tok_s": best_m.get("throughput_tok_s"),
                "prefill_tok_s": best_m.get("prefill_tok_s"),
                "overall_tok_s": best_m.get("overall_tok_s"),
                "ttft_avg_s": best_m.get("ttft_avg_s") or best_m.get("ttft_s"),
                "ttft_p50_s": best_m.get("ttft_p50_s"),
                "ttft_p95_s": best_m.get("ttft_p95_s"),
                "decode_time_s": best_m.get("decode_time_s"),
                "tokens_out_total": best_m.get("tokens_out_total"),
                "accuracy": best_m.get("accuracy"),
                "timestamp": best_m.get("timestamp"),
            },
            "paths": {
                "metrics_path": best_row.get("metrics_path"),
                "logs_path": best_row.get("logs_path"),
            },
            "state": best_row.get("state"),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        out_path = registry.write_job_best(job_name, metric, payload)
        return payload
    
    else:
        # Return all runs sorted by metric
        results = []
        
        for row, m in registry.iter_job_metrics(job_name):
            if row.get("state") != "COMPLETED":
                continue
            
            # Skip if metric is missing
            if metric not in m or m[metric] is None:
                continue
            
            # Apply passed filter
            if filter_passed and (m.get("accuracy") is None or m.get("accuracy") <= 0):
                continue
                
            results.append({
                "run_id": row["run_id"],
                "gpu_pool": row["gpu_pool"],
                "config": m.get("config", {}),
                "model": m.get("model"),
                "metrics": {
                    "throughput_tok_s": m.get("throughput_tok_s"),
                    "prefill_tok_s": m.get("prefill_tok_s"),
                    "overall_tok_s": m.get("overall_tok_s"),
                    "ttft_avg_s": m.get("ttft_avg_s") or m.get("ttft_s"),
                    "ttft_p50_s": m.get("ttft_p50_s"),
                    "ttft_p95_s": m.get("ttft_p95_s"),
                    "decode_time_s": m.get("decode_time_s"),
                    "tokens_out_total": m.get("tokens_out_total"),
                    "accuracy": m.get("accuracy"),
                    "timestamp": m.get("timestamp"),
                },
                "paths": {
                    "metrics_path": row.get("metrics_path"),
                    "logs_path": row.get("logs_path"),
                },
                "state": row.get("state"),
            })
        
        if not results:
            raise HTTPException(404, f"No completed runs with metric '{metric}' found for job '{job_name}'" +
                              (" that passed evaluation" if filter_passed else ""))
        
        # Sort by the specified metric
        reverse = (mode == "max")
        results.sort(key=lambda x: x["metrics"].get(metric) or 0, reverse=reverse)
        
        return {
            "job_name": job_name,
            "metric": metric,
            "mode": mode,
            "filter_passed": filter_passed,
            "count": len(results),
            "runs": results
        }