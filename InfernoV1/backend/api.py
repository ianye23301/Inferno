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

# --- helpers (put near your imports) ---
ALIASES_TO_CANON = {
    "throughput_tok_s": "overall_tok_s",
    "output_tok_s": "decode_tok_s",
    "decode_time_s": "decode_time_total_s",
    "ttft_s": "ttft_avg_s",
    "time_to_first_token_s": "ttft_avg_s",
}

# given a metric name from the client, return the canonical name you store/emit
def canon_name(name: str) -> str:
    return ALIASES_TO_CANON.get(name, name)

# resolve a metric value from a row's metrics dict with alias fallback
def get_metric_value(metrics: dict, name: str):
    # exact hit
    v = metrics.get(name)
    if v is not None:
        return v
    # alias fallback
    alt = ALIASES_TO_CANON.get(name)
    if alt is not None:
        return metrics.get(alt)
    return None

# make a copy of metrics that includes only canon keys and adds computed/alias fields if you want
def project_metrics(m: dict) -> dict:
    # prefer canon keys; compute a minimal compat set
    out = {
        "decode_tok_s": m.get("decode_tok_s"),
        "prefill_tok_s": m.get("prefill_tok_s"),
        "overall_tok_s": m.get("overall_tok_s"),
        "ttft_avg_s": m.get("ttft_avg_s") or m.get("ttft_s") or m.get("time_to_first_token_s"),
        "ttft_p50_s": m.get("ttft_p50_s"),
        "ttft_p95_s": m.get("ttft_p95_s"),
        "tpot_avg_s": m.get("tpot_avg_s"),
        "decode_time_total_s": m.get("decode_time_total_s") or m.get("decode_time_s"),
        "tokens_out_total": m.get("tokens_out_total"),
        "accuracy": m.get("accuracy"),
        "timestamp": m.get("timestamp"),
    }
    # optional: include legacy fields for downstreams that still expect them
    if out["overall_tok_s"] is not None and "throughput_tok_s" not in m:
        out["throughput_tok_s"] = out["overall_tok_s"]
    if out["decode_tok_s"] is not None and "output_tok_s" not in m:
        out["output_tok_s"] = out["decode_tok_s"]
    if out["decode_time_total_s"] is not None and "decode_time_s" not in m:
        out["decode_time_s"] = out["decode_time_total_s"]
    return out



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
    metric: str = "decode_tok_s", 
    mode: str = "max",
    return_all: bool = False,
    filter_passed: bool = False
):
    metric = canon_name(metric)
    if mode not in ("max", "min"):
        raise HTTPException(400, "mode must be 'max' or 'min'")

    if not return_all:
        candidates = []
        for row, m in registry.iter_job_metrics(job_name):
            val = get_metric_value(m, metric)
            if val is None:
                continue
            if filter_passed and (m.get("accuracy") is None or m.get("accuracy") <= 0):
                continue
            candidates.append((row, m, val))

        if not candidates:
            raise HTTPException(
                404,
                f"No runs found for job '{job_name}' with metric '{metric}'" +
                (" that passed evaluation" if filter_passed else "")
            )

        best_row, best_m, best_val = None, None, None

        def better(cur_m, cur_val, best_m, best_val):
            if best_val is None:
                return True
            if mode == "max":
                if cur_val != best_val:
                    return cur_val > best_val
            else:
                if cur_val != best_val:
                    return cur_val < best_val
            # tie-breakers
            ca, ba = (cur_m.get("accuracy"), best_m.get("accuracy"))
            if (ca is not None) or (ba is not None):
                ca_ = float("-inf") if ca is None else ca
                ba_ = float("-inf") if ba is None else ba
                if ca_ != ba_:
                    return ca_ > ba_
            ctt = cur_m.get("ttft_avg_s") or cur_m.get("ttft_s") or cur_m.get("time_to_first_token_s")
            btt = best_m.get("ttft_avg_s") or best_m.get("ttft_s") or best_m.get("time_to_first_token_s")
            if (ctt is not None) and (btt is not None) and ctt != btt:
                return ctt < btt
            return False

        for row, m, val in candidates:
            if better(m, val, best_m, best_val):
                best_row, best_m, best_val = row, m, val

        payload = {
            "job_name": job_name,
            "metric": metric,
            "mode": mode,
            "filter_passed": filter_passed,
            "run_id": best_row["run_id"],
            "gpu_pool": best_row["gpu_pool"],
            "config": best_m.get("config", {}),
            "model": best_m.get("model"),
            "metrics": project_metrics(best_m),
            "paths": {
                "metrics_path": best_row.get("metrics_path"),
                "logs_path": best_row.get("logs_path"),
            },
            "state": best_row.get("state"),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        registry.write_job_best(job_name, metric, payload)
        return payload

    # return_all branch
    results = []
    for row, m in registry.iter_job_metrics(job_name):
        if row.get("state") != "COMPLETED":
            continue
        val = get_metric_value(m, metric)
        if val is None:
            continue
        if filter_passed and (m.get("accuracy") is None or m.get("accuracy") <= 0):
            continue

        results.append({
            "run_id": row["run_id"],
            "gpu_pool": row["gpu_pool"],
            "config": m.get("config", {}),
            "model": m.get("model"),
            "metrics": project_metrics(m),
            "paths": {
                "metrics_path": row.get("metrics_path"),
                "logs_path": row.get("logs_path"),
            },
            "state": row.get("state"),
        })

    if not results:
        raise HTTPException(
            404,
            f"No completed runs with metric '{metric}' found for job '{job_name}'" +
            (" that passed evaluation" if filter_passed else "")
        )

    reverse = (mode == "max")
    results.sort(
        key=lambda x: get_metric_value(x["metrics"], metric) or float("-inf"),
        reverse=reverse
    )

    return {
        "job_name": job_name,
        "metric": metric,
        "mode": mode,
        "filter_passed": filter_passed,
        "count": len(results),
        "runs": results
    }
