from __future__ import annotations
import json, subprocess, tempfile, shutil
from pathlib import Path
from typing import Dict, Any
from settings import RUNS_DIR
from logger import log_event
from engines.vllm_args import build_modal_payload
from engines.trtllm_args import build_modal_payload as build_trtllm_payload

def _fn_name_for(pool: str, count: int) -> str:
    base = pool.lower().replace("-", "")   # e.g. "A100-80GB" -> "a10080gb"
    suffix = "" if count == 1 else f"x{count}"
    return f"backend.runners.vllm_modal::bench_{base}{suffix}"
    
class ModalDriver:
    """
    Launches a Modal function via the Modal CLI and streams stdout to logs.
    Keeps the control plane simple; later you can switch to SDK for async and cancel.
    """
    def launch(self, run: Dict[str, Any]) -> str:
        # Return a pseudo job id for bookkeeping (CLI is blocking per run in this MVP)
        return f"cli-{run.get('run_id','unknown')}"

    def status(self, modal_job_id: str) -> str:
        return "UNKNOWN"

    def cancel(self, modal_job_id: str) -> None:
        # Not supported in CLI blocking mode (MVP)
        pass
    

        # --- replace execute_locally_and_collect(...) with this version ---
    def execute_locally_and_collect(self, run_id: str, spec: Dict[str, Any], config: Dict[str, Any]) -> str:
        # Ensure modal CLI exists
        if not shutil.which("modal"):
            raise RuntimeError("Modal CLI not found. Install with `pip install modal` and run `modal token new`.")

        engine = (spec.get("engine") or "vllm").lower()

        folder = RUNS_DIR / run_id
        folder.mkdir(parents=True, exist_ok=True)
        logs_path = folder / "logs.txt"
        metrics_path = folder / "metrics.json"

        # in ModalDriver.execute_locally_and_collect(...)
        if engine == "trtllm":
            payload = build_trtllm_payload(spec, config)
            tp = int(payload.get("tensor_parallel", 1))
            fn_by_pool_and_count = {
                ("B200", 1): "backend.runners.trtllm_modal::bench_b200",
                ("B200", 2): "backend.runners.trtllm_modal::bench_b200_tp2",
                ("B200", 4): "backend.runners.trtllm_modal::bench_b200_tp4",
                ("B200", 8): "backend.runners.trtllm_modal::bench_b200_tp8",
            }
            # If user didn’t specify num_gpus, infer from TP
            gpu_pool = spec.get("gpu_pool", "B200")
            num_gpus = int(spec.get("num_gpus", tp))
            fn = fn_by_pool_and_count.get((gpu_pool, num_gpus))
        else:
            payload = build_modal_payload(spec, config)
            fn_by_pool_and_count = {
                ("H100", 1): "backend.runners.vllm_modal::bench_h100",
                ("H100", 2): "backend.runners.vllm_modal::bench_h100x2",
                ("H100", 4): "backend.runners.vllm_modal::bench_h100x4",
                ("H200", 1): "backend.runners.vllm_modal::bench_h200",
                ("H200", 2): "backend.runners.vllm_modal::bench_h200x2",
                ("B200", 1): "backend.runners.vllm_modal::bench_b200",
                ("B200", 2): "backend.runners.vllm_modal::bench_b200x2",
                ("A100-80GB", 1): "backend.runners.vllm_modal::bench_a100",
                ("A100-80GB", 2): "backend.runners.vllm_modal::bench_a100x2",
            }

        gpu_pool = spec.get("gpu_pool", "H100")
        num_gpus = int(spec.get("num_gpus", 1))
        fn = fn_by_pool_and_count.get((gpu_pool, num_gpus))
        if not fn:
            avail = sorted({k for k in fn_by_pool_and_count.keys() if k[0] == gpu_pool})
            raise RuntimeError(f"No Modal function defined for pool={gpu_pool} with num_gpus={num_gpus}. Available: {avail}")

        # Call exactly the function in `fn`; do NOT prefix it again
        cmd = [
            "modal", "run", "-m",
            fn,
            "--args", json.dumps([payload]),   # function receives a single positional arg; we pass [payload]
        ]

        log_event("modal_cli_start", run_id=run_id, cmd=" ".join(cmd))
        proc = subprocess.Popen(cmd, cwd="..", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        last_metrics: Dict[str, Any] | None = None
        with logs_path.open("a") as logf:
            for line in proc.stdout:  # stream logs
                logf.write(line)
                logf.flush()
                try:
                    obj = json.loads(line.strip())
                    if obj.get("event") == "metrics":
                        # Expect a unified metrics dict (see runners below)
                        last_metrics = obj.get("data")
                except Exception:
                    pass

        rc = proc.wait()
        log_event("modal_cli_finish", run_id=run_id, returncode=rc)

        if last_metrics is None:
            last_metrics = {
                "model": spec.get("model"),
                "gpu": spec.get("gpu_pool"),
                "config": config,
                "throughput_tok_s": 0.0,
                "ttft_s": 0.0,
                "accuracy": None,
                "timestamp": None,
            }

        metrics_path.write_text(json.dumps(last_metrics, indent=2))
        return str(metrics_path)
