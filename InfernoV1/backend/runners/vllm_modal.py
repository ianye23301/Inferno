# backend/runners/vllm_modal.py

from __future__ import annotations
import time, json, os
from datetime import datetime
import modal
from vllm import LLM, SamplingParams

app = modal.App("inferno-vllm-bench-mock")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .pip_install(
        # match vLLM 0.5.0 requirements
        "torch==2.3.0",
        "vllm==0.5.0",
        "transformers",     # 4.5x is fine
        "accelerate",
        "numpy"
    )
)

@app.function(image=image, gpu="A100", timeout=60*30)
def bench_a100(args):
    return _bench_impl(args)

@app.function(image=image, gpu="H100", timeout=60*30)
def bench_h100(args):
    return _bench_impl(args)

@app.function(image=image, gpu="H200", timeout=60*30)
def bench_h200(args):
    return _bench_impl(args)

@app.function(image=image, gpu="B200", timeout=60*30)
def bench_b200(args):
    return _bench_impl(args)


def _coerce_args(args):
    """Accept dict or JSON string from CLI and return a dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            # Fall back to empty spec if someone passed a non-JSON string
            return {}
    # Unexpected type: also fall back
    return {}


def _bench_impl(args):
    args = _coerce_args(args)

    model_name = args.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
    gpu = args.get("gpu_pool", "H100")
    cfg = args.get("config", {})

    batch_size = int(cfg.get("batch_size", 1))
    tp = int(cfg.get("tensor_parallel", 1))
    quant = cfg.get("quantization", "none")

    # Optional: configure quantization / fp8
    if quant == "fp8":
        os.environ["VLLM_FP8_ENABLED"] = "1"

    prompts = ["The quick brown fox jumps over the lazy dog."] * batch_size
    sampling = SamplingParams(temperature=0.8, max_tokens=64)

    # --- Benchmark TTFT (load + first token)
    start_load = time.time()
    llm = LLM(model=model_name, tensor_parallel_size=tp)
    ttft = time.time() - start_load

    # --- Benchmark throughput
    start_gen = time.time()
    outputs = llm.generate(prompts, sampling)
    total_time = time.time() - start_gen
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / total_time

    metrics = {
        "model": model_name,
        "gpu": gpu,
        "config": cfg,
        "throughput_tok_s": throughput,
        "ttft_s": ttft,
        "accuracy": None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    print(json.dumps({"event": "metrics", "data": metrics}))
    return metrics