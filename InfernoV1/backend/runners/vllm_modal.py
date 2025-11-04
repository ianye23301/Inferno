# backend/runners/vllm_modal.py

from __future__ import annotations
import time, json
from datetime import datetime
import modal


app = modal.App("inferno-vllm-bench-mock")
# Make sure these are set BEFORE importing vllm/transformers.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .pip_install("numpy==1.26.4")          # install first to avoid upgrades to 2.x
    .pip_install(
        "torch==2.3.0",                    # required by vllm==0.5.0
        "vllm==0.5.0",
        "transformers==4.45.2",
        "accelerate==1.11.0",
        "pillow<11",
    )
    .env({
        "TRANSFORMERS_NO_TF": "1",         # stop TF imports from transformers
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "PYTHONNOUSERSITE": "1",           # ignore user site-packages
    })
)


HF_SECRET = modal.Secret.from_name("hf-token")

@app.function(image=image, gpu="A100", timeout=60*30, secrets=[HF_SECRET])
def bench_a100(args):
    return _bench_impl(args)

@app.function(image=image, gpu="H100", timeout=60*30, secrets=[HF_SECRET])
def bench_h100(args):
    return _bench_impl(args)

@app.function(image=image, gpu="H200", timeout=60*30, secrets=[HF_SECRET])
def bench_h200(args):
    return _bench_impl(args)

@app.function(image=image, gpu="B200", timeout=60*30, secrets=[HF_SECRET])
def bench_b200(args):
    return _bench_impl(args)

@app.function(image=image, gpu="H100", secrets=[HF_SECRET])
def env_check():
    import os, torch, numpy, transformers, vllm
    from huggingface_hub import whoami, login

    print("HUGGINGFACE_HUB_TOKEN in env:", "HUGGINGFACE_HUB_TOKEN" in os.environ)
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("No HUGGINGFACE_HUB_TOKEN in environment.")

    # optional but nice: establish an auth cache for the process
    login(token=token, add_to_git_credential=False)
    print("HF whoami:", whoami())  # no args needed after login
    print("PYTHONNOUSERSITE=", os.environ.get("PYTHONNOUSERSITE"))
    print("TRANSFORMERS_NO_TF=", os.environ.get("TRANSFORMERS_NO_TF"))
    print("torch", torch.__version__)
    print("numpy", numpy.__version__)
    print("transformers", transformers.__version__)
    print("vllm", vllm.__version__)



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
    import os
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # ⬇️ Import vLLM only inside the worker process (remote GPU)
    from vllm import LLM, SamplingParams

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