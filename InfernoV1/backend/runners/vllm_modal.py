# backend/runners/vllm_modal.py
from __future__ import annotations
import os, json, time
from datetime import datetime
from typing import Any, Dict

import modal

APP_NAME = "inferno-vllm-bench-mock"
HF_SECRET_NAME = "hf-token"  # Secret must define HUGGINGFACE_HUB_TOKEN or HF_TOKEN

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .pip_install("numpy==1.26.4")  # keep <2 for ABI stability
    .pip_install(
        # vLLM 0.6.x + CUDA 12.1 compatible stack
        "torch==2.4.0",
        "vllm==0.6.2",
        "transformers==4.45.2",
        "accelerate==1.11.0",
        "pillow<11",
        "huggingface_hub>=0.24.0",
        "ray==2.51.1",
    )
    .env({
        "TRANSFORMERS_NO_TF": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "PYTHONNOUSERSITE": "1",
    })
)

HF_SECRET = modal.Secret.from_name(HF_SECRET_NAME)


# ---------------------------
# Helpers
# ---------------------------

def _get_hf_token() -> str:
    tok = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError(
            f"Missing Hugging Face token. Provide it via Modal secret '{HF_SECRET_NAME}' "
            "with key HUGGINGFACE_HUB_TOKEN or HF_TOKEN."
        )
    # mirror for downstream libs
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", tok)
    os.environ.setdefault("HF_TOKEN", tok)
    return tok


def _check_hf_access(model_id: str):
    from huggingface_hub import whoami, model_info
    tok = _get_hf_token()
    try:
        print("HF whoami:", whoami(token=tok))
    except Exception as e:
        print("HF whoami() failed:", repr(e))
    _ = model_info(model_id, token=tok)
    print(f"HF access OK: {model_id}")


def _coerce_args(args: Any) -> Dict[str, Any]:
    # Accept dict directly
    if isinstance(args, dict):
        return args
    # Modal CLI --args '[{...}]'
    if isinstance(args, list):
        if len(args) == 1 and isinstance(args[0], dict):
            return args[0]
        if len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 1 and isinstance(args[0][0], dict):
            return args[0][0]
        return {}
    if isinstance(args, str):
        try:
            obj = json.loads(args)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
                return obj[0]
        except Exception:
            pass
    return {}


# ---------------------------
# Sanity / env check
# ---------------------------

@app.function(image=image, gpu="H100:1", timeout=60*30, secrets=[HF_SECRET])
def env_check():
    import torch, numpy, transformers, vllm
    print("PYTHONNOUSERSITE=", os.environ.get("PYTHONNOUSERSITE"))
    print("TRANSFORMERS_NO_TF=", os.environ.get("TRANSFORMERS_NO_TF"))
    print("torch", torch.__version__)
    print("numpy", numpy.__version__)
    print("transformers", transformers.__version__)
    print("vllm", vllm.__version__)

    # Token presence
    print("Has HUGGINGFACE_HUB_TOKEN:", "HUGGINGFACE_HUB_TOKEN" in os.environ)
    print("Has HF_TOKEN:", "HF_TOKEN" in os.environ)

    _check_hf_access("meta-llama/Llama-3.1-8B-Instruct")


# ---------------------------
# Core bench impl
# ---------------------------

def _bench_impl(args: Any):
    # env hygiene
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    from vllm import LLM, SamplingParams
    from transformers.utils.hub import cached_file
    import pathlib

    payload = _coerce_args(args)
    model_name = payload.get("model", "meta-llama/Llama-3.1-8B-Instruct")
    cfg        = payload.get("config", {}) or {}
    run_env    = payload.get("env", {}) or {}
    num_gpus   = int(payload.get("num_gpus", 1))
    dataset    = payload.get("dataset")  # optional: list or path to jsonl

    # apply per-run env
    for k, v in run_env.items():
        os.environ[str(k)] = str(v)

    batch_size = int(cfg.get("batch_size", 1))
    tp         = int(cfg.get("tensor_parallel", 1))
    quant      = cfg.get("quantization", "none")

    # Ensure HF token exists (and warms cache access)
    tok = _get_hf_token()
    _ = cached_file(model_name, "config.json", token=tok)
    print(f"Fetched {model_name}/config.json OK")

    # Single vs multi-gpu backends
    if num_gpus <= 1:
        os.environ["VLLM_USE_RAY"] = "0"
        dist_backend = "mp"
        if tp != 1:
            print(f"[inferno] For single-GPU container, forcing tensor_parallel=1 (was {tp}).")
            tp = 1
    else:
        os.environ.pop("VLLM_USE_RAY", None)
        dist_backend = "ray"

    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tp,
        distributed_executor_backend=dist_backend,
        download_dir=os.environ.get("HF_HOME", "/root/.cache/huggingface"),
        trust_remote_code=False,
        # If you hit issues with long contexts, uncomment:
        # enable_chunked_prefill=False,
    )
    if quant and quant != "none":
        llm_kwargs["quantization"] = quant

    # --- TTFT
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    ttft = time.time() - t0

    # --- Throughput
    prompts = ["The quick brown fox jumps over the lazy dog."] * batch_size
    sampling = SamplingParams(temperature=0.8, max_tokens=64)

    t1 = time.time()
    outputs = llm.generate(prompts, sampling)
    gen_time = time.time() - t1
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / gen_time if gen_time > 0 else 0.0

    # --- Minimal optional accuracy (substring EM)
    accuracy = None
    if dataset:
        def _load_ds(ds):
            if isinstance(ds, list):
                return ds
            p = pathlib.Path(str(ds))
            if p.exists():
                return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            return []
        eval_recs = _load_ds(dataset)[:32]
        if eval_recs:
            eval_prompts = [r["prompt"] for r in eval_recs]
            eval_outs = llm.generate(eval_prompts, SamplingParams(temperature=0.0, max_tokens=64))
            hits = 0
            for r, out in zip(eval_recs, eval_outs):
                text = out.outputs[0].text
                ans = (r.get("answer_substr") or "").strip()
                if ans and ans.lower() in text.lower():
                    hits += 1
            accuracy = hits / len(eval_recs)

    metrics = {
        "model": model_name,
        "config": cfg,
        "throughput_tok_s": throughput,
        "ttft_s": ttft,
        "accuracy": accuracy,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    print(json.dumps({"event": "metrics", "data": metrics}))
    return metrics


# ---------------------------
# Dynamic function registration (new GPU string syntax)
# ---------------------------

_GPU_POOLS = ["H100", "H200", "B200", "A100-80GB"]
_COUNTS_BY_POOL = {
    "H100": [1, 2, 4],
    "H200": [1, 2],
    "B200": [1, 2],
    "A100-80GB": [1, 2],
}

def _register_modal_fn(name: str, gpu_str: str):
    @app.function(image=image, gpu=gpu_str, timeout=60*30, secrets=[HF_SECRET], serialized=True)
    def _runner(args):
        return _bench_impl(args)
    _runner.__name__ = name
    globals()[name] = _runner
    return _runner

# Generate: bench_h100, bench_h100x2, bench_h100x4, bench_h200, bench_h200x2, ...
for pool in _GPU_POOLS:
    for c in _COUNTS_BY_POOL[pool]:
        gpu_str = f"{pool}:{c}"
        suffix = "" if c == 1 else f"x{c}"
        fn_name = f"bench_{pool.lower().replace('-', '')}{suffix}"
        _register_modal_fn(fn_name, gpu_str)
