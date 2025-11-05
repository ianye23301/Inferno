# backend/runners/vllm_modal.py
from __future__ import annotations
import json, time, os
from datetime import datetime
import modal


APP_NAME = "inferno-vllm-bench-mock"
HF_SECRET_NAME = "hf-token"  # Secret must define HUGGINGFACE_HUB_TOKEN or HF_TOKEN

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .pip_install("numpy==1.26.4")  # keep <2 to dodge ABI drama
    .pip_install(
        # ✅ Match vLLM 0.6.2's exact requirement:
        "torch==2.4.0",               # cu121 wheel
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

def _mk_prompt(base_tokens: int) -> str:
    base = ("Generate a runnable Python terminal game with a main() and replay loop. "
            "Keep commentary minimal and use standard library only. ")
    filler = ("Provide clean structure and deterministic behavior. " * 200)
    need_chars = max(0, base_tokens * 4 - len(base))  # ~4 chars/token heuristic
    return base + filler[:need_chars]

def _get_hf_token() -> str:
    """Read HF token from either env var name and mirror it to both names."""
    tok = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError(
            f"Missing Hugging Face token. Provide it via Modal secret '{HF_SECRET_NAME}' "
            "with key HUGGINGFACE_HUB_TOKEN or HF_TOKEN."
        )
    # Mirror to both env names so every lib (and child processes) can find it.
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", tok)
    os.environ.setdefault("HF_TOKEN", tok)
    return tok

def _check_hf_access(model_id: str):
    """Lightweight check that the token works and can see the model."""
    from huggingface_hub import whoami, model_info
    tok = _get_hf_token()
    print("HF whoami:", whoami(token=tok))
    # Throws if gated/no access:
    _ = model_info(model_id, token=tok)
    print(f"HF access OK: {model_id}")

def _coerce_args(args):
    # Accept dict directly
    if isinstance(args, dict):
        return args

    # If Modal delivers a list, unwrap it
    if isinstance(args, list):
        if len(args) == 1 and isinstance(args[0], dict):
            return args[0]
        # Handle double-nested [[payload]]
        if len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 1 and isinstance(args[0][0], dict):
            return args[0][0]
        return {}

    # If stringified JSON, parse and unwrap
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


@app.function(image=image, gpu="H100", timeout=60*30, secrets=[HF_SECRET])
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

    # Verify token + model access (adjust model if you use a different one)
    _check_hf_access("meta-llama/Llama-3.1-8B-Instruct")

# --- H100 variants ---
@app.function(image=image, gpu="H100", timeout=60*30, secrets=[HF_SECRET])
def bench_h100(args): return _bench_impl(args)

@app.function(image=image, gpu="H100:2", timeout=60*30, secrets=[HF_SECRET])
def bench_h100x2(args): return _bench_impl(args)

@app.function(image=image, gpu="H100:4", timeout=60*30, secrets=[HF_SECRET])
def bench_h100x4(args): return _bench_impl(args)

# --- H200 variants (use string format, not mgpu.H200) ---
@app.function(image=image, gpu="H200", timeout=60*30, secrets=[HF_SECRET])
def bench_h200(args): return _bench_impl(args)

@app.function(image=image, gpu="H200:2", timeout=60*30, secrets=[HF_SECRET])
def bench_h200x2(args): return _bench_impl(args)

# --- B200 variants (use string format, not mgpu.B200) ---
@app.function(image=image, gpu="B200", timeout=60*30, secrets=[HF_SECRET])
def bench_b200(args): return _bench_impl(args)

@app.function(image=image, gpu="B200:2", timeout=60*30, secrets=[HF_SECRET])
def bench_b200x2(args): return _bench_impl(args)

# --- A100-80GB variants ---
@app.function(image=image, gpu="A100", timeout=60*30, secrets=[HF_SECRET])
def bench_a100(args): return _bench_impl(args)

@app.function(image=image, gpu="A100:2", timeout=60*30, secrets=[HF_SECRET])
def bench_a100x2(args): return _bench_impl(args)

    
def _bench_impl(args):
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    from vllm import LLM, SamplingParams
    import json, pathlib

    args = _coerce_args(args)
    model_name = args.get("model", "meta-llama/Llama-3.1-8B-Instruct")
    cfg        = args.get("config", {}) or {}
    run_env    = args.get("env", {}) or {}
    num_gpus   = int(args.get("num_gpus", 1))
    dataset    = args.get("dataset")  # Optional evaluation dataset

    for k, v in run_env.items():
        os.environ[str(k)] = str(v)

    batch_size = int(cfg.get("batch_size", 1))
    tp         = int(cfg.get("tensor_parallel", 1))
    quant      = cfg.get("quantization", "none")

    # Verify the container has the requested GPU count (sanity check).
    try:
        import torch
        vis = torch.cuda.device_count()
        if vis != num_gpus:
            print(f"[inferno] WARNING: container sees {vis} GPUs but num_gpus={num_gpus}. Proceeding.")
    except Exception as _:
        pass

    # Backend selection: mp for 1 GPU, ray for >1 GPUs
    if num_gpus <= 1:
        os.environ["VLLM_USE_RAY"] = "0"
        dist_backend = "mp"
        # Keep TP=1 on single GPU
        if tp != 1:
            print(f"[inferno] For single-GPU container, forcing tensor_parallel=1 (was {tp}).")
            tp = 1
    else:
        # multi-GPU: use ray (default). Ensure env doesn't force-disable it.
        os.environ.pop("VLLM_USE_RAY", None)
        dist_backend = "ray"

    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tp,
        distributed_executor_backend=dist_backend,
        download_dir=os.environ.get("HF_HOME", "/root/.cache/huggingface"),
        trust_remote_code=False,
        # If chunked prefill causes issues, uncomment:
        # enable_chunked_prefill=False,
    )
    if quant and quant != "none":
        llm_kwargs["quantization"] = quant

    input_tokens = int(cfg.get("input_tokens", 250))
    max_new_tokens = int(cfg.get("max_new_tokens", 2048))
    temperature = float(cfg.get("temperature", 0.7))
    top_p = float(cfg.get("top_p", 0.9))

    single_prompt = _mk_prompt(input_tokens)
    prompts = [single_prompt] * batch_size

    # Build LLM (measure model init separately, not TTFT)
    t_init0 = time.time()
    llm = LLM(**llm_kwargs)
    init_time = time.time() - t_init0  # optional debug

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p
    )

    # Non-streaming generate; use per-request metrics for TTFT if available
    t_start = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    t_end = time.time()

    # Aggregate metrics
        # Aggregate metrics
    total_new_tokens = 0
    ttft_values = []

    def _norm_seconds(x: float) -> float:
        # normalize ms/ns if someone logged those
        if x > 1e6:   # likely ns
            return x / 1e9
        if x > 1e3:   # likely ms
            return x / 1e3
        return x

    for o in outputs:
        # count generated tokens (first hypothesis)
        if o.outputs and len(o.outputs) > 0:
            total_new_tokens += len(o.outputs[0].token_ids)

        m = getattr(o, "metrics", None)
        if m is None:
            continue

        # Try direct latency first
        ft = None
        for attr in ("first_token_latency", "first_token_latency_s"):
            v = getattr(m, attr, None) if not isinstance(m, dict) else m.get(attr)
            if v is not None:
                ft = _norm_seconds(float(v))
                break

        # Fallback: derive from absolute times
        if ft is None:
            # vLLM often has these as seconds; normalize just in case
            if isinstance(m, dict):
                ft_time  = m.get("first_token_time")
                req_time = m.get("request_start_time")
            else:
                ft_time  = getattr(m, "first_token_time", None)
                req_time = getattr(m, "request_start_time", None)

            if ft_time is not None and req_time is not None:
                ft = _norm_seconds(float(ft_time)) - _norm_seconds(float(req_time))

        if ft is not None and ft >= 0:
            ttft_values.append(ft)

    wall = max(1e-6, t_end - t_start)

    if ttft_values:
        ttft_s = sum(ttft_values) / len(ttft_values)
        # Clamp pathological cases (e.g., derived > wall due to clock mismatch)
        if ttft_s >= wall:
            ttft_s = wall * 0.9
        decode_time = max(1e-6, wall - ttft_s)
    else:
        # conservative fallback: split wall time by input/share
        approx_prefill = wall * (input_tokens / max(1.0, input_tokens + total_new_tokens))
        ttft_s = min(approx_prefill, wall * 0.9)
        decode_time = max(1e-6, wall - ttft_s)

    throughput = (total_new_tokens / decode_time) if total_new_tokens > 0 else 0.0

    metrics = {
        "model": model_name,
        "config": cfg | {
            "batch_size": batch_size,
            "tensor_parallel": tp,
            "quantization": quant,
            "input_tokens": input_tokens,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "throughput_tok_s": throughput,
        "ttft_s": ttft_s,
        "accuracy": None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        # "model_load_s": init_time,
    }
    print(json.dumps({"event": "metrics", "data": metrics}))
    return metrics