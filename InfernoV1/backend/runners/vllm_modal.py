# backend/runners/vllm_modal.py
from __future__ import annotations
import json, time, os
from datetime import datetime
import modal
from modal import gpu as mgpu


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
        # ✅ Match vLLM 0.6.2’s exact requirement:
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

@app.function(image=image, gpu=mgpu.H100(count=2), timeout=60*30, secrets=[HF_SECRET])
def bench_h100x2(args): return _bench_impl(args)

@app.function(image=image, gpu=mgpu.H100(count=4), timeout=60*30, secrets=[HF_SECRET])
def bench_h100x4(args): return _bench_impl(args)

# --- H200 variants ---
@app.function(image=image, gpu="H200", timeout=60*30, secrets=[HF_SECRET])
def bench_h200(args): return _bench_impl(args)

@app.function(image=image, gpu=mgpu.H200(count=2), timeout=60*30, secrets=[HF_SECRET])
def bench_h200x2(args): return _bench_impl(args)

# --- B200 variants ---
@app.function(image=image, gpu="B200", timeout=60*30, secrets=[HF_SECRET])
def bench_b200(args): return _bench_impl(args)

@app.function(image=image, gpu=mgpu.B200(count=2), timeout=60*30, secrets=[HF_SECRET])
def bench_b200x2(args): return _bench_impl(args)

# --- A100-80GB variants ---
@app.function(image=image, gpu="A100", timeout=60*30, secrets=[HF_SECRET])
def bench_a100(args): return _bench_impl(args)

@app.function(image=image, gpu=mgpu.A100(count=2, memory="80GB"), timeout=60*30, secrets=[HF_SECRET])
def bench_a100x2(args): return _bench_impl(args)

    
def _bench_impl(args):
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    from vllm import LLM, SamplingParams
    from transformers.utils.hub import cached_file
    import json, pathlib  # <- unconditionally here, not inside an if

    args = _coerce_args(args)
    model_name = args.get("model", "meta-llama/Llama-3.1-8B-Instruct")
    cfg        = args.get("config", {}) or {}
    run_env    = args.get("env", {}) or {}
    num_gpus   = int(args.get("num_gpus", 1))

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
        # multi-GPU: use ray (default). Ensure env doesn’t force-disable it.
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

    t0  = time.time()
    llm = LLM(**llm_kwargs)
    ttft = time.time() - t0

    t1 = time.time()
    outputs = llm.generate(prompts, sampling)
    gen_time = time.time() - t1
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / gen_time if gen_time > 0 else 0.0

    # --- Minimal opt-in accuracy harness (substring EM)
    # Accept either:
    #  - dataset = [{"prompt": "...", "answer_substr": "..."}]
    #  - dataset = path to a JSONL with those fields
    accuracy = None
    if dataset:
        import json, pathlib
        def _load_ds(ds):
            if isinstance(ds, list): return ds
            p = pathlib.Path(str(ds))
            if p.exists():
                return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            return []
        eval_recs = _load_ds(dataset)[:32]  # small, fast sanity-check eval
        if eval_recs:
            eval_prompts = [r["prompt"] for r in eval_recs]
            eval_outs = llm.generate(eval_prompts, SamplingParams(temperature=0.0, max_tokens=64))
            hits = 0
            for r, out in zip(eval_recs, eval_outs):
                text = out.outputs[0].text
                if r.get("answer_substr") and r["answer_substr"].lower() in text.lower():
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
