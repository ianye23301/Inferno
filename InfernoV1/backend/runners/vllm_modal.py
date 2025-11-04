# backend/runners/vllm_modal.py
from __future__ import annotations
import json, time, os
from datetime import datetime
import modal

APP_NAME = "inferno-vllm-bench-mock"
HF_SECRET_NAME = "hf-token"  # make sure this secret exists (see notes below)

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    # Pin NumPy < 2 to avoid binary-compat issues with TF-triggered deps.
    .pip_install("numpy==1.26.4")
    .pip_install(
        "torch==2.3.0",
        "vllm==0.5.0",
        "transformers==4.45.2",
        "accelerate==1.11.0",
        "pillow<11",
        "huggingface_hub>=0.24.0",
        "ray==2.51.1",
    )
    .env({
        # Keep TF out of transformers so it doesn't import the TF stack.
        "TRANSFORMERS_NO_TF": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "PYTHONNOUSERSITE": "1",
        # Do NOT put tokens here.
    })
)

HF_SECRET = modal.Secret.from_name(HF_SECRET_NAME)

def _get_hf_token() -> str:
    """Read HF token from either env var name."""
    tok = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError(
            "Missing Hugging Face token. Provide it via Modal secret "
            f"'{HF_SECRET_NAME}' with key HUGGINGFACE_HUB_TOKEN or HF_TOKEN."
        )
    return tok

def _login_hf_and_check():
    # Authenticate and verify access to the repo you’ll use.
    from huggingface_hub import login, whoami, model_info
    token = _get_hf_token()
    # Log in for the current process; no git credential noise.
    login(token=token, add_to_git_credential=False)
    print("HF whoami:", whoami())  # proves the token works

    # Ensure we can actually see the *exact* model you’ll load with vLLM.
    # Change this if you use a different repo.
    _ = model_info("meta-llama/Llama-3.1-8B-Instruct")
    print("HF access OK: meta-llama/Llama-3.1-8B-Instruct")

def _init_ray_with_hf_env():
    import ray
    token = _get_hf_token()
    # Some libs check one name or the other; set both for safety.
    env_vars = {
        "HUGGINGFACE_HUB_TOKEN": token,
        "HF_TOKEN": token,
        "TRANSFORMERS_NO_TF": os.environ.get("TRANSFORMERS_NO_TF", "1"),
        "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3"),
        "HF_HOME": os.environ.get("HF_HOME", "/root/.cache/huggingface"),
        "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1"),
    }
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": env_vars})

def _coerce_args(args):
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {}
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

    # Check token presence + HF access:
    print("Has HUGGINGFACE_HUB_TOKEN:", "HUGGINGFACE_HUB_TOKEN" in os.environ)
    print("Has HF_TOKEN:", "HF_TOKEN" in os.environ)

    _login_hf_and_check()

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

def _bench_impl(args):
    # Keep TF disabled inside the worker process too.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # Authenticate & propagate env to Ray workers BEFORE LLM creation.
    _login_hf_and_check()
    _init_ray_with_hf_env()

    from vllm import LLM, SamplingParams
    from transformers.utils.hub import cached_file

    args = _coerce_args(args)
    model_name = args.get("model", "meta-llama/Llama-3.1-8B-Instruct")
    cfg = args.get("config", {})
    batch_size = int(cfg.get("batch_size", 1))
    tp = int(cfg.get("tensor_parallel", 1))
    quant = cfg.get("quantization", "none")

    # Optional: enable fp8 path
    if quant == "fp8":
        os.environ["VLLM_FP8_ENABLED"] = "1"

    # Preflight: check we can fetch config.json with this process’ token.
    _ = cached_file(model_name, "config.json", token=_get_hf_token())
    print(f"Fetched {model_name}/config.json OK")

    prompts = ["The quick brown fox jumps over the lazy dog."] * batch_size
    sampling = SamplingParams(temperature=0.8, max_tokens=64)

    # --- TTFT (load + first token init)
    t0 = time.time()
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        # Optional: set download dir to keep cache stable across runs
        download_dir=os.environ.get("HF_HOME", "/root/.cache/huggingface"),
        trust_remote_code=False,
    )
    ttft = time.time() - t0

    # --- Throughput
    t1 = time.time()
    outputs = llm.generate(prompts, sampling)
    gen_time = time.time() - t1
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / gen_time if gen_time > 0 else 0.0

    metrics = {
        "model": model_name,
        "config": cfg,
        "throughput_tok_s": throughput,
        "ttft_s": ttft,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    print(json.dumps({"event": "metrics", "data": metrics}))
    return metrics
