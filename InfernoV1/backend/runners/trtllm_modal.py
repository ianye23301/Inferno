import modal, json, time
from pathlib import Path

APP_NAME = "inferno-trtllm-qwen"
HF_SECRET_NAME = "hf-token"
ENGINE_VOL = "trtllm-engines"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(ENGINE_VOL, create_if_missing=True)

image = modal.Image.from_registry("nvcr.io/nvidia/tensorrt-llm/release:1.0.0")

def _coerce_args(args):
    """Handle args being passed as list, dict, or string"""
    # If it's already a dict, return it
    if isinstance(args, dict):
        return args
    
    # If it's a list, check what's inside
    if isinstance(args, list):
        if not args:  # Empty list
            return {}
        if isinstance(args[0], dict):  # [{}]
            return args[0]
        # If list of other things, return empty dict
        return {}
    
    # If it's a string, try to parse JSON
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except:
            pass
    
    # Default: return empty dict
    return {}

    
@app.function(
    image=image,
    gpu="B200",
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/engines": vol},
    timeout=60*30
)
def bench_b200(args=None):
    from tensorrt_llm import LLM, SamplingParams, BuildConfig
    import time, json, os
    from datetime import datetime

    args = _coerce_args(args) if args is not None else {}

    model = args.get("model", "Qwen/Qwen2.5-Coder-14B")
    dtype = args.get("dtype", "float16")     # "float16", "bfloat16", "fp8"
    tp = int(args.get("tensor_parallel", 1))
    max_seq = int(args.get("max_seq_len", 4096))
    max_new_tokens = int(args.get("max_new_tokens", 512))
    temperature = float(args.get("temperature", 0.8))
    top_p = float(args.get("top_p", 0.95))
    batch_size = int(args.get("batch_size", 1))
    input_tokens = int(args.get("input_tokens", 128))

    build_config = BuildConfig(max_seq_len=max_seq, strongly_typed=True)
    build_config.plugin_config.use_paged_context_fmha = True

    llm_dtype = "fp8" if dtype.lower() == "fp8" else dtype

    # Optional: honor HF token if provided via secret (parity with vLLM)
    tok = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if tok:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", tok)
        os.environ.setdefault("HF_TOKEN", tok)

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype=llm_dtype,
        build_config=build_config,
    )

    # Rough token->chars expansion; prioritize output-heavy prompts
    prompt_text = "Generate a Python function to calculate fibonacci numbers. " * max(1, input_tokens // 10)
    prompts = [prompt_text] * batch_size

    sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    t1 = time.perf_counter()

    # Aggregate across batch
    total_new_tokens = 0
    ttft_values = []

    def _norm_seconds(x: float) -> float:
        if x > 1e6: return x / 1e9
        if x > 1e3: return x / 1e3
        return x

    for o in outputs:
        if o.outputs and len(o.outputs) > 0:
            total_new_tokens += len(o.outputs[0].token_ids)
        m = getattr(o, "metrics", None)
        if not m:
            continue
        # Prefer explicit first-token latency if present
        ft = None
        for attr in ("first_token_latency", "first_token_latency_s"):
            v = getattr(m, attr, None) if not isinstance(m, dict) else m.get(attr)
            if v is not None:
                ft = _norm_seconds(float(v))
                break
        if ft is None:
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

    wall = max(1e-6, t1 - t0)
    if ttft_values:
        ttft_s = sum(ttft_values) / len(ttft_values)
        if ttft_s >= wall:
            ttft_s = wall * 0.9
        decode_time = max(1e-6, wall - ttft_s)
    else:
        approx_prefill = wall * (input_tokens / max(1.0, input_tokens + total_new_tokens))
        ttft_s = min(approx_prefill, wall * 0.9)
        decode_time = max(1e-6, wall - ttft_s)

    throughput = (total_new_tokens / decode_time) if total_new_tokens > 0 else 0.0

    metrics = {
        "model": model,
        "gpu": "B200",
        "config": {
            "dtype": dtype,
            "tensor_parallel": tp,
            "max_seq_len": max_seq,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "input_tokens": input_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "throughput_tok_s": throughput,
        "ttft_s": ttft_s,
        "accuracy": None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    print(json.dumps({"event": "metrics", "data": metrics}))
    return metrics