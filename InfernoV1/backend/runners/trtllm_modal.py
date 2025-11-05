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
    """Simple benchmark using LLM API"""
    from tensorrt_llm import LLM, SamplingParams, BuildConfig
    
    args = _coerce_args(args) if args is not None else {}
    
    model = args.get("model", "Qwen/Qwen2.5-Coder-14B")
    dtype = args.get("dtype", "float16")  # "float16", "bfloat16", "fp8"
    tp = args.get("tensor_parallel", 1)
    max_seq = args.get("max_seq_len", 4096)
    max_new_tokens = args.get("max_new_tokens", 512)
    temperature = args.get("temperature", 0.8)
    top_p = args.get("top_p", 0.95)
    batch_size = args.get("batch_size", 1)  # FIXED: Now extracted
    input_tokens = args.get("input_tokens", 128)  # FIXED: Now extracted
    
    # Build config for engine optimization
    build_config = BuildConfig(
        max_seq_len=max_seq,
        strongly_typed=True,
    )
    
    # Enable optimizations
    build_config.plugin_config.use_paged_context_fmha = True
    
    # FP8 quantization - just pass "fp8" directly to LLM
    # In TensorRT-LLM 1.0, quantization is handled automatically
    if dtype == "fp8":
        llm_dtype = "fp8"  # LLM API handles FP8 automatically
    else:
        llm_dtype = dtype
    
    # LLM API handles engine building and caching automatically
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype=llm_dtype,
        build_config=build_config,
    )
    
    # FIXED: Generate prompt of specified input length
    # Approximate tokens (rough: 1 token ≈ 4 chars)
    prompt_text = "Generate a Python function to calculate fibonacci numbers. " * (input_tokens // 10)
    
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens
    )
    
    # FIXED: Support batch_size > 1
    prompts = [prompt_text] * batch_size
    
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    t1 = time.perf_counter()
    
    # FIXED: Aggregate metrics across batch
    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    elapsed = t1 - t0
    
    # FIXED: Calculate TTFT (time to first token) if available
    ttft_s = None
    if hasattr(outputs[0].outputs[0], 'ttft'):
        ttft_s = outputs[0].outputs[0].ttft
    
    result = {
        "model": model,
        "gpu": "B200",
        "config": {
            "dtype": dtype,
            "tensor_parallel": tp,
            "max_seq_len": max_seq,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,  # FIXED: Now recorded
            "input_tokens": input_tokens,  # FIXED: Now recorded
            "temperature": temperature,
            "top_p": top_p,
        },
        "metrics": {
            "tokens_generated": total_tokens,
            "time_s": round(elapsed, 3),
            "throughput_tok_s": round(total_tokens / elapsed, 2) if elapsed > 0 else 0,
            "ttft_s": round(ttft_s, 4) if ttft_s else None,  # FIXED: Include TTFT
        },
        "timestamp": time.time(),
    }
    
    print(json.dumps({"event": "metrics", "data": result}))
    return result