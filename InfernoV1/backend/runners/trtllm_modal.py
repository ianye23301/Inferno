import modal, json, time
from pathlib import Path

APP_NAME = "inferno-trtllm-qwen"
HF_SECRET_NAME = "hf-token"
ENGINE_VOL = "trtllm-engines"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(ENGINE_VOL, create_if_missing=True)

image = modal.Image.from_registry(
    "nvcr.io/nvidia/tensorrt-llm/release:1.0.0"  # Older stable version
)
def _coerce_args(args):
    if isinstance(args, dict): return args
    if isinstance(args, list) and args and isinstance(args[0], dict): return args[0]
    try: return json.loads(args) if isinstance(args, str) else {}
    except: return {}

@app.function(
    image=image, 
    gpu="B200",
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/engines": vol}, 
    timeout=60*30
)
def bench_b200(args):
    """Simple benchmark using LLM API"""
    from tensorrt_llm import LLM, SamplingParams, BuildConfig
    
    args = _coerce_args(args)
    
    model = args.get("model", "Qwen/Qwen2.5-Coder-14B")
    dtype = args.get("dtype", "float16")  # "float16", "bfloat16", "fp8"
    tp = args.get("tensor_parallel", 1)
    max_seq = args.get("max_seq_len", 4096)
    max_new_tokens = args.get("max_new_tokens", 512)
    temperature = args.get("temperature", 0.8)
    top_p = args.get("top_p", 0.95)
    
    # Build config for engine optimization
    build_config = BuildConfig(
        max_seq_len=max_seq,
        strongly_typed=True,
    )
    
    # Enable optimizations
    build_config.plugin_config.use_paged_context_fmha = True
    
    # FP8 quantization
    if dtype == "fp8":
        from tensorrt_llm import QuantConfig, QuantAlgo
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
        dtype = "float16"  # base dtype
    
    # LLM API handles engine building and caching automatically
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype=dtype,
        build_config=build_config,
    )
    
    prompt = "Generate a Python function to calculate fibonacci numbers."
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens
    )
    
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    t1 = time.perf_counter()
    
    tokens = len(outputs[0].outputs[0].token_ids)
    elapsed = t1 - t0
    
    result = {
        "model": model,
        "gpu": "B200",
        "config": {
            "dtype": args.get("dtype"),
            "tensor_parallel": tp,
            "max_seq_len": max_seq,
            "max_new_tokens": max_new_tokens,
        },
        "metrics": {
            "tokens_generated": tokens,
            "time_s": round(elapsed, 3),
            "throughput_tok_s": round(tokens / elapsed, 2) if elapsed > 0 else 0,
        },
        "timestamp": time.time(),
    }
    
    print(json.dumps({"event": "metrics", "data": result}))
    return result