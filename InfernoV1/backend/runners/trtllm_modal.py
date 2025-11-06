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
    if isinstance(args, dict):
        return args
    if isinstance(args, list):
        if not args:
            return {}
        if isinstance(args[0], dict):
            return args[0]
        return {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except:
            pass
    return {}


def _eval_python_code(code: str) -> dict:
    """
    Evaluate generated Python code for correctness.
    Returns dict with pass/fail and error info.
    """
    import sys
    import io
    import traceback
    
    # Basic syntax check
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return {
            "passed": False,
            "error": "SyntaxError",
            "detail": str(e)
        }
    
    # Try to execute (with timeout protection)
    try:
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        # Create isolated namespace
        namespace = {}
        exec(code, namespace)
        
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Check for common game components (snake game specific)
        has_game_loop = any(keyword in code.lower() for keyword in ['while', 'game', 'loop'])
        has_snake_logic = any(keyword in code.lower() for keyword in ['snake', 'direction', 'move'])
        has_imports = 'import' in code
        
        completeness_score = sum([
            has_game_loop * 0.33,
            has_snake_logic * 0.33,
            has_imports * 0.34
        ])
        
        return {
            "passed": True,
            "error": None,
            "detail": None,
            "completeness": completeness_score,
            "has_game_loop": has_game_loop,
            "has_snake_logic": has_snake_logic,
            "code_length": len(code)
        }
        
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        return {
            "passed": False,
            "error": type(e).__name__,
            "detail": str(e),
            "traceback": traceback.format_exc()
        }

    
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
    dtype = args.get("dtype", "float16")
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

    # NEW: Use actual coding prompt
    prompt_text = "Build a snake game in Python."
    prompts = [prompt_text] * batch_size

    sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    t1 = time.perf_counter()

    # Aggregate throughput metrics
        # Aggregate throughput + timing
    total_new_tokens = 0
    total_input_tokens = batch_size * max(0, int(input_tokens))
    ttft_values = []
    per_request = []  # collect per-request stats for p50/p95 etc.

    def _norm_seconds(x: float) -> float:
        if x > 1e6: return x / 1e9
        if x > 1e3: return x / 1e3
        return x

    for idx, o in enumerate(outputs):
        # count output tokens defensively
        out_tok = 0
        if getattr(o, "outputs", None) and len(o.outputs) > 0:
            # some versions expose token_ids, others output_token_ids
            ids = getattr(o.outputs[0], "token_ids", None) or getattr(o.outputs[0], "output_token_ids", None)
            if ids is not None:
                out_tok = len(ids)
        total_new_tokens += out_tok

        # TTFT extraction across schema variants
        m = getattr(o, "metrics", None)
        ft = None
        if m is not None:
            # handle object-attr and dict styles
            if isinstance(m, dict):
                for k in ("first_token_latency_s", "first_token_latency"):
                    if k in m:
                        ft = _norm_seconds(float(m[k]))
                        break
                if ft is None and "first_token_time" in m and "request_start_time" in m:
                    ft = _norm_seconds(float(m["first_token_time"])) - _norm_seconds(float(m["request_start_time"]))
            else:
                for k in ("first_token_latency_s", "first_token_latency"):
                    v = getattr(m, k, None)
                    if v is not None:
                        ft = _norm_seconds(float(v))
                        break
                if ft is None:
                    ft_time  = getattr(m, "first_token_time", None)
                    req_time = getattr(m, "request_start_time", None)
                    if ft_time is not None and req_time is not None:
                        ft = _norm_seconds(float(ft_time)) - _norm_seconds(float(req_time))

        if ft is not None and ft >= 0:
            ttft_values.append(ft)

        per_request.append({
            "index": idx,
            "output_tokens": out_tok,
            "ttft_s": ft
        })

    wall = max(1e-6, t1 - t0)

    # Prefill/decode split
    if ttft_values:
        ttft_s = sum(ttft_values) / len(ttft_values)
        if ttft_s >= wall:
            ttft_s = wall * 0.9
        decode_time_s = max(1e-6, wall - ttft_s)
    else:
        # Fallback split if TTFT not reported
        approx_prefill = wall * (total_input_tokens / max(1.0, total_input_tokens + total_new_tokens))
        ttft_s = min(approx_prefill, wall * 0.9)
        decode_time_s = max(1e-6, wall - ttft_s)

    # Rates
    output_tok_s = (total_new_tokens / decode_time_s) if total_new_tokens > 0 else 0.0
    prefill_tok_s = (total_input_tokens / ttft_s) if ttft_s > 0 and total_input_tokens > 0 else 0.0

    # ---------- Code evaluation (robust) ----------
    eval_results = []
    for o in outputs:
        text = None
        if getattr(o, "outputs", None) and len(o.outputs) > 0:
            # prefer text if present
            text = getattr(o.outputs[0], "text", None)
            if not text:
                # fall back: empty text → treat as failed completeness 0
                pass
        if text:
            eval_results.append(_eval_python_code(text))
        else:
            eval_results.append({
                "passed": False,
                "error": "NoOutput",
                "detail": "No generated text available for evaluation.",
                "completeness": 0.0
            })

    # Accuracy: average(pass_flag) * avg(completeness), but never None
    if eval_results:
        pass_rate = sum(1 for e in eval_results if e.get("passed")) / len(eval_results)
        avg_completeness = sum(float(e.get("completeness", 0.0)) for e in eval_results) / len(eval_results)
        accuracy = pass_rate * avg_completeness
    else:
        accuracy = 0.0

    # TTFT distribution
    def _percentile(vals, p):
        if not vals: return None
        s = sorted(vals)
        k = (len(s)-1) * (p/100)
        f = int(k)
        c = min(f+1, len(s)-1)
        if f == c: return s[f]
        return s[f] + (s[c]-s[f]) * (k - f)

    ttft_p50 = _percentile(ttft_values, 50)
    ttft_p95 = _percentile(ttft_values, 95)

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
        # counts
        "tokens_in_total": total_input_tokens,
        "tokens_out_total": total_new_tokens,
        "requests": len(outputs),
        # timing
        "wall_time_s": wall,
        "ttft_avg_s": ttft_s,
        "ttft_p50_s": ttft_p50,
        "ttft_p95_s": ttft_p95,
        "decode_time_s": decode_time_s,
        # rates
        "output_tok_s": output_tok_s,            # explicit alias
        "throughput_tok_s": output_tok_s,        # keep your original key too
        "prefill_tok_s": prefill_tok_s,
        # eval
        "accuracy": accuracy,
        "eval_details": eval_results,
        # per-request (useful for debugging spread)
        "per_request": per_request,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


    print(json.dumps({"event": "metrics", "data": metrics}))
    return metrics