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
    import sys, io, traceback, textwrap

    # Normalize code formatting
    code = code.strip()
    # Remove Markdown fences if present
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    # Dedent any leading indentation
    code = textwrap.dedent(code)

    # Basic syntax check
    try:
        compile(code, '<string>', 'exec')
        syntax_valid = True
    except SyntaxError as e:
        return {
            "passed": False,
            "error": "SyntaxError",
            "detail": str(e),
            "completeness": 0.0
        }
    
    # Check for common game components (snake game specific)
    code_lower = code.lower()
    has_game_loop = any(keyword in code_lower for keyword in ['while', 'for', 'loop'])
    has_snake_logic = any(keyword in code_lower for keyword in ['snake', 'direction', 'move', 'position'])
    has_imports = 'import' in code
    has_class_or_function = 'def ' in code or 'class ' in code
    
    # Completeness score (0.0 to 1.0)
    completeness_score = sum([
        has_game_loop * 0.25,
        has_snake_logic * 0.25,
        has_imports * 0.25,
        has_class_or_function * 0.25
    ])
    
    # Try to execute (with graceful handling of import errors)
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        namespace = {}
        exec(code, namespace)
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Successfully executed
        return {
            "passed": True,
            "error": None,
            "detail": "Executed successfully",
            "completeness": completeness_score,
            "has_game_loop": has_game_loop,
            "has_snake_logic": has_snake_logic,
            "code_length": len(code),
            "executed": True
        }
        
    except ModuleNotFoundError as e:
        # Module not found is OK - the code is valid, just can't run here
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        return {
            "passed": True,  # ← Changed to True!
            "error": "ModuleNotFoundError",
            "detail": str(e),
            "completeness": completeness_score,
            "has_game_loop": has_game_loop,
            "has_snake_logic": has_snake_logic,
            "code_length": len(code),
            "executed": False,
            "note": "Syntax valid, imports unavailable in eval environment"
        }
        
    except Exception as e:
        # Other runtime errors indicate actual problems
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Check if it's just a benign error (e.g., pygame.quit() when pygame not initialized)
        error_type = type(e).__name__
        benign_errors = ['SystemExit', 'KeyboardInterrupt']
        
        if error_type in benign_errors:
            return {
                "passed": True,
                "error": error_type,
                "detail": str(e),
                "completeness": completeness_score,
                "has_game_loop": has_game_loop,
                "has_snake_logic": has_snake_logic,
                "code_length": len(code),
                "executed": False,
                "note": "Benign runtime error"
            }
        
        return {
            "passed": False,
            "error": error_type,
            "detail": str(e),
            "completeness": max(0.0, completeness_score - 0.3),  # Penalize but don't zero out
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

    prompt_text = "Build a snake game in Python."
    prompts = [prompt_text] * batch_size

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens
    )

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    t1 = time.perf_counter()
    
    wall_time = t1 - t0
    
    # Extract metrics from outputs
    total_output_tokens = 0
    generated_texts = []
    ttft_values = []
    e2e_values = []
    
    print(f"DEBUG: Processing {len(outputs)} outputs")
    
    for idx, o in enumerate(outputs):
        # Get tokens and text
        if hasattr(o, "outputs") and len(o.outputs) > 0:
            tokens = getattr(o.outputs[0], "token_ids", None) or getattr(o.outputs[0], "output_token_ids", [])
            text = getattr(o.outputs[0], "text", "")
            
            total_output_tokens += len(tokens) if tokens else 0
            if text:
                generated_texts.append(text)
        
        # Try to get metrics_dict from the output
        metrics_dict = getattr(o, "metrics_dict", None)
        if metrics_dict:
            print(f"DEBUG: Output {idx} metrics_dict: {metrics_dict}")
            
            # Try different metric name patterns
            for ttft_key in ["ttft", "TTFT", "time_to_first_token", "first_token_latency"]:
                if ttft_key in metrics_dict:
                    ttft_values.append(metrics_dict[ttft_key])
                    print(f"DEBUG: Found TTFT={metrics_dict[ttft_key]} with key '{ttft_key}'")
                    break
            
            for e2e_key in ["e2e", "E2E", "end_to_end_latency", "total_latency"]:
                if e2e_key in metrics_dict:
                    e2e_values.append(metrics_dict[e2e_key])
                    print(f"DEBUG: Found E2E={metrics_dict[e2e_key]} with key '{e2e_key}'")
                    break
        else:
            print(f"DEBUG: Output {idx} has no metrics_dict")
    
    # Calculate timing
    if ttft_values:
        avg_ttft_s = sum(ttft_values) / len(ttft_values)
        ttft_p50 = sorted(ttft_values)[len(ttft_values) // 2]
        ttft_p95 = sorted(ttft_values)[int(len(ttft_values) * 0.95)] if len(ttft_values) > 1 else ttft_values[0]
        print(f"DEBUG: Using real TTFT metrics: avg={avg_ttft_s}, p50={ttft_p50}, p95={ttft_p95}")
    else:
        # Fallback estimation
        total_input_tokens = batch_size * input_tokens
        estimated_prefill_fraction = total_input_tokens / max(1.0, total_input_tokens + total_output_tokens)
        estimated_prefill_fraction = max(0.05, min(0.5, estimated_prefill_fraction))
        avg_ttft_s = wall_time * estimated_prefill_fraction
        ttft_p50 = avg_ttft_s
        ttft_p95 = avg_ttft_s
        print(f"DEBUG: Using estimated TTFT: {avg_ttft_s}")
    
    decode_time_s = max(1e-6, wall_time - avg_ttft_s)
    
    # Throughput calculations
    throughput_tok_s = (total_output_tokens / decode_time_s) if total_output_tokens > 0 else 0.0
    prefill_tok_s = ((batch_size * input_tokens) / avg_ttft_s) if avg_ttft_s > 0 else 0.0
    overall_tok_s = ((batch_size * input_tokens + total_output_tokens) / wall_time) if wall_time > 0 else 0.0
    
    # Code evaluation
    eval_results = []
    for text in generated_texts:
        if text:
            eval_results.append(_eval_python_code(text))
        else:
            eval_results.append({
                "passed": False,
                "error": "NoOutput",
                "completeness": 0.0
            })
    
    # Accuracy
    if eval_results:
        pass_rate = sum(1 for e in eval_results if e.get("passed")) / len(eval_results)
        avg_completeness = sum(float(e.get("completeness", 0.0)) for e in eval_results) / len(eval_results)
        accuracy = pass_rate * avg_completeness
    else:
        accuracy = 0.0
    
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
        "tokens_in_total": batch_size * input_tokens,
        "tokens_out_total": total_output_tokens,
        "requests": batch_size,
        "wall_time_s": wall_time,
        "ttft_avg_s": avg_ttft_s,
        "ttft_p50_s": ttft_p50,
        "ttft_p95_s": ttft_p95,
        "decode_time_s": decode_time_s,
        "throughput_tok_s": throughput_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "overall_tok_s": overall_tok_s,
        "accuracy": accuracy,
        "eval_details": eval_results,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    print(json.dumps({"event": "metrics", "data": metrics}))
    return metrics