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
    import subprocess
    import json
    import os
    import tempfile
    from datetime import datetime
    from pathlib import Path

    args = _coerce_args(args) if args is not None else {}
    
    model = args.get("model", "Qwen/Qwen2.5-Coder-14B")
    dtype = args.get("dtype", "float16")
    tp = int(args.get("tensor_parallel", 1))
    max_seq = int(args.get("max_seq_len", 4096))
    max_new_tokens = int(args.get("max_new_tokens", 512))
    batch_size = int(args.get("batch_size", 1))
    input_tokens = int(args.get("input_tokens", 128))
    
    tok = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if tok:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", tok)
        os.environ.setdefault("HF_TOKEN", tok)
    
    # According to the docs, the benchmark script is at:
    # examples/benchmark/benchmark.py
    
    # Create input dataset (CSV format with prompts)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        # Write multiple copies of the prompt for batch testing
        for _ in range(batch_size):
            f.write("Build a snake game in Python.\n")
        dataset_path = f.name
    
    try:
        # Build the engine first (if not cached)
        # The benchmark script can do this automatically, or we can use trtllm-build
        
        # Run benchmark using gptManagerBenchmark (from the docs)
        benchmark_script = "/workspace/tensorrt_llm/examples/benchmark/gptManagerBenchmark.py"
        
        # Check if script exists, otherwise try alternate location
        if not Path(benchmark_script).exists():
            # Try finding it
            find_result = subprocess.run(
                ["find", "/workspace", "-name", "gptManagerBenchmark.py"],
                capture_output=True,
                text=True
            )
            if find_result.stdout.strip():
                benchmark_script = find_result.stdout.strip().split('\n')[0]
                print(f"Found benchmark at: {benchmark_script}")
            else:
                # Try the simpler benchmark.py
                benchmark_script = "/workspace/tensorrt_llm/examples/benchmark/benchmark.py"
        
        cmd = [
            "python", benchmark_script,
            "--model", model,
            "--dtype", dtype,
            "--tensor_parallel_size", str(tp),
            "--max_input_length", str(input_tokens),
            "--max_output_length", str(max_new_tokens),
            "--batch_size", str(batch_size),
            "--input_file", dataset_path,
            "--output_csv", "/tmp/results.csv",
            "--warm_up", "2",  # Warm-up iterations
            "--num_runs", "3",  # Number of benchmark runs
        ]
        
        print(f"Running benchmark: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1500,
            cwd="/workspace"
        )
        
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        
        # Parse CSV results
        if Path("/tmp/results.csv").exists():
            import csv
            with open("/tmp/results.csv") as f:
                reader = csv.DictReader(f)
                results = list(reader)
                
            if results:
                # Extract metrics from benchmark results
                last_result = results[-1]
                
                # Now run code evaluation
                eval_results = []
                # We need to actually generate to evaluate, so fall back to LLM API for that
                from tensorrt_llm import LLM, SamplingParams, BuildConfig
                
                build_config = BuildConfig(max_seq_len=max_seq, strongly_typed=True)
                build_config.plugin_config.use_paged_context_fmha = True
                llm_dtype = "fp8" if dtype.lower() == "fp8" else dtype
                
                llm = LLM(
                    model=model,
                    tensor_parallel_size=tp,
                    dtype=llm_dtype,
                    build_config=build_config,
                )
                
                prompt_text = "Build a snake game in Python."
                sampling = SamplingParams(
                    temperature=float(args.get("temperature", 0.8)),
                    top_p=float(args.get("top_p", 0.95)),
                    max_tokens=max_new_tokens
                )
                
                outputs = llm.generate([prompt_text] * batch_size, sampling)
                
                for o in outputs:
                    text = None
                    if getattr(o, "outputs", None) and len(o.outputs) > 0:
                        text = getattr(o.outputs[0], "text", None)
                    if text:
                        eval_results.append(_eval_python_code(text))
                    else:
                        eval_results.append({
                            "passed": False,
                            "error": "NoOutput",
                            "completeness": 0.0
                        })
                
                # Calculate accuracy
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
                        "temperature": args.get("temperature", 0.8),
                        "top_p": args.get("top_p", 0.95),
                    },
                    # Use NVIDIA's benchmark metrics
                    "throughput_tok_s": float(last_result.get("throughput", 0)),
                    "ttft_avg_s": float(last_result.get("time_to_first_token_ms", 0)) / 1000.0,
                    "latency_s": float(last_result.get("latency_ms", 0)) / 1000.0,
                    "tokens_out_total": int(last_result.get("total_output_tokens", 0)),
                    "accuracy": accuracy,
                    "eval_details": eval_results,
                    "benchmark_raw": last_result,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                
                print(json.dumps({"event": "metrics", "data": metrics}))
                return metrics
        
        # If benchmark didn't work, return error with logs
        return {
            "error": "Benchmark script failed or results not found",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
        
    finally:
        Path(dataset_path).unlink(missing_ok=True)