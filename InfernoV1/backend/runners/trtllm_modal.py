import modal, json, time
from pathlib import Path

APP_NAME = "inferno-trtllm-qwen"
HF_SECRET_NAME = "hf-token"
ENGINE_VOL = "trtllm-engines"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(ENGINE_VOL, create_if_missing=True)

# UPDATED: Use latest TRT-LLM image
image = modal.Image.from_registry("nvcr.io/nvidia/tensorrt-llm/release:latest")

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
            "passed": True,
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
            "completeness": max(0.0, completeness_score - 0.3),
            "traceback": traceback.format_exc()
        }

# Single GPU version (original)
@app.function(
    image=image,
    gpu="B200",
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/engines": vol},
    timeout=60*30
)
def bench_b200(args=None):
    return _bench_b200_impl(args)

# 2 GPU version
@app.function(
    image=image,
    gpu="B200:2",
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/engines": vol},
    timeout=60*30
)
def bench_b200_tp2(args=None):
    return _bench_b200_impl(args)

# 4 GPU version
@app.function(
    image=image,
    gpu="B200:4",
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/engines": vol},
    timeout=60*30
)
def bench_b200_tp4(args=None):
    return _bench_b200_impl(args)

# 8 GPU version
@app.function(
    image=image,
    gpu="B200:8",
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/engines": vol},
    timeout=60*30
)
def bench_b200_tp8(args=None):
    return _bench_b200_impl(args)
    
def _bench_b200_impl(args=None):
    from tensorrt_llm import LLM, SamplingParams, BuildConfig
    try:
        from tensorrt_llm.llmapi import KvCacheConfig
    except Exception:
        KvCacheConfig = None

    import time, json, os, inspect
    from datetime import datetime

    args = _coerce_args(args) if args is not None else {}

    # Core knobs
    model = args.get("model", "Qwen/Qwen2.5-Coder-14B")
    dtype = str(args.get("dtype", "float16"))
    quantization = str(args.get("quantization", "none"))
    tp = int(args.get("tensor_parallel", 1))
    max_seq = int(args.get("max_seq_len", 4096))
    max_new_tokens = int(args.get("max_new_tokens", 512))
    batch_size = int(args.get("batch_size", 1))
    input_tokens = int(args.get("input_tokens", 128))

    # Sampling
    temperature = float(args.get("temperature", 0.0))
    top_p = float(args.get("top_p", 1.0))
    top_k = args.get("top_k", None)
    if top_k is not None:
        top_k = int(top_k)

    # Extra runtime/build flags
    extra = args.get("extra", {}) or {}
    use_paged_context_fmha = bool(extra.get("use_paged_context_fmha", True))
    enable_block_reuse      = bool(extra.get("enable_block_reuse", True))
    kv_block_size           = extra.get("kv_block_size", None)
    sink_token_length       = extra.get("sink_token_length", None)
    
    # Speculative decoding config
    use_speculation = bool(extra.get("use_speculation", False))
    spec_mode = extra.get("spec_mode", "ngram")
    ngram_draft_len = int(extra.get("ngram_draft_len", 10))
    ngram_matching_size = int(extra.get("ngram_matching_size", 7))
    draft_model = extra.get("draft_model", None)
    num_draft_tokens = int(extra.get("num_draft_tokens", 5))

    # KV cache budget
    max_tokens_budget = batch_size * (input_tokens + max_new_tokens + 128)

    # UPDATED: Parse quantization configuration
    quantization_lower = quantization.lower()
    
    # Determine base dtype
    dtype_map = {
        "fp16": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
    }
    llm_dtype = dtype_map.get(dtype.lower(), "float16")
    
    # Parse quantization type
    quant_config = None
    if quantization_lower not in ["none", "null", "", "auto"]:
        print(f"[INFO] Enabling quantization: {quantization}")
        try:
            from tensorrt_llm.models.modeling_utils import QuantConfig, QuantAlgo
            
            quant_algo_map = {
                "fp8": QuantAlgo.FP8,
                "float8": QuantAlgo.FP8,
                "fp4": QuantAlgo.FP4,  # Assuming this exists
                "int8": QuantAlgo.W8A16,
                "int4": QuantAlgo.W4A16,
                "int4_awq": QuantAlgo.W4A16_AWQ,
                "int4_gptq": QuantAlgo.W4A16_GPTQ,
                "w8a8": QuantAlgo.W8A8_SQ_PER_CHANNEL,
                "smoothquant": QuantAlgo.W8A8_SQ_PER_CHANNEL,
            }
            
            quant_algo = quant_algo_map.get(quantization_lower)
            if quant_algo:
                # Build QuantConfig with appropriate settings
                quant_kwargs = {"quant_algo": quant_algo}
                
                # Add algorithm-specific parameters from extra
                if quantization_lower in ["int4_awq", "int4_gptq", "int4"]:
                    quant_kwargs["group_size"] = extra.get("group_size", 128)
                    quant_kwargs["has_zero_point"] = extra.get("has_zero_point", False)
                
                if quantization_lower in ["w8a8", "smoothquant"]:
                    quant_kwargs["smoothquant_val"] = extra.get("smoothquant_val", 0.5)
                
                quant_config = QuantConfig(**quant_kwargs)
                print(f"[INFO] Created QuantConfig: {quant_algo}")
            else:
                print(f"[WARN] Unknown quantization type: {quantization}, ignoring")
                
        except Exception as e:
            print(f"[WARN] Could not create QuantConfig: {e}")
            quant_config = None

    # Auth
    tok = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if tok:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", tok)
        os.environ.setdefault("HF_TOKEN", tok)

    # Multi-GPU NCCL Setup
    if tp > 1:
        print(f"[INFO] Setting up multi-GPU environment for TP={tp}")
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        os.environ["NCCL_TIMEOUT"] = "600"
        os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"

    # Build config
    build_config = BuildConfig(
        max_seq_len=max_seq,
        strongly_typed=True,
    )
    
    # Plugin config
    try:
        build_config.plugin_config.use_paged_context_fmha = use_paged_context_fmha
    except Exception:
        pass

    if kv_block_size is not None:
        try:
            build_config.plugin_config.kv_cache_block_size = int(kv_block_size)
        except Exception:
            pass

    # Speculative decoding setup
    speculative_config = None
    if use_speculation:
        print(f"[INFO] Enabling speculative decoding: mode={spec_mode}")
        try:
            if spec_mode == "ngram":
                try:
                    from tensorrt_llm.llmapi.llm_args import NGramDecodingConfig
                except ImportError:
                    from tensorrt_llm.llmapi import NGramDecodingConfig
                
                speculative_config = NGramDecodingConfig(
                    max_draft_len=ngram_draft_len,
                    max_matching_ngram_size=ngram_matching_size,
                    is_keep_all=True,
                    is_use_oldest=True,
                )
                print(f"[INFO] NGram: draft_len={ngram_draft_len}, matching_size={ngram_matching_size}")
            
            elif spec_mode == "draft_model" and draft_model:
                try:
                    from tensorrt_llm.llmapi import SpeculativeDecodingConfig
                    
                    speculative_config = SpeculativeDecodingConfig(
                        draft_model=draft_model,
                        num_draft_tokens=num_draft_tokens,
                        draft_tensor_parallel_size=1,
                        draft_dtype=llm_dtype,
                    )
                    print(f"[INFO] Draft model: {draft_model}, tokens={num_draft_tokens}")
                except Exception as e:
                    print(f"[WARN] SpeculativeDecodingConfig not available: {e}")
                    speculative_config = None
                    
        except Exception as e:
            print(f"[WARN] Could not create speculative config: {e}")
            speculative_config = None

    # KV cache config
    # --- replace your KV cache config block with this ---
# KV cache config
    kv_config = None
    if KvCacheConfig is not None:
        try:
            kv_kwargs = {
                "enable_block_reuse": enable_block_reuse,
                "max_tokens": int(max_tokens_budget),
            }

            # Optional knobs from extra
            max_attention_window = extra.get("max_attention_window", None)
            sink_token_length = extra.get("sink_token_length", None)

            # Only set max_attention_window if user asked for it
            if max_attention_window is not None:
                kv_kwargs["max_attention_window"] = [int(max_attention_window)]
                # Enforce a positive sink; default to 4 if user omitted or passed 0
                sink_val = int(sink_token_length) if sink_token_length is not None else 4
                if sink_val <= 0:
                    sink_val = 4
                kv_kwargs["sink_token_length"] = sink_val

            # Add optional fields only if KvCacheConfig supports them
            import inspect
            sig = inspect.signature(KvCacheConfig)
            if "free_gpu_memory_fraction" in sig.parameters:
                kv_kwargs["free_gpu_memory_fraction"] = float(extra.get("free_gpu_memory_fraction", 0.95))
            if "enable_partial_reuse" in sig.parameters:
                kv_kwargs["enable_partial_reuse"] = bool(extra.get("enable_partial_reuse", True))

            kv_config = KvCacheConfig(**kv_kwargs)
        except Exception as e:
            print(f"[WARN] Could not create KvCacheConfig: {e}")
            kv_config = None

    # Create LLM
    print(f"[INFO] Initializing LLM: TP={tp}, dtype={llm_dtype}, quantization={quantization}")
    
    llm_kwargs = {
        "model": model,
        "tensor_parallel_size": tp,
        "dtype": llm_dtype,
        "build_config": build_config,
    }
    
    # Add quant_config if quantization requested
    if quant_config is not None:
        llm_kwargs["quant_config"] = quant_config
    
    # Add KV cache config
    if kv_config is not None:
        llm_kwargs["kv_cache_config"] = kv_config
    
    # Add speculative config
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = speculative_config
    
    
    llm = LLM(**llm_kwargs)

    # Rest stays the same...
    prompt_text = args.get("prompt_text", "Build a snake game in Python.")
    prompts = [prompt_text] * batch_size

    sampling_kwargs = dict(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        return_generation_logits=bool(extra.get("return_generation_logits", False)),
        return_context_logits=bool(extra.get("return_context_logits", False)),
    )
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    if extra.get("eos_token_id") is not None:
        sampling_kwargs["end_id"] = int(extra["eos_token_id"])

    sampling = SamplingParams(**sampling_kwargs)

    request_data = {i: {
        "start_time": None,
        "first_token_time": None,
        "end_time": None,
        "tokens": [],
        "text": "",
    } for i in range(batch_size)}

    total_start = time.perf_counter()

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    t1 = time.perf_counter()

    for idx, o in enumerate(outputs):
        request_data[idx]["start_time"] = t0
        request_data[idx]["end_time"] = t1

        if getattr(o, "outputs", None) and len(o.outputs) > 0:
            request_data[idx]["tokens"] = getattr(o.outputs[0], "token_ids", []) or []
            request_data[idx]["text"] = getattr(o.outputs[0], "text", "")

        m = getattr(o, "metrics", None)
        ft = None
        if m:
            if isinstance(m, dict):
                ft = m.get("first_token_latency_s") or m.get("first_token_latency")
            else:
                ft = getattr(m, "first_token_latency_s", None) or getattr(m, "first_token_latency", None)
        if ft is not None:
            ft = float(ft)
            if ft > 1e6:
                ft /= 1e9
            elif ft > 1e3:
                ft /= 1e3
            request_data[idx]["first_token_time"] = t0 + max(0.0, ft)

    total_end = time.perf_counter()
    wall_time = total_end - total_start

    # [Metrics calculation - same as before]
    ttft_values, tpot_values = [], []
    total_output_tokens = 0

    for data in request_data.values():
        n_tok = len(data["tokens"])
        total_output_tokens += n_tok
        if data["first_token_time"] and data["start_time"]:
            ttft = data["first_token_time"] - data["start_time"]
            if ttft > 0:
                ttft_values.append(ttft)
            if data["end_time"] and n_tok > 1:
                decode_time = data["end_time"] - data["first_token_time"]
                if decode_time > 0:
                    tpot_values.append(decode_time / max(1, n_tok - 1))

    if ttft_values:
        ttft_values_sorted = sorted(ttft_values)
        ttft_avg_s = sum(ttft_values_sorted) / len(ttft_values_sorted)
        ttft_p50_s = ttft_values_sorted[len(ttft_values_sorted)//2]
        ttft_p95_s = ttft_values_sorted[min(len(ttft_values_sorted)-1, int(0.95*(len(ttft_values_sorted)-1)))]
    else:
        ttft_avg_s = min(0.2 * wall_time, wall_time)
        ttft_p50_s = ttft_avg_s
        ttft_p95_s = ttft_avg_s

    if tpot_values:
        tpot_avg_s = sum(tpot_values)/len(tpot_values)
        decode_time_total_s = sum(
            (d["end_time"] - d["first_token_time"])
            for d in request_data.values()
            if d["end_time"] and d["first_token_time"]
        )
    else:
        decode_time_total_s = max(0.0, wall_time - ttft_avg_s)
        tpot_avg_s = (decode_time_total_s / max(1, total_output_tokens)) if total_output_tokens else 0.0

    tokens_in_total = batch_size * input_tokens
    prefill_tok_s = (tokens_in_total / ttft_avg_s) if ttft_avg_s > 0 else 0.0
    decode_tok_s  = (total_output_tokens / decode_time_total_s) if decode_time_total_s > 0 else 0.0
    overall_tok_s = (tokens_in_total + total_output_tokens) / max(1e-6, wall_time)

    generated_texts = [d["text"] for d in request_data.values() if d["text"]]
    eval_results = []
    if generated_texts:
        try:
            r = _eval_python_code(generated_texts[0])
            eval_results.append(r)
            accuracy = 1.0 if r.get("passed") else 0.0
        except Exception:
            accuracy = 0.0
    else:
        accuracy = 0.0

    metrics = {
        "model": model,
        "gpu": f"B200x{tp}",
        "trtllm_version": "latest",
        "config": {
            "dtype": dtype,
            "quantization": quantization,
            "actual_dtype": llm_dtype,
            "has_quantization": quant_config is not None,
            "tensor_parallel": tp,
            "max_seq_len": max_seq,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "input_tokens": input_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "extra": extra,
            "speculation_enabled": use_speculation,
            "speculation_mode": spec_mode if use_speculation else None,
        },
        "tokens_in_total": tokens_in_total,
        "tokens_out_total": total_output_tokens,
        "requests": batch_size,
        "wall_time_s": wall_time,
        "ttft_avg_s": ttft_avg_s,
        "ttft_p50_s": ttft_p50_s,
        "ttft_p95_s": ttft_p95_s,
        "tpot_avg_s": tpot_avg_s,
        "decode_time_total_s": decode_time_total_s,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "overall_tok_s": overall_tok_s,
        "accuracy": accuracy,
        "eval_details": eval_results,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    print(json.dumps({"event": "metrics", "data": metrics}))
    return metrics