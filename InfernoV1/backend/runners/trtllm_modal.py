# backend/runners/trtllm_qwen_modal.py
import os, json, time
from typing import Dict, Any
from pathlib import Path
import modal
from backend.eval.python import eval_python

APP_NAME = "inferno-trtllm-qwen"
HF_SECRET_NAME = "hf-token"
ENGINE_VOL = "trtllm-engines"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(ENGINE_VOL, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget")
    # Base deps first
    .pip_install(
        "numpy==1.26.4",
        "transformers==4.45.2",
        "huggingface_hub>=0.24.0",
        # torch is needed because we call .cuda() on tokenized input
        "torch==2.4.0",
    )
    # Then install TRT-LLM from NVIDIA index using extra_options
    .pip_install(
        "tensorrt-llm==0.20.0",
        extra_options=["--extra-index-url", "https://pypi.nvidia.com"],
    )
    .run_commands("git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM /opt/TensorRT-LLM")
    .env({"TOKENIZERS_PARALLELISM": "false", "NCCL_P2P_DISABLE": "1"})
)

def _mk_prompt(base_tokens: int) -> str:
    base = ("Generate a runnable Python terminal game with a main() and replay loop. "
            "Keep commentary minimal and use standard library only. ")
    filler = ("Provide clean structure and deterministic behavior. " * 200)
    # crude char approximation: ~4 chars/token
    need_chars = max(0, base_tokens * 4 - len(base))
    return base + filler[:need_chars]

def _engine_tag(model: str, dtype: str, tp: int, lookahead: int, max_seq: int) -> str:
    safe = model.replace("/", "_").lower()
    return f"{safe}_{dtype}_tp{tp}_la{lookahead}_ms{max_seq}"

@app.function(image=image, gpu="B200", secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
             volumes={"/engines": vol}, timeout=60*30)
def _ensure_engine(args: Dict[str, Any]) -> str:
    import subprocess, os
    model = args.get("model", "Qwen/Qwen2.5-Coder-14B")
    dtype = args.get("dtype", "fp8")
    tp = int(args.get("tensor_parallel", 1))
    lookahead = int(args.get("lookahead", 6))
    max_seq = int(args.get("max_seq_len", 6144))

    tag = _engine_tag(model, dtype, tp, lookahead, max_seq)
    engine_dir = Path("/engines") / tag
    if engine_dir.exists() and any(engine_dir.iterdir()):
        return str(engine_dir)

    ckpt_dir = Path("/engines") / f"{tag}_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_TOKEN"] = env.get("HF_TOKEN") or env.get("HUGGINGFACE_TOKEN", "")

    # Convert HF -> TRT-LLM
    conv = [
        "python", "/opt/TensorRT-LLM/examples/llama/convert_checkpoint.py",
        "--model_dir", model,
        "--output_dir", str(ckpt_dir),
        "--dtype", dtype,
        "--use_parallel_embedding",
    ]
    subprocess.check_call(conv, env=env)

    # Build engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    build = [
        "trtllm-build",
        "--checkpoint_dir", str(ckpt_dir),
        "--output_dir", str(engine_dir),
        "--tp_size", str(tp),
        "--pp_size", "1",
        "--max_seq_len", str(max_seq),
        "--use_paged_context_fmha",
        "--enable_context_fmha",
        "--enable_chunked_context",
    ]
    if dtype == "fp8":
        build += ["--use_fp8", "--use_fp8_kv_cache"]
    if lookahead > 0:
        build += ["--lookahead_max_steps", str(lookahead)]
    subprocess.check_call(build)
    return str(engine_dir)

def _bench_impl(args: Dict[str, Any]) -> Dict[str, Any]:
    from tensorrt_llm.runtime import ModelRunner, KvCacheConfig, LookaheadDecodingConfig, SamplingConfig
    from transformers import AutoTokenizer
    model = args.get("model", "Qwen/Qwen2.5-Coder-14B")
    dtype = args.get("dtype", "fp8")
    tp = int(args.get("tensor_parallel", 1))
    lookahead = int(args.get("lookahead", 6))
    max_seq = int(args.get("max_seq_len", 6144))
    input_tokens = int(args.get("input_tokens", 250))
    max_new_tokens = int(args.get("max_new_tokens", 2048))
    temperature = float(args.get("temperature", 0.2))
    top_p = float(args.get("top_p", 0.95))

    engine_dir = _ensure_engine.call({
        "model": model, "dtype": dtype, "tensor_parallel": tp,
        "lookahead": lookahead, "max_seq_len": max_seq
    })
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True)
    kv = KvCacheConfig(fp8_kv_cache=True, enable_block_reuse=True)
    look = LookaheadDecodingConfig(max_steps=lookahead) if lookahead > 0 else None
    runner = ModelRunner.from_dir(engine_dir, kv_cache_config=kv, lookahead_config=look,
                                  cuda_graph_mode="static", enable_overlap_schedule=True)

    # controlled-length prompt
    prompt = args.get("prompt") or _mk_prompt(input_tokens)
    # simple chat wrapper
    sys = "You are Qwen2.5-Coder. Return only code unless asked."
    text = f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tok(text, return_tensors="pt").input_ids.cuda()

    samp = SamplingConfig(temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)

    t0 = time.perf_counter()
    it_first = None
    tokens = 0
    for out in runner.generate_stream(input_ids=input_ids, sampling_config=samp):
        if it_first is None and out.text_delta:
            it_first = time.perf_counter()
        if out.text_delta:
            tokens += tok(out.text_delta, add_special_tokens=False, return_tensors="pt").input_ids.shape[-1]
    t1 = time.perf_counter()

    ttft_s = (it_first - t0) if it_first else 0.0
    gen_time = (t1 - (it_first or t0))
    tps = (tokens / gen_time) if gen_time > 0 and tokens else 0.0

    return {
        "model": model,
        "gpu": "B200",
        "config": {
            "engine": "trtllm", "dtype": dtype, "tensor_parallel": tp,
            "lookahead": lookahead, "input_tokens": input_tokens,
            "max_new_tokens": max_new_tokens
        },
        "throughput_tok_s": round(tps, 2),
        "ttft_s": round(ttft_s, 3),
        "accuracy": None,
        "timestamp": time.time(),
    }


def _coerce_args(args):
    # Accept dict directly
    if isinstance(args, dict):
        return args
    # If Modal delivers a list, unwrap it
    if isinstance(args, list):
        if len(args) == 1 and isinstance(args[0], dict):
            return args[0]
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


# --- Modal entrypoints per pool to match your CLI map ---
@app.function(image=image, gpu="B200", secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
             volumes={"/engines": vol}, timeout=60*30)
def bench_b200(args: Dict[str, Any]) -> None:
    res = _bench_impl(args)
    print(json.dumps({"event":"metrics","data":res}))
