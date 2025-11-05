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
    .apt_install(
        "git",
        "wget",
        "openmpi-bin",
        "libopenmpi-dev",
    )
    # ensure no rogue 'cuda' package
    .run_commands("python -m pip uninstall -y cuda || true")
    # core wheels
    .pip_install(
        "numpy==1.26.4",
        "transformers==4.45.2",
        "huggingface_hub>=0.24.0",
        "torch==2.4.0",
        "mpi4py==3.1.6",
    )
    # NVIDIA wheels
    .run_commands(
        "python -m pip install --extra-index-url https://pypi.nvidia.com "
        "cuda-python==12.6.0 tensorrt-llm==0.21.0"
    )
    .run_commands("git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM /opt/TensorRT-LLM")
    # 🔒 safe smoke test (pure import — no driver needed)
    .run_commands(
        'python -c "from cuda import cuda, cudart; print(\'cuda-python import OK\')"'
    )
    .env({
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_P2P_DISABLE": "1",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openmpi/lib:${LD_LIBRARY_PATH}",
        "PYTHONNOUSERSITE": "1",
    })
)
@app.function(image=image, gpu="B200", timeout=60*10)
def _sanity_runtime():
    # Runs on a GPU worker → has libcuda.so.1
    from cuda import cudart
    import tensorrt_llm
    # optional: touch the driver to verify
    err, ndev = cudart.cudaGetDeviceCount()
    if err != 0 or ndev < 1:
        raise RuntimeError(f"CUDA device not available: err={err}, ndev={ndev}")
    return {"tensorrt_llm": tensorrt_llm.__version__, "devices": ndev}


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
             
def _ensure_engine(args) -> str:
    _ = _sanity_runtime.remote()

    args = _coerce_args(args)
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

    from huggingface_hub import snapshot_download
    model_dir = Path("/engines") / "models" / model.replace("/", "_")
    if not model_dir.exists():
        print(f"Downloading {model} from HuggingFace...")
        snapshot_download(
            repo_id=model,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    ckpt_dir = Path("/engines") / f"{tag}_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_TOKEN"] = env.get("HF_TOKEN") or env.get("HUGGINGFACE_TOKEN", "")

    # Convert HF -> TRT-LLM
    conv = [
        "python", "/opt/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py",
        "--model_dir", str(model_dir),
        "--output_dir", str(ckpt_dir),
        "--dtype", "bfloat16",
        "--tp_size", str(tp),  # Add this line
        "--use_parallel_embedding",
    ]
    subprocess.check_call(conv, env=env)

    # Build engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    build = [
        "trtllm-build",
        "--checkpoint_dir", str(ckpt_dir),
        "--output_dir", str(engine_dir),
        "--max_seq_len", str(max_seq),
        "--gpt_attention_plugin", "auto",
        "--gemm_plugin", "auto",
        "--context_fmha", "enable",
        "--use_paged_context_fmha", "enable",
        "--remove_input_padding", "enable",
    ]

    # Apply FP8 quantization if requested
    if dtype == "fp8":
        build += [
            "--strongly_typed",
            "--gemm_plugin", "fp8",
            "--gpt_attention_plugin", "fp8",
        ]

    if lookahead > 0:
        build += ["--lookahead_max_steps", str(lookahead)]

    subprocess.check_call(build)
    return str(engine_dir)

def _bench_impl(args: Dict[str, Any]) -> Dict[str, Any]:
    from tensorrt_llm.runtime import ModelRunner, SamplingConfig
    from tensorrt_llm.bindings.executor import KvCacheConfig, LookaheadDecodingConfig
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

    engine_dir = _ensure_engine.remote({
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
    enc = tok(text, add_special_tokens=True)
    input_ids = enc["input_ids"]  # list[int] is fine for TRT-LLM

    samp = SamplingConfig(temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)

    t0 = time.perf_counter()
    it_first = None
    tokens = 0
    for out in runner.generate_stream(input_ids=input_ids, sampling_config=samp):
        if it_first is None and out.text_delta:
            it_first = time.perf_counter()
        if out.text_delta:
            tokens += len(tok(out.text_delta, add_special_tokens=False)["input_ids"])
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
def bench_b200(args):
    args = _coerce_args(args)
    res = _bench_impl(args)
    print(json.dumps({"event":"metrics","data":res}))