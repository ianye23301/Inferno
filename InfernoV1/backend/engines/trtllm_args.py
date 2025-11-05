# engines/trtllm_args.py
from __future__ import annotations
from typing import Dict, Any

DEFAULTS = {
    "dtype": "fp8",
    "tensor_parallel": 1,
    "lookahead": 6,
    "input_tokens": 250,
    "max_new_tokens": 2048,
    "temperature": 0.2,
    "top_p": 0.95,
    "max_seq_len": 6144,  # input+output cap used for engine build
}

def build_modal_payload(spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror the vLLM payload shape but with TRT-LLM knobs."""
    p = {**DEFAULTS, **(spec or {}), **(config or {})}
    return {
        "model": p.get("model", "Qwen/Qwen2.5-Coder-14B"),
        "dtype": p["dtype"],
        "tensor_parallel": int(p["tensor_parallel"]),
        "lookahead": int(p["lookahead"]),
        "input_tokens": int(p["input_tokens"]),
        "max_new_tokens": int(p["max_new_tokens"]),
        "temperature": float(p["temperature"]),
        "top_p": float(p["top_p"]),
        "max_seq_len": int(p["max_seq_len"]),
        "prompt": p.get("prompt"),  # optional override
    }
