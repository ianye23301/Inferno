from __future__ import annotations
from typing import Dict, Any

DEFAULTS = {
    "dtype": "float16",          # Base dtype (float16, bfloat16, float32)
    "quantization": "none",      # Quantization type (fp8, fp4, int8, int4_awq, none)
    "tensor_parallel": 1,
    "batch_size": 1,
    "input_tokens": 250,
    "max_new_tokens": 2048,
    "temperature": 0.2,
    "top_p": 0.95,
    "max_seq_len": 6144,
    "lookahead": 0,
}

def build_modal_payload(spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the flat payload that backend.runners.trtllm_modal expects.
    
    Handles both dtype and quantization separately:
    - dtype: base precision (float16, bfloat16, float32)
    - quantization: quantization algorithm (fp8, fp4, int8, int4_awq, none)
    """
    p = {**DEFAULTS, **(spec or {}), **(config or {})}
    
    return {
        "model": p.get("model", "Qwen/Qwen2.5-Coder-14B"),
        "dtype": str(p["dtype"]),
        "quantization": str(p["quantization"]),
        "tensor_parallel": int(p["tensor_parallel"]),
        "batch_size": int(p["batch_size"]),
        "input_tokens": int(p["input_tokens"]),
        "max_new_tokens": int(p["max_new_tokens"]),
        "temperature": float(p["temperature"]),
        "top_p": float(p["top_p"]),
        "max_seq_len": int(p["max_seq_len"]),
        "lookahead": int(p.get("lookahead", 0)),
        "prompt": p.get("prompt"),
        "extra": p.get("extra", {}),
    }