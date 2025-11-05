from __future__ import annotations
from typing import Dict, Any

DEFAULTS = {
    "dtype": "fp8",
    "tensor_parallel": 1,
    "batch_size": 1,          # ← needed by TRT runner
    "input_tokens": 250,
    "max_new_tokens": 2048,
    "temperature": 0.2,
    "top_p": 0.95,
    "max_seq_len": 6144,
    # keep if your TRT runner actually consumes it; otherwise omit
    "lookahead": 6,
}

def build_modal_payload(spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the flat payload that backend.runners.trtllm_qwen_modal expects."""
    p = {**DEFAULTS, **(spec or {}), **(config or {})}
    return {
        "model": p.get("model", "Qwen/Qwen2.5-Coder-14B"),
        "dtype": str(p["dtype"]),
        "tensor_parallel": int(p["tensor_parallel"]),
        "batch_size": int(p["batch_size"]),          # ← add
        "input_tokens": int(p["input_tokens"]),
        "max_new_tokens": int(p["max_new_tokens"]),
        "temperature": float(p["temperature"]),
        "top_p": float(p["top_p"]),
        "max_seq_len": int(p["max_seq_len"]),
        # include only if your TRT runner reads it; otherwise drop to avoid confusion
        "lookahead": int(p["lookahead"]),
        # optional override; TRT runner currently ignores it but harmless
        "prompt": p.get("prompt"),
        # you do NOT need num_gpus here unless your TRT runner starts using it
    }
