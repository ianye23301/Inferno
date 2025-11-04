from __future__ import annotations
from typing import Dict, Any

def build_modal_payload(spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": spec.get("model"),
        "gpu_pool": spec.get("gpu_pool"),
        "config": config,
        # NEW: propagate base_env for per-run env injection
        "env": spec.get("base_env", {}),
        # Optional: pass dataset through so the runner can compute accuracy
        "dataset": spec.get("dataset"),
    }
