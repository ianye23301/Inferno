from __future__ import annotations
from typing import Dict, Any

def build_modal_payload(spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": spec.get("model"),
        "gpu_pool": spec.get("gpu_pool"),
        "num_gpus": int(spec.get("num_gpus", 1)),   # NEW
        "config": config,
        "env": spec.get("base_env", {}),
        "dataset": spec.get("dataset"),
    }