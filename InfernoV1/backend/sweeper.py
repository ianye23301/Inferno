from itertools import product
import random
from typing import Dict, Any, List
from models import Sweep

def _axes_from_sweep(s: Sweep) -> Dict[str, List[Any]]:
    axes = {k: v for k, v in {
        "batch_size": s.batch_size,
        "tensor_parallel": s.tensor_parallel,
        "quantization": s.quantization,
        "input_tokens": s.input_tokens,
        "max_new_tokens": s.max_new_tokens,
        "max_seq_len": s.max_seq_len,     # ADD
        "dtype": s.dtype,
        "lookahead": s.lookahead,
        "temperature": s.temperature,      # ADD
        "top_p": s.top_p,                  # ADD
        **(s.extra or {})
    }.items() if v}
    return axes


def grid_points(s: Sweep) -> List[Dict[str, Any]]:
    axes = _axes_from_sweep(s)
    if not axes:
        return [{}]
    keys, vals = zip(*axes.items())
    return [dict(zip(keys, comb)) for comb in product(*vals)]


def random_points(s: Sweep, n: int | None) -> List[Dict[str, Any]]:
    allp = grid_points(s)
    if n and n < len(allp):
        return random.sample(allp, n)
    return allp