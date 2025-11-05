# backend/eval/python_eval.py
import ast, subprocess, tempfile, textwrap, time, os
from pathlib import Path

def eval_python(code: str, params) -> dict:
    out = {"parsed": False, "has_main": False, "ran": False, "banned_hits": [], "loc": 0, "smoke": []}
    try:
        ast.parse(code)
        out["parsed"] = True
    except Exception as e:
        return {**out, "reason": f"AST error: {e}", "ok": False}

    out["loc"] = sum(1 for _ in code.splitlines() if _.strip())
    if params.require_main:
        out["has_main"] = ("def main" in code)

    for b in params.banned:
        if b in code:
            out["banned_hits"].append(b)

    if out["banned_hits"]:
        return {**out, "reason": "banned imports/APIs", "ok": False}

    # Run with isolation
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "prog.py"
            p.write_text(code)
            inp = None
            if params.smoke_tests:
                inp = "\n".join(params.smoke_tests) + "\n"
            proc = subprocess.run(
                ["python3", "-S", "-I", str(p)],
                input=inp,
                capture_output=True,
                text=True,
                timeout=params.run_timeout_s
            )
            out["ran"] = (proc.returncode == 0)
            out["stdout"] = proc.stdout[:2000]
            out["stderr"] = proc.stderr[:1000]
            out["smoke"] = [{"in": s, "ok": out["ran"]} for s in (params.smoke_tests or [])]
    except Exception as e:
        out["ran"] = False
        out["stderr"] = str(e)

    ok = out["parsed"] and (out["has_main"] or not params.require_main) and out["ran"] and (out["loc"] >= params.min_loc)
    return {**out, "ok": ok}
