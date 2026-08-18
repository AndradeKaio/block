import os
import subprocess
from pathlib import Path
from typing import Dict, Optional


_PERF_EVENTS = "cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses,context-switches"

_PERF_REMAP = {
    "cache-references": "cache_refs",
    "cache-misses": "cache_misses",
    "branch-instructions": "branch_instr",
    "branch-misses": "branch_misses",
    "context-switches": "ctx_switches",
}


def compile_taco(
    kernel_h: str, label: str, out_dir: str, script_dir: str
) -> Optional[str]:
    src = Path(script_dir) / "bench_taco.c"
    exe = str(Path(out_dir) / f"bench_{label}")
    cmd = [
        "gcc",
        "-O3",
        "-march=native",
        "-fopenmp",
        f'-DTACO_KERNEL_H="{kernel_h}"',
        f"-I{script_dir}",
        "-o",
        exe,
        str(src),
        "-lm",
    ]
    print(f"  Compiling {label}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  Failed:\n{e.stderr.decode().strip()}")
        return None
    return exe


def _parse_stdout(stdout: str) -> Dict:
    raw = {}
    for line in stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                raw[k.strip()] = int(v.strip())
            except ValueError:
                pass
    return raw


def _parse_perf_stat(stderr: str) -> Dict:
    result = {}
    for line in stderr.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        val_str = parts[0].strip()
        if not val_str or val_str.startswith("<"):
            continue
        try:
            val = int(val_str.replace(",", ""))
            key = _PERF_REMAP.get(parts[2].strip(), parts[2].strip())
            if key:
                result[key] = val
        except ValueError:
            pass
    result["hw_available"] = result.get("cycles", 0) > 0
    c = result.get("cycles", 0)
    result["IPC"] = result["instructions"] / c if c > 0 else None
    cr = result.get("cache_refs", 0)
    result["cache_miss_rate"] = result["cache_misses"] / cr if cr > 0 else None
    bi = result.get("branch_instr", 0)
    result["branch_miss_rate"] = result["branch_misses"] / bi if bi > 0 else None
    return result


def run_taco(
    exe: str, a_path: str, b_path: str, n_runs: int, perf: bool = False
) -> Optional[Dict]:
    env = {**os.environ}

    r = subprocess.run(
        [exe, a_path, b_path, str(n_runs)], capture_output=True, text=True, env=env
    )
    if r.returncode != 0:
        print(f"  run failed: {r.stderr.strip()[:120]}")
        return None

    raw = _parse_stdout(r.stdout)
    runs = []
    for i in range(n_runs):
        a = raw.get(f"run_{i}_assemble_ns", 0) / 1e9
        c = raw.get(f"run_{i}_compute_ns", 0) / 1e9
        runs.append({"assemble_s": a, "compute_s": c})
        print(
            f"    run {i + 1}/{n_runs}  assemble={a * 1e3:.2f}ms  compute={c * 1e3:.2f}ms"
        )

    result = {
        "mean_assemble_s": raw.get("mean_assemble_ns", 0) / 1e9,
        "mean_compute_s": raw.get("mean_compute_ns", 0) / 1e9,
        "A_nnz": raw.get("A_nnz", 0),
        "runs": runs,
    }

    if perf:
        pr = subprocess.run(
            ["perf", "stat", "-e", _PERF_EVENTS, "-x", ",", exe, a_path, b_path, "1"],
            capture_output=True,
            text=True,
            env=env,
        )
        if pr.returncode == 0:
            result.update(_parse_perf_stat(pr.stderr))

    return result
