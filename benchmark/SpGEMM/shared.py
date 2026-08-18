import csv
import ctypes
import ctypes.util
import fcntl
import os
import struct
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from blocks import Block


ROW_MAJOR = 101
NO_TRANS = 111

PERF_FIELDS = [
    "cycles",
    "instructions",
    "IPC",
    "cache_refs",
    "cache_misses",
    "cache_miss_rate",
    "branch_instr",
    "branch_misses",
    "branch_miss_rate",
    "ctx_switches",
    "hw_available",
]


def load_blas() -> Optional[ctypes.CDLL]:
    for name in ("openblas", "blas", "cblas"):
        path = ctypes.util.find_library(name)
        if path:
            try:
                lib = ctypes.CDLL(path)
                _ = lib.cblas_dgemm
                return lib
            except (OSError, AttributeError):
                continue
    for root, _, files in os.walk(os.path.dirname(np.__file__)):
        for f in files:
            if ("blas" in f.lower() or "openblas" in f.lower()) and ".so" in f:
                try:
                    lib = ctypes.CDLL(os.path.join(root, f))
                    _ = lib.cblas_dgemm
                    return lib
                except (OSError, AttributeError):
                    continue
    return None


def setup_gemm(lib: ctypes.CDLL):
    for name, scalar_t in (
        ("cblas_dgemm", ctypes.c_double),
        ("cblas_sgemm", ctypes.c_float),
    ):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            scalar_t,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            scalar_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]


def timer(fn: Callable, n: int = 10):
    times, result = [], None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, float(np.mean(times))


_NR = {"x86_64": 298, "aarch64": 241}
_IOC_ENABLE = 0x2400
_IOC_DISABLE = 0x2401
_IOC_RESET = 0x2403
_HW = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
_SW = [(1, 3)]
_EV_NAMES = [
    "cycles",
    "instructions",
    "cache_refs",
    "cache_misses",
    "branch_instr",
    "branch_misses",
    "ctx_switches",
]
_EV_SPEC = _HW + _SW
_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _pev_open(ev_type, config):
    import platform

    nr = _NR.get(platform.machine())
    if nr is None:
        return -1
    buf = bytearray(120)
    struct.pack_into("I", buf, 0, ev_type)
    struct.pack_into("I", buf, 4, 120)
    struct.pack_into("Q", buf, 8, config)
    struct.pack_into("Q", buf, 40, 1 | (1 << 5) | (1 << 6))
    attr = ctypes.create_string_buffer(bytes(buf))
    fd = _libc.syscall(
        ctypes.c_long(nr),
        attr,
        ctypes.c_int(0),
        ctypes.c_int(-1),
        ctypes.c_int(-1),
        ctypes.c_ulong(0),
    )
    return int(fd) if fd >= 0 else -1


class PerfCounters:
    def __init__(self):
        self._fds = {}
        self.hw_available = False

    def __enter__(self):
        hw_ok = 0
        for name, (ev_type, config) in zip(_EV_NAMES, _EV_SPEC):
            fd = _pev_open(ev_type, config)
            if fd >= 0:
                self._fds[name] = fd
                if ev_type == 0:
                    hw_ok += 1
        self.hw_available = hw_ok > 0
        for fd in self._fds.values():
            fcntl.ioctl(fd, _IOC_RESET, 0)
            fcntl.ioctl(fd, _IOC_ENABLE, 0)
        return self

    def __exit__(self, *_):
        for fd in self._fds.values():
            try:
                fcntl.ioctl(fd, _IOC_DISABLE, 0)
            except Exception:
                pass

    def counts(self):
        result = {n: 0 for n in _EV_NAMES}
        for name, fd in self._fds.items():
            result[name] = struct.unpack("Q", os.read(fd, 8))[0]
        return result

    def close(self):
        for fd in self._fds.values():
            try:
                os.close(fd)
            except Exception:
                pass
        self._fds.clear()


def measure(fn: Callable, n: int = 10, perf: bool = False):
    result, t = timer(fn, n)
    out = {"result": result, "time_s": t}
    if not perf:
        return out
    pc = PerfCounters()
    with pc:
        fn()
    c = pc.counts()
    pc.close()
    cr = c.get("cache_refs", 0)
    bi = c.get("branch_instr", 0)
    cy = c.get("cycles", 0)
    out.update(
        {
            "hw_available": pc.hw_available,
            "cycles": cy,
            "instructions": c["instructions"],
            "IPC": c["instructions"] / cy if cy > 0 else None,
            "cache_refs": cr,
            "cache_misses": c["cache_misses"],
            "cache_miss_rate": c["cache_misses"] / cr if cr > 0 else None,
            "branch_instr": bi,
            "branch_misses": c["branch_misses"],
            "branch_miss_rate": c["branch_misses"] / bi if bi > 0 else None,
            "ctx_switches": c["ctx_switches"],
        }
    )
    return out


def save_csv(path: str, rows: list, config: dict, perf: bool = False):
    base = [
        "M",
        "K",
        "N",
        "blocks_A",
        "blocks_B",
        "block_h_min",
        "block_h_max",
        "block_w_min",
        "block_w_max",
        "n_pairs",
        "n_groups",
        "method",
        "time_ms",
    ]
    fieldnames = base + (PERF_FIELDS if perf else [])
    write_header = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            entry = {
                **config,
                "method": row["method"],
                "time_ms": f"{row['time_s'] * 1e3:.4f}",
            }
            if perf:
                for pf in PERF_FIELDS:
                    v = row.get(pf)
                    entry[pf] = (
                        f"{v:.6g}"
                        if isinstance(v, float)
                        else ("" if v is None else str(v))
                    )
            w.writerow(entry)


def generate_matrices(M, N, n_blocks, h_range, w_range, seed, dtype):
    rng = np.random.default_rng(seed)
    placed = []
    blocks = []
    for _ in range(n_blocks * 50):
        if len(blocks) == n_blocks:
            break
        h = int(rng.integers(h_range[0], h_range[1] + 1))
        w = int(rng.integers(w_range[0], w_range[1] + 1))
        h, w = min(h, M), min(w, N)
        r0 = int(rng.integers(0, M - h + 1))
        c0 = int(rng.integers(0, N - w + 1))
        r1, c1 = r0 + h, c0 + w
        if any(
            r0 < pr1 and r1 > pr0 and c0 < pc1 and c1 > pc0
            for pr0, pr1, pc0, pc1 in placed
        ):
            continue
        placed.append((r0, r1, c0, c1))
        blocks.append(Block(r=r0, c=c0, h=h, w=w))

    coo_r, coo_c, coo_v, flat_parts = [], [], [], []
    offset = 0
    for b in blocks:
        b.offset = offset
        vals = rng.standard_normal((b.h, b.w)).astype(dtype)
        flat_parts.append(vals.ravel())
        ri, ci = np.mgrid[b.r : b.r + b.h, b.c : b.c + b.w]
        coo_r.append(ri.ravel())
        coo_c.append(ci.ravel())
        coo_v.append(vals.ravel())
        offset += b.h * b.w

    flat = np.concatenate(flat_parts) if flat_parts else np.empty(0, dtype=dtype)
    rows = np.concatenate(coo_r).tolist() if coo_r else []
    cols = np.concatenate(coo_c).tolist() if coo_c else []
    vals = np.concatenate(coo_v).tolist() if coo_v else []
    return blocks, flat, rows, cols, vals


def write_mtx(path: str, M: int, N: int, rows, cols, vals):
    entries = sorted(zip(rows, cols, vals))
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{M} {N} {len(entries)}\n")
        for r, c, v in entries:
            f.write(f"{r + 1} {c + 1} {v:.17g}\n")
