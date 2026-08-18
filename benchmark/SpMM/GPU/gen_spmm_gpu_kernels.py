#!/usr/bin/env python3
"""gen_spmm_gpu_kernels.py — stub.

No per-matrix kernel specialisation in this implementation.
The benchmark harness calls this script before compiling
prisma_gpu_spmm_bench.cu; it expects the three generated headers to exist in
--out-dir.  We create empty stubs so the compile succeeds without any
#include of these files needed (prisma_gpu_spmm_bench.cu does not include them).
"""

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes",   default="")
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--out-dir",  required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Create empty stubs so any accidental #include won't fail to find the file.
    for fname in [
        "spmm_gpu_shape_table.hpp",
        "spmm_kernels_generated.cuh",
        "spmm_gpu_dispatch_table.cuh",
    ]:
        (out / fname).write_text("// auto-generated stub (no specialisation)\n")

    shapes = [s for s in args.shapes.split(",") if s]
    print(f"gen_spmm_gpu_kernels: stub — {len(shapes)} shape(s), no codegen")

if __name__ == "__main__":
    main()
