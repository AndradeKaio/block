#!/usr/bin/env python3
"""
gen_spgemm_kernels.py — Generate named-register AVX-512/AVX2 specializations
of gemm_fixed<M,K,N,float> and the DISPATCH() include file.

Two outputs (always written together, same directory):
  spgemm_kernels_generated.hpp  — template specializations with named registers
  spgemm_dispatch_generated.hpp — DISPATCH(m, k, n) lines

Usage:
    python3 gen_spgemm_kernels.py --shapes 5x4x4,1x2x4 --out-dir /tmp/out

    # Print register budget per shape and exit:
    python3 gen_spgemm_kernels.py --shapes 5x4x4,1x2x4 --verify
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# SIMD helpers — select ISA lane width and intrinsic names from N
# ---------------------------------------------------------------------------

def _isa_for_n(n: int) -> tuple[str, int]:
    """Return (isa_tag, lane_floats) for the widest SIMD that fits N.

    Under AVX-512: prefer __m512 (16 floats) when N%16==0.
    Fallback to __m256 (N%8==0) or __m128 (N%4==0).
    """
    if n % 16 == 0:
        return "avx512", 16
    if n % 8 == 0:
        return "avx2_256", 8
    if n % 4 == 0:
        return "avx2_128", 4
    return "scalar", 1


def _simd_names(isa: str) -> dict:
    """Return intrinsic name fragments for the given ISA tag."""
    if isa == "avx512":
        return dict(
            type="__m512",
            load="_mm512_loadu_ps",
            store="_mm512_storeu_ps",
            fma="_mm512_fmadd_ps",
            bcast=lambda ptr: f"_mm512_set1_ps(*({ptr}))",
            lanes=16,
            total_regs=32,
        )
    if isa == "avx2_256":
        return dict(
            type="__m256",
            load="_mm256_loadu_ps",
            store="_mm256_storeu_ps",
            fma="_mm256_fmadd_ps",
            bcast=lambda ptr: f"_mm256_broadcast_ss({ptr})",
            lanes=8,
            total_regs=16,
        )
    if isa == "avx2_128":
        return dict(
            type="__m128",
            load="_mm_loadu_ps",
            store="_mm_storeu_ps",
            fma="_mm_fmadd_ps",
            bcast=lambda ptr: f"_mm_broadcast_ss({ptr})",
            lanes=4,
            total_regs=16,
        )
    return {}  # scalar — no specialization emitted


# ---------------------------------------------------------------------------
# Kernel code generation
# ---------------------------------------------------------------------------

def _gen_kernel(M: int, K: int, N: int, s: dict) -> list[str]:
    """Generate named-register gemm_fixed<M,K,N,float> for one ISA."""
    lanes = s["lanes"]
    total_regs = s["total_regs"]
    NV = N // lanes          # number of SIMD vectors covering N

    # Register budget: M*NV accumulators + NV B-vectors + batch A-broadcasts
    spare = total_regs - M * NV - NV
    batch = max(1, min(M, spare))

    ty = s["type"]
    comment = (
        f"// gemm_fixed<{M},{K},{N},float>  {ty}: "
        f"{M} rows × {NV} regs = {M*NV} accumulators; "
        f"B {NV} regs; A hoisted batch={batch} ({M*NV}+{NV}+{batch}≤{total_regs})"
    )
    lines = [
        comment,
        "template <>",
        f"inline void gemm_fixed<{M}, {K}, {N}, float>(",
        "        const float* __restrict__ A, int lda,",
        "        const float* __restrict__ B, int ldb,",
        "        float*       __restrict__ C, int ldc) {",
    ]

    # Load accumulators: c{row}_{vec}
    for i in range(M):
        for v in range(NV):
            lines.append(f"    {ty} c{i}_{v} = {s['load']}(C + {i}*ldc + {v*lanes});")

    lines.append(f"    for (int p = 0; p < {K}; ++p) {{")

    # Load B row vectors: b{vec}
    for v in range(NV):
        lines.append(f"        {ty} b{v} = {s['load']}(B + p*ldb + {v*lanes});")

    # A-broadcast batches: a{k}
    row = 0
    first_batch = True
    while row < M:
        end = min(row + batch, M)
        bh = end - row

        if first_batch:
            for k in range(bh):
                ptr = f"A + {row+k}*lda + p"
                lines.append(f"        {ty} a{k} = {s['bcast'](ptr)};")
            first_batch = False
        else:
            for k in range(bh):
                ptr = f"A + {row+k}*lda + p"
                lines.append(f"        a{k} = {s['bcast'](ptr)};")

        for k in range(bh):
            for v in range(NV):
                lines.append(f"        c{row+k}_{v} = {s['fma']}(a{k}, b{v}, c{row+k}_{v});")

        row = end

    lines.append("    }")

    # Store accumulators
    for i in range(M):
        for v in range(NV):
            lines.append(f"    {s['store']}(C + {i}*ldc + {v*lanes}, c{i}_{v});")
    lines.append("}")
    return lines


# ---------------------------------------------------------------------------
# Per-shape kernel emit with ISA guards
# ---------------------------------------------------------------------------

def _emit_shape(M: int, K: int, N: int) -> list[str]:
    """Emit ISA-guarded specialization blocks for one (M, K, N) shape."""
    # AVX-512 block: widest ISA that fits N
    isa512, lanes512 = "avx512", 16
    isa256, lanes256 = "avx2_256", 8
    isa128, lanes128 = "avx2_128", 4

    lines = []

    # Under __AVX512F__ — use widest available
    if N % 16 == 0:
        avx512_s = _simd_names("avx512")
        avx2_s   = _simd_names("avx2_256")
        lines += ["#if defined(__AVX512F__)"]
        lines += _gen_kernel(M, K, N, avx512_s)
        lines += ["#elif defined(__AVX2__) && defined(__FMA__)"]
        lines += _gen_kernel(M, K, N, avx2_s)
        lines += ["#endif"]
    elif N % 8 == 0:
        avx2_s = _simd_names("avx2_256")
        lines += ["#if defined(__AVX2__) && defined(__FMA__)"]
        lines += _gen_kernel(M, K, N, avx2_s)
        lines += ["#endif"]
    elif N % 4 == 0:
        avx2_s = _simd_names("avx2_128")
        lines += ["#if defined(__AVX2__) && defined(__FMA__)"]
        lines += _gen_kernel(M, K, N, avx2_s)
        lines += ["#endif"]
    else:
        lines += [f"// shape ({M},{K},{N}): N not divisible by 4 — no SIMD specialization"]

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_shapes(s: str) -> list[tuple[int, int, int]]:
    shapes = []
    seen = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.lower().split("x")
        if len(parts) != 3:
            sys.exit(f"Invalid shape '{token}' — expected MxKxN (e.g. 5x4x4)")
        try:
            M, K, N = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            sys.exit(f"Invalid shape '{token}' — expected integers")
        if (M, K, N) in seen:
            continue
        seen.add((M, K, N))
        shapes.append((M, K, N))
    return shapes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", required=True,
                    help="Comma-separated MxKxN shapes (e.g. '5x4x4,1x2x4')")
    ap.add_argument("--out-dir", default=".",
                    help="Directory for generated files (default: cwd)")
    ap.add_argument("--verify", action="store_true",
                    help="Print register budget per shape and exit without writing")
    args = ap.parse_args()

    shapes = parse_shapes(args.shapes)
    if not shapes:
        sys.exit("No valid shapes provided")

    if args.verify:
        print(f"{'Shape':>14}  {'ISA':>8}  {'lanes':>5}  {'NV':>3}  {'acc':>4}  {'batch':>5}")
        for M, K, N in shapes:
            isa, lanes = _isa_for_n(N)
            if isa == "scalar":
                print(f"  {M}x{K}x{N:>2}         scalar    —    —     — (N not divisible by 4)")
                continue
            s = _simd_names(isa)
            NV = N // s["lanes"]
            spare = s["total_regs"] - M * NV - s["lanes"] // 4  # rough
            batch = max(1, min(M, s["total_regs"] - M * NV - NV))
            print(f"  {M}x{K}x{N:<2}  {isa:>12}  {s['lanes']:>5}  {NV:>3}  {M*NV:>4}  {batch:>5}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernels_path  = out_dir / "spgemm_kernels_generated.hpp"
    dispatch_path = out_dir / "spgemm_dispatch_generated.hpp"

    # --- spgemm_kernels_generated.hpp ---
    kern_lines = [
        "// AUTO-GENERATED by gen_spgemm_kernels.py — do not edit.",
        "// Named SIMD register variables (c{row}_{vec}) guarantee register allocation.",
        "#pragma once",
        "#include <immintrin.h>",
        "",
    ]
    for M, K, N in shapes:
        kern_lines += _emit_shape(M, K, N)
        kern_lines.append("")

    kernels_path.write_text("\n".join(kern_lines))

    # --- spgemm_dispatch_generated.hpp ---
    dispatch_lines = [
        "// AUTO-GENERATED by gen_spgemm_kernels.py — do not edit.",
        "// Included inside gemm() via DISPATCH(m, k, n) macro.",
    ]
    for M, K, N in shapes:
        dispatch_lines.append(f"DISPATCH({M}, {K}, {N})")

    dispatch_path.write_text("\n".join(dispatch_lines) + "\n")

    print(f"Wrote {len(shapes)} shapes:")
    print(f"  kernels  → {kernels_path}")
    print(f"  dispatch → {dispatch_path}")


if __name__ == "__main__":
    main()
