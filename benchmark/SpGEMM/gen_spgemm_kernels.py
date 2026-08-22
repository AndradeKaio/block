#!/usr/bin/env python3
"""
gen_spgemm_kernels.py — Generate named-register AVX-512/AVX2 specializations
of gemm_fixed<M,K,N,double> and the DISPATCH() include file.

Double precision throughout — Prisma SpGEMM CPU (prisma_cpu_bench.cpp's
Scalar) is double, matching TACO's own bench_taco.c (both Bv/Cv/A_t->vals
are `double`), so the two are actually comparable at a tight tolerance
instead of only "within float32 precision". Previously this generator
targeted float (_ps intrinsics, 16/8/4-wide lanes); this is a full rewrite
to _pd intrinsics with the correspondingly narrower lane widths (double is
twice the width of float per SIMD register, so half as many fit per
vector: 8/4/2 instead of 16/8/4).

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
    """Return (isa_tag, lane_doubles) for the widest SIMD whose lane width
    fits at least once in N.

    Under AVX-512: prefer __m512d (8 doubles) when N>=8.
    Fallback to __m256d (N>=4) or __m128d (N>=2); N==1 has no usable lane
    width at all ("scalar").

    N need NOT be an exact multiple of the chosen lane width -- mined block
    widths have no reason to be even (core/block_mining.hpp's MineParams has
    no parity constraint at all), so requiring an exact multiple silently
    dropped roughly half of real top-N shapes from codegen entirely (see
    _gen_kernel's scalar-tail handling of N % lanes leftover columns).
    """
    if n >= 8:
        return "avx512", 8
    if n >= 4:
        return "avx2_256", 4
    if n >= 2:
        return "avx2_128", 2
    return "scalar", 1


def _simd_names(isa: str) -> dict:
    """Return intrinsic name fragments for the given ISA tag."""
    if isa == "avx512":
        return dict(
            type="__m512d",
            load="_mm512_loadu_pd",
            store="_mm512_storeu_pd",
            fma="_mm512_fmadd_pd",
            bcast=lambda ptr: f"_mm512_set1_pd(*({ptr}))",
            lanes=8,
            total_regs=32,
        )
    if isa == "avx2_256":
        return dict(
            type="__m256d",
            load="_mm256_loadu_pd",
            store="_mm256_storeu_pd",
            fma="_mm256_fmadd_pd",
            bcast=lambda ptr: f"_mm256_broadcast_sd({ptr})",
            lanes=4,
            total_regs=16,
        )
    if isa == "avx2_128":
        return dict(
            type="__m128d",
            load="_mm_loadu_pd",
            store="_mm_storeu_pd",
            fma="_mm_fmadd_pd",
            # No _mm_broadcast_sd intrinsic exists (AVX's broadcast-from-
            # memory set is _mm256_broadcast_sd/_mm256_broadcast_ss/
            # _mm_broadcast_ss only) -- _mm_loaddup_pd (SSE3, available
            # under immintrin.h whenever AVX2 is enabled) loads one double
            # from memory and duplicates it into both __m128d lanes, the
            # same "load from ptr, broadcast" semantics.
            bcast=lambda ptr: f"_mm_loaddup_pd({ptr})",
            lanes=2,
            total_regs=16,
        )
    return {}  # scalar — no specialization emitted


# ---------------------------------------------------------------------------
# Kernel code generation
# ---------------------------------------------------------------------------


def _gen_row_tile(row0: int, tile_rows: int, K: int, s: dict, NV: int, rem: int,
                  lanes: int, ty: str | None, batch: int, indent: str) -> list[str]:
    """Generate one row-tile's worth of the accumulator load / K-loop / store
    body, for absolute rows [row0, row0+tile_rows). Register names (c{i}_{v},
    t{i}_{j}, a{k}) are local-to-tile (i in [0, tile_rows)) so the caller must
    wrap each tile's output in its own {} scope to let names repeat across
    tiles -- that's what makes row-tiling actually free registers instead of
    just renaming the same overflow.
    """
    lines = []

    for i in range(tile_rows):
        for v in range(NV):
            lines.append(f"{indent}{ty} c{i}_{v} = {s['load']}(C + {row0 + i}*ldc + {v * lanes});")
    if rem:
        for i in range(tile_rows):
            for j in range(rem):
                lines.append(f"{indent}double t{i}_{j} = C[{row0 + i}*ldc + {NV * lanes + j}];")

    lines.append(f"{indent}for (int p = 0; p < {K}; ++p) {{")

    for v in range(NV):
        lines.append(f"{indent}    {ty} b{v} = {s['load']}(B + p*ldb + {v * lanes});")
    if rem:
        for j in range(rem):
            lines.append(f"{indent}    double bt{j} = B[p*ldb + {NV * lanes + j}];")

    if NV > 0:
        row = 0
        first_batch = True
        while row < tile_rows:
            end = min(row + batch, tile_rows)
            bh = end - row

            if first_batch:
                for k in range(bh):
                    ptr = f"A + {row0 + row + k}*lda + p"
                    lines.append(f"{indent}    {ty} a{k} = {s['bcast'](ptr)};")
                first_batch = False
            else:
                for k in range(bh):
                    ptr = f"A + {row0 + row + k}*lda + p"
                    lines.append(f"{indent}    a{k} = {s['bcast'](ptr)};")

            for k in range(bh):
                for v in range(NV):
                    lines.append(
                        f"{indent}    c{row + k}_{v} = {s['fma']}(a{k}, b{v}, c{row + k}_{v});"
                    )

            row = end

    if rem:
        for i in range(tile_rows):
            for j in range(rem):
                lines.append(f"{indent}    t{i}_{j} += A[{row0 + i}*lda + p] * bt{j};")

    lines.append(f"{indent}}}")

    for i in range(tile_rows):
        for v in range(NV):
            lines.append(f"{indent}{s['store']}(C + {row0 + i}*ldc + {v * lanes}, c{i}_{v});")
    if rem:
        for i in range(tile_rows):
            for j in range(rem):
                lines.append(f"{indent}C[{row0 + i}*ldc + {NV * lanes + j}] = t{i}_{j};")

    return lines


def _gen_kernel(M: int, K: int, N: int, s: dict) -> list[str]:
    """Generate named-register gemm_fixed<M,K,N,double> for one ISA.

    N need not be an exact multiple of the lane width: NV = N // lanes full
    SIMD vectors are generated as before, and any remainder (rem = N % lanes)
    trailing columns get plain scalar named-register accumulators computed
    by an FMA loop appended alongside the vectorized part -- so a shape is
    never skipped just because its width doesn't divide evenly by a lane
    count. When NV == 0 (N smaller than the narrowest lane, or the "scalar"
    ISA with lanes=1 used for N==1) this degenerates to a pure scalar
    kernel: no SIMD registers/intrinsics are emitted at all.

    M is row-tiled: mined block heights have no upper bound (core/
    block_mining.hpp's MineParams has no size cap), so a shape like
    M=27 previously requested M*NV named accumulator registers
    unconditionally -- e.g. 27*2=54 registers on a 16-register AVX2 target,
    ~3.4x the physical register file, causing severe spilling that could
    make the "specialized" kernel slower than gemm_fallback's plain loop,
    which never tries to hold that many values live at once. Each row-tile
    gets its own {}-scoped block reusing the same register names and redoing
    the full K-loop (re-reading B from memory each tile) -- cheap here since
    B for these shapes (K*N doubles) is small enough to stay resident in L1
    regardless of how many times it's re-read.
    """
    lanes = s["lanes"]
    total_regs = s["total_regs"]
    NV = N // lanes  # number of SIMD vectors covering N
    rem = N - NV * lanes  # leftover scalar columns (0 if N % lanes == 0)
    ty = s.get("type")

    if NV > 0:
        # Per-row register cost: NV SIMD accumulators + (rem>0 ? 1 : 0) for
        # the scalar tail cluster (scalar doubles also live in the same
        # xmm/ymm/zmm file on x86-64, so they compete for the same budget).
        # Reserve NV registers for B (shared across all rows in a tile) and
        # at least 1 for A-broadcast batching; the rest is split evenly
        # across rows to find how many rows fit in one tile.
        per_row = NV + (1 if rem else 0)
        rows_per_tile = max(1, min(M, (total_regs - NV) // (per_row + 1)))
        batch = max(1, min(rows_per_tile, total_regs - rows_per_tile * NV - NV))
        n_tiles = -(-M // rows_per_tile)  # ceil
        comment = (
            f"// gemm_fixed<{M},{K},{N},double>  {ty}: {rows_per_tile} rows/tile x "
            f"{NV} regs = {rows_per_tile * NV} accumulators/tile"
            + (f" + {rem} scalar tail col" if rem else "")
            + f"; B {NV} regs; A hoisted batch={batch}; {n_tiles} tile(s) over M={M}"
        )
    else:
        rows_per_tile = M
        batch = 0
        comment = (
            f"// gemm_fixed<{M},{K},{N},double>  pure scalar "
            f"(N={N} smaller than the narrowest SIMD lane width)"
        )

    lines = [
        comment,
        "template <>",
        f"inline void gemm_fixed<{M}, {K}, {N}, double>(",
        "        const double* __restrict__ A, int lda,",
        "        const double* __restrict__ B, int ldb,",
        "        double*       __restrict__ C, int ldc) {",
    ]

    row0 = 0
    while row0 < M:
        tile_rows = min(rows_per_tile, M - row0)
        lines.append("    {")
        lines.extend(_gen_row_tile(row0, tile_rows, K, s, NV, rem, lanes, ty,
                                   batch, "        "))
        lines.append("    }")
        row0 += tile_rows

    lines.append("}")
    return lines


# ---------------------------------------------------------------------------
# Per-shape kernel emit with ISA guards
# ---------------------------------------------------------------------------


def _emit_shape(M: int, K: int, N: int) -> list[str]:
    """Emit ISA-guarded specialization blocks for one (M, K, N) shape.

    Every shape gets a real generated kernel now -- previously any N not
    divisible by the chosen lane width fell through to just a comment (see
    _isa_for_n's docstring for why that silently dropped roughly half of
    real mined shapes from codegen entirely).
    """
    lines = []
    isa, lanes = _isa_for_n(N)

    if isa == "scalar":
        # N == 1: no SIMD lane fits at all. Pure scalar, no #if guard
        # needed -- _gen_kernel degenerates to scalar-only whenever NV == 0,
        # which passing a lanes value larger than N guarantees.
        lines += _gen_kernel(M, K, N, {"lanes": N + 1, "total_regs": 0})
        return lines

    s = _simd_names(isa)
    if isa == "avx512":
        avx2_s = _simd_names("avx2_256")
        lines += ["#if defined(__AVX512F__)"]
        lines += _gen_kernel(M, K, N, s)
        lines += ["#elif defined(__AVX2__) && defined(__FMA__)"]
        lines += _gen_kernel(M, K, N, avx2_s)
        lines += ["#endif"]
    else:
        lines += ["#if defined(__AVX2__) && defined(__FMA__)"]
        lines += _gen_kernel(M, K, N, s)
        lines += ["#endif"]

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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--shapes",
        required=True,
        help="Comma-separated MxKxN shapes (e.g. '5x4x4,1x2x4')",
    )
    ap.add_argument(
        "--out-dir", default=".", help="Directory for generated files (default: cwd)"
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Print register budget per shape and exit without writing",
    )
    args = ap.parse_args()

    shapes = parse_shapes(args.shapes)
    if not shapes:
        sys.exit("No valid shapes provided")

    if args.verify:
        print(
            f"{'Shape':>14}  {'ISA':>8}  {'lanes':>5}  {'NV':>3}  {'rem':>3}  "
            f"{'rows/tile':>9}  {'acc/tile':>8}  {'batch':>5}  {'tiles':>5}"
        )
        for M, K, N in shapes:
            isa, lanes = _isa_for_n(N)
            if isa == "scalar":
                print(
                    f"  {M}x{K}x{N:>2}         scalar    —    —    {N:>3}"
                    f"          —         —      —      1 (pure scalar kernel)"
                )
                continue
            s = _simd_names(isa)
            NV = N // s["lanes"]
            rem = N - NV * s["lanes"]
            per_row = NV + (1 if rem else 0)
            rows_per_tile = max(1, min(M, (s["total_regs"] - NV) // (per_row + 1)))
            batch = max(1, min(rows_per_tile,
                               s["total_regs"] - rows_per_tile * NV - NV))
            n_tiles = -(-M // rows_per_tile)
            print(
                f"  {M}x{K}x{N:<2}  {isa:>12}  {s['lanes']:>5}  {NV:>3}  {rem:>3}  "
                f"{rows_per_tile:>9}  {rows_per_tile * NV:>8}  {batch:>5}  {n_tiles:>5}"
            )
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernels_path = out_dir / "spgemm_kernels_generated.hpp"
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
