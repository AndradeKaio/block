#!/usr/bin/env python3
"""
gen_spmm_kernels.py — Generate named-register AVX-512/AVX2 specializations
of spmm_chunk<H,W,NR> and the spmm_dispatch() table.

Two outputs (always written together, same directory):
  spmm_kernels_generated.hpp  — template specializations with named registers
  spmm_dispatch_table.hpp     — spmm_dispatch() function body

Usage:
    # Use hardcoded shape table (default):
    python3 gen_spmm_kernels.py

    # Use specific shapes (auto-selects NR from H):
    python3 gen_spmm_kernels.py --shapes 6x6,5x4,18x6,3x7

    # Print register budget per shape and exit:
    python3 gen_spmm_kernels.py --verify [--shapes ...]
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default shape table — (H, W, NR512, NR256_or_None)
# Used when --shapes is not given.
# ---------------------------------------------------------------------------

_DEFAULT_SHAPES = [
    # H    W   NR512  NR256
    ( 6,   6,   32,    8),   # conf5/conf6, t520, pkustk02, cegb*, lock*
    (18,   6,    8,    None), # cyl6, lock3491  (H=18 → NR512=8; skip AVX2)
    ( 5,   4,   32,    8),   # olm1000/2000/5000
    ( 6,  12,   32,    8),   # s1rmt3m1, s3rmt3m1
    ( 3,   6,   32,    8),   # opt1
    ( 6,   3,   32,    8),   # msc10848
    ( 5,   5,   32,    8),   # k3plates
    ( 6,   2,   32,    8),   # Kuu, cegb3024
    ( 9,   2,   24,    4),   # linverse
    (12,   4,   16,    4),   # nasa2146
    ( 8,   2,   24,    4),   # lung1
    (10,   2,   16,    4),   # ex9
    (13,   5,   16,    4),   # man_5976
    ( 3,   7,   32,    8),   # bundle1
    (18,  18,    8,    None), # cyl6 biggest_pattern
    ( 8,   8,   24,    4),   # Kuu biggest_pattern
    (12,  12,   16,    4),   # nasa2146-adjacent
]


# ---------------------------------------------------------------------------
# NR budget rules — maximise FMA chains within register budget.
# AVX-512 (32 ZMM): keep H*(NR/8) ≤ 28  (leaves 4 for b-vecs + scratch)
# AVX2    (16 YMM): keep H*(NR/4) ≤ 12  (leaves 4 for b-vecs + scratch)
# ---------------------------------------------------------------------------

def nr_for_h(h: int) -> tuple[int, int | None]:
    """Return (nr512, nr256) for a given H."""
    if h <= 6:
        return 32, 8
    elif h <= 9:
        return 24, 4
    elif h <= 13:
        return 16, 4
    else:
        return 8, None   # H ≥ 14; skip AVX2 specialization


# ---------------------------------------------------------------------------
# Code generation helpers
# ---------------------------------------------------------------------------

def gen_avx512(H: int, W: int, NR: int) -> list[str]:
    """Generate AVX-512 spmm_chunk specialisation with hoisted A-broadcasts.

    AVX-512 has 32 ZMM registers.  The budget after pinning H*NV accumulator
    registers and NV B-vector registers is:

        spare = 32 - H*NV - NV

    We broadcast each A[i*lda+p] element into a named ZMM once per (i,p) pair
    and reuse it across all NV FMAs for that row, rather than letting the
    compiler find the CSE opportunity.  Rows are processed in batches of
    min(H, spare) so we never exceed 32 named ZMMs.
    """
    NV = NR // 8
    spare = 32 - H * NV - NV          # ZMMs available for A-broadcasts
    batch = max(1, min(H, spare))      # rows per A-hoisting batch

    acc_count = H * NV
    b_count = NV
    comment = (
        f"// {H}×{W}, NR={NR}: {H} rows × {NV} ZMM = {acc_count} accumulators; "
        f"A hoisted in batches of {batch} rows ({acc_count}+{b_count}+{batch}≤32)"
    )
    lines = [
        comment,
        "template <>",
        f"inline void spmm_chunk<{H}, {W}, {NR}>(",
        "        const double* __restrict__ A, int lda,",
        "        const double* __restrict__ B, int ldb,",
        "        double*       __restrict__ C, int ldc) {",
    ]
    # Load accumulator registers.
    for i in range(H):
        for v in range(NV):
            lines.append(f"    __m512d c{i}_{v} = _mm512_loadu_pd(C + {i}*ldc + {v*8});")

    lines.append(f"    for (int p = 0; p < {W}; ++p) {{")

    # Load B-vectors once per p.
    for v in range(NV):
        lines.append(f"        __m512d b{v} = _mm512_loadu_pd(B + p*ldb + {v*8});")

    # Emit rows in A-broadcast batches.
    row = 0
    first_batch = True
    while row < H:
        end = min(row + batch, H)
        batch_h = end - row

        if first_batch:
            # Declare A-broadcast variables on first batch.
            for k in range(batch_h):
                lines.append(f"        __m512d a{k} = _mm512_set1_pd(A[{row+k}*lda + p]);")
            first_batch = False
        else:
            # Reuse the same a0..a{batch-1} variable names (assignment, not declaration).
            for k in range(batch_h):
                lines.append(f"        a{k} = _mm512_set1_pd(A[{row+k}*lda + p]);")

        for k in range(batch_h):
            for v in range(NV):
                lines.append(f"        c{row+k}_{v} = _mm512_fmadd_pd(a{k}, b{v}, c{row+k}_{v});")

        row = end

    lines.append("    }")

    # Store accumulator registers.
    for i in range(H):
        for v in range(NV):
            lines.append(f"    _mm512_storeu_pd(C + {i}*ldc + {v*8}, c{i}_{v});")
    lines.append("}")
    return lines


def gen_avx2(H: int, W: int, NR: int) -> list[str]:
    """Generate AVX2+FMA spmm_chunk specialisation with hoisted A-broadcasts.

    AVX2 has 16 YMM registers.  Same batching logic as gen_avx512 but using
    16-register budget and YMM (4-double) vectors.
    """
    NV = NR // 4
    spare = 16 - H * NV - NV
    batch = max(1, min(H, spare))

    acc_count = H * NV
    b_count = NV
    comment = (
        f"// {H}×{W}, NR={NR}: {H} rows × {NV} YMM = {acc_count} accumulators; "
        f"A hoisted in batches of {batch} rows ({acc_count}+{b_count}+{batch}≤16)"
    )
    lines = [
        comment,
        "template <>",
        f"inline void spmm_chunk<{H}, {W}, {NR}>(",
        "        const double* __restrict__ A, int lda,",
        "        const double* __restrict__ B, int ldb,",
        "        double*       __restrict__ C, int ldc) {",
    ]
    for i in range(H):
        for v in range(NV):
            lines.append(f"    __m256d c{i}_{v} = _mm256_loadu_pd(C + {i}*ldc + {v*4});")

    lines.append(f"    for (int p = 0; p < {W}; ++p) {{")

    for v in range(NV):
        lines.append(f"        __m256d b{v} = _mm256_loadu_pd(B + p*ldb + {v*4});")

    row = 0
    first_batch = True
    while row < H:
        end = min(row + batch, H)
        batch_h = end - row

        if first_batch:
            for k in range(batch_h):
                lines.append(f"        __m256d a{k} = _mm256_broadcast_sd(A + {row+k}*lda + p);")
            first_batch = False
        else:
            for k in range(batch_h):
                lines.append(f"        a{k} = _mm256_broadcast_sd(A + {row+k}*lda + p);")

        for k in range(batch_h):
            for v in range(NV):
                lines.append(f"        c{row+k}_{v} = _mm256_fmadd_pd(a{k}, b{v}, c{row+k}_{v});")

        row = end

    lines.append("    }")

    for i in range(H):
        for v in range(NV):
            lines.append(f"    _mm256_storeu_pd(C + {i}*ldc + {v*4}, c{i}_{v});")
    lines.append("}")
    return lines


def gen_dispatch_table(shapes: list[tuple]) -> list[str]:
    """Generate the spmm_dispatch() function body.

    Dispatches on a switch over a packed (H,W) key rather than a linear
    if-chain: a switch over sparse-but-clustered integer cases compiles to a
    jump table (or a compact binary-search of range checks, compiler's
    choice) giving O(1)-ish, branch-predictor-friendly dispatch, instead of
    up to len(shapes) sequential compares every call. This is called once
    per block, and block counts run into the tens of thousands on some
    matrices, so the difference between "1 comparison" and "10 comparisons"
    per call is a real, if modest, amount of avoidable branching in the hot
    path. H and W are packed as (H << 16) | W — block dimensions here are at
    most a few hundred, far under the 16-bit-per-field headroom this needs.
    Falls back to gemm_fallback for shapes not in the table.
    """
    lines = [
        "// AUTO-GENERATED by gen_spmm_kernels.py — do not edit.",
        "// spmm_dispatch — runtime (H,W) → spmm_kernel<H,W,NR> dispatch via a",
        "// switch on a packed (H,W) key (jump table, not a linear if-chain).",
        "// NR is selected per ISA to maximise independent FMA chains.",
        "// Falls back to gemm_fallback for shapes not in the table.",
        "inline void spmm_dispatch(int H, int W, int N,",
        "                          const double* A, int lda,",
        "                          const double* B, int ldb,",
        "                          double*       C, int ldc) {",
        "  switch ((H << 16) | W) {",
    ]
    for H, W, nr512, nr256 in shapes:
        nr256_val = nr256 if nr256 is not None else nr512
        lines.append(f"    case ({H} << 16) | {W}:")
        lines.append("#if defined(__AVX512F__)")
        lines.append(f"      spmm_detail::spmm_kernel<{H}, {W}, {nr512}>(N, A, lda, B, ldb, C, ldc);")
        lines.append("#else")
        lines.append(f"      spmm_detail::spmm_kernel<{H}, {W}, {nr256_val}>(N, A, lda, B, ldb, C, ldc);")
        lines.append("#endif")
        lines.append("      return;")
    lines += [
        "    default: break;",
        "  }",
        "  benchmark_core::cpu_detail::gemm_fallback(H, W, N, A, lda, B, ldb, C, ldc);",
        "}",
    ]
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_shapes_arg(s: str, min_area: int = 1) -> list[tuple[int, int, int, int | None]]:
    """Parse 'H1xW1,H2xW2,...' into shape tuples with auto-selected NR.

    Shapes where H*W < min_area are silently dropped.  min_area=1 keeps all.
    """
    shapes = []
    seen = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            h_str, w_str = token.lower().split("x")
            H, W = int(h_str), int(w_str)
        except ValueError:
            sys.exit(f"Invalid shape '{token}' — expected HxW (e.g. 6x6)")
        if (H, W) in seen:
            continue
        if H * W < min_area:
            continue
        seen.add((H, W))
        nr512, nr256 = nr_for_h(H)
        shapes.append((H, W, nr512, nr256))
    return shapes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="",
                    help="Comma-separated list of HxW shapes (e.g. '6x6,5x4,18x6'). "
                         "NR is auto-selected from H. Omit to use the built-in table.")
    ap.add_argument("--min-area", type=int, default=1, dest="min_area",
                    help="Minimum block area (H*W) required to generate a specialization. "
                         "Shapes below this threshold are dropped even when listed via "
                         "--shapes. Default: 1 (no filter; use 4 to exclude singletons "
                         "and thin slabs that don't benefit over gemm_fallback).")
    ap.add_argument("--verify", action="store_true",
                    help="Print register counts per shape and exit without writing files.")
    ap.add_argument("--out", default=None,
                    help="Path for spmm_kernels_generated.hpp "
                         "(default: next to this script). "
                         "spmm_dispatch_table.hpp is written to the same directory.")
    args = ap.parse_args()

    # Resolve shape list.
    if args.shapes:
        shapes = parse_shapes_arg(args.shapes, min_area=args.min_area)
    else:
        # Apply min_area filter to the built-in table too.
        shapes = [(H, W, nr512, nr256)
                  for H, W, nr512, nr256 in _DEFAULT_SHAPES
                  if H * W >= args.min_area]

    if args.verify:
        print(f"{'Shape':>12}  {'NR512':>5}  {'ZMM acc':>7}  {'NR256':>5}  {'YMM acc':>7}")
        for H, W, nr512, nr256 in shapes:
            zmm = H * (nr512 // 8)
            if nr256 is not None:
                ymm_s = str(H * (nr256 // 4))
            else:
                ymm_s = "skip"
            nr256_disp = "—" if nr256 is None else str(nr256)
            print(f"  {H:>2}×{W:<2}       {nr512:>5}  {zmm:>7}  {nr256_disp:>5}  {ymm_s:>7}")
        return

    # Resolve output paths.
    script_dir = Path(__file__).parent
    out_kernels  = Path(args.out) if args.out else script_dir / "spmm_kernels_generated.hpp"
    out_dispatch = out_kernels.parent / "spmm_dispatch_table.hpp"

    # --- spmm_kernels_generated.hpp ---
    kern_sections = [
        "// AUTO-GENERATED by gen_spmm_kernels.py — do not edit.\n"
        "// Named SIMD register variables (c{row}_{vec}) guarantee register allocation.\n"
        "#pragma once\n",
    ]

    avx512_lines = []
    for H, W, nr512, _ in shapes:
        avx512_lines += gen_avx512(H, W, nr512)
        avx512_lines.append("")

    avx2_lines = []
    for H, W, _, nr256 in shapes:
        if nr256 is None:
            avx2_lines.append(
                f"// {H}×{W}: skipped for AVX2 (H too large for named-register budget)."
            )
            avx2_lines.append("")
        else:
            avx2_lines += gen_avx2(H, W, nr256)
            avx2_lines.append("")

    kern_sections.append("#if defined(__AVX512F__)\n")
    kern_sections.append("\n".join(avx512_lines))
    kern_sections.append("#elif defined(__AVX2__) && defined(__FMA__)\n")
    kern_sections.append("\n".join(avx2_lines))
    kern_sections.append("#endif  // __AVX512F__ / __AVX2__\n")

    out_kernels.write_text("\n".join(kern_sections))

    # --- spmm_dispatch_table.hpp ---
    out_dispatch.write_text("\n".join(gen_dispatch_table(shapes)) + "\n")

    print(f"Wrote {len(shapes)} shapes → {out_kernels.name}, {out_dispatch.name}")
    n512 = sum(1 for H, W, nr, _ in shapes for _ in range(H * (nr // 8)))
    print(f"  AVX-512 named accumulator registers: {n512}")


if __name__ == "__main__":
    main()
