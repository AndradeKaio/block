#!/usr/bin/env python3
"""
validate_bsp.py — Sanity-check that BSP block coverage equals MTX NNZ.

For every done matrix in the output directory, verifies:
    sum(block_h[i] * block_w[i] - block_imps[i])  ==  actual MTX NNZ

The MTX NNZ is counted directly from the file (not the DB) to catch any
mismatch between what was downloaded and what ssgetpy reported. Symmetric /
Hermitian matrices are expanded to full (same as the miner does).

Usage:
  python validate_bsp.py --output-dir /data
  python validate_bsp.py --output-dir /data --matrix ct20stif
  python validate_bsp.py --output-dir /data --fail-fast
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

# ── MTX NNZ counter ───────────────────────────────────────────────────────────

def _mtx_nnz(path: Path) -> int:
    """
    Return the actual NNZ as the miner sees it:
      - general / unsymmetric: declared NNZ from header
      - symmetric / hermitian: 2 * declared − diagonal_count  (full expansion)

    Reads only the header + row/col indices (skips values). Fast even for
    large files because no floating-point parsing is done.
    """
    is_sym = False
    header_parsed = False
    declared = 0
    diag = 0

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip()

            if line.startswith("%"):
                low = line.lower()
                if "symmetric" in low or "hermitian" in low:
                    is_sym = True
                continue

            if not header_parsed:
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"bad MTX header in {path}")
                # M N [NNZ] — NNZ may be absent for some files
                declared = int(parts[2]) if len(parts) >= 3 else 0
                header_parsed = True
                if not is_sym:
                    return declared
                continue

            # symmetric: scan for diagonal entries (r == c)
            parts = line.split()
            if len(parts) >= 2 and parts[0] == parts[1]:
                diag += 1

    return 2 * declared - diag


# ── BSP NNZ counter ───────────────────────────────────────────────────────────

def _bsp_nnz(path: Path):
    """
    Return sum(h*w - imps) over all blocks.
    Reads only block_h, block_w, block_imps (skips values).
    Returns (computed_nnz, n_blocks, n_singles, total_imps).
    """
    try:
        import h5py
    except ImportError:
        sys.exit("h5py is required: pip install h5py")

    with h5py.File(str(path), "r") as f:
        h    = f["block_h"][:]
        w    = f["block_w"][:]
        imps = f["block_imps"][:]

    h    = h.astype(np.int64)
    w    = w.astype(np.int64)
    imps = imps.astype(np.int64)

    nnz_per_block = h * w - imps
    n_singles     = int(np.sum((h == 1) & (w == 1)))
    total_imps    = int(imps.sum())
    computed_nnz  = int(nnz_per_block.sum())

    return computed_nnz, len(h), n_singles, total_imps


# ── DB helpers ────────────────────────────────────────────────────────────────

def fetch_done(con, name_filter=None):
    where = "AND m.name = ?" if name_filter else ""
    params = (name_filter,) if name_filter else ()
    return con.execute(f"""
        SELECT m.name, m.grp, m.nnz AS db_nnz
        FROM matrices m
        WHERE m.status = 'done'
          {where}
        ORDER BY m.name
    """, params).fetchall()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate BSP NNZ coverage against MTX source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="/data",
                   help="Directory containing progress.db and matrix subdirs")
    p.add_argument("--matrix", default=None,
                   help="Validate a single matrix by name")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop on first mismatch")
    return p.parse_args()


def main():
    args     = parse_args()
    out_root = Path(args.output_dir)
    db_path  = out_root / "progress.db"

    if not db_path.exists():
        sys.exit(f"No progress.db at {db_path}")

    con  = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = fetch_done(con, args.matrix)
    con.close()

    if not rows:
        sys.exit("No done matrices found.")

    passed  = 0
    failed  = 0
    skipped = 0

    W = 34  # name column width

    print(f"\n{'matrix':<{W}}  {'mtx_nnz':>12}  {'bsp_nnz':>12}  "
          f"{'blocks':>8}  {'singles':>8}  {'imps':>10}  status")
    print("─" * 110)

    for row in rows:
        name = row["name"]
        grp  = row["grp"] or ""
        mat_dir = out_root / grp / name

        mtx_path = mat_dir / f"{name}.mtx"
        bsp_path_ = mat_dir / f"{name}.bsp"

        # ── check files exist ─────────────────────────────────────────────────
        if not mtx_path.exists():
            print(f"{name:<{W}}  {'—':>12}  {'—':>12}  {'—':>8}  {'—':>8}  {'—':>10}  "
                  f"\033[33mSKIP (no mtx)\033[0m")
            skipped += 1
            continue

        if not bsp_path_.exists():
            print(f"{name:<{W}}  {'—':>12}  {'—':>12}  {'—':>8}  {'—':>8}  {'—':>10}  "
                  f"\033[33mSKIP (no bsp)\033[0m")
            skipped += 1
            continue

        # ── count NNZ ────────────────────────────────────────────────────────
        try:
            mtx_nnz = _mtx_nnz(mtx_path)
        except Exception as e:
            print(f"{name:<{W}}  ERROR reading mtx: {e}")
            skipped += 1
            continue

        try:
            bsp_nnz, n_blocks, n_singles, total_imps = _bsp_nnz(bsp_path_)
        except Exception as e:
            print(f"{name:<{W}}  ERROR reading bsp: {e}")
            skipped += 1
            continue

        # ── compare ───────────────────────────────────────────────────────────
        ok = (mtx_nnz == bsp_nnz)
        status = "\033[32mPASS\033[0m" if ok else \
                 f"\033[31mFAIL  delta={bsp_nnz - mtx_nnz:+,}\033[0m"

        print(f"{name:<{W}}  {mtx_nnz:>12,}  {bsp_nnz:>12,}  "
              f"{n_blocks:>8,}  {n_singles:>8,}  {total_imps:>10,}  {status}")

        if ok:
            passed += 1
        else:
            failed += 1
            if args.fail_fast:
                print("\n[fail-fast] stopping on first mismatch.")
                break

    # ── summary ───────────────────────────────────────────────────────────────
    print("─" * 110)
    total = passed + failed + skipped
    print(f"\n{total} checked  —  "
          f"\033[32m{passed} passed\033[0m  "
          f"\033[31m{failed} failed\033[0m  "
          f"\033[33m{skipped} skipped\033[0m\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
