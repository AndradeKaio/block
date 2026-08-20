#!/usr/bin/env python3
"""
validate_bsp.py — Sanity-check that BSP block coverage equals MTX NNZ, and
(with --deep) that the BSP actually holds the same VALUES as the MTX.

Two ways to pick which matrices to check:
  --matrices-csv CSV   pure filesystem job: 'name'/'group' columns from the
                        CSV are used to build out-dir/group/name/name.{mtx,bsp}
                        directly. No progress.db, no ssgetpy -- just the two
                        files on disk.
  (default)             every 'done' matrix in --output-dir's progress.db.

For each matrix, the default (fast) check verifies:
    sum(block_h[i] * block_w[i] - block_imps[i])  ==  actual MTX NNZ

The MTX NNZ is counted directly from the file (not the DB) to catch any
mismatch between what was downloaded and what ssgetpy reported. Symmetric /
Hermitian matrices are expanded to full (same as the miner does).

This NNZ-count check only catches SIZE mismatches, not VALUE or POSITION
mismatches with the same total count -- e.g. mine_matrix.cpp's read_mtx once
had a bug (scanning every %-comment line for "symmetric"/"pattern" instead of
just the format banner) that mirrored off-diagonal entries at wrong positions
for matrices whose comments happened to mention "symmetric" in prose. That
bug is now fixed, but it's exactly the kind of corruption a count-only check
can't catch reliably: --deep adds a real value-level comparison (BSP values
vs. scipy's trusted MTX reader, sparse-scale so it stays usable on large
matrices) and is off by default only because it's slower (parses every float,
not just structural counts) -- turn it on whenever you don't already trust
the miner's output, e.g. after re-mining or when investigating a suspicious
matrix.

Usage:
  python validate_bsp.py --output-dir /data
  python validate_bsp.py --output-dir /data --matrix ct20stif
  python validate_bsp.py --output-dir /data --fail-fast
  python validate_bsp.py --output-dir /data --deep
  python validate_bsp.py --output-dir /data --matrix ct20stif --deep
  python validate_bsp.py --output-dir /data --matrices-csv matrices_no_singles_32k_32k.csv --deep
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

# ── MTX NNZ counter ───────────────────────────────────────────────────────────

def _mtx_nnz(path: Path) -> int:
    """
    Return the actual UNIQUE-POSITION NNZ, matching what mine_matrix.cpp now
    produces:
      - symmetric / hermitian matrices are expanded to full (mirrored)
      - duplicate (row, col) listings count ONCE, not once per listing

    Duplicate coordinate listings are legal MatrixMarket (common in
    FEM-assembled matrices, meant to be summed at that position -- see
    mine_matrix.cpp's coo_to_csr comment) so declared header NNZ can
    legitimately be HIGHER than the true unique-position count; counting
    declared NNZ directly (the old approach) would flag that as a false
    mismatch against the now-correctly-deduplicated .bsp.

    Reads only row/col indices (skips values, so no floating-point parsing),
    but does need a full scan now rather than an early return for the
    unsymmetric case, since duplicates aren't a symmetric-only phenomenon.
    """
    is_sym = False
    header_parsed = False
    positions = set()

    with open(path, "r", errors="replace") as f:
        for i, raw in enumerate(f):
            line = raw.rstrip()

            if line.startswith("%"):
                # Only line 0 is the actual %%MatrixMarket banner (format-
                # defined); every %-comment line after it is free-text
                # documentation and must NOT be scanned for these keywords --
                # matrices routinely use "symmetric" in prose without being
                # symmetric-format. Same bug as mine_matrix.cpp's read_mtx
                # (see that file's comment) -- fixed here too: this check
                # used to share the miner's false-positive, so a spurious
                # "symmetric" match here matched the miner's own spurious
                # expansion and the NNZ counts lined up even though the
                # underlying matrix was wrong.
                if i == 0:
                    low = line.lower()
                    if "symmetric" in low or "hermitian" in low:
                        is_sym = True
                continue

            if not header_parsed:
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"bad MTX header in {path}")
                header_parsed = True
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            r, c = parts[0], parts[1]
            positions.add((r, c))
            if is_sym and r != c:
                positions.add((c, r))

    return len(positions)


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


# ── Deep (value-level) comparison ───────────────────────────────────────────────

def _load_bsp_as_csr(path: Path):
    """Load a .bsp (HDF5 block-sparse) file into a scipy CSR matrix, values
    upcast to float64. Same schema/logic used across the SpMM/SpGEMM
    validators (e.g. suite-sparse/validate_spmm_cpu.py's _load_bsp_as_csr)."""
    import h5py
    import scipy.sparse

    with h5py.File(str(path), "r") as f:
        M   = int(f.attrs["matrix_rows"])
        N   = int(f.attrs["matrix_cols"])
        br  = f["block_r"][:]
        bc  = f["block_c"][:]
        bh  = f["block_h"][:]
        bw  = f["block_w"][:]
        bo  = f["block_offsets"][:]
        vals = f["values"][:].astype(np.float64)

    rows_list, cols_list, data_list = [], [], []
    for k in range(len(br)):
        r, c, h, w, off = int(br[k]), int(bc[k]), int(bh[k]), int(bw[k]), int(bo[k])
        block = vals[off: off + h * w].reshape(h, w)
        ri, ci = np.nonzero(block)
        rows_list.append(ri + r)
        cols_list.append(ci + c)
        data_list.append(block[ri, ci])

    if rows_list:
        all_r = np.concatenate(rows_list)
        all_c = np.concatenate(cols_list)
        all_d = np.concatenate(data_list)
    else:
        all_r = all_c = all_d = np.array([], dtype=np.float64)

    return scipy.sparse.csr_matrix((all_d, (all_r, all_c)), shape=(M, N), dtype=np.float64)


def _compare_bsp_values(bsp_path: Path, mtx_path: Path,
                        rtol: float = 1e-5, atol: float = 1e-6):
    """Compare the BSP's actual values against scipy's own (trusted) MTX
    reader -- deliberately NOT reusing any hand-written symmetric/pattern
    detection for the reference side, so a bug in that detection logic can't
    hide from this check the way it hid from the NNZ-count check.

    Stays sparse-scale (never densifies the full M×N matrix) so it's usable
    on large mined matrices: the diff itself is sparse, and the per-entry
    tolerance is evaluated only at the diff's nonzero positions.

    The returned bsp_nnz is NOT the same quantity as _bsp_nnz()'s fast count
    above: that one counts non-implicit STORAGE POSITIONS (h*w - imps) per
    block; this one counts stored values that are actually != 0 (via
    _load_bsp_as_csr's np.nonzero() filtering, matching how
    read_matrix_binsparse<double> / the compute kernels themselves treat a
    block -- a stored 0.0 at a non-implicit position contributes nothing to
    SpMM/SpGEMM either way). A block CAN legitimately store an explicit zero
    at a real (non-implicit) position, so these two counts differing is
    NOT itself a failure signal -- only n_mismatched/max_abs_err below are.

    Returns (ok, n_mismatched, max_abs_err, bsp_nnz, mtx_nnz).
    """
    import scipy.io
    import scipy.sparse
    import warnings

    S_bsp = _load_bsp_as_csr(bsp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_mtx = scipy.io.mmread(str(mtx_path)).tocsr().astype(np.float64)

    if S_bsp.shape != S_mtx.shape:
        return False, -1, float("nan"), S_bsp.nnz, S_mtx.nnz

    diff = (S_bsp - S_mtx).tocoo()
    if diff.nnz == 0:
        return True, 0, 0.0, S_bsp.nnz, S_mtx.nnz

    ref_at_diff = np.asarray(S_mtx[diff.row, diff.col]).ravel()
    abs_diff = np.abs(diff.data)
    scale = atol + rtol * np.abs(ref_at_diff)
    bad = abs_diff > scale

    return (not bad.any(), int(bad.sum()), float(abs_diff.max()), S_bsp.nnz, S_mtx.nnz)


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


def load_name_group_pairs_from_csv(csv_path: Path) -> list[dict]:
    """--matrices-csv mode: name+group straight from the CSV, no DB and no
    ssgetpy involved -- the job is just "does this dataset path's .bsp match
    its .mtx", and the CSV already has everything needed to build that path
    (out_root/group/name/name.{mtx,bsp}). Returns dicts (not sqlite3.Row) so
    the main loop's row["name"]/row["grp"] access works identically for
    either source."""
    import csv as _csv
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        if reader.fieldnames is None or "name" not in reader.fieldnames \
                or "group" not in reader.fieldnames:
            sys.exit(f"{csv_path} needs 'name' and 'group' columns; got: {reader.fieldnames}")
        return [{"name": row["name"], "grp": row["group"]} for row in reader]


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
    p.add_argument("--matrices-csv", default=None, dest="matrices_csv", metavar="CSV",
                   help="Validate exactly the matrices in this CSV's 'name'/'group' "
                        "columns (e.g. the same list passed to mine_matrices.py's own "
                        "--matrices-csv) by checking out-dir/group/name/name.{mtx,bsp} "
                        "directly. Does NOT touch progress.db or ssgetpy at all -- pure "
                        "filesystem comparison. Overrides --matrix / the whole-DB scan.")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop on first mismatch")
    p.add_argument("--deep", action="store_true",
                   help="Also compare actual BSP values against scipy's MTX reader "
                        "(slower -- parses every float instead of just structural "
                        "counts -- but catches value/position corruption a matching "
                        "NNZ count can miss)")
    p.add_argument("--rtol", type=float, default=1e-5,
                   help="--deep relative tolerance (default 1e-5, float32 storage "
                        "truncation is ~1e-7 relative; this leaves margin)")
    p.add_argument("--atol", type=float, default=1e-6,
                   help="--deep absolute tolerance (default 1e-6)")
    return p.parse_args()


def main():
    args     = parse_args()
    out_root = Path(args.output_dir)

    if args.matrices_csv:
        # Pure filesystem job: name+group from the CSV, no DB, no ssgetpy --
        # just check whether out_root/group/name/name.{mtx,bsp} agree.
        rows = load_name_group_pairs_from_csv(Path(args.matrices_csv))
    else:
        db_path = out_root / "progress.db"
        if not db_path.exists():
            sys.exit(f"No progress.db at {db_path}")
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        rows = fetch_done(con, args.matrix)
        con.close()

    if not rows:
        sys.exit("No matrices found to validate.")

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

        # ── deep (value-level) compare ─────────────────────────────────────────
        if args.deep:
            try:
                deep_ok, n_bad, max_err, deep_bsp_nnz, deep_mtx_nnz = \
                    _compare_bsp_values(bsp_path_, mtx_path, args.rtol, args.atol)
                if deep_ok:
                    print(f"{'':<{W}}  \033[32mdeep PASS\033[0m  "
                          f"(nonzero-valued entries via scipy, NOT directly "
                          f"comparable to bsp_nnz above: {deep_bsp_nnz:,}/{deep_mtx_nnz:,})")
                else:
                    print(f"{'':<{W}}  \033[31mdeep FAIL\033[0m  "
                          f"{n_bad} entries differ beyond tolerance  "
                          f"max_err={max_err:.3g}  "
                          f"(nonzero-valued entries via scipy, NOT directly "
                          f"comparable to bsp_nnz above: {deep_bsp_nnz:,}/{deep_mtx_nnz:,})")
                    ok = False
            except Exception as e:
                print(f"{'':<{W}}  \033[31mdeep ERROR\033[0m  {e}")
                ok = False

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
