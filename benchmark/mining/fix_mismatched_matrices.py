#!/usr/bin/env python3
"""
fix_mismatched_matrices.py — Detect and fix matrices whose canonical .mtx
(the one find_mtx()/benchmark_spmm.py would pick) or paired .bsp doesn't
match the matrix's true shape.

Root cause (confirmed by direct inspection): mine_matrices.py's download
step (process_matrix(), lines ~304-323) does:
    candidates = list(mat_dir.rglob("*.mtx"))
    if candidates[0] != mtx_path:
        candidates[0].rename(mtx_path)
picking whichever .mtx rglob() returns first (filesystem order, not
guaranteed) when a SuiteSparse archive extracts to a nested directory or
bundles more than one .mtx (companion RHS/E/B/C/D system matrices). For
TSOPF_FS_b9_c1 this put a 2454x1 RHS vector at the canonical
TSOPF_FS_b9_c1.mtx path while the real 2454x2454 matrix sits one level
deeper, still correctly shaped. For inlet, the canonical .mtx is actually
fine (11730x11730, matches SuiteSparse metadata exactly) but its .bsp
(shape [1,11730], 64 stored) was mined from something else entirely --
a stale/mismatched .bsp, not a wrong .mtx.

This script checks BOTH failure modes per matrix:
  1. WRONG_MTX  — canonical .mtx shape != expected (rows,cols) from the
                  input CSV's ssgetpy-sourced metadata, but some OTHER
                  .mtx under the matrix's directory tree does match.
  2. STALE_BSP  — canonical .mtx shape is correct, but the paired .bsp's
                  own recorded (matrix_rows,matrix_cols) doesn't match it.
  3. NO_MATCH   — canonical .mtx is wrong and no candidate under the
                  directory matches expected shape either (needs a fresh
                  download, not fixable by rearranging local files).
  4. OK         — no issue.

Usage:
  python fix_mismatched_matrices.py matrices.csv                 # report only
  python fix_mismatched_matrices.py matrices.csv --fix           # apply fixes
  python fix_mismatched_matrices.py matrices.csv --fix --data-root /path
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_DATA_ROOT = Path("/home/kaio/datasets/suite-sparse")
_MINER = _SCRIPT_DIR / "mine_matrix"


def read_mtx_header(path: Path) -> tuple[int, int, int, bool] | None:
    """Return (rows, cols, nnz, is_symmetric) from a MatrixMarket header,
    without reading the data. None if unparseable."""
    try:
        with open(path, "r", errors="replace") as f:
            first = f.readline()
            if not first.startswith("%%MatrixMarket"):
                return None
            is_symmetric = "symmetric" in first.lower() or "hermitian" in first.lower()
            for line in f:
                if line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    return None
                return int(parts[0]), int(parts[1]), int(parts[2]), is_symmetric
    except (OSError, ValueError):
        return None
    return None


def read_bsp_shape(path: Path) -> tuple[int, int] | None:
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return int(f.attrs["matrix_rows"]), int(f.attrs["matrix_cols"])
    except Exception:
        return None


def find_mtx_canonical(mat_dir: Path, name: str) -> Path | None:
    """Mirrors benchmark_spmm.py's find_mtx(): flat path first, else first
    rglob match."""
    flat = mat_dir / f"{name}.mtx"
    if flat.is_file() and flat.stat().st_size > 0:
        return flat
    candidates = sorted(mat_dir.rglob(f"{name}.mtx"))
    return candidates[0] if candidates else None


def scan_matrix(data_root: Path, name: str, group: str,
                exp_rows: int, exp_cols: int) -> dict:
    mat_dir = data_root / group / name if group else data_root / name
    result = {"name": name, "group": group, "mat_dir": mat_dir,
             "status": "OK", "detail": ""}

    if not mat_dir.is_dir():
        result["status"] = "MISSING_DIR"
        return result

    canonical = find_mtx_canonical(mat_dir, name)
    if canonical is None:
        result["status"] = "NO_MTX"
        return result
    result["canonical_mtx"] = canonical

    hdr = read_mtx_header(canonical)
    if hdr is None:
        result["status"] = "UNREADABLE_MTX"
        return result
    rows, cols, nnz, _ = hdr
    result["canonical_shape"] = (rows, cols)

    mtx_ok = (rows == exp_rows and cols == exp_cols)

    if not mtx_ok:
        # Search every other .mtx under the tree for a shape match.
        all_candidates = sorted(set(mat_dir.rglob("*.mtx")) - {canonical})
        matches = []
        for cand in all_candidates:
            h = read_mtx_header(cand)
            if h and h[0] == exp_rows and h[1] == exp_cols:
                matches.append(cand)
        if len(matches) == 1:
            result["status"] = "WRONG_MTX"
            result["correct_mtx"] = matches[0]
            result["detail"] = (f"canonical={canonical.relative_to(mat_dir)} "
                                f"is {rows}x{cols} (expected {exp_rows}x{exp_cols}); "
                                f"found correct shape at "
                                f"{matches[0].relative_to(mat_dir)}")
        elif len(matches) > 1:
            result["status"] = "AMBIGUOUS"
            result["candidates"] = matches
            result["detail"] = (f"canonical is {rows}x{cols}, wrong; "
                                f"{len(matches)} other files also match "
                                f"{exp_rows}x{exp_cols} -- can't pick automatically")
        else:
            result["status"] = "NO_MATCH"
            result["detail"] = (f"canonical is {rows}x{cols} (expected "
                                f"{exp_rows}x{exp_cols}); no other .mtx under "
                                f"{mat_dir} matches either -- needs re-download")
        return result

    # .mtx is correct -- now check the paired .bsp.
    bsp_path = mat_dir / f"{name}.bsp"
    if not bsp_path.exists():
        result["status"] = "NO_BSP"
        return result
    bsp_shape = read_bsp_shape(bsp_path)
    if bsp_shape is None:
        result["status"] = "UNREADABLE_BSP"
        return result
    if bsp_shape != (exp_rows, exp_cols):
        result["status"] = "STALE_BSP"
        result["bsp_shape"] = bsp_shape
        result["detail"] = (f".mtx is correct ({rows}x{cols}) but .bsp records "
                            f"{bsp_shape} -- needs re-mining")
    return result


def apply_fix(r: dict, dry_run: bool) -> str:
    mat_dir: Path = r["mat_dir"]
    name = r["name"]
    canonical = mat_dir / f"{name}.mtx"
    bsp_path = mat_dir / f"{name}.bsp"

    if r["status"] == "WRONG_MTX":
        correct = r["correct_mtx"]
        wrong_backup = canonical.with_suffix(".mtx.wrong")
        if dry_run:
            return (f"[DRY] would move {canonical.name} -> {wrong_backup.name}, "
                    f"copy {correct.relative_to(mat_dir)} -> {canonical.name}, "
                    f"re-mine {bsp_path.name}")
        if not wrong_backup.exists():
            shutil.move(str(canonical), str(wrong_backup))
        shutil.copy(str(correct), str(canonical))
    elif r["status"] != "STALE_BSP":
        return f"[SKIP] status={r['status']}, not auto-fixable"

    if dry_run:
        return f"[DRY] would re-mine {bsp_path.name} from corrected {canonical.name}"

    cmd = [str(_MINER), str(canonical), str(bsp_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        return f"[FAIL] mine_matrix: {proc.stderr.strip()[-500:]}"
    new_shape = read_bsp_shape(bsp_path)
    return f"[FIXED] re-mined, new .bsp shape={new_shape}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    ap.add_argument("--fix", action="store_true",
                    help="apply fixes (default: report only)")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(r for r in f if not r.startswith("#")))

    results = []
    for row in rows:
        r = scan_matrix(args.data_root, row["name"], row.get("group", ""),
                        int(row["rows"]), int(row["cols"]))
        results.append(r)

    by_status: dict[str, list] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)

    print(f"Scanned {len(results)} matrices under {args.data_root}\n")
    for status in ["OK", "STALE_BSP", "WRONG_MTX", "AMBIGUOUS", "NO_MATCH",
                   "NO_BSP", "MISSING_DIR", "NO_MTX", "UNREADABLE_MTX",
                   "UNREADABLE_BSP"]:
        group = by_status.get(status, [])
        if not group:
            continue
        print(f"=== {status} ({len(group)}) ===")
        for r in group:
            if status == "OK":
                continue
            print(f"  {r['name']:<20} {r.get('detail', '')}")
        print()

    if args.fix:
        fixable = by_status.get("WRONG_MTX", []) + by_status.get("STALE_BSP", [])
        print(f"--- Applying fixes to {len(fixable)} matrices ---")
        for r in fixable:
            msg = apply_fix(r, dry_run=False)
            print(f"  {r['name']:<20} {msg}")
    else:
        fixable = by_status.get("WRONG_MTX", []) + by_status.get("STALE_BSP", [])
        if fixable:
            print(f"{len(fixable)} matrices are auto-fixable with --fix "
                 f"(WRONG_MTX + STALE_BSP)")


if __name__ == "__main__":
    main()
