#!/usr/bin/env python3
"""
count_runs.py — Contiguous-NNZ row analysis for SuiteSparse matrices.

For each .mtx file under --data-dir (layout: <group>/<name>/<name>.mtx), loads the
matrix and analyses every row for its longest unbroken run of consecutive column
indices. Rows with identical NNZ column positions share a hash.

Per-matrix stats written to --output CSV:
  group, name, rows, cols, nnz,
  max_run        — longest contiguous run in any row
  mean_max_run   — average per-row max run
  n_unique_rows  — distinct row patterns (by exact column-position hash)
  n_zero_rows    — rows with no NNZ
  top1_count     — how many rows share the most common pattern
  top1_hash      — hash of that pattern
  top1_run       — its max run

Usage:
  python count_runs.py --data-dir ~/datasets/suite-sparse --output runs.csv
  python count_runs.py --data-dir ~/datasets/suite-sparse --output runs.csv \\
      --workers 8 --detail-dir /tmp/run_details
  python count_runs.py --data-dir ~/datasets/suite-sparse --output runs.csv \\
      --matrix bcsstk27 HB/bcsstk28 --verbose
"""

import argparse
import csv
import hashlib
import json
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Per-row analysis
# ---------------------------------------------------------------------------

_EMPTY_HASH = "0" * 16


def _analyze_row(cols: np.ndarray) -> tuple[int, str]:
    """Return (max_run, hex_hash) for one row's sorted column index array."""
    if len(cols) == 0:
        return 0, _EMPTY_HASH
    if len(cols) == 1:
        max_run = 1
    else:
        diffs = np.diff(cols)
        breaks = np.where(diffs != 1)[0]
        if len(breaks) == 0:
            max_run = len(cols)
        else:
            ends   = np.concatenate([breaks, [len(cols) - 1]])
            starts = np.concatenate([[-1], breaks])
            max_run = int((ends - starts).max())
    h = hashlib.blake2b(cols.tobytes(), digest_size=8).hexdigest()
    return max_run, h


# ---------------------------------------------------------------------------
# Per-matrix analysis
# ---------------------------------------------------------------------------

def _analyze_matrix(group: str, name: str, mtx_path: Path,
                    top: int, verbose: bool, verbose_limit: int) -> dict | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mat = scipy.io.mmread(str(mtx_path))
            except Exception:
                # Harwell-Boeing format (no %%MatrixMarket banner)
                mat = scipy.io.hb_read(str(mtx_path))
        A = scipy.sparse.csr_matrix(mat)
    except Exception as e:
        tqdm.write(f"[FAIL] {group}/{name}: {e}")
        return None

    M, N = A.shape
    indptr  = A.indptr
    indices = A.indices

    row_runs: list[int] = []
    hash_to_run: dict[str, int] = {}
    hash_counts: Counter = Counter()

    do_verbose = verbose and M <= verbose_limit
    if verbose and M > verbose_limit:
        tqdm.write(f"  [skip verbose] {name}: {M} rows > limit {verbose_limit}")

    for r in range(M):
        cols = indices[indptr[r]:indptr[r + 1]]
        run, h = _analyze_row(cols)
        row_runs.append(run)
        hash_counts[h] += 1
        if h not in hash_to_run:
            hash_to_run[h] = run
        if do_verbose:
            tqdm.write(f"  row={r}  max_run={run}  hash={h}")

    n_zero_rows  = int(row_runs.count(0))
    max_run      = int(max(row_runs)) if row_runs else 0
    mean_max_run = round(float(np.mean(row_runs)), 4) if row_runs else 0.0
    n_unique     = len(hash_counts)

    top1_hash, top1_count = hash_counts.most_common(1)[0] if hash_counts else (_EMPTY_HASH, 0)
    top1_run = hash_to_run.get(top1_hash, 0)

    patterns = [
        {"hash": h, "count": c, "max_run": hash_to_run[h]}
        for h, c in hash_counts.most_common(top)
    ]

    return {
        "group":         group,
        "name":          name,
        "rows":          M,
        "cols":          N,
        "nnz":           int(A.nnz),
        "max_run":       max_run,
        "mean_max_run":  mean_max_run,
        "n_unique_rows": n_unique,
        "n_zero_rows":   n_zero_rows,
        "top1_count":    top1_count,
        "top1_hash":     top1_hash,
        "top1_run":      top1_run,
        "_patterns":     patterns,   # not written to CSV
    }


# ---------------------------------------------------------------------------
# Dataset walking
# ---------------------------------------------------------------------------

def _find_matrices(data_dir: Path, name_filter: set[str]) -> list[tuple[str, str, Path]]:
    """
    Walk data_dir for <group>/<name>/<name>.mtx files.
    Deduplicates by matrix name (first group wins).
    """
    seen: set[str] = set()
    results: list[tuple[str, str, Path]] = []
    for mtx_path in sorted(data_dir.glob("*/*/*.mtx")):
        name  = mtx_path.stem
        group = mtx_path.parent.parent.name
        if mtx_path.parent.name != name:
            continue  # skip sub-files / multi-file matrices
        if name in seen:
            continue  # deduplicate by name
        if name_filter and name not in name_filter and f"{group}/{name}" not in name_filter:
            continue
        seen.add(name)
        results.append((group, name, mtx_path))
    return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "group", "name", "rows", "cols", "nnz",
    "max_run", "mean_max_run",
    "n_unique_rows", "n_zero_rows",
    "top1_count", "top1_hash", "top1_run",
]


def _load_done_names(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add(row.get("name", ""))
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Contiguous-NNZ row analysis for SuiteSparse matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="~/datasets/suite-sparse",
                   help="Root of the dataset (group/name/name.mtx layout)")
    p.add_argument("--output", default="runs.csv",
                   help="Output CSV path")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel worker threads")
    p.add_argument("--top", type=int, default=10,
                   help="Top N patterns to include in --detail-dir JSON")
    p.add_argument("--detail-dir", default="", metavar="DIR",
                   help="If set, write per-matrix JSON detail files here")
    p.add_argument("--verbose", action="store_true",
                   help="Print one line per row to stdout")
    p.add_argument("--verbose-limit", type=int, default=5000, dest="verbose_limit",
                   help="Skip per-row output for matrices with more rows than this")
    p.add_argument("--matrix", nargs="*", default=[], metavar="NAME",
                   help="Filter: one or more 'name' or 'group/name' strings")
    p.add_argument("--force", action="store_true",
                   help="Re-analyse matrices already present in the output CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir   = Path(args.data_dir).expanduser()
    out_path   = Path(args.output)
    detail_dir = Path(args.detail_dir) if args.detail_dir else None

    if not data_dir.exists():
        raise SystemExit(f"data-dir not found: {data_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    name_filter = set(args.matrix) if args.matrix else set()

    matrices = _find_matrices(data_dir, name_filter)
    if not matrices:
        raise SystemExit("No .mtx files found matching the given filters.")

    done_names: set[str] = set() if args.force else _load_done_names(out_path)
    todo = [(g, n, p) for g, n, p in matrices if n not in done_names]
    skipped = len(matrices) - len(todo)

    print(f"Data dir  : {data_dir}")
    print(f"Output    : {out_path}")
    print(f"Matrices  : {len(todo)}" + (f"  ({skipped} skipped)" if skipped else ""))
    print(f"Workers   : {args.workers}")
    print()

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    f_csv = open(out_path, "a", newline="")
    writer = csv.DictWriter(f_csv, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    n_ok = n_fail = 0

    def _worker(task):
        g, n, p = task
        return _analyze_matrix(g, n, p, args.top, args.verbose, args.verbose_limit)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t for t in todo}
        with tqdm(total=len(todo), unit="mat") as bar:
            for fut in as_completed(futures):
                g, n, _ = futures[fut]
                bar.set_postfix_str(n)
                result = fut.result()
                if result is None:
                    n_fail += 1
                else:
                    writer.writerow(result)
                    f_csv.flush()
                    n_ok += 1
                    tqdm.write(
                        f"[DONE] {g}/{n}  max_run={result['max_run']}"
                        f"  unique_rows={result['n_unique_rows']}"
                        f"  top1_count={result['top1_count']}"
                    )
                    if detail_dir:
                        out_json = detail_dir / g / f"{n}.json"
                        out_json.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_json, "w") as jf:
                            json.dump({
                                "name":          result["name"],
                                "group":         result["group"],
                                "rows":          result["rows"],
                                "cols":          result["cols"],
                                "nnz":           result["nnz"],
                                "max_run":       result["max_run"],
                                "mean_max_run":  result["mean_max_run"],
                                "n_unique_rows": result["n_unique_rows"],
                                "n_zero_rows":   result["n_zero_rows"],
                                "patterns":      result["_patterns"],
                            }, jf, indent=2)
                bar.update(1)

    f_csv.close()
    print(f"\n{n_ok} done  |  {n_fail} failed  |  {out_path}")


if __name__ == "__main__":
    main()
