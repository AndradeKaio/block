#!/usr/bin/env python3
"""
suite-sparse/remine_and_validate.py — Re-mine a matrix list, then validate
the .bsp files it just overwrote, in one command.

Thin orchestrator over two independently-usable pieces:
  mining/mine_matrices.py --matrices-csv ... --force   (re-mine)
  mining/validate_bsp.py  --matrices-csv ... --deep     (validate the result)

The CSV only needs 'name' and 'group' columns -- e.g.
suite-sparse/matrices_no_singles_32k_32k.csv works as-is. mine_matrices.py
resolves each matrix's SuiteSparse id itself (its local progress.db cache
first, an ssgetpy name lookup as fallback) and writes an 'id' column back
into the CSV as a side effect of running; validate_bsp.py never needs an id
at all -- it just compares the dataset path's .mtx and .bsp directly.

Runs each step as a subprocess with output streamed live (not captured) --
both are long-running against real hardware/network, and you want to see
mining/validation progress as it happens, not after the fact.

Usage:
  python remine_and_validate.py matrices_no_singles_32k_32k.csv
  python remine_and_validate.py matrices.csv --output-dir /home/kaio/datasets/suite-sparse
  python remine_and_validate.py matrices.csv --skip-mine       # validate only
  python remine_and_validate.py matrices.csv --skip-validate   # mine only
  python remine_and_validate.py matrices.csv --workers 8 --rtol 1e-4
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_MINING_DIR = _SCRIPT_DIR.parent / "mining"
_DEFAULT_OUTPUT_DIR = "/home/kaio/datasets/suite-sparse"  # matches _DATA_ROOT
                                                            # used across every
                                                            # other suite-sparse
                                                            # benchmark/validate
                                                            # script


def _check_csv(csv_path: Path) -> None:
    with open(csv_path, newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames
    missing = [c for c in ("name", "group") if fieldnames is None or c not in fieldnames]
    if missing:
        sys.exit(f"{csv_path} is missing column(s) {missing}; got: {fieldnames}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("csv_path", metavar="MATRICES.csv",
                     help="needs 'name' and 'group' columns; 'id' is resolved "
                          "and written back automatically by the mining step")
    ap.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR,
                     help=f"dataset root, passed to both steps (default: {_DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--skip-mine", action="store_true", help="only run validation")
    ap.add_argument("--skip-validate", action="store_true", help="only run mining")
    ap.add_argument("--workers", type=int, default=4, help="mine_matrices.py --workers")
    ap.add_argument("--rtol", type=float, default=1e-5, help="validate_bsp.py --deep --rtol")
    ap.add_argument("--atol", type=float, default=1e-6, help="validate_bsp.py --deep --atol")
    ap.add_argument("--no-deep", action="store_true",
                     help="skip the slower value-level check, NNZ-count only "
                          "(not recommended right after a re-mine -- see "
                          "validate_bsp.py's own docstring for why the fast "
                          "check alone can miss real corruption)")
    ap.add_argument("--fail-fast", action="store_true", help="validate_bsp.py --fail-fast")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        sys.exit(f"Not found: {csv_path}")
    _check_csv(csv_path)

    if not args.skip_mine:
        cmd = [
            sys.executable, str(_MINING_DIR / "mine_matrices.py"),
            "--matrices-csv", str(csv_path),
            "--force",
            "--output-dir", args.output_dir,
            "--workers", str(args.workers),
        ]
        print(f"$ {' '.join(cmd)}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"mine_matrices.py exited {r.returncode} -- stopping "
                     f"before validation (pass --skip-mine to validate what's "
                     f"already there instead).")
        print()

    if not args.skip_validate:
        cmd = [
            sys.executable, str(_MINING_DIR / "validate_bsp.py"),
            "--matrices-csv", str(csv_path),
            "--output-dir", args.output_dir,
            "--rtol", str(args.rtol),
            "--atol", str(args.atol),
        ]
        if not args.no_deep:
            cmd.append("--deep")
        if args.fail_fast:
            cmd.append("--fail-fast")
        print(f"$ {' '.join(cmd)}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
