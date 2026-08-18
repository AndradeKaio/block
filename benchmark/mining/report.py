#!/usr/bin/env python3
"""
report.py — Show aggregate block-pattern mining results.

Usage:
  python report.py --output-dir /data
  python report.py --output-dir /data --matrix ct20stif
"""

import argparse
import csv
import pickle
import sqlite3
import sys
from collections import Counter
from pathlib import Path


W = 78

def hr(char="─"): print(char * W)
def section(title): print(); hr("═"); print(f"  {title}"); hr("═")

def fmt_int(n):   return f"{n:,}" if n is not None else "—"
def fmt_float(f): return f"{f:.1f}" if f is not None else "—"
def fmt_pct(f):   return f"{f*100:.1f}%" if f is not None else "—"

def col_widths(rows, headers):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    return widths

def print_table(headers, rows, aligns=None):
    if not rows:
        print("  (no data)")
        return
    aligns = aligns or ["<"] * len(headers)
    widths = col_widths(rows, headers)
    fmt = "  " + "  ".join(f"{{:{a}{w}}}" for a, w in zip(aligns, widths))
    print(fmt.format(*headers))
    print("  " + "  ".join("─" * w for w in widths))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def _make_stub_factory():
    class _StubFactory:
        def __new__(cls, *a, **kw): return object.__new__(cls)
        def __init__(self, *a, **kw): pass
        def __setstate__(self, s):
            if isinstance(s, dict):
                self.__dict__.update(s)
            elif isinstance(s, tuple) and len(s) == 2:
                # slotted class: (dict_state, slots_state)
                d, sl = s
                if d:  self.__dict__.update(d)
                if sl: self.__dict__.update(sl)
        def num_nonzeros(self):
            return getattr(self, "h", 0) * getattr(self, "w", 0) - getattr(self, "imperfections", 0)
    return _StubFactory

_StubFactory = _make_stub_factory()

class _ForgivingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _StubFactory

def _load_patterns(path):
    try:
        with open(path, "rb") as f:
            return _ForgivingUnpickler(f).load()
    except Exception as e:
        print(f"  [warn] could not load {path.name}: {e}")
        return []

def _biggest_shape(out_root, grp, name):
    """Return (shape_str, count_int) for the pattern with highest nnz in the original ordering."""
    pkl = out_root / grp / name / "original_patterns.pkl"
    if not pkl.exists():
        return "—", None
    patterns = _load_patterns(pkl)
    if not patterns:
        return "—", None
    best = max(patterns, key=lambda p: p.num_nonzeros())
    bh, bw = best.h, best.w
    count = sum(1 for p in patterns if p.h == bh and p.w == bw)
    return f"{bh}x{bw}", count


def connect(db_path):
    if not db_path.exists():
        sys.exit(f"No progress.db found at {db_path}. Run mine_matrices.py first.")
    return sqlite3.connect(str(db_path))

def query(con, sql, params=()):
    cur = con.execute(sql, params)
    return [d[0] for d in cur.description], cur.fetchall()


def report_aggregate_table(con, out_root, name_filter=None):
    section("AGGREGATE TABLE  (original ordering)")

    where = "AND m.name = ?" if name_filter else ""
    params = (name_filter,) if name_filter else ()

    _, rows = query(con, f"""
        SELECT
            m.name,
            m.grp,
            m.rows,
            m.cols,
            o.dominant_shape,
            o.n_large,
            CAST(m.nnz AS REAL) / (CAST(m.rows AS REAL) * m.cols) AS density,
            o.dominant_share,
            o.padding_zeros,
            o.covered_nnz
        FROM matrices m
        LEFT JOIN results o ON o.matrix_id = m.id AND o.ordering = 'original'
        WHERE m.status = 'done' {where}
        ORDER BY m.nnz DESC
    """, params)

    if not rows:
        print("  No completed matrices.")
        return

    hdr = ["matrix", "common pattern", "count", "biggest pattern", "count",
           "patterns (orig)", "% sparsity", "padding zeros", "padding %"]
    tbl = []
    for name, grp, rows_, cols, dom_shape, n_large, density, dom_share, padding_zeros, covered_nnz in rows:
        sparsity = (1 - (density or 0)) * 100
        dom_count = fmt_int(int(dom_share * n_large)) if (dom_share and n_large) else "—"
        big_shape, big_count = _biggest_shape(out_root, grp or "", name)
        if padding_zeros is not None and covered_nnz:
            pad_fmt = f"+{padding_zeros:,}"
            pad_pct = f"{padding_zeros / covered_nnz * 100:.1f}%"
        else:
            pad_fmt = "—"
            pad_pct = "—"
        tbl.append((
            name or "—",
            dom_shape or "—",
            dom_count,
            big_shape,
            fmt_int(big_count),
            fmt_int(n_large),
            f"{sparsity:.2f}%",
            pad_fmt,
            pad_pct,
        ))
    print_table(hdr, tbl, ["<", "<", ">", "<", ">", ">", ">", ">", ">"])


def report_csv(con, out_root, out, names=None):
    writer = csv.writer(out)
    writer.writerow([
        "name", "group", "rows", "cols", "shape", "nnz",
        "common_pattern", "common_count",
        "biggest_pattern", "biggest_count",
        "n_patterns", "sparsity_pct",
    ])

    where  = f"AND m.name IN ({','.join('?'*len(names))})" if names else ""
    params = list(names) if names else []

    _, rows = query(con, f"""
        SELECT m.name, m.grp, m.rows, m.cols, m.nnz,
               o.dominant_shape, o.dominant_count, o.n_large,
               CAST(m.nnz AS REAL) / (CAST(m.rows AS REAL) * m.cols),
               o.covered_nnz, o.padding_zeros
        FROM matrices m
        LEFT JOIN results o ON o.matrix_id = m.id AND o.ordering = 'original'
        WHERE m.status = 'done'
        {where}
        ORDER BY m.nnz DESC
    """, params)
    for name, grp, rows_, cols, nnz, dom_shape, dom_count, n_large, density, covered_nnz, padding_zeros in rows:
        if not dom_count:
            continue
        big_shape, big_count = _biggest_shape(out_root, grp or "", name)
        sparsity = round((1 - (density or 0)) * 100, 4)
        writer.writerow([
            name, grp, rows_, cols, f"{rows_}x{cols}", nnz,
            dom_shape, dom_count,
            big_shape, big_count,
            n_large, sparsity,
        ])


def parse_args():
    p = argparse.ArgumentParser(
        description="Report block-pattern mining results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="/data", dest="output_dir",
                   help="Directory containing progress.db and matrix subdirs")
    p.add_argument("--matrix", default=None,
                   help="Filter to a single matrix by name (text mode)")
    p.add_argument("--names", nargs="+", default=None, metavar="NAME",
                   help="Filter CSV to these matrix names")
    p.add_argument("--format", choices=["text", "csv"], default="text",
                   help="Output format")
    return p.parse_args()


def main():
    args    = parse_args()
    out     = Path(args.output_dir)
    db_path = out / "progress.db"
    con     = connect(db_path)

    if args.format == "csv":
        report_csv(con, out, sys.stdout, names=args.names)
        con.close()
        return

    print()
    hr("═")
    print(f"  BLOCK PATTERN MINING REPORT")
    print(f"  output dir : {out}")
    print(f"  database   : {db_path}")
    hr("═")

    report_aggregate_table(con, out, name_filter=args.matrix)

    print()
    hr()
    con.close()


if __name__ == "__main__":
    main()
