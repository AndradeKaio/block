#!/usr/bin/env python3
"""
plot_stats.py — Interactive visual report of block-pattern mining results.

Combines progress.db summaries with per-block detail from .bsp HDF5 files.
Outputs a self-contained HTML file (requires: plotly, h5py).

Usage:
  python plot_stats.py --output-dir /data --output mining_stats.html
  pip install plotly h5py   # if not already installed
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── DB helpers ────────────────────────────────────────────────────────────────


def connect(db_path):
    if not db_path.exists():
        sys.exit(f"No progress.db found at {db_path}. Run mine_matrices.py first.")
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def _size_clause(max_rows, max_cols, max_singles=None):
    clauses, params = [], []
    if max_rows is not None:
        clauses.append("m.rows <= ?")
        params.append(max_rows)
    if max_cols is not None:
        clauses.append("m.cols <= ?")
        params.append(max_cols)
    if max_singles is not None:
        clauses.append("r.n_singles <= ?")
        params.append(max_singles)
    sql = (" AND " + " AND ".join(clauses)) if clauses else ""
    return sql, params


def fetch_all_done(con, max_rows=None, max_cols=None, max_singles=None):
    size_sql, size_params = _size_clause(max_rows, max_cols, max_singles)
    return con.execute(
        f"""
        SELECT
            m.id, m.name, m.grp, m.nnz, m.rows, m.cols,
            r.n_patterns, r.n_large, r.dominant_shape,
            r.dominant_count, r.dominant_share, r.max_nnz, r.mean_nnz,
            r.padding_zeros, r.covered_nnz, r.total_padding, r.n_singles,
            r.mining_time,
            (m.finished_at - m.started_at) AS wall_time
        FROM matrices m
        JOIN results r ON r.matrix_id = m.id AND r.ordering = 'original'
        WHERE m.status = 'done'
          AND m.rows = m.cols
          AND r.n_patterns IS NOT NULL
          {size_sql}
    """,
        size_params,
    ).fetchall()


def fetch_matrix_status(con, names):
    """Return status rows for the given matrix names."""
    placeholders = ",".join("?" * len(names))
    return con.execute(
        f"""
        SELECT
            m.id, m.name, m.grp, m.nnz, m.rows, m.cols, m.status,
            m.started_at, m.finished_at,
            r.n_patterns, r.n_large, r.dominant_shape,
            r.dominant_count, r.dominant_share, r.n_singles,
            r.mining_time,
            (m.finished_at - m.started_at) AS wall_time
        FROM matrices m
        LEFT JOIN results r ON r.matrix_id = m.id AND r.ordering = 'original'
        WHERE m.id IN ({placeholders})
    """,
        names,
    ).fetchall()


def _print_rows_table(rows, show_status=False, missing=None):
    col_w = max((len(r["name"]) for r in rows), default=10) if rows else 10
    id_w  = max((len(str(r["id"])) for r in rows), default=4) if rows else 4
    id_w  = max(id_w, 4)  # at least wide enough for "ID"
    status_col = f"  {'Status':<10}" if show_status else ""
    header = (
        f"{'ID':>{id_w}}  {'Matrix':<{col_w}}{status_col}  {'Rows':>8}  {'Cols':>8}  "
        f"{'NNZ':>12}  {'DomShape':<10}  {'Coverage':>10}  {'Singles':>10}  {'Time(s)':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        status_val = f"  {(r['status'] or 'unknown'):<10}" if show_status else ""
        coverage = ""
        if r["nnz"] and r["n_singles"] is not None:
            pct = (r["nnz"] - r["n_singles"]) / r["nnz"] * 100
            coverage = f"{pct:.1f}%"
        mtime = f"{r['mining_time']:.1f}" if r["mining_time"] else "-"
        print(
            f"{(r['id'] or 0):>{id_w}}  {r['name']:<{col_w}}{status_val}  "
            f"{(r['rows'] or 0):>8,}  {(r['cols'] or 0):>8,}  {(r['nnz'] or 0):>12,}  "
            f"{(r['dominant_shape'] or '-'):<10}  {coverage:>10}  "
            f"{(r['n_singles'] or 0):>10,}  {mtime:>8}"
        )

    for n in (missing or []):
        print(f"{'?':>{id_w}}  {n:<{col_w}}  {'NOT FOUND':<10}")


def print_matrix_status(con, names):
    rows = fetch_matrix_status(con, names)
    found = {r["name"] for r in rows}
    missing = [n for n in names if n not in found]
    name_order = {n: i for i, n in enumerate(names)}
    sorted_rows = sorted(rows, key=lambda x: name_order.get(x["name"], 9999))
    _print_rows_table(sorted_rows, show_status=True, missing=missing)


def fetch_top(con, top, max_rows=None, max_cols=None, max_coverage=False, max_singles=None):
    """
    max_coverage=False (default): rank by dominant_count DESC, max_nnz DESC.
    max_coverage=True:            rank by (nnz - n_singles) / nnz DESC.
    """
    size_sql, size_params = _size_clause(max_rows, max_cols, max_singles)
    if max_coverage:
        order = "CAST(m.nnz - r.n_singles AS REAL) / m.nnz DESC, r.n_large DESC"
        extra_where = "AND r.n_singles IS NOT NULL AND m.nnz > 0"
    else:
        order = "r.dominant_count DESC, r.max_nnz DESC"
        extra_where = "AND r.dominant_count IS NOT NULL"
    return con.execute(
        f"""
        SELECT
            m.id, m.name, m.grp, m.nnz, m.rows, m.cols,
            r.n_patterns, r.n_large, r.dominant_shape,
            r.dominant_count, r.dominant_share, r.max_nnz, r.mean_nnz,
            r.padding_zeros, r.covered_nnz, r.total_padding, r.n_singles,
            r.mining_time,
            CAST(m.nnz - COALESCE(r.n_singles, 0) AS REAL) / m.nnz AS block_coverage
        FROM matrices m
        JOIN results r ON r.matrix_id = m.id AND r.ordering = 'original'
        WHERE m.status = 'done'
          AND m.rows = m.cols
          AND r.dominant_shape IS NOT NULL
          AND r.dominant_shape != '1x1'
          {extra_where}
          {size_sql}
        ORDER BY {order}
        LIMIT ?
    """,
        size_params + [top],
    ).fetchall()


# ── BSP loader ────────────────────────────────────────────────────────────────


def load_bsp(path):
    """Return (h, w, imps) numpy arrays. Raises on failure."""
    try:
        import h5py
    except ImportError:
        raise ImportError("pip install h5py")
    with h5py.File(str(path), "r") as f:
        h = f["block_h"][:]
        w = f["block_w"][:]
        imps = f["block_imps"][:]
    return h.astype(np.int32), w.astype(np.int32), imps.astype(np.int64)


def bsp_path(out_root, grp, name):
    return Path(out_root) / grp / name / f"{name}.bsp"


def load_bsp_for_rows(out_root, top_rows):
    results = []
    for r in top_rows:
        p = bsp_path(out_root, r["grp"] or "", r["name"])
        if not p.exists():
            print(f"  [warn] BSP not found: {p}")
            results.append(None)
            continue
        try:
            results.append(load_bsp(p))
        except Exception as e:
            print(f"  [warn] BSP unreadable ({e}): {p}")
            results.append(None)
    return results


# ── Colour helpers ────────────────────────────────────────────────────────────

_PALETTE = [
    "#f4a261",
    "#e76f51",
    "#2a9d8f",
    "#457b9d",
    "#e9c46a",
    "#a8dadc",
    "#c77dff",
    "#90e0ef",
    "#f28482",
    "#b7e4c7",
]
_shape_color_cache = {}


def shape_color(s):
    if s not in _shape_color_cache:
        _shape_color_cache[s] = _PALETTE[len(_shape_color_cache) % len(_PALETTE)]
    return _shape_color_cache[s]


# ── Panel builders (each returns one or more plotly traces + layout hints) ────


def traces_leaderboard(top_rows):
    """Panel 1: horizontal bar — dominant_count per matrix."""
    names = [r["name"] for r in top_rows]
    counts = [r["dominant_count"] or 0 for r in top_rows]
    shapes = [r["dominant_shape"] or "?" for r in top_rows]
    shares = [r["dominant_share"] or 0 for r in top_rows]
    colors = [shape_color(s) for s in shapes]

    hover = [
        f"<b>{n}</b><br>shape: {sh}<br>count: {c:,}<br>share: {s * 100:.1f}%"
        for n, sh, c, s in zip(names, shapes, counts, shares)
    ]

    import plotly.graph_objects as go

    return [
        go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{sh}  {s * 100:.0f}%" for sh, s in zip(shapes, shares)],
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
            name="dominant count",
            showlegend=False,
        )
    ]


def traces_nnz_vs_blocks(all_rows, top_ids):
    """Panel 2: scatter log-log — NNZ vs n_large."""
    import plotly.graph_objects as go

    bg = [r for r in all_rows if r["id"] not in top_ids]
    fg = [r for r in all_rows if r["id"] in top_ids]

    def make_scatter(rows, highlighted):
        nnz = [r["nnz"] or 1 for r in rows]
        nlarge = [r["n_large"] or 1 for r in rows]
        share = [r["dominant_share"] or 0 for r in rows]
        mtime = [r["mining_time"] or 0 for r in rows]
        hover = [
            f"<b>{r['name']}</b><br>NNZ: {r['nnz']:,}<br>"
            f"n_large: {r['n_large']:,}<br>share: {(r['dominant_share'] or 0) * 100:.1f}%<br>"
            f"shape: {r['dominant_shape'] or '?'}<br>mine: {r['mining_time'] or 0:.1f}s"
            for r in rows
        ]
        return go.Scatter(
            x=nnz,
            y=nlarge,
            mode="markers",
            marker=dict(
                size=10 if highlighted else 6,
                color=share,
                colorscale="Plasma",
                cmin=0,
                cmax=1,
                line=dict(
                    width=1.5 if highlighted else 0,
                    color="white" if highlighted else "rgba(0,0,0,0)",
                ),
                showscale=highlighted,
                colorbar=dict(title="dominant_share", x=1.02, thickness=12, len=0.4)
                if highlighted
                else None,
            ),
            opacity=1.0 if highlighted else 0.45,
            hovertext=hover,
            hoverinfo="text",
            name="top-10" if highlighted else "all",
            showlegend=highlighted,
        )

    return [make_scatter(bg, False), make_scatter(fg, True)]


def traces_biggest_block(top_rows, bsp_data):
    """
    Panel 3: biggest block per matrix (h×w − imps).
    Uses BSP when available; falls back to max_nnz from DB.
    """
    import plotly.graph_objects as go

    names, biggest_nnz, biggest_shape, shape_counts, sources = [], [], [], [], []

    for r, bsp in zip(top_rows, bsp_data):
        name = r["name"]
        names.append(name)
        if bsp is not None:
            h, w, imps = bsp
            nnz_per_block = h.astype(np.int64) * w.astype(np.int64) - imps
            idx = int(np.argmax(nnz_per_block))
            bh, bw = int(h[idx]), int(w[idx])
            biggest_nnz.append(int(nnz_per_block[idx]))
            biggest_shape.append(f"{bh}×{bw}")
            shape_counts.append(int(np.sum((h == bh) & (w == bw))))
            sources.append("BSP")
        else:
            biggest_nnz.append(r["max_nnz"] or 0)
            biggest_shape.append(r["dominant_shape"] or "?")
            shape_counts.append(None)
            sources.append("DB")

    hover = [
        f"<b>{n}</b><br>biggest block: {sh}<br>NNZ: {nnz:,}<br>"
        + (f"appears: {cnt:,}×<br>" if cnt is not None else "")
        + f"source: {src}"
        for n, sh, nnz, cnt, src in zip(
            names, biggest_shape, biggest_nnz, shape_counts, sources
        )
    ]
    colors = [shape_color(sh) for sh in biggest_shape]

    return [
        go.Bar(
            x=biggest_nnz,
            y=names,
            orientation="h",
            marker_color=colors,
            text=biggest_shape,
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
            name="biggest block NNZ",
            showlegend=False,
        )
    ]


def traces_padding_per_matrix(top_rows):
    """
    Panel 4: exact total_padding per matrix (all zeros added across all blocks).
    Also shows padding as % of (covered_nnz + total_padding).
    """
    import plotly.graph_objects as go

    names = [r["name"] for r in top_rows]
    padding = [r["total_padding"] or 0 for r in top_rows]
    covered = [r["covered_nnz"] or 0 for r in top_rows]

    pct = []
    for pad, cov in zip(padding, covered):
        total = cov + pad
        pct.append(pad / total * 100 if total > 0 else 0.0)

    hover = [
        f"<b>{n}</b><br>total padding: {p:,} zeros<br>"
        f"covered NNZ: {c:,}<br>overhead: {pc:.1f}%"
        for n, p, c, pc in zip(names, padding, covered, pct)
    ]

    # color by overhead %
    return [
        go.Bar(
            x=padding,
            y=names,
            orientation="h",
            marker=dict(
                color=pct,
                colorscale=[[0, "#2a9d8f"], [0.3, "#e9c46a"], [1.0, "#e63946"]],
                showscale=True,
                colorbar=dict(title="overhead %", x=1.02, thickness=12, len=0.4),
            ),
            text=[f"{pc:.1f}%" for pc in pct],
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
            name="total padding",
            showlegend=False,
        )
    ]


def traces_composition(top_rows):
    """Panel 5: stacked bar — covered_nnz | padding_zeros | n_singles."""
    import plotly.graph_objects as go

    names = [r["name"] for r in top_rows]
    covered = [r["covered_nnz"] or 0 for r in top_rows]
    padding = [r["padding_zeros"] or 0 for r in top_rows]
    singles = [r["n_singles"] or 0 for r in top_rows]

    return [
        go.Bar(
            x=covered,
            y=names,
            orientation="h",
            marker_color="#2a9d8f",
            name="covered NNZ",
            hovertemplate="<b>%{y}</b><br>covered NNZ: %{x:,}<extra></extra>",
        ),
        go.Bar(
            x=padding,
            y=names,
            orientation="h",
            marker_color="#e63946",
            name="padding zeros",
            hovertemplate="<b>%{y}</b><br>padding zeros: %{x:,}<extra></extra>",
        ),
        go.Bar(
            x=singles,
            y=names,
            orientation="h",
            marker_color="#f4a261",
            name="singletons",
            hovertemplate="<b>%{y}</b><br>singletons: %{x:,}<extra></extra>",
        ),
    ]


def traces_mining_time(all_rows, top_ids):
    """Panel 6: scatter log-log — NNZ vs mining_time."""
    import plotly.graph_objects as go

    valid = [
        r for r in all_rows if r["nnz"] and r["mining_time"] and r["mining_time"] > 0
    ]
    bg = [r for r in valid if r["id"] not in top_ids]
    fg = [r for r in valid if r["id"] in top_ids]

    def make_scatter(rows, highlighted):
        nnz = [r["nnz"] for r in rows]
        ttime = [r["mining_time"] for r in rows]
        eff = [(r["n_large"] or 0) / max(r["nnz"], 1) for r in rows]
        hover = [
            f"<b>{r['name']}</b><br>NNZ: {r['nnz']:,}<br>"
            f"time: {r['mining_time']:.2f}s<br>"
            f"n_large/NNZ: {(r['n_large'] or 0) / max(r['nnz'], 1):.4f}"
            for r in rows
        ]
        return go.Scatter(
            x=nnz,
            y=ttime,
            mode="markers",
            marker=dict(
                size=10 if highlighted else 6,
                color=eff,
                colorscale="Viridis",
                line=dict(
                    width=1.5 if highlighted else 0,
                    color="white" if highlighted else "rgba(0,0,0,0)",
                ),
                showscale=highlighted,
                colorbar=dict(title="n_large/NNZ", x=1.02, thickness=12, len=0.4)
                if highlighted
                else None,
            ),
            opacity=1.0 if highlighted else 0.45,
            hovertext=hover,
            hoverinfo="text",
            name="top-10" if highlighted else "all",
            showlegend=highlighted,
        )

    return [make_scatter(bg, False), make_scatter(fg, True)]


# ── Assemble figure ───────────────────────────────────────────────────────────


def _params_label(args):
    parts = [f"output-dir: {args.output_dir}", f"top: {args.top}"]
    parts.append(
        "ranking: max-coverage" if args.max_coverage else "ranking: dominant-count"
    )
    if args.max_rows is not None:
        parts.append(f"max-rows: {args.max_rows:,}")
    if args.max_cols is not None:
        parts.append(f"max-cols: {args.max_cols:,}")
    if args.max_singles is not None:
        parts.append(f"max-singles: {args.max_singles:,}")
    if args.no_bsp:
        parts.append("no-bsp")
    return "  |  ".join(parts)


def traces_summary_table(top_rows):
    """Panel 7: tabular summary of selected matrices."""
    import plotly.graph_objects as go

    coverages, mtimes = [], []
    for r in top_rows:
        pct = (r["nnz"] - (r["n_singles"] or 0)) / r["nnz"] * 100 if r["nnz"] else 0
        coverages.append(f"{pct:.1f}%")
        mtimes.append(f"{r['mining_time']:.1f}" if r["mining_time"] else "-")

    header_vals = ["Name", "Dom Shape", "Coverage", "Singles", "NNZ", "n_large", "Time (s)"]
    cell_vals = [
        [r["name"]                          for r in top_rows],
        [r["dominant_shape"] or "?"         for r in top_rows],
        coverages,
        [f"{r['n_singles'] or 0:,}"         for r in top_rows],
        [f"{r['nnz'] or 0:,}"               for r in top_rows],
        [f"{r['n_large'] or 0:,}"           for r in top_rows],
        mtimes,
    ]

    return go.Table(
        header=dict(
            values=header_vals,
            fill_color="#2a2a2a",
            font=dict(color="#cccccc", size=12),
            align="left",
            line_color="#444",
        ),
        cells=dict(
            values=cell_vals,
            fill_color="#181818",
            font=dict(color="#cccccc", size=11),
            align=["left", "center", "right", "right", "right", "right", "right"],
            line_color="#333",
        ),
    )


def build_figure(top_rows, all_rows, bsp_data, args=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    top_ids = {r["id"] for r in top_rows}
    n = len(top_rows)

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "xy"},    {"type": "xy"}],
            [{"type": "xy"},    {"type": "xy"}],
            [{"type": "xy"},    {"type": "xy"}],
            [{"type": "table", "colspan": 2}, None],
        ],
        subplot_titles=[
            "Top matrices — most frequent pattern",
            "NNZ vs large blocks found (all matrices)",
            "Biggest block per matrix",
            "Exact padding per matrix (all blocks)",
            "Block composition per matrix",
            "Mining time vs NNZ (all matrices)",
            "Selected matrix summary",
        ],
        horizontal_spacing=0.18,
        vertical_spacing=0.08,
        row_heights=[0.28, 0.28, 0.28, 0.16],
    )

    # Panel 1
    for t in traces_leaderboard(top_rows):
        fig.add_trace(t, row=1, col=1)

    # Panel 2
    for t in traces_nnz_vs_blocks(all_rows, top_ids):
        fig.add_trace(t, row=1, col=2)

    # Panel 3
    for t in traces_biggest_block(top_rows, bsp_data):
        fig.add_trace(t, row=2, col=1)

    # Panel 4
    for t in traces_padding_per_matrix(top_rows):
        fig.add_trace(t, row=2, col=2)

    # Panel 5
    for t in traces_composition(top_rows):
        fig.add_trace(t, row=3, col=1)

    # Panel 6
    for t in traces_mining_time(all_rows, top_ids):
        fig.add_trace(t, row=3, col=2)

    # Panel 7 — summary table
    fig.add_trace(traces_summary_table(top_rows), row=4, col=1)

    # Log axes for scatter panels
    fig.update_xaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(type="log", row=3, col=2)
    fig.update_yaxes(type="log", row=3, col=2)

    # Stacked bar for composition
    fig.update_layout(barmode="stack")

    # Invert y-axes on all horizontal bar charts so top = best
    for row, col in [(1, 1), (2, 1), (2, 2), (3, 1)]:
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    height_per_matrix = max(28, 600 // n)
    bar_height = n * height_per_matrix
    table_height = max(200, n * 22 + 40)

    fig.update_layout(
        height=max(1600, bar_height * 3 + table_height + 300),
        width=1600,
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#181818",
        font=dict(color="#cccccc", size=11),
        title=dict(
            text=f"Block Mining Statistics — {len(all_rows)} matrices done, "
            f"top {n} selected",
            font=dict(size=15, color="white"),
            x=0.5,
        ),
        annotations=[
            dict(
                text=_params_label(args) if args else "",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.0,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=10, color="#888888"),
            )
        ],
        legend=dict(bgcolor="#222", bordercolor="#444", borderwidth=1),
    )

    # Dark grid on all subplots
    fig.update_xaxes(gridcolor="#2a2a2a", zerolinecolor="#444")
    fig.update_yaxes(gridcolor="#2a2a2a", zerolinecolor="#444")

    # Axis labels
    fig.update_xaxes(title_text="dominant block count", row=1, col=1)
    fig.update_xaxes(title_text="NNZ", row=1, col=2)
    fig.update_yaxes(title_text="n_large blocks", row=1, col=2)
    fig.update_xaxes(title_text="biggest block NNZ (h×w − imps)", row=2, col=1)
    fig.update_xaxes(title_text="total padding (zero cells added)", row=2, col=2)
    fig.update_xaxes(title_text="NNZ count", row=3, col=1)
    fig.update_xaxes(title_text="NNZ", row=3, col=2)
    fig.update_yaxes(title_text="mining time (s)", row=3, col=2)

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive mining statistics report (HTML output).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir", default="/data", help="Directory containing progress.db"
    )
    p.add_argument("--output", default="mining_stats.html", help="Output HTML path")
    p.add_argument(
        "--top", type=int, default=10, help="Number of top matrices to feature"
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Exclude matrices with more than this many rows",
    )
    p.add_argument(
        "--max-cols",
        type=int,
        default=None,
        help="Exclude matrices with more than this many columns",
    )
    p.add_argument(
        "--max-coverage",
        action="store_true",
        help="Rank by block NNZ coverage (nnz−singletons)/nnz; "
        "default ranks by dominant pattern count",
    )
    p.add_argument(
        "--no-bsp",
        action="store_true",
        help="Skip BSP file loading (panels 3 will use DB fallback)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Ignored (kept for backwards compatibility)",
    )
    p.add_argument(
        "--max-singles",
        type=int,
        default=None,
        help="Exclude matrices with more than this many singleton NNZ entries",
    )
    p.add_argument(
        "--matrices",
        nargs="+",
        metavar="NAME",
        help="Print status for these matrix names and exit",
    )
    return p.parse_args()


def main():
    try:
        import plotly
    except ImportError:
        sys.exit("plotly is required: pip install plotly")

    args = parse_args()
    out_root = Path(args.output_dir)
    db_path = out_root / "progress.db"

    con = connect(db_path)

    if args.matrices:
        print_matrix_status(con, args.matrices)
        con.close()
        return

    all_rows = fetch_all_done(con, args.max_rows, args.max_cols, args.max_singles)
    top_rows = fetch_top(con, args.top, args.max_rows, args.max_cols, args.max_coverage, args.max_singles)
    con.close()

    if not all_rows:
        sys.exit("No completed matrices in DB.")
    if not top_rows:
        sys.exit("No qualifying matrices (need non-1x1 dominant shapes).")

    print(f"All done: {len(all_rows)}   Top selected: {len(top_rows)}")
    _print_rows_table(top_rows)

    if args.no_bsp:
        bsp_data = [None] * len(top_rows)
    else:
        bsp_data = load_bsp_for_rows(out_root, top_rows)

    out_path = Path(args.output)

    if out_path.suffix.lower() == ".csv":
        import csv
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "name", "group", "rows", "cols", "shape", "nnz",
                "common_pattern", "common_count",
                "biggest_pattern", "biggest_count",
                "n_patterns", "sparsity_pct",
            ])
            for r, bsp in zip(top_rows, bsp_data):
                total_cells = (r["rows"] or 0) * (r["cols"] or 0)
                sparsity = (r["nnz"] / total_cells * 100) if total_cells else ""
                if bsp is not None:
                    h, w, imps = bsp
                    nnz_per_block = h.astype(np.int64) * w.astype(np.int64) - imps
                    idx = int(np.argmax(nnz_per_block))
                    bh, bw = int(h[idx]), int(w[idx])
                    biggest_pattern = f"{bh}x{bw}"
                    biggest_count = int(np.sum((h == bh) & (w == bw)))
                else:
                    biggest_pattern = r["dominant_shape"] or ""
                    biggest_count = r["dominant_count"] or ""
                writer.writerow([
                    r["name"],
                    r["grp"] or "",
                    r["rows"] or 0,
                    r["cols"] or 0,
                    f"{r['rows']}x{r['cols']}",
                    r["nnz"] or 0,
                    r["dominant_shape"] or "",
                    r["dominant_count"] or "",
                    biggest_pattern,
                    biggest_count,
                    r["n_patterns"] or "",
                    round(sparsity, 6) if sparsity != "" else "",
                ])
        print(f"Saved → {out_path}")
        return

    fig = build_figure(top_rows, all_rows, bsp_data, args)

    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
