#!/usr/bin/env python3
"""
plot_perf.py — Plot hardware perf-counter / memory metrics from any of
benchmark_spgemm_cpu.py / benchmark_spmm_cpu.py / benchmark_spmv_cpu.py's
--perf output (see perf_wrap.py). Domain-agnostic: all three scripts emit
the exact same perf_wrap.PERF_CSV_FIELDS columns, so one script covers
whichever CSV you hand it -- it reads the kernel names actually present and
colors/labels only those.

Each perf_wrap.py reading is ONE aggregate value for a kernel's entire
subprocess call (all --runs iterations combined, plus startup) -- unlike
compute_ms/total_ms, it's not a per-run measurement, so every run_id row for
a given (matrix, kernel, threads) carries an identical duplicated value.
This script collapses those duplicates (mean of identical copies == the one
real reading) and does NOT draw error bars the way the timing plots do,
since there is no real per-run variance to show.

If the CSV contains more than one distinct `threads` value (e.g. produced
via --threads-sweep), pass --threads to pick which one to plot -- mixing
thread counts on one plot would silently average unrelated measurements
together.

Usage:
  python plot_perf.py --file spgemm_cpu_results.csv
  python plot_perf.py --file spmm_results.csv --metrics cache_miss_rate,peak_rss_mb
  python plot_perf.py --file spmv_results.csv --threads 16 --out plot.html
  python plot_perf.py --file spmm_results.csv --drop prisma_auto,prisma_static
"""

import argparse
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Visual settings -- union of every kernel name used across all three
# domains' benchmark scripts. Each CSV only ever contains one domain's
# kernel set, so cross-domain color reuse (e.g. "taco" and "taco_cpu" share
# a blue) never collides in practice. Display names follow the canonical
# BLOCKS / BLOCKS (specialized) / BLOCKS (specialized+tiled) scheme agreed
# for the plot_sp*_cpu.py scripts.
# ---------------------------------------------------------------------------

KERNEL_COLORS = {
    # TACO variants → reds
    "taco_cpu":            "#d62728",  # SpGEMM
    "taco_cpu_opt":        "#99000d",  # SpGEMM
    "taco":                "#d62728",  # SpMM / SpMV
    "taco_opt":            "#99000d",  # SpMV
    "taco_opt0":           "#fc9272",  # SpMM
    "taco_opt1":           "#99000d",  # SpMM
    # Prisma (BLOCKS) variants → greens
    "prisma_generic":      "#41ab5d",  # SpGEMM baseline
    "prisma_cpu":          "#41ab5d",  # SpMM / SpMV baseline
    "prisma_top10":        "#006d2c",  # SpGEMM specialized
    "prisma_specialized":  "#006d2c",  # SpMM / SpMV specialized
    "prisma_tiled":        "#00441b",  # SpMM specialized+tiled
    "prisma_static":       "#74c476",  # SpMV static-sched ablation
    "prisma_auto":         "#a1d99b",  # SpMM auto-schedule ablation
}

KERNEL_DISPLAY = {
    "taco_cpu":            "TACO",
    "taco_cpu_opt":        "TACO (opt)",
    "taco":                "TACO",
    "taco_opt":            "TACO opt",
    "taco_opt0":           "TACO opt0",
    "taco_opt1":           "TACO opt1",
    "prisma_generic":      "BLOCKS",
    "prisma_cpu":          "BLOCKS",
    "prisma_top10":        "BLOCKS (specialized)",
    "prisma_specialized":  "BLOCKS (specialized)",
    "prisma_tiled":        "BLOCKS (specialized+tiled)",
    "prisma_static":       "BLOCKS (static sched)",
    "prisma_auto":         "BLOCKS (auto)",
}

DEFAULT_COLOR = "#7f7f7f"

# ---------------------------------------------------------------------------
# Metric definitions: name -> (source column(s), display label)
# ---------------------------------------------------------------------------

_RATE_METRICS = {
    "ipc":              (("instructions", "cycles"), "IPC (instructions/cycle)"),
    "cache_miss_rate":  (("cache_misses", "cache_references"), "Cache miss rate [%]"),
    "branch_miss_rate": (("branch_misses", "branches"), "Branch misprediction rate [%]"),
    "dtlb_miss_rate":   (("dtlb_load_misses", "dtlb_loads"), "dTLB miss rate [%]"),
    "itlb_miss_rate":   (("itlb_load_misses", "itlb_loads"), "iTLB miss rate [%]"),
}

_RAW_METRICS = {
    "cycles":           ("cycles", "Cycles"),
    "instructions":     ("instructions", "Instructions"),
    "cache_references": ("cache_references", "Cache references"),
    "cache_misses":     ("cache_misses", "Cache misses"),
    "branches":         ("branches", "Branches"),
    "branch_misses":    ("branch_misses", "Branch misses"),
    "dtlb_loads":       ("dtlb_loads", "dTLB loads"),
    "dtlb_load_misses": ("dtlb_load_misses", "dTLB load misses"),
    "itlb_loads":       ("itlb_loads", "iTLB loads"),
    "itlb_load_misses": ("itlb_load_misses", "iTLB load misses"),
    "peak_rss_mb":      ("peak_rss_mb", "Peak RSS [MB]"),
}

_ALL_METRIC_NAMES = sorted(set(_RATE_METRICS) | set(_RAW_METRICS))

# Reading order for the "plot everything" default -- raw counter next to its
# derived rate (e.g. cache_references, cache_misses, cache_miss_rate), memory
# last. Not just sorted(_ALL_METRIC_NAMES), which would scatter each rate
# away from the raw counters it's derived from.
_ALL_METRICS_ORDERED = [
    "cycles", "instructions", "ipc",
    "cache_references", "cache_misses", "cache_miss_rate",
    "branches", "branch_misses", "branch_miss_rate",
    "dtlb_loads", "dtlb_load_misses", "dtlb_miss_rate",
    "itlb_loads", "itlb_load_misses", "itlb_miss_rate",
    "peak_rss_mb",
]
assert set(_ALL_METRICS_ORDERED) == set(_ALL_METRIC_NAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot hardware perf-counter / memory metrics from any "
        "benchmark_sp{gemm,mm,v}_cpu.py --perf output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f"Available metrics: {', '.join(_ALL_METRIC_NAMES)}",
    )
    p.add_argument("--file", required=True, help="Benchmark CSV produced with --perf")
    p.add_argument(
        "--metrics",
        default=None,
        help="comma-separated list of metrics to plot, one panel each "
        "(default: all available metrics, see below)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="which 'threads' column value to plot, if the CSV has more "
        "than one (e.g. from --threads-sweep) -- required in that case, "
        "to avoid silently averaging different thread counts together",
    )
    p.add_argument(
        "--drop",
        default="",
        help="comma-separated kernel names to exclude (e.g. ablation "
        "kernels like prisma_auto/prisma_static) -- not dropped "
        "automatically since which kernels are 'ablations' is "
        "domain-specific (prisma_static is a real kept kernel in SpMV, "
        "an ablation in SpMM)",
    )
    p.add_argument("--warmup", action="store_true", default=False,
                    help="Include warmup run (run_id == 0)")
    p.add_argument("--out", default=None,
                   help="Save HTML to this path (also opens in browser if omitted)")
    p.add_argument("--show", action="store_true", default=False,
                   help="Open in browser even when --out is given")
    p.add_argument("--note", default="",
                   help="Text appended to the plot title (e.g. a domain name)")
    return p.parse_args()


def load(path: str, include_warmup: bool, threads: int | None,
         drop_kernels: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()

    required = {"matrix_name", "nnz", "kernel", "run_id", "threads"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"CSV is missing columns: {missing} -- was this produced with "
            f"a version of benchmark_sp{{gemm,mm,v}}_cpu.py that has "
            f"--perf support?"
        )

    df["nnz"] = pd.to_numeric(df["nnz"], errors="coerce")
    for col, _ in _RAW_METRICS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if not include_warmup:
        df = df[df["run_id"] != 0]

    if drop_kernels:
        df = df[~df["kernel"].isin(drop_kernels)]

    thread_values = sorted(df["threads"].dropna().unique().tolist())
    if len(thread_values) > 1:
        if threads is None:
            sys.exit(
                f"CSV has multiple thread counts {thread_values} -- pass "
                f"--threads N to pick one (mixing them would silently "
                f"average unrelated measurements together)."
            )
        df = df[df["threads"] == threads]
        if df.empty:
            sys.exit(f"No rows with threads == {threads} (available: {thread_values})")
    elif threads is not None and thread_values and thread_values[0] != threads:
        sys.exit(f"--threads {threads} not found in CSV (available: {thread_values})")

    if "peak_rss_kb" in df.columns:
        df["peak_rss_mb"] = pd.to_numeric(df["peak_rss_kb"], errors="coerce") / 1024.0

    for name, ((num_col, den_col), _label) in _RATE_METRICS.items():
        if num_col not in df.columns or den_col not in df.columns:
            continue
        scale = 100.0 if name.endswith("_rate") else 1.0
        ratio = scale * df[num_col] / df[den_col]
        df[name] = ratio.replace([float("inf"), float("-inf")], float("nan"))

    return df


def _metric_column(name: str) -> tuple[str, str]:
    if name in _RATE_METRICS:
        return name, _RATE_METRICS[name][1]
    if name in _RAW_METRICS:
        col, label = _RAW_METRICS[name]
        return col, label
    sys.exit(f"Unknown metric '{name}'. Available: {', '.join(_ALL_METRIC_NAMES)}")


def build_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Collapse the duplicated per-run_id perf reading down to one value per
    (matrix, kernel) -- mean of identical duplicates equals the single real
    reading, so this is safe even though it looks like an aggregation."""
    d = df[df[col].notna()]
    agg = (
        d.groupby(["matrix_name", "nnz", "kernel"], as_index=False)
         .agg(value=(col, "mean"))
    )
    return agg


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _fmt_nnz(v):
    if v >= 1_000_000_000: return f"{v/1e9:.1f}B"
    if v >= 1_000_000:     return f"{v/1e6:.1f}M"
    if v >= 1_000:         return f"{v/1e3:.0f}K"
    return str(int(v))


def _add_panel(fig: go.Figure, stats: pd.DataFrame, matrix_order: list,
                nnz_map: dict, kernels_ordered: list, row: int, col: int,
                label: str, legend_shown: set) -> None:
    for kernel in kernels_ordered:
        kdf = stats[stats["kernel"] == kernel].set_index("matrix_name")
        if kdf.empty:
            continue
        color = KERNEL_COLORS.get(kernel, DEFAULT_COLOR)
        display_name = KERNEL_DISPLAY.get(kernel, kernel)
        show = kernel not in legend_shown
        legend_shown.add(kernel)

        values = [kdf.loc[m, "value"] if m in kdf.index else None for m in matrix_order]
        hover = [
            (
                f"<b>{display_name}</b><br>"
                f"matrix: {m}<br>"
                f"nnz: {nnz_map[m]:,.0f}<br>"
                f"{label}: {kdf.loc[m, 'value']:.4g}"
            ) if m in kdf.index else None
            for m in matrix_order
        ]

        fig.add_trace(go.Scatter(
            name=display_name,
            legendgroup=display_name,
            showlegend=show,
            mode="markers",
            x=list(range(len(matrix_order))),
            y=values,
            marker=dict(color=color, size=9, opacity=0.9),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ), row=row, col=col)


def make_figure(df: pd.DataFrame, metrics: list[str], note: str = "",
                 threads: int | None = None) -> go.Figure:
    matrix_order = (
        df[["matrix_name", "nnz"]]
        .drop_duplicates()
        .sort_values("nnz")
        ["matrix_name"]
        .tolist()
    )
    nnz_map = (
        df[["matrix_name", "nnz"]]
        .drop_duplicates()
        .set_index("matrix_name")["nnz"]
        .to_dict()
    )
    n = len(matrix_order)
    step = max(1, round(n / 10))
    tick_vals = list(range(0, n, step))
    tick_labels = [_fmt_nnz(nnz_map[matrix_order[i]]) for i in tick_vals]

    kernels_in_data = sorted(df["kernel"].unique().tolist())
    kernels_ordered = [k for k in KERNEL_COLORS if k in kernels_in_data] + \
                      [k for k in kernels_in_data if k not in KERNEL_COLORS]

    # Wrap into a grid instead of one wide row -- the "plot everything"
    # default is 16 panels, unreadable as a single row.
    ncols = min(4, len(metrics))
    nrows = -(-len(metrics) // ncols)  # ceil division

    labels = [_metric_column(m)[1] for m in metrics]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=labels,
                         horizontal_spacing=0.06, vertical_spacing=0.14)

    legend_shown: set = set()
    for idx, metric in enumerate(metrics):
        row, col = idx // ncols + 1, idx % ncols + 1
        col_name, label = _metric_column(metric)
        stats = build_stats(df, col_name)
        _add_panel(fig, stats, matrix_order, nnz_map, kernels_ordered, row,
                   col, label, legend_shown)
        fig.update_xaxes(
            tickmode="array", tickvals=tick_vals, ticktext=tick_labels,
            title="NNZ (ordered ascending)", gridcolor="#e0e0e0", row=row, col=col,
        )
        log_y = metric not in ("ipc",)  # IPC's dynamic range doesn't need log
        fig.update_yaxes(title=label, gridcolor="#e0e0e0",
                          type="log" if log_y else "linear", row=row, col=col)

    title = "SuiteSparse CPU perf counters"
    if threads is not None:
        title += f" (threads={threads})"
    if note:
        title += f" — {note}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=17), x=0.5),
        legend=dict(
            orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5,
            font=dict(size=14), bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc", borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=max(1100, 330 * ncols),
        height=max(500, 380 * nrows + 150),
        margin=dict(l=70, r=30, t=90, b=140),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = list(_ALL_METRICS_ORDERED)
    for m in metrics:
        _metric_column(m)  # validates early, exits with a clear message

    drop_kernels = {k.strip() for k in args.drop.split(",") if k.strip()}

    df = load(args.file, include_warmup=args.warmup, threads=args.threads,
              drop_kernels=drop_kernels)
    if df.empty:
        sys.exit("No valid rows after filtering — nothing to plot.")

    threads_used = args.threads
    if threads_used is None:
        uniq = df["threads"].dropna().unique().tolist()
        threads_used = uniq[0] if len(uniq) == 1 else None

    print(f"Matrices : {df['matrix_name'].nunique()}")
    print(f"Kernels  : {sorted(df['kernel'].unique())}")
    print(f"Threads  : {threads_used}")
    print(f"Metrics  : {metrics}")
    for m in metrics:
        col, _ = _metric_column(m)
        if col not in df.columns or df[col].notna().sum() == 0:
            print(f"  WARNING: '{m}' has no non-NaN data — was --perf actually "
                  f"used, and did perf have permission to run on that machine?")
    print()

    fig = make_figure(df, metrics, note=args.note, threads=threads_used)

    if args.out:
        fig.write_html(args.out, include_plotlyjs="cdn")
        print(f"Saved → {args.out}")
    if not args.out or args.show:
        fig.show()


if __name__ == "__main__":
    main()
