#!/usr/bin/env python3
"""
suite-sparse/plot_spgemm_gpu.py — Plot suite-sparse GPU SpGEMM benchmark results.

Produces a 1x2 grid: total time on the left, compute time only on the
right, sharing one legend.

X-axis : matrices ordered by NNZ (ascending)
Y-axis : mean time per run [ms], log scale
Columns: one scatter point per contender per matrix

Usage:
  python plot_spgemm_gpu.py --file suite_sparse_results.csv
  python plot_spgemm_gpu.py --file results.csv --warmup --out plot.html
"""

import argparse
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Visual settings
# ---------------------------------------------------------------------------

KERNEL_COLORS = {
    "prisma_tc_tile":       "#17becf",
    "prisma_cuda":          "#2ca02c",
    "tc_spgemm":            "#8c564b",
    "tilespgemm":           "#ff7f0e",
    "cusparse_tilespgemm":  "#d62728",
    "taco_gpu":             "#e377c2",
}

KERNEL_DISPLAY = {
    "prisma_tc_tile":       "BLOCKS (TC tile)",
    "prisma_cuda":          "BLOCKS (CUDA)",
    "tc_spgemm":            "IPDPS'26",
    "tilespgemm":           "PPoPP'22",
    "cusparse_tilespgemm":  "cuSPARSE(PPoPP'22)",
    "taco_gpu":             "TACO GPU",
}

DEFAULT_COLOR = "#7f7f7f"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot suite-sparse GPU SpGEMM benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", required=True,
                   help="Benchmark CSV produced by benchmark_spgemm_gpu.py "
                        "(a SuiteSparse-matrix GPU SpGEMM benchmark driver -- "
                        "currently missing from this repo; SpGEMM/GPU/sweep.py "
                        "benchmarks synthetic matrices with a different CSV "
                        "schema and can't feed this script)")
    p.add_argument("--warmup", action="store_true", default=False,
                   help="Include warmup run (run_id == 0) in the average")
    p.add_argument("--out", default=None,
                   help="Save HTML to this path (also opens in browser if omitted)")
    p.add_argument("--show", action="store_true", default=False,
                   help="Open in browser even when --out is given")
    p.add_argument("--show-min", action="store_true", default=False,
                   help="Plot the minimum time per run instead of the mean")
    return p.parse_args()


def load(path: str, include_warmup: bool) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()

    required = {"matrix_name", "nnz", "kernel", "run_id", "compute_ms", "total_ms"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing columns: {missing}")

    df["compute_ms"] = pd.to_numeric(df["compute_ms"], errors="coerce")
    df["total_ms"]   = pd.to_numeric(df["total_ms"],   errors="coerce")
    df["nnz"]        = pd.to_numeric(df["nnz"],        errors="coerce")

    if not include_warmup:
        df = df[df["run_id"] != 0]

    return df


def build_stats(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df[df[time_col].notna() & (df[time_col] > 0)]
    agg = (
        df.groupby(["matrix_name", "nnz", "kernel"], as_index=False)
          .agg(mean_ms=(time_col, "mean"),
               min_ms=(time_col, "min"),
               max_ms=(time_col, "max"),
               n_runs=(time_col, "count"))
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
              nnz_map: dict, kernels_ordered: list, col: int,
              show_legend: bool, show_min: bool = False) -> None:
    value_col = "min_ms" if show_min else "mean_ms"
    for kernel in kernels_ordered:
        kdf = stats[stats["kernel"] == kernel].set_index("matrix_name")
        if kdf.empty:
            continue
        color        = KERNEL_COLORS.get(kernel, DEFAULT_COLOR)
        display_name = KERNEL_DISPLAY.get(kernel, kernel)

        values = [kdf.loc[m, value_col] if m in kdf.index else None for m in matrix_order]
        lowers = [
            kdf.loc[m, value_col] - kdf.loc[m, "min_ms"] if m in kdf.index else None
            for m in matrix_order
        ]
        uppers = [
            kdf.loc[m, "max_ms"] - kdf.loc[m, value_col] if m in kdf.index else None
            for m in matrix_order
        ]
        hover = [
            (
                f"<b>{display_name}</b><br>"
                f"matrix: {m}<br>"
                f"nnz: {nnz_map[m]:,.0f}<br>"
                f"mean: {kdf.loc[m,'mean_ms']:.2f} ms<br>"
                f"min: {kdf.loc[m,'min_ms']:.2f} ms<br>"
                f"max: {kdf.loc[m,'max_ms']:.2f} ms<br>"
                f"runs: {int(kdf.loc[m,'n_runs'])}"
            ) if m in kdf.index else None
            for m in matrix_order
        ]

        fig.add_trace(go.Scatter(
            name=display_name,
            legendgroup=display_name,
            showlegend=show_legend,
            mode="markers",
            x=list(range(len(matrix_order))),
            y=values,
            error_y=dict(
                type="data",
                symmetric=False,
                array=uppers,
                arrayminus=lowers,
                visible=True,
                color=color,
                thickness=1.2,
                width=4,
            ),
            marker=dict(color=color, size=9, opacity=0.9),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ), row=1, col=col)


def make_figure(stats_total: pd.DataFrame, stats_compute: pd.DataFrame,
                show_min: bool = False) -> go.Figure:
    basis = stats_total if not stats_total.empty else stats_compute
    matrix_order = (
        basis[["matrix_name", "nnz"]]
        .drop_duplicates()
        .sort_values("nnz")
        ["matrix_name"]
        .tolist()
    )
    nnz_map = (
        basis[["matrix_name", "nnz"]]
        .drop_duplicates()
        .set_index("matrix_name")["nnz"]
        .to_dict()
    )

    n = len(matrix_order)
    step = max(1, round(n / 10))
    tick_vals   = list(range(0, n, step))
    tick_labels = [_fmt_nnz(nnz_map[matrix_order[i]]) for i in tick_vals]

    kernels_in_data = set(stats_total["kernel"].unique().tolist()) | set(
        stats_compute["kernel"].unique().tolist()
    )
    kernels_ordered = [k for k in KERNEL_COLORS if k in kernels_in_data] + \
                      [k for k in sorted(kernels_in_data) if k not in KERNEL_COLORS]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Total time", "Compute time only"),
        horizontal_spacing=0.08,
    )

    # Legend entries shown once, preferring the left (total-time) panel.
    legend_shown = set()
    for col, stats in ((1, stats_total), (2, stats_compute)):
        present = set(stats["kernel"].unique().tolist())
        for kernel in kernels_ordered:
            if kernel not in present:
                continue
            show = kernel not in legend_shown
            legend_shown.add(kernel)
            _add_panel(fig, stats[stats["kernel"] == kernel], matrix_order,
                      nnz_map, [kernel], col, show_legend=show, show_min=show_min)

    fig.update_xaxes(
        tickmode="array", tickvals=tick_vals, ticktext=tick_labels,
        title="NNZ (ordered ascending)", gridcolor="#e0e0e0", row=1, col=1,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=tick_vals, ticktext=tick_labels,
        title="NNZ (ordered ascending)", gridcolor="#e0e0e0", row=1, col=2,
    )
    y_axis_label = "Min time [ms]" if show_min else "Mean time [ms]"
    fig.update_yaxes(title=y_axis_label, gridcolor="#e0e0e0", type="log",
                     row=1, col=1)
    fig.update_yaxes(title=y_axis_label, gridcolor="#e0e0e0", type="log",
                     row=1, col=2)

    fig.update_layout(
        title=dict(
            text="SuiteSparse GPU SpGEMM benchmark (A×A)"
            + (" (min)" if show_min else ""),
            font=dict(size=17),
            x=0.5,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=90, b=180),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _report_missing(df: pd.DataFrame, time_col: str) -> None:
    all_kernels = df["kernel"].unique()
    failed = (
        df.groupby(["matrix_name", "kernel"])[time_col]
          .apply(lambda s: s.isna().all() or (s == 0).all())
          .reset_index(name="failed")
    )
    failed = failed[failed["failed"]]
    all_pairs = pd.MultiIndex.from_product(
        [df["matrix_name"].unique(), all_kernels], names=["matrix_name", "kernel"]
    )
    present = df.groupby(["matrix_name", "kernel"]).size().index
    missing_pairs = all_pairs.difference(present)
    if not missing_pairs.empty:
        missing_df = missing_pairs.to_frame(index=False)
        missing_df["failed"] = True
        failed = pd.concat([failed, missing_df], ignore_index=True).drop_duplicates(
            subset=["matrix_name", "kernel"]
        )
    if not failed.empty:
        print(f"\nFailed / no valid data ({time_col}):")
        for mat, grp in failed.groupby("matrix_name"):
            kernels_failed = ", ".join(sorted(grp["kernel"].tolist()))
            print(f"  {mat}: {kernels_failed}")


def main():
    args  = parse_args()
    df    = load(args.file, include_warmup=args.warmup)

    if df.empty:
        sys.exit("No valid rows after filtering — nothing to plot.")

    stats_total   = build_stats(df, "total_ms")
    stats_compute = build_stats(df, "compute_ms")

    print(f"Matrices : {df['matrix_name'].nunique()}")
    print(f"Kernels  : {sorted(df['kernel'].unique())}")
    print(f"Warmup   : {'included' if args.warmup else 'excluded (run_id=0)'}")
    print("Timing   : total time (left panel) + compute time (right panel)")

    _report_missing(df, "compute_ms")
    print()

    fig = make_figure(stats_total, stats_compute, show_min=args.show_min)

    if args.out:
        fig.write_html(args.out, include_plotlyjs="cdn")
        print(f"Saved → {args.out}")
    if not args.out or args.show:
        fig.show()


if __name__ == "__main__":
    main()
