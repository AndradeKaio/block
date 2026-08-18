#!/usr/bin/env python3
"""
plot_gpu.py — Plot suite-sparse GPU benchmark results.

X-axis : matrices ordered by NNZ (ascending)
Y-axis : average total_ms across timed runs
Columns: one bar group per matrix, one color per contender

Usage:
  python plot_gpu.py --file suite_sparse_results.csv
  python plot_gpu.py --file results.csv --warmup --out plot.html
"""

import argparse
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    "prisma_tc_tile":       "PRISMA (TC tile)",
    "prisma_cuda":          "PRISMA (CUDA)",
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
        description="Plot suite-sparse GPU benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", required=True, help="Benchmark CSV produced by benchmark_gpu.py")
    p.add_argument("--warmup", action="store_true", default=False,
                   help="Include warmup run (run_id == 0) in the average")
    p.add_argument("--out", default=None,
                   help="Save HTML to this path (also opens in browser if omitted)")
    p.add_argument("--total-time", action="store_true", default=False, dest="total_time",
                   help="Plot total time (symbolic + compute) instead of compute time only")
    p.add_argument("--show", action="store_true", default=False,
                   help="Open in browser even when --out is given")
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

def make_figure(stats: pd.DataFrame, time_label: str) -> go.Figure:
    # X order: matrices sorted by NNZ ascending
    matrix_order = (
        stats[["matrix_name", "nnz"]]
        .drop_duplicates()
        .sort_values("nnz")
        ["matrix_name"]
        .tolist()
    )

    # NNZ labels for x-axis ticks
    nnz_map = (
        stats[["matrix_name", "nnz"]]
        .drop_duplicates()
        .set_index("matrix_name")["nnz"]
        .to_dict()
    )

    def fmt_nnz(v):
        if v >= 1_000_000_000: return f"{v/1e9:.1f}B"
        if v >= 1_000_000:     return f"{v/1e6:.1f}M"
        if v >= 1_000:         return f"{v/1e3:.0f}K"
        return str(int(v))

    n = len(matrix_order)
    step = max(1, round(n / 10))
    tick_vals   = list(range(0, n, step))
    tick_labels = [fmt_nnz(nnz_map[matrix_order[i]]) for i in tick_vals]

    kernels_in_data = stats["kernel"].unique().tolist()
    kernels_ordered = [k for k in KERNEL_COLORS if k in kernels_in_data] + \
                      [k for k in sorted(kernels_in_data) if k not in KERNEL_COLORS]

    fig = go.Figure()
    seen = set()

    for kernel in kernels_ordered:
        kdf = stats[stats["kernel"] == kernel].set_index("matrix_name")
        color        = KERNEL_COLORS.get(kernel, DEFAULT_COLOR)
        display_name = KERNEL_DISPLAY.get(kernel, kernel)
        show_legend  = kernel not in seen
        seen.add(kernel)

        means  = [kdf.loc[m, "mean_ms"] if m in kdf.index else None for m in matrix_order]
        lowers = [
            kdf.loc[m, "mean_ms"] - kdf.loc[m, "min_ms"] if m in kdf.index else None
            for m in matrix_order
        ]
        uppers = [
            kdf.loc[m, "max_ms"] - kdf.loc[m, "mean_ms"] if m in kdf.index else None
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
            y=means,
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
        ))

    fig.update_layout(
        title=dict(
            text=f"SuiteSparse GPU SpGEMM benchmark (A×A) — {time_label}",
            font=dict(size=17),
            x=0.5,
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_labels,
            title="NNZ (ordered ascending)",
            gridcolor="#e0e0e0",
        ),
        yaxis=dict(
            title=f"Mean {time_label} [ms]",
            gridcolor="#e0e0e0",
            type="log",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=90, b=140),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    df    = load(args.file, include_warmup=args.warmup)

    if df.empty:
        sys.exit("No valid rows after filtering — nothing to plot.")

    time_col   = "total_ms"   if args.total_time else "compute_ms"
    time_label = "total time" if args.total_time else "compute time"

    stats = build_stats(df, time_col)

    print(f"Matrices : {stats['matrix_name'].nunique()}")
    print(f"Kernels  : {sorted(stats['kernel'].unique())}")
    print(f"Warmup   : {'included' if args.warmup else 'excluded (run_id=0)'}")
    print(f"Timing   : {time_label} ({time_col})")

    # Report matrices where at least one contender has no valid runs
    all_kernels = df["kernel"].unique()
    failed = (
        df.groupby(["matrix_name", "kernel"])[time_col]
          .apply(lambda s: s.isna().all() or (s == 0).all())
          .reset_index(name="failed")
    )
    failed = failed[failed["failed"]]
    # Also flag matrices missing entirely for a kernel
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
        print("\nFailed / no valid data:")
        for mat, grp in failed.groupby("matrix_name"):
            kernels_failed = ", ".join(sorted(grp["kernel"].tolist()))
            print(f"  {mat}: {kernels_failed}")
    print()

    fig = make_figure(stats, time_label)

    if args.out:
        fig.write_html(args.out, include_plotlyjs="cdn")
        print(f"Saved → {args.out}")
    if not args.out or args.show:
        fig.show()


if __name__ == "__main__":
    main()
