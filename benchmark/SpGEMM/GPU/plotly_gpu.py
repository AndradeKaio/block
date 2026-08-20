#!/usr/bin/env python3
"""
plotly_gpu.py — Interactive visualisation of sweep.py benchmark results.

Two browser figures (total time and compute-only time):
  - Row    : block range  (h_min, h_max, w_min, w_max)
  - Column : block density
  - X-axis : matrix size M (log scale)
  - Y-axis : mean time [ms], log scale, shared per row
  - Color  : contender / kernel
  - Error  : vertical bar min–max across timed runs
  - Hover  : kernel, M, mean ms, min, max

Usage
  python plotly_gpu.py --csv results.csv                        # opens two browser tabs
  python plotly_gpu.py --csv results.csv \
      --out-total total.html --out-compute compute.html         # also saves HTML files
"""

import argparse
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------

CSV_PATH = "results.csv"
SKIP_WARMUP = True  # exclude run_id == 0 (warmup)
MATRIX_SIZE_FILTER = None  # None = auto-pick the largest M present in data

# ---------------------------------------------------------------------------
# Fixed visual mappings
# ---------------------------------------------------------------------------

KERNEL_COLORS = {
    "prisma_tc_tile":  "#2ca02c",
    "prisma_tc_block": "#1f77b4",
    "prisma_cuda":     "#9467bd",
    "tilespgemm": "#ff7f0e",
    "cusparse_tilespgemm": "#d62728",
    "taco_gpu": "#e377c2",
    "tc_spgemm": "#8c564b",
}

KERNEL_DISPLAY = {
    "tc_spgemm": "IPDPS'26",
    "tilespgemm": "PPoPP'22",
    "cusparse_tilespgemm": "cuSPARSE(PPoPP'22)",
}

DEFAULT_COLOR = "#7f7f7f"

# ---------------------------------------------------------------------------
# Data loading  (identical logic to suite-sparse/plot_spgemm_gpu.py)
# ---------------------------------------------------------------------------

_NUMERIC = [
    "M",
    "K",
    "N",
    "blocks_A",
    "blocks_B",
    "block_h_min",
    "block_h_max",
    "block_w_min",
    "block_w_max",
    "block_density",
    "run_id",
    "symbolic_ms",
    "compute_ms",
    "total_ms",
    "n_pairs",
    "n_groups",
    "n_tc_descs",
    "n_cuda_descs",
]

_GROUP_COLS = [
    "M",
    "K",
    "N",
    "blocks_A",
    "block_density",
    "block_h_min",
    "block_h_max",
    "block_w_min",
    "block_w_max",
    "kernel",
]


def load_and_filter(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#")
    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["total_ms", "compute_ms"], how="all")
    if SKIP_WARMUP:
        df = df[df["run_id"] != 0]

    m = int(MATRIX_SIZE_FILTER if MATRIX_SIZE_FILTER is not None else df["M"].max())
    df = df[df["M"] == m].copy()

    if df.empty:
        sys.exit(f"No rows remain after filtering to M={m}.")

    print(f"Matrix size filter : M = {m}")
    print(f"Rows after filter  : {len(df)}")
    print(f"Kernels found      : {sorted(df['kernel'].unique())}")
    print(f"n_blocks values    : {sorted(df['blocks_A'].unique())}")
    print(f"Block densities    : {sorted(df['block_density'].unique())}")
    return df


def compute_stats(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    sub = df.dropna(subset=[time_col])
    return (
        sub.groupby(_GROUP_COLS)[time_col]
        .agg(t_min="min", t_med="mean", t_max="max")
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------


def make_figure(stats: pd.DataFrame, time_col: str, title: str) -> go.Figure:
    row_keys = sorted(
        stats[["block_h_min", "block_h_max", "block_w_min", "block_w_max"]]
        .drop_duplicates()
        .itertuples(index=False, name=None),
        key=lambda t: t[0],
    )
    col_keys = sorted(stats["blocks_A"].unique())
    x_vals = sorted(stats["block_density"].unique())

    n_rows, n_cols = len(row_keys), len(col_keys)
    if n_rows == 0 or n_cols == 0:
        fig = go.Figure()
        fig.update_layout(title_text=f"{title} — no data")
        return fig

    # Column headers only on the top row; blank for every other row.
    subplot_titles = []
    for i in range(n_rows):
        for j, nb in enumerate(col_keys):
            subplot_titles.append(f"blocks = {int(nb)}" if i == 0 else "")

    v_spacing = 0.10
    h_spacing = 0.04

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=h_spacing,
        vertical_spacing=v_spacing,
    )

    # Compute y-centre for each row (paper coords, 0=bottom 1=top).
    cell_h = (1.0 - v_spacing * (n_rows - 1)) / n_rows

    def row_center_y(i):
        return 1.0 - i * (cell_h + v_spacing) - cell_h / 2.0

    kernels_ordered = list(KERNEL_COLORS.keys()) + [
        k for k in sorted(stats["kernel"].unique()) if k not in KERNEL_COLORS
    ]
    seen_kernels: set = set()

    for i, rng in enumerate(row_keys):
        h_min, h_max, w_min, w_max = rng

        for j, n_blocks in enumerate(col_keys):
            cell = stats[
                (stats["block_h_min"] == h_min)
                & (stats["block_h_max"] == h_max)
                & (stats["block_w_min"] == w_min)
                & (stats["block_w_max"] == w_max)
                & (stats["blocks_A"] == n_blocks)
            ]

            for kernel in kernels_ordered:
                kdf = cell[cell["kernel"] == kernel].sort_values("block_density")
                if kdf.empty:
                    continue

                color = KERNEL_COLORS.get(kernel, DEFAULT_COLOR)
                display_name = KERNEL_DISPLAY.get(kernel, kernel)
                first_occurrence = kernel not in seen_kernels
                seen_kernels.add(kernel)

                xs = kdf["block_density"].tolist()
                y_med = kdf["t_med"].tolist()
                y_min = kdf["t_min"].tolist()
                y_max = kdf["t_max"].tolist()

                err_above = [hi - med for hi, med in zip(y_max, y_med)]
                err_below = [med - lo for med, lo in zip(y_med, y_min)]

                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=y_med,
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=err_above,
                            arrayminus=err_below,
                            thickness=1.2,
                            width=4,
                        ),
                        mode="lines+markers",
                        line=dict(color=color, width=1.6),
                        marker=dict(color=color, size=6),
                        name=display_name,
                        legendgroup=display_name,
                        showlegend=first_occurrence,
                        customdata=list(zip(y_min, y_max)),
                        hovertemplate=(
                            f"<b>{display_name}</b><br>"
                            "density=%{x:.2f}<br>"
                            "mean=%{y:.3f} ms<br>"
                            "min=%{customdata[0]:.3f} ms<br>"
                            "max=%{customdata[1]:.3f} ms"
                            "<extra></extra>"
                        ),
                    ),
                    row=i + 1,
                    col=j + 1,
                )

            # Linear x-axis (density 0–1); log y-axis.
            fig.update_xaxes(
                tickvals=x_vals,
                ticktext=[f"{x:.2f}" for x in x_vals],
                tickangle=45,
                showgrid=True,
                gridcolor="lightgrey",
                row=i + 1,
                col=j + 1,
            )
            fig.update_yaxes(
                type="log",
                showgrid=True,
                gridcolor="lightgrey",
                row=i + 1,
                col=j + 1,
            )

        # Row label on the left margin.
        fig.add_annotation(
            text=f"h=[{h_min},{h_max}]  w=[{w_min},{w_max}]",
            xref="paper",
            yref="paper",
            x=-0.03,
            y=row_center_y(i),
            showarrow=False,
            textangle=-90,
            font=dict(size=10),
            xanchor="right",
            yanchor="middle",
        )

    fig.update_layout(
        title_text=title,
        title_font_size=14,
        height=300 * n_rows + 150,
        width=310 * n_cols + 180,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.04,
            yanchor="bottom",
            font=dict(size=14),
        ),
        margin=dict(l=110, r=30, t=120, b=40),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Interactive Plotly benchmark grid")
    p.add_argument("--csv", default=CSV_PATH, help="input CSV file")
    p.add_argument("--out-total", default="", help="save total-time figure as HTML")
    p.add_argument("--out-compute", default="", help="save compute-time figure as HTML")
    args = p.parse_args()

    df = load_and_filter(args.csv)

    fig1 = make_figure(
        compute_stats(df, "total_ms"),
        "total_ms",
        "Total time  (symbolic + compute) [ms]",
    )
    fig2 = make_figure(
        compute_stats(df, "compute_ms"),
        "compute_ms",
        "Compute time  (numeric phase only) [ms]",
    )

    if args.out_total:
        fig1.write_html(args.out_total)
        print(f"Saved: {args.out_total}")
    if args.out_compute:
        fig2.write_html(args.out_compute)
        print(f"Saved: {args.out_compute}")

    fig1.show()
    fig2.show()


if __name__ == "__main__":
    main()
