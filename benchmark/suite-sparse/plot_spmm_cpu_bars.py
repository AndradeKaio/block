#!/usr/bin/env python3
"""
plot_spmm_cpu_bars.py — Bar-chart view of suite-sparse CPU SpMM benchmark
results.

One grouped bar per kernel per matrix instead of a scatter point. Each
kernel's bar has two stacked segments: compute time (the kernel's own
color, on the bottom) and symbolic time (a single shared gray, stacked on
top) -- so enabling/growing the symbolic cost for a kernel visibly makes
its bar taller, without needing a second panel. bar height = compute_ms +
symbolic_ms = total_ms.

Log Y-axis, matching plot_spmm_cpu.py's scatter plot. Note: on a log axis
a stacked segment's on-screen height no longer reads as proportional to its
share of the bar (log compresses the top of a tall bar much more than the
bottom) -- each segment is still positioned at its correct numeric value,
just not perceptually "additive" by eye the way it would be on a linear
axis. Traded off in favor of keeping matrices with very different time
scales all readable in one chart, same reasoning as the scatter plot.

X-axis : matrices ordered by NNZ (ascending)
Y-axis : mean time per run [ms], log scale, compute + symbolic stacked

Usage:
  python plot_spmm_cpu_bars.py --file spmm_results.csv
  python plot_spmm_cpu_bars.py --file results.csv --warmup --out plot.html
"""

import argparse
import sys

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Visual settings
# ---------------------------------------------------------------------------

KERNEL_COLORS = {
    # TACO variants → blues
    "taco":                "#1f77b4",
    "taco_opt0":            "#6baed6",
    "taco_opt1":            "#08519c",
    # Prisma variants → greens
    "prisma_cpu":           "#41ab5d",
    "prisma_auto":          "#a1d99b",
    "prisma_specialized":   "#006d2c",
    "prisma_static":        "#74c476",
    "prisma_tiled":         "#00441b",
}

KERNEL_DISPLAY = {
    "taco":                "TACO",
    "taco_opt0":            "TACO opt0",
    "taco_opt1":            "TACO opt1",
    "prisma_cpu":           "BLOCKS (CPU)",
    "prisma_auto":          "BLOCKS (auto)",
    "prisma_specialized":   "BLOCKS (specialized)",
    "prisma_static":        "BLOCKS (static sched)",
    "prisma_tiled":         "BLOCKS (tiled)",
}

DEFAULT_COLOR  = "#7f7f7f"
SYMBOLIC_COLOR = "#bdbdbd"  # one shared color for every kernel's symbolic segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Bar-chart view of suite-sparse CPU SpMM benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", required=True, help="Benchmark CSV produced by benchmark_spmm_cpu.py")
    p.add_argument("--warmup", action="store_true", default=False,
                   help="Include warmup run (run_id == 0) in the average")
    p.add_argument("--out", default=None,
                   help="Save HTML to this path (also opens in browser if omitted)")
    p.add_argument("--show", action="store_true", default=False,
                   help="Open in browser even when --out is given")
    p.add_argument("--note", default="",
                   help="Text appended to the plot title (e.g. 'AVX-512, dual-acc')")
    return p.parse_args()


def load(path: str, include_warmup: bool) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()

    required = {"matrix_name", "nnz", "kernel", "run_id", "symbolic_ms", "compute_ms"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing columns: {missing}")

    for col in ("symbolic_ms", "compute_ms", "nnz"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not include_warmup:
        df = df[df["run_id"] != 0]

    return df


def build_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["compute_ms"].notna() & (df["compute_ms"] > 0)]
    agg = df.groupby(["matrix_name", "nnz", "kernel"], as_index=False).agg(
        compute_mean=("compute_ms", "mean"),
        symbolic_mean=("symbolic_ms", "mean"),
        n_runs=("compute_ms", "count"),
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


def make_figure(stats: pd.DataFrame, note: str = "") -> go.Figure:
    matrix_order = (
        stats[["matrix_name", "nnz"]]
        .drop_duplicates()
        .sort_values("nnz")["matrix_name"]
        .tolist()
    )
    nnz_map = (
        stats[["matrix_name", "nnz"]]
        .drop_duplicates()
        .set_index("matrix_name")["nnz"]
        .to_dict()
    )

    n = len(matrix_order)
    step = max(1, round(n / 10))
    tick_vals = list(range(0, n, step))
    tick_labels = [_fmt_nnz(nnz_map[matrix_order[i]]) for i in tick_vals]

    kernels_in_data = stats["kernel"].unique().tolist()
    kernels_ordered = [k for k in KERNEL_COLORS if k in kernels_in_data] + \
                      [k for k in sorted(kernels_in_data) if k not in KERNEL_COLORS]

    fig = go.Figure()
    symbolic_legend_shown = False

    for kernel in kernels_ordered:
        kdf = stats[stats["kernel"] == kernel].set_index("matrix_name")
        color = KERNEL_COLORS.get(kernel, DEFAULT_COLOR)
        display_name = KERNEL_DISPLAY.get(kernel, kernel)

        compute_vals  = [kdf.loc[m, "compute_mean"]  if m in kdf.index else 0 for m in matrix_order]
        symbolic_vals = [kdf.loc[m, "symbolic_mean"] if m in kdf.index else 0 for m in matrix_order]

        hover_compute = [
            (
                f"<b>{display_name}</b><br>"
                f"matrix: {m}<br>"
                f"nnz: {nnz_map[m]:,.0f}<br>"
                f"compute: {kdf.loc[m, 'compute_mean']:.3f} ms<br>"
                f"symbolic: {kdf.loc[m, 'symbolic_mean']:.3f} ms<br>"
                f"total: {kdf.loc[m, 'compute_mean'] + kdf.loc[m, 'symbolic_mean']:.3f} ms<br>"
                f"runs: {int(kdf.loc[m, 'n_runs'])}"
            ) if m in kdf.index else None
            for m in matrix_order
        ]
        hover_symbolic = [
            (
                f"<b>{display_name} — symbolic</b><br>"
                f"matrix: {m}<br>"
                f"symbolic: {kdf.loc[m, 'symbolic_mean']:.3f} ms"
            ) if m in kdf.index else None
            for m in matrix_order
        ]

        # Same offsetgroup for both segments of one kernel -> they stack
        # into one bar; different kernels get different offsetgroups -> those
        # bars sit side by side (barmode="group" below).
        fig.add_trace(go.Bar(
            name=display_name,
            legendgroup=kernel,
            offsetgroup=kernel,
            x=list(range(len(matrix_order))),
            y=compute_vals,
            marker_color=color,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_compute,
        ))
        fig.add_trace(go.Bar(
            name="Symbolic",
            legendgroup="__symbolic__",
            offsetgroup=kernel,
            showlegend=not symbolic_legend_shown,
            x=list(range(len(matrix_order))),
            y=symbolic_vals,
            marker_color=SYMBOLIC_COLOR,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_symbolic,
        ))
        symbolic_legend_shown = True

    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        title=dict(
            text="SuiteSparse CPU SpMM benchmark — compute + symbolic (stacked)"
            + (f" — {note}" if note else ""),
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
            title="Mean time [ms] (compute + symbolic)",
            gridcolor="#e0e0e0",
            type="log",
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


def _report_missing(df: pd.DataFrame) -> None:
    all_kernels = df["kernel"].unique()
    failed = (
        df.groupby(["matrix_name", "kernel"])["compute_ms"]
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
        print("\nFailed / no valid data:")
        for mat, grp in failed.groupby("matrix_name"):
            kernels_failed = ", ".join(sorted(grp["kernel"].tolist()))
            print(f"  {mat}: {kernels_failed}")


def main():
    args = parse_args()
    df = load(args.file, include_warmup=args.warmup)

    if df.empty:
        sys.exit("No valid rows after filtering — nothing to plot.")

    stats = build_stats(df)

    print(f"Matrices : {df['matrix_name'].nunique()}")
    print(f"Kernels  : {sorted(df['kernel'].unique())}")
    print(f"Warmup   : {'included' if args.warmup else 'excluded (run_id=0)'}")
    print("Timing   : compute (bottom segment) + symbolic (top segment, stacked)")

    _report_missing(df)
    print()

    fig = make_figure(stats, note=args.note)

    if args.out:
        fig.write_html(args.out, include_plotlyjs="cdn")
        print(f"Saved → {args.out}")
    if not args.out or args.show:
        fig.show()


if __name__ == "__main__":
    main()
