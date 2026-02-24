"""
Create a single-panel plot of FRD_infty (WLS intercept) across five generator categories:
  - lines:        lambda_lines_2d & lambda_points_line_2d > 0
  - rectangles:   lambda_rectangle_2d & lambda_points_rectangle_2d > 0
  - circles:      lambda_circle & lambda_points_circle > 0
  - rect_outline: lambda_rect_outline_2d & lambda_points_rect_outline_2d > 0
  - clutter:      only clutter (lambda_clutter > 0) and NO shapes active

A generator is assigned to EXACTLY ONE category, using a prioritized order
that can be configured via --category_order (first matching category wins).

Notes:
- FRD_infty := intercept from WLS with y ~ const + 1/n, weights sqrt(n) over dataset_size.
- Points within the same category share the EXACT same x position (no horizontal jitter).
- This script also writes a CSV with ranks, parameters, and FRD statistics next to the plot.
"""

# std-lib imports
import argparse
import ast
from pathlib import Path
import sys

# 3rd-party imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import colormaps

# ----------------------------- Global settings ------------------------------

LABEL_SIZE = 19
TICK_SIZE  = 16

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
})

RELEVANT_COLUMNS_DEFAULT = [
    "lambda_lines_2d", "lambda_points_line_2d", "lambda_clutter",
    "lambda_circle", "lambda_points_circle",
    "lambda_rectangle_2d", "lambda_points_rectangle_2d",
    "lambda_rect_outline_2d", "lambda_points_rect_outline_2d",
]

# Default prioritized category order (first match wins)
CATEGORY_ORDER_DEFAULT = ["lines", "rect_outline", "rectangles",  "circles"]  # "clutter" optional

# Pretty labels for the x-axis
CATEGORY_DISPLAY = {
    "lines":        "Lines only",
    "rectangles":   "Rectangles only",
    "circles":      "Circles only",
    "rect_outline": "RectOutline only",
    "clutter":      "Clutter-only",
}

# ------------------------------ Helper functions ----------------------------

def round_key_tuple(d, keys, ndigits=6):
    """Return a stable color key tuple from a dict by rounding floats slightly."""
    return tuple(round(float(d.get(k, 0.0)), ndigits) for k in keys)

def detect_shapes(params):
    """Return booleans indicating which shapes are active."""
    has_lines        = (params.get("lambda_lines_2d", 0) > 0) and (params.get("lambda_points_line_2d", 0) > 0)
    has_rectangles   = (params.get("lambda_rectangle_2d", 0) > 0) and (params.get("lambda_points_rectangle_2d", 0) > 0)
    has_circles      = (params.get("lambda_circle", 0) > 0) and (params.get("lambda_points_circle", 0) > 0)
    has_rect_outline = (params.get("lambda_rect_outline_2d", 0) > 0) and (params.get("lambda_points_rect_outline_2d", 0) > 0)
    has_clutter      = (params.get("lambda_clutter", 0) > 0)
    return has_lines, has_rectangles, has_circles, has_rect_outline, has_clutter

def assign_category(params, category_order):
    """
    Assign EXACTLY ONE category based on prioritized order.
      - 'lines'/'rectangles'/'circles'/'rect_outline': if the respective shape is active.
      - 'clutter': only if NO shapes are active but clutter > 0.
    The first category in 'category_order' that matches is chosen.
    """
    has_lines, has_rectangles, has_circles, has_rect_outline, has_clutter = detect_shapes(params)
    flags = {
        "lines":        has_lines,
        "rectangles":   has_rectangles,
        "circles":      has_circles,
        "rect_outline": has_rect_outline,
        "clutter":      (has_clutter and not (has_lines or has_rectangles or has_circles or has_rect_outline)),
    }
    for cat in category_order:
        if flags.get(cat, False):
            return cat
    return None

def wls_frd_infty_per_comparison(df_one_fd):
    """
    Run WLS per (comparison_params):
        y = mean(FD) ~ const + 1/n, weights = sqrt(n).
    Return a list of dict rows containing FRD_infty (intercept), CI, p-value, and parameters.
    """
    rows = []
    for (comp_params, _fd), df in df_one_fd.groupby(["comparison_params", "feature_dim"]):
        comp_gen_dict = ast.literal_eval(comp_params)

        agg = df.groupby("dataset_size").agg({
            "frechet_distance": ["mean", "count", "std"]
        }).reset_index()
        agg.columns = ["dataset_size", "fd_mean", "fd_count", "fd_std"]

        x = agg["dataset_size"].astype(float)
        y = agg["fd_mean"].astype(float)
        if len(x) < 2:
            # Insufficient points for a meaningful regression; skip this combination
            continue

        weights = np.sqrt(x)
        X = pd.DataFrame({"inv_n": 1.0 / x})
        X_const = sm.add_constant(X)
        model = sm.WLS(y, X_const, weights=weights).fit()

        ci_low, ci_high = model.conf_int().loc["const", 0], model.conf_int().loc["const", 1]

        row = {
            "feature_dim": int(df["feature_dim"].iloc[0]),
            "FD": float(model.params["const"]),
            "-CI": float(ci_low),
            "+CI": float(ci_high),
            "FD_pval": float(model.pvalues["const"]),
        }
        row.update({k: float(comp_gen_dict.get(k, 0.0)) for k in RELEVANT_COLUMNS_DEFAULT})
        rows.append(row)
    return rows

# ----------------------------------- Main -----------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-panel FRD∞ plot over generator categories."
    )
    parser.add_argument("--exp_id", "-e", required=True,
                        help="Experiment folder name under ./experiments/results/fd_different_parameters")
    parser.add_argument("--feature_dims", "-d", default="4096",
                        help="Comma-separated feature dimensions (e.g., '128,256'). Default: '256'.")
    parser.add_argument("--category_order", "-o",
                        default=",".join(CATEGORY_ORDER_DEFAULT),
                        help="Prioritized category order, e.g., 'lines,rectangles,circles,rect_outline,clutter'.")
    parser.add_argument("--outfile", "-f", default="plot_by_category.png",
                        help="Output filename (saved under <exp_path>/plots/).")
    args = parser.parse_args()

    base_path   = Path("./experiments/results/fd_different_parameters") / args.exp_id
    plots_path  = base_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    data_path   = base_path / "results_cumalative_sampling.csv"
    out_path    = plots_path / args.outfile
    csv_out     = plots_path / "frd_by_category_ranks.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError("CSV is empty.")

    # Parse reference generator (must be unique)
    groups = list(data.groupby("reference_params"))
    if len(groups) != 1:
        raise ValueError(f"Expected exactly 1 'reference_params' group, found: {len(groups)}")
    ref_gen_str  = groups[0][0]
    ref_gen_dict = ast.literal_eval(ref_gen_str)
    ref_params   = {k: float(ref_gen_dict.get(k, 0.0)) for k in RELEVANT_COLUMNS_DEFAULT}

    # Filter to selected feature dimensions
    try:
        feature_dims = sorted({int(x.strip()) for x in args.feature_dims.split(",") if x.strip() != ""})
    except Exception as ex:
        raise ValueError(f"Failed to parse --feature_dims: {args.feature_dims}") from ex

    if feature_dims:
        data = data[data["feature_dim"].isin(feature_dims)]
        if data.empty:
            raise ValueError(f"No data for feature_dim(s)={feature_dims}")

    # Compute WLS per comparison_params
    result_rows = []
    for fd in sorted(data["feature_dim"].unique()):
        df_fd = data[data["feature_dim"] == fd]
        result_rows.extend(wls_frd_infty_per_comparison(df_fd))

    if not result_rows:
        raise ValueError("No WLS results (possibly too few points per dataset_size).")

    result = pd.DataFrame(result_rows)

    # Assign categories (exactly one per combination) based on prioritized order
    category_order = [c.strip() for c in args.category_order.split(",") if c.strip()]
    valid_cats = {"lines", "rectangles", "circles", "rect_outline", "clutter"}
    invalid = [c for c in category_order if c not in valid_cats]
    if invalid:
        raise ValueError(f"Unknown categories in --category_order: {invalid}. Allowed: {sorted(valid_cats)}")

    categories = []
    for _, row in result.iterrows():
        params = {k: float(row.get(k, 0.0)) for k in RELEVANT_COLUMNS_DEFAULT}
        cat = assign_category(params, category_order)
        categories.append(cat)
    result["category"] = categories
    result = result[~result["category"].isna()].copy()
    if result.empty:
        raise ValueError("No combinations could be assigned to categories. Check parameters and rules.")

    # ------------------------------- Ranks CSV --------------------------------
    # Compute ranks: lower FRD is better.
    # - overall_rank: global rank across all rows (ties handled with 'min')
    # - category_rank: rank within the same category
    result["overall_rank"] = result["FD"].rank(method="min", ascending=True).astype(int)
    result["category_rank"] = (
        result.groupby("category")["FD"]
              .rank(method="min", ascending=True)
              .astype(int)
    )

    # Choose output columns: ranks + stats + parameters
    out_cols = (
        ["overall_rank", "category_rank", "category", "feature_dim", "FD", "-CI", "+CI", "FD_pval"]
        + RELEVANT_COLUMNS_DEFAULT
    )
    # Ensure all columns exist (robustness)
    out_cols = [c for c in out_cols if c in result.columns]

    result_sorted = result.sort_values(["overall_rank", "category", "FD"], ascending=[True, True, True])
    result_sorted.to_csv(csv_out, index=False)

    # ------------------------------- Plotting --------------------------------

    # Stable colors per full parameter combination
    cmap = colormaps["tab20"]
    keys = sorted({
        round_key_tuple({k: result.iloc[i][k] for k in RELEVANT_COLUMNS_DEFAULT}, RELEVANT_COLUMNS_DEFAULT)
        for i in range(len(result))
    })
    palette = list(cmap.colors)
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(keys)}

    def color_for_row(r):
        key = round_key_tuple({k: r[k] for k in RELEVANT_COLUMNS_DEFAULT}, RELEVANT_COLUMNS_DEFAULT)
        return color_map.get(key, "C0")

    # Map category to x position (according to given order)
    x_positions = {cat: i for i, cat in enumerate(category_order)}
    x_labels    = [CATEGORY_DISPLAY.get(cat, cat) for cat in category_order]

    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)

    # Baseline & y-scale
    ax.axhline(0.0, color="k", lw=2, alpha=0.75)
    ax.set_yscale('symlog', linthresh=100.0, linscale=0.5)
    ax.set_ylim([-110, 1e5])

    # Whisker cap width in x units
    wl = 0.1

    # Reference generator category (for annotation)
    ref_cat = assign_category(ref_params, category_order)

    for _, row in result.iterrows():
        cat = row["category"]
        if cat not in x_positions:
            continue
        x_center = x_positions[cat]
        # Exact vertical alignment (no jitter)
        xval = x_center

        yval = float(row["FD"])
        ci1  = float(row["-CI"])
        ci2  = float(row["+CI"])
        col  = color_for_row(row)

        # Your original marker logic preserved (square for "matching" sum rule)
        if (
            row['lambda_lines_2d'] * row['lambda_points_line_2d']
            + row['lambda_clutter']
            + row['lambda_circle'] * row['lambda_points_circle']
            + row['lambda_rectangle_2d'] * row['lambda_points_rectangle_2d']
            + row['lambda_rect_outline_2d'] * row['lambda_points_rect_outline_2d']
            == ref_params['lambda_lines_2d'] * ref_params['lambda_points_line_2d']
            + ref_params['lambda_clutter']
        ):
            marker = 's'
        else:
            marker = 'o'

        # CI whiskers
        ax.plot([xval, xval], [ci1, ci2], '-', c=col, lw=2)
        ax.plot([xval - wl, xval + wl], [ci1, ci1], '-', c=col, lw=2)
        ax.plot([xval - wl, xval + wl], [ci2, ci2], '-', c=col, lw=2)

        # Point
        ax.scatter(xval, yval, color=col, s=150, marker=marker, linewidth=1.2, alpha=1)

    ax.set_xticks(list(x_positions.values()), x_labels)
    ax.set_xlabel("Generator category (prioritized)")
    ax.set_ylabel(r"$\mathrm{FRD}_\infty$")
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"[OK] Plot saved to: {out_path}")
    print(f"[OK] Ranks CSV saved to: {csv_out}")

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"[ERROR] {ex}", file=sys.stderr)
        sys.exit(1)