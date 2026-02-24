
# std-lib
import argparse
import ast
from pathlib import Path

# 3rd-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import colormaps

# project
from RadarDataGen.Metrics.log_likelihood import poisson_entropy


"""



 python3 ./experiments/results/fd_different_parameters/plots/plot_frd_to_loglike.py -d 4096 -e exp4
"""


TITLE_SIZE   = 20
LABEL_SIZE   = 18
TICK_SIZE    = 15
LEGEND_SIZE  = 12

plt.rcParams.update({
    "font.size": TICK_SIZE,          
    "axes.titlesize": TITLE_SIZE,   
    "axes.labelsize": LABEL_SIZE,    
    "xtick.labelsize": TICK_SIZE,    
    "ytick.labelsize": TICK_SIZE,    
    "legend.fontsize": LEGEND_SIZE,  
})


# ============================================================
# Helpers
# ============================================================

FEATURES = ["lambda_lines_2d", "lambda_points_line_2d", "lambda_clutter"]

def parse_params(s: str):
    d = ast.literal_eval(s)
    return {k: d[k] for k in FEATURES}

def parse_series(s: str):
    d = ast.literal_eval(s)
    return pd.Series({k: d[k] for k in FEATURES})

def expected_nll(gen_params):
    """Return theoretical minimal NLL for the reference generator."""
    return (
        poisson_entropy(gen_params["lambda_lines_2d"]) +
        poisson_entropy(gen_params["lambda_points_line_2d"]) +
        poisson_entropy(gen_params["lambda_clutter"])
    )

def build_color_map(keys):
    cmap = colormaps["tab20"]
    palette = cmap.colors
    return {k: palette[i % len(palette)] for i, k in enumerate(sorted(keys))}

def same_intensity(row, gen):
    lhs = row["lambda_lines_2d"] * row["lambda_points_line_2d"] + row["lambda_clutter"]
    rhs = gen["lambda_lines_2d"] * gen["lambda_points_line_2d"] + gen["lambda_clutter"]
    return np.isclose(lhs, rhs)


def dict_from_param_str(param_str):
    """
        Parses a comparison_params string into a dict with the 3 lambdas.
    """
    d = ast.literal_eval(param_str)
    return {k: d[k] for k in FEATURES}


# ============================================================
# Plot Function
# ============================================================

def plot_frd_vs_shifted_nll(merged, gen_params, out_path):

    # Build color map
    all_keys = set(zip(
        merged["lambda_lines_2d"],
        merged["lambda_points_line_2d"],
        merged["lambda_clutter"]
    ))
    color_map = build_color_map(all_keys)

    exp_nll = expected_nll(gen_params)
    merged["shifted_NLL"] = merged["NLL"] - exp_nll
    fig, ax = plt.subplots(figsize=(8.5, 7), tight_layout=True)

    for _, row in merged.iterrows():
        marker = "s" if same_intensity(row, gen_params) else "o"
        key = (row["lambda_lines_2d"], row["lambda_points_line_2d"], row["lambda_clutter"])
        col = color_map[key]
        
        if row["lambda_lines_2d"] != 0:
        
            ax.errorbar(
                row["FD"], row["shifted_NLL"],
                #yerr=row["std_ll"],
                fmt=marker,
                ms=9,
                color=col,
                ecolor=col,
                elinewidth=1.8,
                capsize=4,
                alpha=0.95,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{FRD}_\infty$")
    ax.set_ylabel(r"$\mathrm{NLL} - \mathbb{E}[\mathrm{NLL}_{\mathrm{ref}}]$")
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=300)
    print(f"[OK] Saved → {out_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FRD∞ vs shifted NLL")
    parser.add_argument("--exp_id", "-e", required=True)
    parser.add_argument("--feature_dim", "-d", type=int, default=256)
    args = parser.parse_args()

    # Paths
    base = Path("./experiments/results/fd_different_parameters") / args.exp_id
    path_ll = base / "results_ll.csv"
    path_frd = base / "results_cumalative_sampling.csv"
    out_path = base / "plots" / "plot_frd_vs_shifted_nll.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load LL
    ll_df = pd.read_csv(path_ll)
    ref = ast.literal_eval(ll_df["reference_params"].iloc[0])
    gen_params = {k: ref[k] for k in FEATURES}

    ll_agg = (
        ll_df.groupby("comparison_params", as_index=False)
             .agg(mean_ll=("log_like", "mean"),
                  std_ll=("log_like", "std"))
    )
    ll_agg[FEATURES] = ll_agg["comparison_params"].apply(parse_series)
    ll_agg["NLL"] = -ll_agg["mean_ll"]

    # Load FRD
    frd_df = pd.read_csv(path_frd)
    frd_df = frd_df[frd_df["feature_dim"] == args.feature_dim]

    rows = []
    for comp, g in frd_df.groupby("comparison_params"):
        ddf = g.groupby("dataset_size")["frechet_distance"].mean().reset_index()
        n = ddf["dataset_size"]
        y = ddf["frechet_distance"]

        X = pd.DataFrame({"inv_n": 1/n})
        model = sm.WLS(y, sm.add_constant(X), weights=np.sqrt(n)).fit()

        lam_dict = dict_from_param_str(comp)
        ci = model.conf_int().loc["const"].tolist()
        rows.append({
            "comparison_params": comp,
            "FD": model.params["const"],
            "FD_minus_CI": ci[0],
            "FD_plus_CI": ci[1],
            **lam_dict
        })

    frd_est = pd.DataFrame(rows)

    # Merge LL + FRD
    merged = pd.merge(
        frd_est,
        ll_agg[["comparison_params", "NLL", "std_ll"] + FEATURES],
        on=["comparison_params"] + FEATURES,
        how="inner"
    )

    # Plot
    plot_frd_vs_shifted_nll(merged, gen_params, out_path)
