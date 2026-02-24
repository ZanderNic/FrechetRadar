
# std-lib imports
import argparse
import ast
from pathlib import Path

# 3rd-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import colormaps

# project imports
from RadarDataGen.Metrics.log_likelihood import poisson_entropy





"""



python3 ./experiments/results/fd_different_parameters/plots/plot_ba.py --exp_id exp_div_num_point_line_2 -c lambda_points_line_2d -d 4096;

python3 ./experiments/results/fd_different_parameters/plots/plot_ba.py --exp_id exp_div_num_clutter_2 -c lambda_clutter -d 4096;

python3 ./experiments/results/fd_different_parameters/plots/plot_ba.py --exp_id exp_div_lines_2 -c lambda_lines_2d -d 4096;

"""




TITLE_SIZE   = 20
LABEL_SIZE   = 16
TICK_SIZE    = 13
LEGEND_SIZE  = 12

plt.rcParams.update({
    "font.size": TICK_SIZE,          
    "axes.titlesize": TITLE_SIZE,   
    "axes.labelsize": LABEL_SIZE,    
    "xtick.labelsize": TICK_SIZE,    
    "ytick.labelsize": TICK_SIZE,    
    "legend.fontsize": LEGEND_SIZE,  
})



# ============================================
# Utility Helpers
# ============================================

FEATURES = ["lambda_lines_2d", "lambda_points_line_2d", "lambda_clutter"]

def parse_params(s: str):
    d = ast.literal_eval(s)
    return pd.Series({c: d.get(c, np.nan) for c in FEATURES})


def build_color_map(keys):
    cmap = colormaps["tab20"]
    palette = cmap.colors
    return {k: palette[i % len(palette)] for i, k in enumerate(sorted(keys))}


def expected_nll(gen_params):
    """Expected negative log likelihood from Poisson entropies (3 components)."""
    return (
        poisson_entropy(gen_params["lambda_lines_2d"])
        + poisson_entropy(gen_params["lambda_points_line_2d"])
        + poisson_entropy(gen_params["lambda_clutter"])
    )


def same_intensity(row, gen_params):
    lhs = row["lambda_lines_2d"] * row["lambda_points_line_2d"] + row["lambda_clutter"]
    rhs = gen_params["lambda_lines_2d"] * gen_params["lambda_points_line_2d"] + gen_params["lambda_clutter"]
    return np.isclose(lhs, rhs)


# ============================================
# Plotting Functions
# ============================================

def plot_frd(ax, merged, relevant_column, color_map, lambda_ref):
    ax.axhline(0, color="black", lw=1.6, alpha=0.9)
    ax.set_yscale("symlog", linthresh=100.0, linscale=0.5)

    for _, row in merged.iterrows():
        key = (row["lambda_lines_2d"], row["lambda_points_line_2d"], row["lambda_clutter"])
        col = color_map[key]

        ax.errorbar(
            row[relevant_column], row["FD"],
            yerr=[[row["FD"] - row["FD_minus_CI"]],
                  [row["FD_plus_CI"] - row["FD"]]],
            fmt="o", ms=10, color=col, mfc=col,
            elinewidth=2.0, capsize=5, alpha=1
        )
    
    xticks = list(ax.get_xticks())
    xticklabels = [str(t) for t in xticks]

    if any(np.isclose(t, lambda_ref) for t in xticks):
        idx = np.argmin(np.abs(np.array(xticks) - lambda_ref))
        xticklabels[idx] = r"$\lambda_\mathrm{Ref}$"
    else:
        xticks.append(lambda_ref)
        xticklabels.append(r"$\lambda_\mathrm{Ref}$")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


    ax.set_xlabel(relevant_column)
    ax.set_ylabel(r"$\mathrm{FRD}_\infty$")
    ax.grid(True)


def plot_nll(ax, merged, relevant_column, color_map, expected_ll, lambda_ref):
    ax.axhline(expected_ll, color="black", lw=1.6, alpha=0.9)

    for _, row in merged.iterrows():
        key = (row["lambda_lines_2d"], row["lambda_points_line_2d"], row["lambda_clutter"])
        col = color_map[key]

        ax.errorbar(
            row[relevant_column], row["NLL"],
            yerr=row["std_ll"],
            fmt="o", ms=10, color=col, mfc=col,
            elinewidth=2.0, capsize=5, alpha=1
        )

    
    xticks = list(ax.get_xticks())
    xticklabels = [str(t) for t in xticks]

    if any(np.isclose(t, lambda_ref) for t in xticks):
        idx = np.argmin(np.abs(np.array(xticks) - lambda_ref))
        xticklabels[idx] = r"$\lambda_\mathrm{Ref}$"
    else:
        xticks.append(lambda_ref)
        xticklabels.append(r"$\lambda_\mathrm{Ref}$")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


    ax.set_xlabel(relevant_column)
    ax.set_ylabel(r"$\mathrm{NLL}$")
    ax.grid(True)


def plot_frd_vs_nll(ax, merged, color_map, expected_ll):

    ax.set_xscale("log")
    ax.set_yscale("log")

    for _, row in merged.iterrows():
        key = (row["lambda_lines_2d"], row["lambda_points_line_2d"], row["lambda_clutter"])
        col = color_map[key]

        ax.errorbar(
            row["FD"], row["NLL"] - expected_ll,
            yerr=row["std_ll"],
            fmt="o", ms=10, color=col, mfc=col,
            elinewidth=2.0, capsize=5, alpha=1
        )

    ax.set_xlabel(r"$\mathrm{FRD}_\infty$")
    ax.set_ylabel(r"$\mathrm{NLL} - \mathbb{E}[\mathrm{NLL}_{\mathrm{ref}}]$")
    ax.grid(True)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined FRD/NLL plotting tool.")
    parser.add_argument("--exp_id", "-e", required=True)
    parser.add_argument("--relevant_column", "-c", required=True, choices=FEATURES)
    parser.add_argument("--feature_dim", "-d", type=int, default=256)
    args = parser.parse_args()

    base_path = Path("./experiments/results/fd_different_parameters") / args.exp_id
    plot_dir = base_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load Data
    ll_df = pd.read_csv(base_path / "results_ll.csv")
    frd_df = pd.read_csv(base_path / "results_cumalative_sampling.csv")

    # Reference generator parameters
    ref_gen = ast.literal_eval(ll_df["reference_params"].iloc[0])
    gen_params = {k: ref_gen[k] for k in FEATURES}

    # Aggregate LL
    ll_agg = (
        ll_df.groupby("comparison_params", as_index=False)
        .agg(mean_ll=("log_like", "mean"), std_ll=("log_like", "std"))
    )
    ll_agg[FEATURES] = ll_agg["comparison_params"].apply(parse_params)
    ll_agg["NLL"] = -ll_agg["mean_ll"]

    # Aggregate FRD
    frd_df = frd_df[frd_df["feature_dim"] == args.feature_dim]
    rows = []

    for comp, g in frd_df.groupby("comparison_params"):
        ddf = g.groupby("dataset_size")["frechet_distance"].mean().reset_index()
        n = ddf["dataset_size"]
        y = ddf["frechet_distance"]
        X = pd.DataFrame({"inv_n": 1 / n})
        model = sm.WLS(y, sm.add_constant(X), weights=np.sqrt(n)).fit()
        comp_dict = ast.literal_eval(comp)

        ci = model.conf_int().loc["const"].values.tolist()
        rows.append({
            "comparison_params": comp,
            "FD": model.params["const"],
            "FD_minus_CI": ci[0],
            "FD_plus_CI": ci[1],
            **comp_dict
        })

    frd_est = pd.DataFrame(rows)

    # Merge datasets
    merged = pd.merge(frd_est, ll_agg, on=["comparison_params"] + FEATURES, how="inner")

    # Color map
    all_keys = set(zip(merged["lambda_lines_2d"], merged["lambda_points_line_2d"], merged["lambda_clutter"]))
    color_map = build_color_map(all_keys)

    # Compute expected baseline NLL
    exp_negative_ll = expected_nll(gen_params)

    # PLOTS
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

    plot_frd(axes[0], merged, args.relevant_column, color_map, lambda_ref = gen_params[args.relevant_column])
    plot_nll(axes[1], merged, args.relevant_column, color_map, exp_negative_ll,  lambda_ref = gen_params[args.relevant_column])
    plot_frd_vs_nll(axes[2], merged, color_map, exp_negative_ll)

    fig.savefig(plot_dir / f"combined_plot_{args.relevant_column}.png", dpi=300)
    print(f"[OK] saved → {plot_dir / f'combined_plot_{args.relevant_column}.png'}")