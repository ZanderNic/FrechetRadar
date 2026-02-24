
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import colormaps

from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer
from RadarDataGen.Data_Generator.pseudo_radar_points import pseudo_radar_points

"""
Generates three plots for an experiment analyzing the Frechet Distance (FRD∞)
using a radar discretizer:

1) Example grid/image generated from pseudo_radar_points
2) FRD∞ vs. dataset size (log–log), including a WLS regression in 1/N
3) Rescaled FRD∞ vs. dataset size (log–log), including a WLS regression in 1/N

Expected files inside the experiment directory:
- results_cumalative_sampling.csv   -> columns: dataset_size, feature_dim, frechet_distance
- setting.json                      -> contains "discretizer_params" and "reference_generators"

Example usage:
    python3 ./experiments/results/fd_same_parameters/plots/plot_exp.py --exp exp_circle
    python3 ./experiments/results/fd_same_parameters/plots/plot_exp.py --exp exp_mixed_data
    python3 ./experiments/results/fd_same_parameters/plots/plot_exp.py --exp exp_rectangel_filled
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



def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_example_grids(axes, discretizer, config):
    for i in range(1):
        points = pseudo_radar_points(**config["reference_generators"][0])
        
        mask = (
            (points[:, 0] >= discretizer.x_min) & (points[:, 0] <= discretizer.x_max)
            & (points[:, 1] >= discretizer.y_min) & (points[:, 1] <= discretizer.y_max)
        )
        points = points[mask]

        grid = discretizer.points_to_grid(points)
        image = discretizer.grid_to_image(grid, swap_xy=True, invert_rows=True)
        
        axes[i].matshow(image, cmap="viridis")
        axes[i].set_title("Beispiel Realisierung " + str(i + 1))
        axes[i].xaxis.set_ticks_position('bottom')

def plot_fid_vs_size(axes, df):
    unique_features = sorted(df["feature_dim"].unique())
    cmap = colormaps.get_cmap("tab10")

    for idx, dim in enumerate(unique_features):
        sub = df[df["feature_dim"] == dim].groupby("dataset_size")["frechet_distance"].mean().reset_index()

        x = sub["dataset_size"].to_numpy(dtype=float)
        y = sub["frechet_distance"].to_numpy(dtype=float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        weights = x

        axes.scatter(x, y, marker="o", color=cmap(idx), label=f"dim={dim}")
        X_with_intercept = sm.add_constant(1 / x)
        model = sm.WLS(y, X_with_intercept, weights=weights).fit()
        axes.plot(x, model.predict(X_with_intercept), "--", color=cmap(idx))

    axes.set_xscale("log")
    axes.set_yscale("log")
    xticks = df["dataset_size"].unique()
    axes.set_xticks(xticks, [f'{int(x_i):d}' for x_i in xticks], rotation=45)
    axes.grid(True)
    axes.set_xlabel(r"Anzahl Samples")
    axes.set_ylabel(r"FRD$_\infty$")
    axes.set_title(r"FRD$_\infty$ vs Datensatz Größe")
    axes.legend()


def plot_rescaled_fid(axes, df, alpha=1.90):
    unique_features = sorted(df["feature_dim"].unique())
    cmap = colormaps.get_cmap("tab10")

    for idx, dim in enumerate(unique_features):
        sub = df[df["feature_dim"] == dim].groupby("dataset_size")["frechet_distance"].mean().reset_index()

        x = sub["dataset_size"].to_numpy(dtype=float)
        y = sub["frechet_distance"].to_numpy(dtype=float)
        y = [fdist * (1 / (dim * ((dim + 3) / 2))) for fdist in sub["frechet_distance"]] 
        weights = x

        axes.scatter(x, y, marker="o", color=cmap(idx), label=f"dim={dim}")
        X_with_intercept = sm.add_constant(1 / x)
        model = sm.WLS(y, X_with_intercept, weights=weights).fit()
        axes.plot(x, model.predict(X_with_intercept), "--", color=cmap(idx))

    axes.set_xscale("log")
    axes.set_yscale("log")
    xticks = df["dataset_size"].unique()
    axes.set_xticks(xticks, [f'{int(x_i):d}' for x_i in xticks], rotation=45)
    axes.grid(True)
    axes.set_xlabel(r"Anzahl Samples")
    axes.set_ylabel(r"Skalierte FRD$_\infty$")
    
    axes.set_title(
        rf"FRD$_\infty$ umskaliert mit $\frac{{dim * (dim + 3)}}{{2}}$"
    )

    axes.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribution plots for selected dimensions.")
    parser.add_argument("--exp_id", "-e", required=True, help="Experiment folder name")
    args = parser.parse_args()

    base_path = Path("./experiments/results/fd_same_parameters") / args.exp_id
    path_plots = base_path / "figs"
    path_plots.mkdir(parents=True, exist_ok=True)

    path_fid = base_path / "results_cumalative_sampling.csv"
    path_config = base_path / "setting.json"

    if not path_fid.exists() or not path_config.exists():
        raise FileNotFoundError("Required files not found.")

    df = pd.read_csv(path_fid)
    config = load_config(path_config)
    discretizer = RadarDiscretizer(**config["discretizer_params"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), tight_layout=True)
    axes = axes.flatten()
   
    plot_example_grids(axes, discretizer, config)
    plot_fid_vs_size(axes[1], df)
    plot_rescaled_fid(axes[2], df)

    fig.savefig(path_plots / "plot_ba.png", dpi=300)
    plt.close(fig)

