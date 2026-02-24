# std lib
import argparse
import math
import os

# third party
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import colormaps




"""

    Examples:
        python3 ./experiments/results/Diff_model_train/plots/plot_frd_curves.py --exp_id dit/test_dit_x0 

        python3 ./experiments/results/Diff_model_train/plots/plot_frd_curves.py --exp_id dit/test_dit_x0 
"""




def plot_frd_vs_n(ax, dff, group_col, fd_col, colors):
    for idx, (group, dfg) in enumerate(dff.groupby(group_col)):
        ddf = (
            dfg.groupby("dataset_size", as_index=False)
               .agg(mean_fd=(fd_col, "mean"))
               .sort_values("dataset_size")
        )

        x = ddf["dataset_size"].to_numpy()
        y = ddf["mean_fd"].to_numpy()

        weights = np.sqrt(x)
        X = sm.add_constant(1.0 / x)
        model = sm.WLS(y, X, weights=weights).fit()

        color = colors[idx % len(colors)]

        ax.scatter(x, y, s=40, marker="o", color=color, alpha=0.9)
        ax.plot(x, model.predict(X), "--", color=color, linewidth=2,
                label=f"Trained for {group} Batches")


    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    xticks = np.sort(dff["dataset_size"].unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x_i):d}" for x_i in xticks], rotation=45)

    ax.set_xlabel(r"Sample size $N$")
    ax.set_ylabel(r"Fréchet Radar Distance $FRD(P\Vert Q)$")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=8)


if __name__ == "__main__":
    group_col = "trained_batches"
    fd_col = "frechet_distance"

    parser = argparse.ArgumentParser(description="Plot FID vs Dataset Size")
    parser.add_argument("--exp_id", "-e", required=True, help="Experiment folder name")
    args = parser.parse_args()

    base_path = "./experiments/results/Diff_model_train/"
    path_plots = os.path.join(base_path, str(args.exp_id), "plots")
    path_fid = os.path.join(base_path, str(args.exp_id), "result_resampling.csv")

    os.makedirs(path_plots, exist_ok=True)

    df = pd.read_csv(path_fid)

    feature_dims = sorted(df["feature_dim"].unique())
    n_dims = len(feature_dims)

    cmap = colormaps["tab20"]
    colors = cmap.colors

    n_cols = max(1, math.ceil(math.sqrt(n_dims)))
    n_rows = max(1, math.ceil(n_dims / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(9 * n_cols, 8 * n_rows),
                             tight_layout=True)

    axes = np.array(axes).reshape(-1)

    (f"[INFO] Found {n_dims} feature_dims: {feature_dims}")


    for i, fd in enumerate(feature_dims):
        ax = axes[i]
        dff = df[df["feature_dim"] == fd]

        plot_frd_vs_n(ax, dff, group_col, fd_col, colors)

        ax.set_title(f"d = {fd}")

    for j in range(n_dims, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Frechet Distance vs Dataset Size", y=1.02, fontsize=12)

    outfile = os.path.join(
        path_plots,
        "fid_curves_model_train_grid.png" if n_dims > 1 else "fid_curves_model_train.png"
    )
    plt.savefig(outfile, dpi=300)

    print(f"[INFO] plot saved at {outfile}")