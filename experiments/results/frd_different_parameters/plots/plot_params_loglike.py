# std-lib imports
import argparse
from pathlib import Path
import ast

# 3-party imports 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import colormaps

# projekt imports 
from RadarDataGen.Metrics.log_likelihood import poisson_entropy

# Settings plot ####################################
LABEL_SIZE   = 17
TICK_SIZE    = 14

plt.rcParams.update({
    "font.size": TICK_SIZE,          
    "axes.labelsize": LABEL_SIZE,    
    "xtick.labelsize": TICK_SIZE,    
    "ytick.labelsize": TICK_SIZE,    
})

####################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribution plots for selected dimensions.")
    parser.add_argument("--exp_id", "-e", required=True, help="Experiment folder name")
    args = parser.parse_args()

    base_path = Path("./experiments/results/fd_different_parameters") / args.exp_id
    path_plots = base_path / "plots"
    path_plots.mkdir(parents=True, exist_ok=True)

    data_path = base_path / "results_ll.csv"
    plot_path = path_plots / "plot_params_loglike.png"
    
    if not data_path.exists():
        raise FileNotFoundError("Required files not found.")

    data = pd.read_csv(data_path)

    required = {"log_like", "reference_params", "comparison_params", "dataset_size"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV fehlt Spalten: {missing}")

    features_order = ["lambda_lines_2d", "lambda_points_line_2d", "lambda_clutter"]

    groups = list(data.groupby("reference_params"))
    if len(groups) != 1:
        raise ValueError(f"Erwarte genau 1 reference_params, gefunden: {len(groups)}")
    ref_gen_str = groups[0][0]
    ref_gen_dict = ast.literal_eval(ref_gen_str)
    gen_params = {c: ref_gen_dict[c] for c in features_order}

    agg = (
        data.groupby(["comparison_params", "dataset_size"])
            .agg(mean_ll=("log_like", "mean"),
                 std_ll=("log_like", "std"),
                 var_ll=("log_like", "var"),
                 n=("log_like", "size"))
            .reset_index()
    )
   
    def parse_params(s):
        d = ast.literal_eval(s)
        return pd.Series({c: d.get(c, np.nan) for c in features_order})

    param_df = agg["comparison_params"].apply(parse_params)
    result = pd.concat([agg, param_df], axis=1)

    if result.empty:
        raise ValueError("Keine Ergebnisse zum Plotten.")

    nrows, ncols = 1, len(features_order)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5), tight_layout=True, sharey=True)

    if ncols == 1:
        axes = np.array([axes])

    cmap = colormaps["tab20"]

    def params_key(row):
        return (row["lambda_lines_2d"], row["lambda_points_line_2d"], row["lambda_clutter"])

    all_keys = sorted({params_key(r) for _, r in result.iterrows()})

    palette = cmap.colors 
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(all_keys)}

    expected_ll = (
            poisson_entropy(gen_params["lambda_lines_2d"]) +
            poisson_entropy(gen_params["lambda_points_line_2d"]) +
            poisson_entropy(gen_params["lambda_clutter"])
        )

    def same_intensity(row):
        lhs = row["lambda_lines_2d"] * row["lambda_points_line_2d"] + row["lambda_clutter"]
        rhs = gen_params["lambda_lines_2d"] * gen_params["lambda_points_line_2d"] + gen_params["lambda_clutter"]
       
        return np.isclose(lhs, rhs, rtol=0, atol=1e-12)

    for col_idx, feature in enumerate(features_order):
        ax = axes[col_idx]

        if feature == "lambda_lines_2d":
            xlabel, wl, xlim = r"$\lambda_\mathrm{Lines}$", 1, [2.5, 42.5]
        elif feature == "lambda_points_line_2d":
            xlabel, wl, xlim = r"$\lambda_\mathrm{Points/Line}$", 1, [0, 45]
        elif feature == "lambda_clutter":
            xlabel, wl, xlim = r"$\lambda_\mathrm{Clutter}$", 2.5, [-5, 85]
        else:
            xlabel, wl = feature, 1
            xlim = [float(result[feature].min()), float(result[feature].max())]

        ax.plot(xlim, [expected_ll, expected_ll], 'k-', lw=2, alpha=0.75)

        for idx, row in result.iterrows():
            xval = row[feature]
            yval = -row["mean_ll"]
            yerr = row["std_ll"] if not np.isnan(row["std_ll"]) else 0.0
            
            key = (row["lambda_lines_2d"], row["lambda_points_line_2d"], row["lambda_clutter"])
            col = color_map[key]

            marker = "s" if same_intensity(row) else "o"

            if xval > xlim[0] and xval < xlim[1]:
                ax.scatter(xval, yval, marker=marker, color=col, s=120, zorder=3)

                if yerr > 0:
                    ax.plot([xval, xval], [yval - yerr, yval + yerr], "-", c=col, lw=2, alpha=0.9, zorder=2)
                    ax.plot([xval - wl, xval + wl], [yval - yerr, yval - yerr], "-", c=col, lw=2, alpha=0.9)
                    ax.plot([xval - wl, xval + wl], [yval + yerr, yval + yerr], "-", c=col, lw=2, alpha=0.9)


        xticks = list(ax.get_xticks())
        xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
        if gen_params[feature] in xticks:
            pos = xticks.index(gen_params[feature])
            xticklabels[pos] = r"$\lambda_\mathrm{Ref}$"
        else:
            xticks.append(gen_params[feature])
            xticklabels.append(r"$\lambda_\mathrm{Ref}$")

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        ax.set_xlabel(xlabel)
        ax.set_xlim(xlim)

        ax.set_ylim([expected_ll - 5, 85]) #max(-result["mean_ll"]) * 1.1])
        ax.grid(True, alpha=0.3)

        if col_idx == 0:
            ax.set_ylabel("Negative Log-Likelihood")

    fig.savefig(plot_path, dpi=300)
    print(f"Plot saved at: {plot_path}")
