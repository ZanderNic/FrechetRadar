# std-lib imports 
import argparse
import ast
from pathlib import Path
import math

# 3 party imports 
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from matplotlib import colormaps


LABEL_SIZE   = 19
TICK_SIZE    = 16

plt.rcParams.update({
    "font.size": TICK_SIZE,          
    "axes.labelsize": LABEL_SIZE,    
    "xtick.labelsize": TICK_SIZE,    
    "ytick.labelsize": TICK_SIZE,    
})


if __name__ == "__main__":

    # Settings ##################################
    relevant_columns = ["lambda_lines_2d", "lambda_points_line_2d", "lambda_clutter"]
    #relevant_columns = ["lambda_lines_2d", "lambda_points_line_2d", "lambda_clutter", "lambda_circle", "lambda_points_circle",  "lambda_rectangle_2d", "lambda_points_rectangle_2d", "lambda_rect_outline_2d", "lambda_points_rect_outline_2d"]
    feature_dims_all = [4096] #sorted(data["feature_dim"].unique())
    ###############################################

    parser = argparse.ArgumentParser(description="Distribution plots for selected dimensions.")
    parser.add_argument("--exp_id", "-e", required=True, help="Experiment folder name")
    args = parser.parse_args()

    base_path = Path("./experiments/results/fd_different_parameters") / args.exp_id
    path_plots = base_path / "plots"
    path_plots.mkdir(parents=True, exist_ok=True)

    data_path = base_path / "results_cumalative_sampling.csv"
    plot_path = path_plots / "plot_params.png"
    
    if not data_path.exists():
        raise FileNotFoundError("Required files not found.")

    data = pd.read_csv(data_path)

    # reference generator
    df_ref = data.groupby("reference_params")

    groups = list(df_ref)
    if len(groups) == 1:
        ref_gen_str = groups[0][0]
        ref_gen_dict = ast.literal_eval(ref_gen_str)
        gen_params = {c: ref_gen_dict.get(c, 0) for c in relevant_columns}
    else:
        raise ValueError("More than one reference generator.")

    # compute results per feature_dim
    all_comp_dicts = []
    results_by_dim = []
    for feature_dim in feature_dims_all:
        df_sorted = data[data.feature_dim == feature_dim].sort_values(by="comparison_params")
        res = []
        for (comp_params, fd), df in df_sorted.groupby(["comparison_params", "feature_dim"]):
            ddf = df.groupby('dataset_size', as_index=False).describe()
            comp_gen_dict = ast.literal_eval(comp_params)

            x = ddf['dataset_size']
            y = ddf['frechet_distance']["mean"]
            weights = np.sqrt(x)

            X = pd.DataFrame({'inv_n': 1 / x})
            X_with_intercept = sm.add_constant(X)
            model = sm.WLS(y, X_with_intercept, weights=weights).fit()

            comp_dict = {c: comp_gen_dict.get(c, 0) for c in relevant_columns}
            all_comp_dicts.append(comp_dict)

            res.append({
                'feature_dim': feature_dim,
                'FD': model.params['const'],
                '-CI': model.conf_int().loc['const', 0],
                '+CI': model.conf_int().loc['const', 1],
                'FD_pval': model.pvalues['const'],
                **comp_dict
            })

        result = pd.DataFrame(res)
        if not result.empty:
            results_by_dim.append((feature_dim, result))

    if len(results_by_dim) == 0:
        raise ValueError("No results to plot.")

    # figure with rows = feature_dims, cols = 3 features
    if len(relevant_columns) > 3 and len(feature_dims_all) == 1:
        nrows = math.ceil(len(relevant_columns) / 3)
        ncols = 3 

    else:
        nrows = len(results_by_dim)
        ncols = len(relevant_columns)


    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), tight_layout=True, sharey=True)
    axes = axes.reshape(-1)

    cmap = colormaps["tab20"]
    all_keys = sorted([
        tuple(int(d[string]) for string in relevant_columns) for d in all_comp_dicts
    ])

    palette = cmap.colors 
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(all_keys)}

    for row_idx in range(nrows):
        axi = axes[row_idx * nrows]
        axi.set_ylabel(r"$\text{FRD}_\infty$")

        for col_idx, feature in enumerate(relevant_columns[row_idx * ncols: (row_idx + 1) * (ncols)]):
            axi = axes[row_idx * nrows + col_idx]
            
            print(f"Plotting feature: {feature}")

            if feature == "lambda_lines_2d":
                xlabel, wl, xlim, ticks = r"$\lambda_{\mathrm{Lines}}$", 1, [-4, 40], [0, 40, 10]
            elif feature == "lambda_points_line_2d":
                xlabel, wl, xlim, ticks = r"$\lambda_{\mathrm{Points/Line}}$", 1, [-4, 50], [0, 50, 10]
            elif feature == "lambda_clutter":
                xlabel, wl, xlim, ticks = r"$\lambda_{\mathrm{Clutter}}$", 2.5, [-8, 80], [0, 80, 20]
            elif feature == "lambda_rectangle_2d":
                xlabel, wl, xlim, ticks = r"$\lambda_{\mathrm{Rectangle}}$", 0.1, [-4, 50], [0, 50, 10] #[-1, 4], [0, 4, 1]
            elif feature == "lambda_points_rectangle_2d":
                xlabel, wl, xlim, ticks = r"$\lambda_{\mathrm{Points/Rectangle}}$", 2.5, [-4, 50], [0, 50, 10]#[-20, 120], [0, 120, 60]
            elif feature == "lambda_circle":
                xlabel, wl, xlim, ticks = r"$\lambda_{\mathrm{Circle}}$", 1,[-4, 50], [0, 50, 10]# [-3, 12], [0, 12, 4]
            elif feature == "lambda_points_circle":
                xlabel, wl, xlim, ticks = r"$\lambda_{\mathrm{Points/Circle}}$", 2.5, [-4, 50], [0, 50, 10]#[-3, 50], [0, 50, 10]
            elif feature == "lambda_rect_outline_2d":
                xlabel, wl, xlim, ticks = r"$\lambda_{\mathrm{RectOutline}}$", 1, [-4, 50], [0, 50, 10]#[-3, 12], [0, 12, 4]
            elif feature == "lambda_points_rect_outline_2d":
                xlabel, wl, xlim, ticks = r"$\lambda_{\mathrm{Points/RectOut}}$", 2.5, [-4, 50], [0, 50, 10]#[-6, 60], [0, 60, 20]
            else:
                xlabel, wl = feature, 1
                xlim = [float(result[feature].min()), float(result[feature].max())]
                ticks = [int(xlim[0]), int(xlim[1]), max(1, int((xlim[1]-xlim[0]) // 3))]

            wl = (xlim[1] - xlim[0])  / 25

            axi.plot(xlim, [0, 0], 'k-', lw=2, alpha=0.75)
            axi.set_yscale('symlog', linthresh=100.0, linscale=0.5)
            axi.set_ylim([-110, 1e7])

            for idx, row in result.iterrows():
                xval = row[feature]
                yval = row["FD"]
                ci1, ci2 = row[['-CI', '+CI']]
                
                key = tuple(int(row.get(feature, 0)) for feature in relevant_columns)
                col = color_map[key]

                if (
                    row.get('lambda_lines_2d', 0) * row.get('lambda_points_line_2d', 0)
                    + row.get('lambda_clutter', 0)
                    + row.get('lambda_circle', 0) * row.get('lambda_points_circle', 0)
                    + row.get('lambda_rectangle_2d', 0) * row.get('lambda_points_rectangle_2d', 0)
                    + row.get('lambda_rect_outline_2d', 0) * row.get('lambda_points_rect_outline_2d', 0)
                    ==
                    gen_params.get('lambda_lines_2d', 0) * gen_params.get('lambda_points_line_2d', 0)
                    + gen_params.get('lambda_clutter', 0)
                    + gen_params.get('lambda_circle', 0) * gen_params.get('lambda_points_circle', 0)
                    + gen_params.get('lambda_rectangle_2d', 0) * gen_params.get('lambda_points_rectangle_2d', 0)
                    + gen_params.get('lambda_rect_outline_2d', 0) * gen_params.get('lambda_points_rect_outline_2d', 0)
                ):
                    marker = 's'
                else:
                    marker = 'o'


                if xval > xlim[0] and xval < xlim[1]:
                    axi.scatter(xval, yval, marker=marker, color=col, s=150)
                    axi.plot([xval, xval], [ci1, ci2], '-', c=col, lw=2)
                    axi.plot([xval - wl, xval + wl], [ci1, ci1], '-', c=col, lw=2)
                    axi.plot([xval - wl, xval + wl], [ci2, ci2], '-', c=col, lw=2)

            xticks = list(range(ticks[0], ticks[1] + 1, ticks[2]))
        
            xticklabels = list(range(ticks[0], ticks[1] + 1, ticks[2]))
            if gen_params[feature] in xticks:
                xticklabels[xticks.index(gen_params[feature])] = r'$\lambda_\mathrm{Ref}$'
            else:
                xticks.append(gen_params[feature])
                xticklabels.append(r'$\lambda_\mathrm{Ref}$')

            axi.set_xticks(xticks, xticklabels)
            axi.set_xlabel(xlabel)
            axi.set_xlim(xlim)
            axi.grid(True, alpha=0.3)


    fig.savefig(plot_path, dpi=300)