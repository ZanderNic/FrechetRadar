# std-lib imports
import argparse
import os
import ast

# third-party imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import colormaps

"""
    This script loads FRD evaluation data for different generator parameter settings
    and produces comparison plots across feature dimensions. It visualizes parameter
    sweeps, confidence intervals, and the training progress of the diffusion model,
    saving all results as a consolidated figure for analysis.

    Examples:
        python3 ./experiments/results/Diff_model_train/plots/plot_comparison_param_gen.py --exp_id u_net/exp_big_model_x0

        python3 ./experiments/results/Diff_model_train/plots/plot_comparison_param_gen.py --exp_id dit/test_dit_x0
"""


# ===============================================================
# ------------------------- CONFIG -------------------------------
# ===============================================================

YLIM = [-110, 1e6]
FEATURES_ORDER = ['lambda_lines_2d', 'lambda_points_line_2d', 'lambda_clutter']

# FID training plotting settings
FID_TRAIN_COLOR_LINE = "#FF4208"     # Aumovio Orange
FID_TRAIN_COLOR_POINT = "#8000FF"    # Aumovio Purple
FID_TRAIN_MARKER = "D"
FID_TRAIN_BATCH_STRIDE = 100_000     # Only draw every 100k


TITLE_SIZE   = 20
LABEL_SIZE   = 18
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

# ===============================================================
# -------------------- HELPER FUNCTIONS --------------------------
# ===============================================================

def fit_wls_const(x, y):
    if len(y) < 2:
        return float(y[0]), np.nan, np.nan, np.nan

    weights = np.sqrt(x)
    X = pd.DataFrame({'inv_n': 1.0 / x})
    X_with_intercept = sm.add_constant(X)
    model = sm.WLS(y, X_with_intercept, weights=weights).fit()

    const = model.params['const']
    ci_low, ci_high = model.conf_int().loc['const']
    pval = model.pvalues['const']

    return const, ci_low, ci_high, pval


def compute_frd_group(df_group):
    ddf = df_group.groupby("dataset_size")["frechet_distance"].mean().sort_index()
    x = ddf.index.to_numpy(float)
    y = ddf.to_numpy(float)

    return fit_wls_const(x, y)


def load_reference_generator(data, relevant_columns):
    grouped = list(data.groupby("reference_params"))
    if len(grouped) != 1:
        raise ValueError("More than one reference generator present.")

    ref_dict = ast.literal_eval(grouped[0][0])
    return {k: ref_dict[k] for k in relevant_columns}


def compute_results_by_feature_dim(data, fid_df, relevant_columns):
    feature_dims = sorted(set(data.feature_dim) & set(fid_df.feature_dim))
    results = []

    for feature_dim in feature_dims:
        df_sorted = data[data.feature_dim == feature_dim].sort_values("comparison_params")

        rows = []
        for comp_params, df_group in df_sorted.groupby("comparison_params"):
            comp_dict = ast.literal_eval(comp_params)
            fd, ci_low, ci_high, pval = compute_frd_group(df_group)

            rows.append({
                "feature_dim": feature_dim,
                "FD": fd,
                "-CI": ci_low,
                "+CI": ci_high,
                "FD_pval": pval,
                **{c: comp_dict[c] for c in relevant_columns}
            })

        results.append((feature_dim, pd.DataFrame(rows)))

    return results


def build_generator_color_map(result):
    cmap = colormaps["tab20"]
    palette = cmap.colors

    keys = list(zip(
        result["lambda_lines_2d"],
        result["lambda_points_line_2d"],
        result["lambda_clutter"]
    ))

    unique_keys = sorted(set(keys))
    return {key: palette[i % len(palette)] for i, key in enumerate(unique_keys)}


def plot_errorbar_point(ax, x, y, ci_low, ci_high, color,
                        width=1.0, marker="o", size=120, zorder=2):
    ax.scatter(x, y, color=color, s=size, marker=marker, zorder=zorder)
    ax.plot([x, x], [ci_low, ci_high], color=color, lw=2, zorder=zorder)
    ax.plot([x - width, x + width], [ci_low, ci_low], color=color, lw=2, zorder=zorder)
    ax.plot([x - width, x + width], [ci_high, ci_high], color=color, lw=2, zorder=zorder)


# ===============================================================
# ------------------------ MAIN PLOTTING -------------------------
# ===============================================================

def plot_results(results_by_dim, df_fid_dif, gen_params, path_plots):
    nrows = len(results_by_dim)
    ncols = len(FEATURES_ORDER) + 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18, 6 * nrows),
        tight_layout=True,
        sharey=True,
    )

    if nrows == 1:
        axes = np.array([axes])

    for row_idx, (feature_dim, result) in enumerate(results_by_dim):

        color_map = build_generator_color_map(result)

        # -------------------------------------------------------
        # Parameter sweep plotting
        # -------------------------------------------------------
        for col_idx, feature in enumerate(FEATURES_ORDER):
            ax = axes[row_idx, col_idx]

            if feature == "lambda_lines_2d":
                xlabel, width, xlim = r'$\lambda_\mathrm{Lines}$', 1, [2.5, 42.5]
            elif feature == "lambda_points_line_2d":
                xlabel, width, xlim = r'$\lambda_\mathrm{Points/Line}$', 1, [2.5, 45]
            else:
                xlabel, width, xlim = r'$\lambda_\mathrm{Clutter}$', 1.5, [15, 85]

            ax.set_yscale("symlog", linthresh=100.0, linscale=0.5)
            ax.set_ylim(YLIM)
            ax.plot(xlim, [0, 0], "k-", lw=2, alpha=0.7)

            # --- Plot points (unchanged colors; no recoloring of reference data)
            for _, row in result.iterrows():
                key = (row["lambda_lines_2d"], row["lambda_points_line_2d"], row["lambda_clutter"])
                base_color = color_map[key]

                is_ref = (
                    row["lambda_lines_2d"] == gen_params["lambda_lines_2d"] and
                    row["lambda_points_line_2d"] == gen_params["lambda_points_line_2d"] and
                    row["lambda_clutter"] == gen_params["lambda_clutter"]
                )
                marker = "s" if is_ref else "o"

                plot_errorbar_point(
                    ax=ax,
                    x=row[feature],
                    y=row["FD"],
                    ci_low=row["-CI"],
                    ci_high=row["+CI"],
                    color=base_color,
                    width=width,
                    marker=marker,
                    zorder=6,
                )

            ref_val = gen_params[feature]

            if feature == "lambda_lines_2d":
                ticks = [0, 40, 10]
            elif feature == "lambda_points_line_2d":
                ticks = [0, 50, 10]
            elif feature == "lambda_clutter":
                ticks = [0, 80, 20]
            else:
                start = int(np.floor(xlim[0]))
                stop  = int(np.ceil(xlim[1]))
                step  = max(1, int((stop - start) // 4))
                ticks = [start, stop, step]

            xticks = list(range(ticks[0], ticks[1] + 1, ticks[2]))
            xticklabels = list(range(ticks[0], ticks[1] + 1, ticks[2]))

            if ref_val in xticks:
                xticklabels[xticks.index(ref_val)] = r'$\lambda_\mathrm{Ref}$'
            else:
                xticks.append(ref_val)
                xticklabels.append(r'$\lambda_\mathrm{Ref}$')

            ax.set_xticks(xticks, xticklabels)


            if gen_params[feature] not in xticks:
                xticks.append(gen_params[feature])
                xticklabels.append(r'$\lambda_\mathrm{Ref}$')

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlim(xlim)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.3)

        # -------------------------------------------------------
        # FID TRAINING PLOT
        # -------------------------------------------------------
        ax_fid = axes[row_idx, -1]
        df_fd = df_fid_dif[df_fid_dif.feature_dim == feature_dim]

        fid_rows = []
        for trained_batches, df_group in df_fd.groupby("trained_batches"):
            fd, ci_low, ci_high, pval = compute_frd_group(df_group)
            fid_rows.append({
                "trained_batches": trained_batches,
                "FD": fd,
                "-CI": ci_low,
                "+CI": ci_high,
                "FD_pval": pval
            })

        fid_df = pd.DataFrame(fid_rows)
        fid_df_plot = fid_df[fid_df["trained_batches"] % FID_TRAIN_BATCH_STRIDE == 0]

        # --- LINE FIRST (under everything)
        ax_fid.plot(
            fid_df["trained_batches"],
            fid_df["FD"],
            color=FID_TRAIN_COLOR_LINE,
            lw=2.5,
            zorder=1,
        )

        # --- POINTS ON TOP
        if not fid_df_plot.empty:
            width = (fid_df["trained_batches"].max() - fid_df["trained_batches"].min()) / 30

            for _, row in fid_df_plot.iterrows():
                plot_errorbar_point(
                    ax=ax_fid,
                    x=row["trained_batches"],
                    y=row["FD"],
                    ci_low=row["-CI"],
                    ci_high=row["+CI"],
                    color=FID_TRAIN_COLOR_POINT,
                    width=width,
                    marker=FID_TRAIN_MARKER,
                    size=75,
                    zorder=3,
                )

        ax_fid.plot(
            [df_fd["trained_batches"].min(), df_fd["trained_batches"].max()],
            [0, 0],
            "k-",
            lw=2,
            alpha=0.75,
            zorder=1
        )

        ax_fid.set_yscale("symlog", linthresh=100.0, linscale=0.5)
        ax_fid.set_ylim(YLIM)
        ax_fid.set_xlabel("Trained Batches")
        ax_fid.set_title("Trained Diffusion Model FRD")
        ax_fid.grid(alpha=0.3)
        axes[row_idx, 0].set_ylabel(f"Feature dim: {feature_dim}\nFRD ∞")

    save_path = os.path.join(path_plots, "plot_comp_fid.png")
    fig.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")


# ===============================================================
# ------------------------------ MAIN ---------------------------
# ===============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", "-e", required=True)
    args = parser.parse_args()
    
    base_path = "./experiments/results/Diff_model_train/"
    
    path_plots = os.path.join(base_path, str(args.exp_id), "plots")
    os.makedirs(path_plots, exist_ok=True)

    path_fid = os.path.join(base_path, str(args.exp_id), "result_resampling.csv")
    comparison_data_path = "./experiments/results/frd_different_parameters/exp_div_params/results_cumalative_sampling.csv"

    data = pd.read_csv(comparison_data_path)
    fid_df = pd.read_csv(path_fid)

    relevant_columns = ["lambda_lines_2d", "lambda_points_line_2d", "lambda_clutter"]

    gen_params = load_reference_generator(data, relevant_columns)

    results_by_dim = compute_results_by_feature_dim(data, fid_df, relevant_columns)

    plot_results(results_by_dim, fid_df, gen_params, path_plots)

   