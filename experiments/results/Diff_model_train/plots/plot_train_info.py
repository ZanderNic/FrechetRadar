# std lib imports
import argparse
import os
from typing import Optional, Sequence, Tuple

# 3-party imports
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
import numpy as np
import statsmodels.api as sm


"""
    This script aggregates and visualizes training results from one or multiple experiments.

    It supports plotting:
    - FRD∞ (with asymptotic extrapolation and confidence intervals)
    - Train/Test loss curves (EMA-smoothed)
    - Optional feature/valid-dimension loss components
    - Negative log-likelihood during training

    Multiple experiment IDs can be passed and will be overlaid in the same plots.
    Feature dimensions can be filtered via a command-line argument; if none are given,
    all available feature dimensions are used.

    Important:
    - Each plot type (FRD/Loss/LogLike) is truncated to the largest batch index that is available for *all* included experiments for that specific plot type.
    - Colors are consistent across all subplots.
    - If multiple experiments: color = experiment, linestyle = experiment (stable)
    - If single experiment but multiple feature_dims: color = feature_dim (stable), linestyle = feature_dim
    - If exactly one feature_dim is selected, it is shown in the FRD title (not repeated in the legend).
    - FRD error bars can be thinned (every N points) to improve readability.

    Example usage:
        1) U-Net one model plot
            python3 ./experiments/results/Diff_model_train/plots/plot_train_info.py -e u_net/exp_medium_model_x0 -i frd log_like loss --err_every 0
        
        2) Multiple U-Net models
            python3 ./experiments/results/Diff_model_train/plots/plot_train_info.py -e u_net/exp_medium_model_x0 u_net/exp_small_model_x0 u_net/exp_big_model_x0 -i frd log_like -fd 4096 --err_every 0

        3) Multiple U-Net models with name mapping 
            python3 ./experiments/results/Diff_model_train/plots/plot_train_info.py -e u_net/exp_small_small_model_x0 u_net/exp_small_model_x0 u_net/exp_medium_model_x0 u_net/exp_big_model_x0  -i frd log_like loss -fd 4096 -l M1 M2 M3 M4 --err_every 0

        4) Dit model (just change config)
            python3 ./experiments/results/Diff_model_train/plots/plot_train_info.py -e dit/test_dit_x0 -i frd log_like loss --err_every 0            
"""


####### Settings #################################

alpha_train = 0.9
alpha_test = 0.4

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

linestyles = ["-"] #["-", "--", "-.", ":"]

##################################################


def ema(values: pd.Series, alpha: float = 0.1):
    values = values.reset_index(drop=True)
    if values.empty:
        return []
    s = []
    last = values.iloc[0]
    s.append(last)
    for v in values.iloc[1:]:
        last = (1 - alpha) * last + alpha * v
        s.append(last)
    return s


def build_exp_color_map(exp_ids):
    cols = colormaps["tab10"].colors
    return {eid: cols[i % len(cols)] for i, eid in enumerate(exp_ids)}


def build_fd_color_map(feature_dims):
    cols = colormaps["tab10"].colors
    fds = list(feature_dims)
    return {fd: cols[i % len(cols)] for i, fd in enumerate(fds)}


def _as_int_list(xs: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if xs is None:
        return None
    return tuple(int(x) for x in xs)


def _build_exp_label_map(exp_ids: Sequence[str], labels: Optional[Sequence[str]]) -> dict:
    if labels is None:
        return {eid: eid for eid in exp_ids}
    if len(labels) != len(exp_ids):
        raise SystemExit(
            f"--labels must have the same length as --exp_ids "
            f"({len(labels)} vs {len(exp_ids)})"
        )
    return {eid: lab for eid, lab in zip(exp_ids, labels)}

def _common_max_batch(runs: Sequence[Tuple[str, pd.DataFrame]], batch_col: str) -> float:
    max_vals = []
    for _, df in runs:
        if df is None or df.empty or batch_col not in df.columns:
            continue
        s = pd.to_numeric(df[batch_col], errors="coerce").dropna()
        if not s.empty:
            max_vals.append(float(s.max()))
    return float(min(max_vals)) if max_vals else float("inf")


def _truncate_runs_to_common_max(
    runs: Sequence[Tuple[str, pd.DataFrame]],
    batch_col: str
) -> Sequence[Tuple[str, pd.DataFrame]]:
    if not runs:
        return runs
    common_max = _common_max_batch(runs, batch_col)
    out = []
    for exp_id, df in runs:
        if df is None or df.empty or batch_col not in df.columns:
            out.append((exp_id, df))
            continue
        b = pd.to_numeric(df[batch_col], errors="coerce")
        out.append((exp_id, df[b <= common_max].copy()))
    return out


def plot_frd(
    ax,
    frd_runs: Sequence[Tuple[str, pd.DataFrame]],
    scaled_plot: bool,
    feature_dims_filter: Optional[Tuple[int, ...]] = None,
    exp_color: Optional[dict] = None,
    err_every: int = 0,
    exp_label: Optional[dict] = None,
):
     
    all_fds = sorted({int(fd) for _, df in frd_runs for fd in df["feature_dim"].dropna().unique()})
    if feature_dims_filter is not None:
        all_fds = [fd for fd in all_fds if fd in set(feature_dims_filter)]
    if not all_fds:
        ax.set_title(r"FRD$\infty$ in Abhängigkeit vom Trainingsfortschritt")
        ax.text(0.5, 0.5, "No FRD data after feature_dim filtering.", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
   
    single_fd = (feature_dims_filter is not None and len(feature_dims_filter) == 1)
    fd_in_title = f" (feature_dim={feature_dims_filter[0]})" if single_fd else ""
    ax.set_title(r"FRD$\infty$ in Abhängigkeit vom Trainingsfortschritt" + fd_in_title)

    multi_exp = len(frd_runs) > 1
    color_by_fd = (not multi_exp) and (len(all_fds) > 1)
    fd_color = build_fd_color_map(all_fds) if color_by_fd else None

    for exp_idx, (exp_id, df_frd) in enumerate(frd_runs):
        disp = exp_label.get(exp_id, exp_id) if exp_label is not None else exp_id
        if df_frd is None or df_frd.empty:
            continue

        df_frd = df_frd.copy()
        if feature_dims_filter is not None:
            df_frd = df_frd[df_frd["feature_dim"].isin(feature_dims_filter)]
        if df_frd.empty:
            continue

        base_c = exp_color[exp_id] if (exp_color is not None and exp_id in exp_color) else colormaps["tab10"].colors[exp_idx % 10]
        base_ls = linestyles[exp_idx % len(linestyles)]

        for fd_idx, feature_dim in enumerate(all_fds):
            df_frd_fd = df_frd[df_frd["feature_dim"] == feature_dim]
            if df_frd_fd.empty:
                continue

            if color_by_fd:
                c = fd_color[feature_dim]
                ls = linestyles[fd_idx % len(linestyles)]
            else:
                c = base_c
                ls = base_ls

            res = []
            for trained_batches, dfb in df_frd_fd.groupby("trained_batches"):
                ddf = (
                    dfb.groupby("dataset_size", as_index=True)["frechet_distance"]
                    .agg(mean="mean")
                    .sort_index()
                )
                x = ddf.index.to_numpy(dtype=float)
                y = ddf["mean"].to_numpy(dtype=float) / (feature_dim if scaled_plot else 1)

                if y.size < 2:
                    res.append({"trained_batches": float(trained_batches), "FRD": float(y[0]), "-CI": np.nan, "+CI": np.nan})
                    continue

                weights = np.sqrt(x)
                X = pd.DataFrame({"inv_n": 1.0 / x})
                X_with_intercept = sm.add_constant(X)
                model = sm.WLS(y, X_with_intercept, weights=weights).fit()

                ci = model.conf_int().loc["const"].to_numpy(dtype=float)
                res.append({
                    "trained_batches": float(trained_batches),
                    "FRD": float(model.params["const"]),
                    "-CI": float(ci[0]),
                    "+CI": float(ci[1]),
                })

            result_frd = pd.DataFrame(res).sort_values("trained_batches")
            print(f"For Experiment {exp_idx} and feature dim = {feature_dim}")
            print( result_frd)
            if result_frd.empty:
                continue

            if single_fd:
                label = exp_id if multi_exp else exp_id  
            else:
                label = (f"feature_dim={feature_dim}" if (not multi_exp) else f"{exp_id} | feature_dim={feature_dim}")

            if single_fd:
                label = disp
            else:
                label = (f"feature_dim={feature_dim}" if (not multi_exp)
                        else f"{disp} | feature_dim={feature_dim}")

            ax.plot(
                result_frd["trained_batches"],
                result_frd["FRD"],
                label=label,
                color=c,
                linestyle=ls,
                linewidth=1.6,
            )

            if err_every is not None and err_every != 0:
                mask_ci = result_frd["-CI"].notna() & result_frd["+CI"].notna()
                if mask_ci.any():
                    tmp = result_frd.loc[mask_ci].copy()
                    if err_every > 1:
                        tmp = tmp.iloc[::err_every]

                    xvals = tmp["trained_batches"].to_numpy()
                    yvals = tmp["FRD"].to_numpy()
                    ci1 = tmp["-CI"].to_numpy()
                    ci2 = tmp["+CI"].to_numpy()
                    yerr = np.vstack([yvals - ci1, ci2 - yvals])

                    ax.errorbar(
                        xvals, yvals, yerr=yerr,
                        fmt="s",
                        markersize=5,
                        color=c,
                        elinewidth=1.0,
                        capsize=4,
                        capthick=1.0,
                        linestyle="none",
                        alpha=0.9,
                    )

    ax.set_xlabel("Batch")
    if scaled_plot:
        ax.set_ylabel(r"$\frac{1}{\text{feature\_dim}} \mathrm{FRD}_{\infty}$")
        ax.set_yscale("symlog", linthresh=100.0, linscale=0.5)
    else:
        ax.set_ylabel(r"$\mathrm{FRD}_{\infty}$")
        ax.set_yscale("symlog", linthresh=100.0, linscale=0.5)

    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax.legend(list(uniq.values()), list(uniq.keys()), ncols=1)



def plot_loss(ax, loss_runs: Sequence[Tuple[str, pd.DataFrame]], exp_color: Optional[dict] = None, exp_label: Optional[dict] = None,):
    for exp_idx, (exp_id, df_loss) in enumerate(loss_runs):
        if df_loss is None or df_loss.empty or "Batch" not in df_loss.columns:
            continue

        disp = exp_label.get(exp_id, exp_id) if exp_label is not None else exp_id
        train_mask = df_loss["Train_Loss"].notna() if "Train_Loss" in df_loss.columns else pd.Series(False, index=df_loss.index)
        test_mask  = df_loss["Test_Loss"].notna() if "Test_Loss" in df_loss.columns else pd.Series(False, index=df_loss.index)

        c = exp_color[exp_id] if exp_color is not None else colormaps["tab10"].colors[exp_idx % 10]
        ls = linestyles[exp_idx % len(linestyles)]

        if train_mask.any():
            train_loss_ema = ema(df_loss.loc[train_mask, "Train_Loss"], alpha=alpha_train)
            ax.plot(
                df_loss.loc[train_mask, "Batch"],
                train_loss_ema,
                label=f"{disp} | Train (EMA {alpha_train})",
                linestyle=ls,
                color=c,
                alpha=0.9,
                linewidth=1.6,
            )

        if test_mask.any():
            test_loss_ema = ema(df_loss.loc[test_mask, "Test_Loss"], alpha=alpha_test)
            ax.plot(
                df_loss.loc[test_mask, "Batch"],
                test_loss_ema,
                label=f"{disp} | Test (EMA {alpha_test})",
                linestyle=ls,
                color="orange",
                alpha=0.9,
                linewidth=1.6,
            )

    ax.set_title("Train vs Test Loss")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend(ncols=1)


def plot_more_loss_info(ax, loss_runs: Sequence[Tuple[str, pd.DataFrame]], exp_color: Optional[dict] = None, exp_label: Optional[dict] = None,):

    for exp_idx, (exp_id, df_loss) in enumerate(loss_runs):
        if df_loss is None or df_loss.empty or "Batch" not in df_loss.columns:
            continue

        c = exp_color[exp_id] if exp_color is not None else colormaps["tab10"].colors[exp_idx % 10]
        ls = linestyles[exp_idx % len(linestyles)]

        # Train
        if "Train_Feature_Loss" in df_loss.columns:
            m = df_loss["Train_Feature_Loss"].notna()
            if m.any():
                y = ema(df_loss.loc[m, "Train_Feature_Loss"], alpha=alpha_train)
                ax.plot(df_loss.loc[m, "Batch"], y, label=f"{exp_id} | Train Feature", linestyle=ls, color=c, alpha=0.9, linewidth=1.6)

        if "Train_Valid_Dim_Loss" in df_loss.columns:
            m = df_loss["Train_Valid_Dim_Loss"].notna()
            if m.any():
                y = ema(df_loss.loc[m, "Train_Valid_Dim_Loss"], alpha=alpha_train)
                ax.plot(df_loss.loc[m, "Batch"], y, label=f"{exp_id} | Train Valid-Dim", linestyle=ls, color=c, alpha=0.75, linewidth=1.6)

        # Test
        if "Test_Feature_Loss" in df_loss.columns:
            m = df_loss["Test_Feature_Loss"].notna()
            if m.any():
                y = ema(df_loss.loc[m, "Test_Feature_Loss"], alpha=alpha_test)
                ax.plot(df_loss.loc[m, "Batch"], y, label=f"{exp_id} | Test Feature", linestyle=ls, color=c, alpha=0.55, linewidth=1.6)

        if "Test_Valid_Dim_Loss" in df_loss.columns:
            m = df_loss["Test_Valid_Dim_Loss"].notna()
            if m.any():
                y = ema(df_loss.loc[m, "Test_Valid_Dim_Loss"], alpha=alpha_test)
                ax.plot(df_loss.loc[m, "Batch"], y, label=f"{exp_id} | Test Valid-Dim", linestyle=ls, color=c, alpha=0.45, linewidth=1.6)

    ax.set_title("Feature/Valid-Dim Loss (Train/Test)")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend(ncols=1)


def plot_log_likelihood(ax, ll_runs: Sequence[Tuple[str, pd.DataFrame]], exp_color: Optional[dict] = None, exp_label: Optional[dict] = None):

    for exp_idx, (exp_id, df_ll) in enumerate(ll_runs):
        if df_ll is None or df_ll.empty:
            continue
        if not ("batches" in df_ll.columns and "neg_log_likelihood" in df_ll.columns):
            continue

        c = exp_color[exp_id] if exp_color is not None else colormaps["tab10"].colors[exp_idx % 10]
        ls = linestyles[exp_idx % len(linestyles)]
        disp = exp_label.get(exp_id, exp_id) if exp_label is not None else exp_id

        ax.plot(
            df_ll["batches"],
            df_ll["neg_log_likelihood"],
            label=f"{disp}",
            linestyle=ls,
            color=c,
            linewidth=1.6,
        )

    ax.set_title("Negative Log-Likelihood vs Training Progress")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.grid(True)
    ax.legend(ncols=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FRD/Loss/Log-Likelihood for one or more experiments.")
    parser.add_argument("--exp_ids", "-e", required=True, nargs="+", type=str)
    parser.add_argument("--feature_dims", "-fd", required=False, nargs="+", type=int, default=None)
    parser.add_argument("--infos", "-i", required=False, nargs="+", default=["log_like", "frd", "loss"])
    parser.add_argument("--scaled_plot", required=False, action="store_true")
    parser.add_argument("--err_every", required=False, type=int, default=4, help="Plot FRD errorbars every N points. 0 disables errorbars. (default: 4)")
    parser.add_argument("--labels", "-l", required=False, nargs="+", type=str, default=None, help="Optional display labels for exp_ids (same count).")
    args = parser.parse_args()

    feature_dims_filter = _as_int_list(args.feature_dims)
    exp_label = _build_exp_label_map(args.exp_ids, args.labels)
    exp_color = build_exp_color_map(args.exp_ids)

    frd_runs = []
    loss_runs = []
    ll_runs = []

    base0 = "./experiments/results/Diff_model_train/" + str(args.exp_ids[0])
    path_plots = os.path.join(base0, "plots")
    os.makedirs(path_plots, exist_ok=True)

    for exp_id in args.exp_ids:
        base_path = "./experiments/results/Diff_model_train/" + str(exp_id)
        path_frd  = os.path.join(base_path, "result_resampling.csv")
        path_ll   = os.path.join(base_path, "result_train_loglike.csv")
        path_loss = os.path.join(base_path, "result_train_info.csv")

        if os.path.exists(path_frd) and "frd" in args.infos:
            frd_runs.append((exp_id, pd.read_csv(path_frd)))

        if os.path.exists(path_loss) and ("loss" in args.infos or "feature_loss" in args.infos):
            loss_runs.append((exp_id, pd.read_csv(path_loss)))

        if os.path.exists(path_ll) and "log_like" in args.infos:
            ll_runs.append((exp_id, pd.read_csv(path_ll)))

    # Truncate each plot-type to common maximum batch across all included experiments
    if frd_runs:
        frd_runs = list(_truncate_runs_to_common_max(frd_runs, "trained_batches"))
    if loss_runs:
        loss_runs = list(_truncate_runs_to_common_max(loss_runs, "Batch"))
    if ll_runs:
        ll_runs = list(_truncate_runs_to_common_max(ll_runs, "batches"))

    has_frd = len(frd_runs) > 0 and "frd" in args.infos
    has_ll = len(ll_runs) > 0 and "log_like" in args.infos
    has_loss = len(loss_runs) > 0 and "loss" in args.infos

    has_more_loss_info = False
    if len(loss_runs) > 0 and "feature_loss" in args.infos:
        for _, df_loss in loss_runs:
            if df_loss is None:
                continue
            if (("Test_Feature_Loss" in df_loss.columns and "Test_Valid_Dim_Loss" in df_loss.columns) or
                ("Train_Feature_Loss" in df_loss.columns and "Train_Valid_Dim_Loss" in df_loss.columns)):
                has_more_loss_info = True
                break

    num_axes = sum([has_frd, has_ll, has_loss, has_more_loss_info])
    if num_axes == 0:
        raise SystemExit("No data found for requested infos/exp_ids.")

    # Global common max for consistent xlim across subplots
    cuts = []
    if frd_runs:
        cuts.append(_common_max_batch(frd_runs, "trained_batches"))
    if loss_runs:
        cuts.append(_common_max_batch(loss_runs, "Batch"))
    if ll_runs:
        cuts.append(_common_max_batch(ll_runs, "batches"))
    global_common_max = float(min(cuts)) if cuts else float("inf")

    fig, axes = plt.subplots(num_axes, 1, figsize=(10, 4 * num_axes), sharex=False)
    if num_axes == 1:
        axes = [axes]

    idx = 0
    if has_frd:
        plot_frd(
            axes[idx], frd_runs,
            scaled_plot=args.scaled_plot,
            feature_dims_filter=feature_dims_filter,
            exp_color=exp_color,
            err_every=args.err_every,
            exp_label=exp_label
        )
        idx += 1
    if has_ll:
        plot_log_likelihood(axes[idx], ll_runs, exp_color=exp_color, exp_label=exp_label)
        idx += 1
    if has_loss:
        plot_loss(axes[idx], loss_runs, exp_color=exp_color, exp_label=exp_label)
        idx += 1
    if has_more_loss_info:
        plot_more_loss_info(axes[idx], loss_runs, exp_color=exp_color, exp_label=exp_label)

    # Apply consistent xlim across all subplots
    if np.isfinite(global_common_max):
        for ax in axes:
            ax.set_xlim(0, global_common_max)

    plt.tight_layout()

    fd_tag = "allFD" if feature_dims_filter is None else ("FD_" + "_".join(map(str, feature_dims_filter)))
    exp_tag = "EXPS_" + "_".join(args.exp_ids)
    name = (f"train_info_{exp_tag}_{fd_tag}_" + ("dim_scale" if args.scaled_plot else "log_scale") + ".png" ).replace("/", ":")
    out_path = os.path.join(path_plots, name)
    plt.savefig(out_path, dpi=300)

    print("Plot saved at:", out_path)
