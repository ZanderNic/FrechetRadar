# std lib imports
import argparse
from pathlib import Path

# 3rd party imports
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np


alpha = 1.96

USE_GERMAN = True  

BASE_FS   = 17       
LABEL_FS  = BASE_FS   
TICK_FS   = BASE_FS-1 
TITLE_FS  = BASE_FS+2
LEGEND_FS = BASE_FS-1 


plt.rcParams.update({
    "font.size": BASE_FS,
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
})

TEXTS = {
    True: {   
        "xlabel": r'Stichprobengröße $N$',
        "ylabel": r'Fréchet-Radar-Distanz',
        # "title": rf"FRD umskaliert mit $\frac{{1}}{{dim^\alpha}}$ und $\alpha$ = {alpha}", 1/2 * dim**2 + dim)
        #"title": rf"FRD umskaliert mit $\frac{{1}}{{2}} * dim^2 + 2/3 * dim$",
        "title": rf"FRD umskaliert mit $\frac{{dim * (dim + 3)}}{{2}}$",
        "legend_prefix": 'Feature-Dimension',
        "print_const": "konst",
        "print_slope": "steigung",
    },
    False: {  
        "xlabel": r'Sample size $N$',
        "ylabel": r'Fréchet-Inception-Distance',
        "title": 'Fréchet Inception Distance vs Dataset Size',
        "legend_prefix": 'feature_dim',
        "print_const": "const",
        "print_slope": "slope",
    }
}
T = TEXTS[USE_GERMAN]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribution plots for selected dimensions.")
    parser.add_argument("--exp_id", "-e", required=True, help="Experiment folder name")
    args = parser.parse_args()

    base_path = Path("./experiments/results/fd_same_parameters") / args.exp_id
    path_plots = base_path / "figs"
    path_plots.mkdir(parents=True, exist_ok=True)

    path_fid = base_path / "results_cumalative_sampling.csv"


    df = pd.read_csv(path_fid)
    df = df.drop(columns=["reference_params", "comparison_params"])
    df["count"] = 1

    df = df.groupby(["feature_dim", "dataset_size", "data_dim"], as_index=False).agg(
        frechet_distance=("frechet_distance", "mean"),
        count=("count", "sum"),
        dataset_size=("dataset_size", "mean"),
        dataset_sum=("dataset_size", "sum")
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    unique_features = sorted(set(df["feature_dim"]))
    res = []
    for dim in unique_features:
        x = [ds for fd, ds in zip(df["feature_dim"], df["dataset_size"]) if fd == dim]      
        y = [fdist * (1 / (dim * ((dim + 3) / 2))) for fd, fdist in zip(df["feature_dim"], df["frechet_distance"]) if fd == dim] 
        #y = [fdist * (1 /dim**alpha) for fd, fdist in zip(df["feature_dim"], df["frechet_distance"]) if fd == dim] 
        weights = np.sqrt(np.array(x))
        
        ax.scatter(x, y, marker='o', label=f'{T["legend_prefix"]}={dim}')

        
        X_reg = 1.0 / (np.array(x))
        X_with_intercept = sm.add_constant(X_reg)

        model = sm.WLS(y, X_with_intercept, weights=weights).fit()

        x_arr = np.array(x)
        sort_idx = np.argsort(x_arr)
        x_sorted = x_arr[sort_idx]
        X_pred = sm.add_constant(1 / x_sorted)
        y_pred = model.predict(X_pred)

        ax.plot(x_sorted, y_pred, '--')

        param_names_localized = [T["print_const"], T["print_slope"]]
        for pname, pval, pval_p, CI in zip(param_names_localized, model.params, model.pvalues, model.conf_int()):
            print(f'dim: {dim:04d} {pname}: {pval:>6.4e} CI: {CI[0]:+6.4e} bis {CI[1]:+6.4e} p={pval_p:.2f}')
            res.append([dim, pval])

    ax.set_xscale('log')
    ax.set_yscale('log')

    xticks = np.sort(df["dataset_size"].unique())
    ax.set_xticks(xticks, [f'{int(x_i):d}' for x_i in xticks], rotation=45)

    ax.set_xlabel(T["xlabel"], fontsize=LABEL_FS)
    ax.set_ylabel(T["ylabel"], fontsize=LABEL_FS)
    ax.set_title(T["title"], fontsize=TITLE_FS)

    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)

    leg = ax.legend(title=None, fontsize=LEGEND_FS, markerscale=1.0, frameon=True)
   
    ax.grid(True, linewidth=0.7, alpha=0.7)

    if leg.get_title() is not None:
        leg.get_title().set_fontsize(LEGEND_FS)

    fig.savefig( path_plots / "FRD_model_scaled_plot.png", dpi=300)
