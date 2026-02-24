
# std lib import
import argparse
import ast
from pathlib import Path

# 3 party import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import kendalltau, spearmanr
from matplotlib import colormaps

# projekt imports
from RadarDataGen.Metrics.log_likelihood import poisson_entropy

# Setting
FEATURES = ["lambda_lines_2d", "lambda_points_line_2d", "lambda_clutter"]

## Utils
def parse_params(s: str):
    d = ast.literal_eval(s)
    return pd.Series({c: d.get(c, np.nan) for c in FEATURES})


def build_color_map(all_keys):
    cmap = colormaps["tab20"]
    palette = cmap.colors
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(sorted(all_keys))}
    return color_map

#####

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ranking evaluation FRD vs NLL with consistent colors.")
    parser.add_argument("--exp_id", "-e", required=True)
    parser.add_argument("--feature_dim", "-d", type=int, default=256)
    args = parser.parse_args()

    base_path = Path("./experiments/results/fd_different_parameters") / args.exp_id
    out_dir = base_path / "ranking_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ll_df = pd.read_csv(base_path / "results_ll.csv")
    frd_df = pd.read_csv(base_path / "results_cumalative_sampling.csv")
    frd_df = frd_df[frd_df["feature_dim"] == args.feature_dim]


    # Reference parameters
    ref_gen = ast.literal_eval(ll_df["reference_params"].iloc[0])
    gen_params_ref = {k: ref_gen[k] for k in FEATURES}

    # LL aggregation
    ll_agg = (
        ll_df.groupby("comparison_params", as_index=False)
        .agg(mean_ll=("log_like", "mean"), std_ll=("log_like", "std"))
    )
    ll_agg[FEATURES] = ll_agg["comparison_params"].apply(parse_params)
    ll_agg["NLL"] = -ll_agg["mean_ll"]
  
    # FRD aggregation
    rows = []
    for comp, g in frd_df.groupby("comparison_params"):
        ddf = g.groupby("dataset_size")["frechet_distance"].mean().reset_index()
        n = ddf["dataset_size"]
        y = ddf["frechet_distance"]
        X = pd.DataFrame({"inv_n": 1 / n})

        model = sm.WLS(y, sm.add_constant(X), weights=np.sqrt(n)).fit()
        ci = model.conf_int().loc["const"].tolist()

        comp_dict = ast.literal_eval(comp)
        rows.append({
            "comparison_params": comp,
            "FD": model.params["const"],
            "FD_minus_CI": ci[0],
            "FD_plus_CI": ci[1],
            **comp_dict
        })

    frd_est = pd.DataFrame(rows)

    # MERGE
    merged = pd.merge(frd_est, ll_agg, on=["comparison_params"] + FEATURES, how="inner")

    if merged.empty:
        raise ValueError("Merged dataframe empty!")

    # SAME COLOR MAP AS OTHER PLOTS
    all_keys = set(zip(
        merged["lambda_lines_2d"],
        merged["lambda_points_line_2d"],
        merged["lambda_clutter"]
    ))
    color_map = build_color_map(all_keys)

    #merged = merged[merged["lambda_clutter"] != 0]
    merged = merged[merged["lambda_lines_2d"] != 0]

    # --- Ranking ---
    merged["rank_frd"] = merged["FD"].rank(ascending=True, method="dense")
    merged["rank_nll"] = merged["NLL"].rank(ascending=True, method="dense")
    merged["rank_diff"] = merged["rank_frd"] - merged["rank_nll"]

    # Rank correlations
    tau, p_tau = kendalltau(merged["rank_frd"], merged["rank_nll"])
    rho, p_rho = spearmanr(merged["rank_frd"], merged["rank_nll"])

    rank_diff_abs = (merged["rank_frd"] - merged["rank_nll"]).abs()
    rank_mae = rank_diff_abs.mean()
    rank_rmse = np.sqrt((rank_diff_abs ** 2).mean())

    def topk_metrics(ranks_a, ranks_b, k):
        k = int(min(k, len(ranks_a), len(ranks_b)))
        top_a_idx = set(ranks_a.nsmallest(k).index)
        top_b_idx = set(ranks_b.nsmallest(k).index)
        inter = len(top_a_idx & top_b_idx)
        precision = inter / k if k > 0 else np.nan
        recall = inter / k if k > 0 else np.nan
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1, inter, k

    topk_list = [3, 5, 10]
    topk_results = {}
    for k in topk_list:
        p, r, f1, inter, k_eff = topk_metrics(merged["rank_frd"], merged["rank_nll"], k)
        topk_results[k] = {
            "k_effective": k_eff,
            "precision": p,
            "recall": r,
            "f1": f1,
            "intersection": inter
        }

    def pairwise_stats(r1, r2):
        idx = r1.index.to_list()
        n = len(idx)
        concordant = 0
        discordant = 0
        ties = 0
        for i in range(n):
            for j in range(i+1, n):
                a = np.sign(r1.iloc[i] - r1.iloc[j])
                b = np.sign(r2.iloc[i] - r2.iloc[j])
                if a == 0 or b == 0:
                    ties += 1
                elif a == b:
                    concordant += 1
                else:
                    discordant += 1
        total_comp = concordant + discordant  # Ties ausgeschlossen
        pairwise_acc = concordant / total_comp if total_comp > 0 else np.nan
        inversion_rate = discordant / total_comp if total_comp > 0 else np.nan
        return pairwise_acc, inversion_rate, concordant, discordant, ties, total_comp

    pairwise_accuracy, inversion_rate, n_conc, n_disc, n_ties, n_total = pairwise_stats(
        merged["rank_frd"], merged["rank_nll"]
    )


    goodman_gamma = ((n_conc - n_disc) / (n_conc + n_disc)) if (n_conc + n_disc) > 0 else np.nan

    from scipy.stats import pearsonr
    pearson_r, pearson_p = pearsonr(merged["FD"], merged["NLL"])


    # (a) Summary in rank_summary.txt erweitern
    with open(out_dir / "rank_summary.txt", "a") as f:
        f.write("=== Ranking Evaluation Summary ===\n\n")
        f.write(f"Kendall:   {tau:.4f} (p={p_tau:.4g})\n")
        f.write(f"Spearman:  {rho:.4f} (p={p_rho:.4g})\n")
        f.write(f"Rank MAE:           {rank_mae:.4f}\n")
        f.write(f"Rank RMSE:          {rank_rmse:.4f}\n")
        f.write(f"Pairwise accuracy:  {pairwise_accuracy:.4f}  (concordant={n_conc}, discordant={n_disc}, ties={n_ties})\n")
        f.write(f"Inversion rate:     {inversion_rate:.4f}\n")
        f.write(f"Goodman-Kruskal:  {goodman_gamma:.4f}\n")
        f.write(f"Pearson r (FD,NLL): {pearson_r:.4f} (p={pearson_p:.4g})\n")
        for k, st in topk_results.items():
            f.write(f"Top-{k}  (eff={st['k_effective']}):  precision={st['precision']:.4f}, recall={st['recall']:.4f}, F1={st['f1']:.4f}, intersection={st['intersection']}\n")


    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------

    # Scatter FRD-rank vs NLL-rank
    plt.figure(figsize=(6, 6))
    for _, row in merged.iterrows():
        key = (row["lambda_lines_2d"], row["lambda_points_line_2d"], row["lambda_clutter"])
        if (row['lambda_lines_2d'] * row['lambda_points_line_2d'] + row['lambda_clutter'] == gen_params_ref['lambda_lines_2d'] * gen_params_ref['lambda_points_line_2d'] + gen_params_ref['lambda_clutter']):
            marker = 's'
        else:
            marker = 'o'
      
        plt.scatter(row["rank_frd"], row["rank_nll"], color=color_map[key], s=120, marker=marker)
        
    max_rank = int(max(merged["rank_frd"].max(), merged["rank_nll"].max()))
    plt.plot([1, max_rank], [1, max_rank], "k--", lw=1.5)

    plt.xlabel("FRD-Rang")
    plt.ylabel("NLL-Rang")
    plt.grid(True)
    plt.savefig(out_dir / "rank_scatter.png", dpi=200)
    plt.close()

    # Rank difference plot
    plt.figure(figsize=(10, 4))
    for idx, (_, row) in enumerate(merged.iterrows()):
        key = (row["lambda_lines_2d"], row["lambda_points_line_2d"], row["lambda_clutter"])
        plt.bar(idx, row["rank_diff"], color=color_map[key])

    plt.axhline(0, color="black", lw=1)
    plt.xlabel("Generator Index")
    plt.ylabel("Rank(FRD) − Rank(NLL)")
    plt.title("Ranking Disagreement")
    plt.grid(True)
    plt.savefig(out_dir / "rank_difference.png", dpi=200)
    plt.close()

    # Save ranking table
    merged.to_csv(out_dir / "ranking_results.csv", index=False)

    print(f"[OK] Ranking evaluation complete → {out_dir}")