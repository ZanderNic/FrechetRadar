
# std lib imports
import time
import random
from typing import Dict, List, Tuple
from pathlib import Path

# third-party imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import gammaln
from tqdm import tqdm

# project imports
from RadarDataGen.Data_Generator.pseudo_radar_points import _pseudo_radar_points_with_info
from RadarDataGen.Metrics.log_likelihood import (
    log_likelihood_pseudo_radar_points,
    detect_lines_and_clutter,
    poisson_entropy,
    poisson_ll,
)

# ======================================================================
#                             PLOT SETTINGS
# ======================================================================

TITLE_SIZE   = 20
LABEL_SIZE   = 18
TICK_SIZE    = 16
LEGEND_SIZE  = 14

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

# ======================================================================
#                              HELPERS
# ======================================================================

def set_global_seeds(rng: np.random.Generator):
    py_seed = int(rng.integers(0, 2**32 - 1))
    np_seed = int(rng.integers(0, 2**32 - 1))
    th_seed = int(rng.integers(0, 2**31 - 1))

    random.seed(py_seed)
    np.random.seed(np_seed)
    torch.manual_seed(th_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(th_seed)


def poisson_logpmf(k: np.ndarray, lam: float) -> np.ndarray:
    k = np.asarray(k, dtype=np.int64)
    if lam <= 0.0:
        out = np.full_like(k, fill_value=-np.inf, dtype=float)
        out[k == 0] = 0.0
        return out
    return -lam + k * np.log(lam) - gammaln(k + 1.0)


def summarize_errors(true_arr: np.ndarray, pred_arr: np.ndarray) -> Dict[str, float]:
    true_arr = np.asarray(true_arr, dtype=float)
    pred_arr = np.asarray(pred_arr, dtype=float)
    diff = pred_arr - true_arr
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    if np.std(true_arr) > 0 and np.std(pred_arr) > 0:
        corr = float(np.corrcoef(true_arr, pred_arr)[0, 1])
    else:
        corr = float("nan")
    return {"mae": mae, "rmse": rmse, "bias": bias, "corr": corr}


# ======================================================================
#                               MAIN
# ======================================================================

if __name__ == "__main__":

    out_dir = Path("./tests/test_log_likelihood/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "reference_generators": [{
            "lambda_lines_2d": 10,
            "lambda_points_line_2d": 20,
            "lambda_clutter": 50
        }]
    }

    lam_lines = float(config["reference_generators"][0]["lambda_lines_2d"])
    lam_points_per_line = float(config["reference_generators"][0]["lambda_points_line_2d"])
    lam_clutter = float(config["reference_generators"][0]["lambda_clutter"])

    # Settings
    num_point_clouds = 1000
    num_trials = 10
    tau = 0.001
    min_inliers = 8
    num_workers = 8
    base_seed = 42

    # Accumulators for GT Poisson model
    all_gt_num_lines = []
    all_gt_per_line_counts = []
    all_gt_num_clutter = []

    # Accumulators for detection
    all_det_num_lines = []
    all_det_num_clutter = []
    all_det_per_line_counts = []

    # LL means across trials
    est_ll_means = []
    counts_ll_means = []

    # RNG
    ss = np.random.SeedSequence(base_seed)
    trial_seeds = ss.spawn(num_trials)

    t0_global = time.perf_counter()

    for trial_idx, seq in enumerate(trial_seeds):
        rng_trial = np.random.default_rng(seq)
        set_global_seeds(rng_trial)

        point_clouds = []
        gt_num_lines_trial = []
        gt_per_line_counts_trial = []
        gt_num_clutter_trial = []

        # ------------------ Generate data ------------------
        for _ in tqdm(range(num_point_clouds), desc=f"Trial {trial_idx+1}/{num_trials} sampling", leave=False):
            pts, info = _pseudo_radar_points_with_info(
                lambda_lines=lam_lines,
                lambda_points_line=lam_points_per_line,
                lambda_clutter=lam_clutter
            )
            point_clouds.append(pts)
            gt_num_lines_trial.append(int(info["num_lines"]))
            gt_per_line_counts_trial.append(np.asarray(info["num_points_per_line"], dtype=np.int64))
            gt_num_clutter_trial.append(int(info["num_clutter"]))

        # append to global GT lists
        all_gt_num_lines.extend(gt_num_lines_trial)
        for arr in gt_per_line_counts_trial:
            all_gt_per_line_counts.extend(arr.tolist())
        all_gt_num_clutter.extend(gt_num_clutter_trial)

        # ------------------ Estimated LL ------------------
        est_ll = log_likelihood_pseudo_radar_points(
            config=config,
            point_clouds=point_clouds,
            num_workers=num_workers,
            normalize=True,
            num_inliers=min_inliers,
            tau=tau
        )
        est_ll_means.append(float(est_ll))

        # ------------------ Detection ------------------
        det_num_lines_trial = []
        det_num_clutter_trial = []
        det_per_line_counts_trial = []

        for cloud in tqdm(point_clouds, desc=f"Trial {trial_idx+1}/{num_trials} detecting", leave=False):
            res = detect_lines_and_clutter(cloud, tau=tau, min_inliers=min_inliers)
            det_num_lines_trial.append(len(res["lines"]))
            det_num_clutter_trial.append(len(res["clutter"]))
            for ln in res["lines"]:
                det_per_line_counts_trial.append(len(ln))

        all_det_num_lines.extend(det_num_lines_trial)
        all_det_num_clutter.extend(det_num_clutter_trial)
        all_det_per_line_counts.extend(det_per_line_counts_trial)

    t1_global = time.perf_counter()

    # ======================================================================
    #                    CORRECT GLOBAL TRUE POISSON LL
    # ======================================================================

    gt_lines_arr = np.asarray(all_gt_num_lines, dtype=np.int64)
    gt_points_arr = np.asarray(all_gt_per_line_counts, dtype=np.int64)
    gt_clutter_arr = np.asarray(all_gt_num_clutter, dtype=np.int64)

    ll_real = (
        poisson_ll(gt_lines_arr, lam_lines)
        + poisson_ll(gt_points_arr, lam_points_per_line)
        + poisson_ll(gt_clutter_arr, lam_clutter)
    )

    counts_ll_means = [ll_real]   # single true value

    # Expected baseline
    expected_ll_baseline = -(
        poisson_entropy(lam_lines) +
        poisson_entropy(lam_points_per_line) +
        poisson_entropy(lam_clutter)
    )

    # ======================================================================
    #                       PRINT RESULTS
    # ======================================================================

    print("\n=== Runtime ===")
    print(f"Total time: {t1_global - t0_global:.2f}s\n")

    print("=== Log-Likelihood ===")
    print(f"Estimated LL (mean across trials): {np.mean(est_ll_means):.6f}")
    print(f"True Poisson LL (global):          {ll_real:.6f}")
    print(f"Baseline expected (theory):        {expected_ll_baseline:.6f}\n")

    # ======================================================================
    #                           SAVE PLOTS + REPORTS
    # ======================================================================

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "summary.txt"
    with open(report_path, "w") as f:
        f.write("=== Log-Likelihood ===\n")
        f.write(f"Estimated LL mean across trials: {np.mean(est_ll_means):.6f}\n")
        f.write(f"True Poisson LL (global):        {ll_real:.6f}\n")
        f.write(f"Expected baseline (theory):      {expected_ll_baseline:.6f}\n")

    print("[OK] Completed and wrote summary.")
