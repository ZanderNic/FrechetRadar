
# std lib imports
import argparse
import json
import os
import re
import math

# 3rd party imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

# project imports
from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer
from RadarDataGen.Metrics.log_likelihood import detect_lines_and_clutter


"""
    Analyze detected line and clutter statistics from generated radar samples.

    The script loads diffusion model samples, converts them to radar point clouds,
    detects line structures and clutter, and compares the empirical distributions
    to the ground-truth Poisson models used during data generation.

    To generate samples see: experiments/results/Diff_model_train/plots/generate_samples_from_checkpoints.py

    For each checkpoint, distributions of:
        - lines per point cloud,
        - points per line,
        - clutter points per cloud
    are visualized and saved as plots.

    Examples:
        1) U-Net:
            python3 ./experiments/results/Diff_model_train/plots/plot_lines_clutter_points_from_samples.py --exp_id u_net/exp_small_model_x0

        2) dit:
            python3 ./experiments/results/Diff_model_train/plots/plot_lines_clutter_points_from_samples.py --exp_id dit/test_dit_x0
        
        3) dit with detection params:
            python3 ./experiments/results/Diff_model_train/plots/plot_lines_clutter_points_from_samples.py --exp_id dit/test_dit_x0 --tau 0.1 --min_inliers 8 --valid_threshold 0.6

"""



def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def poisson_overlay(ax, counts: np.ndarray, lam: float, title: str, xlabel: str):
    counts = np.asarray(counts, dtype=np.int64)
    n = counts.size
    k_max_data = int(counts.max()) if n > 0 else 0
    k_max = max(int(lam * 3 + 10), k_max_data, 1)
    x = np.arange(0, k_max + 1)

    if lam > 0.0:
        log_pmf = -lam + x * np.log(lam) - gammaln(x + 1)
        pmf = np.exp(log_pmf)
    else:
        pmf = np.zeros_like(x, dtype=float)
        pmf[x == 0] = 1.0

    bins = np.arange(0, k_max + 2) - 0.5
    ax.hist(counts, bins=bins, color="#8905be", alpha=0.5, density=True, label="Empirical")
    ax.plot(x, pmf, "o-", color="#FF4208", lw=2, label=f"Poisson(λ={lam:.2f})")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, k_max + 0.5)
    ax.legend()


def process_folder(folder, radar_disc, valid_threshold, tau, min_inliers):
    results = []
    for file in os.listdir(folder):
        if file.endswith(".pt"):
            samples = torch.load(os.path.join(folder, file))
            for i in range(samples.shape[0]):
                points = radar_disc.grid_to_points(samples[i].cpu(), valid_threshold=valid_threshold)
                res = detect_lines_and_clutter(points, tau=tau, min_inliers=min_inliers)
                results.append(res)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze detected lines and clutter distributions per checkpoint.")
    parser.add_argument("--exp_id", "-e", required=True, help="Experiment folder name")
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--min_inliers", type=int, default=8)
    parser.add_argument("--valid_threshold", type=float, default=0.6)
    args = parser.parse_args()

    base_path = f"./experiments/results/Diff_model_train/{args.exp_id}"
    path_config = os.path.join(base_path, "setting.json")
    path_samples = os.path.join(base_path, "samples_model")

    config = load_config(path_config)
    radar_disc = RadarDiscretizer(**config["discretizer_params"])

    lam_lines = float(config["data_generator"][0]["lambda_lines_2d"])
    lam_points_per_line = float(config["data_generator"][0]["lambda_points_line_2d"])
    lam_clutter = float(config["data_generator"][0]["lambda_clutter"])

    pattern = re.compile(r"^model_(\d+)_batches$")
    folders = []
    for name in os.listdir(path_samples):
        match = pattern.match(name)
        if match:
            number = int(match.group(1))
            folders.append((number, os.path.join(path_samples, name)))

    folders.sort(key=lambda x: x[0])

    # Process each checkpoint separately
    for num, folder in folders:
        print(f"[INFO] Processing checkpoint: {folder}")
        results = process_folder(folder, radar_disc, args.valid_threshold, args.tau, args.min_inliers)

        # Extract counts
        number_lines = np.array([len(r["lines"]) for r in results], dtype=np.int64)
        per_line_counts = np.array([len(ln) for r in results for ln in r["lines"]], dtype=np.int64)
        number_clutter = np.array([len(r["clutter"]) for r in results], dtype=np.int64)

        # Plot distributions for this checkpoint
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)
        poisson_overlay(axes[0], number_lines, lam_lines, "Lines per cloud", "Number of lines")
        poisson_overlay(axes[1], per_line_counts, lam_points_per_line, "Points per line", "Number of points")
        poisson_overlay(axes[2], number_clutter, lam_clutter, "Clutter points per cloud", "Number of clutter points")

        out_path = os.path.join(folder, "detected_distribution.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        print(f"[INFO] Saved plot at {out_path}")
