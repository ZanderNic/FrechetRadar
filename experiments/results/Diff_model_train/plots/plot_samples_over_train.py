
# std lib imports
import argparse
import json
import os
import re
import math

# 3rd party imports
import torch
import matplotlib.pyplot as plt

# project imports
from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer



"""
    Visualize generated radar samples from diffusion models.

    The script loads sampled grids from different training checkpoints,
    converts them to radar point clouds, and visualizes both the grid
    representation and the resulting point clouds.

    Command-line arguments:
        --exp_id (str, required):
            Experiment directory inside Diff_model_train.

        --valid_threshold (float, optional):
            Threshold for converting grid values to valid radar points.

        --per_checkpoint (flag):
            If set, saves plots separately inside each checkpoint folder.
            Otherwise, a global overview plot is created.

        --num_samples_per_checkpoint (int, optional):
            Number of samples to visualize per checkpoint when
            --per_checkpoint is enabled.


    Examples:
        1) U-Net per checkpoint 
            python3 ./e

        2) DIT per Checkpoint:
            python3 ./experiments/results/Diff_model_train/plots/plot_samples_over_train.py --exp_id dit/test_dit_x0 --per_checkpoint

        2) DIT over train:
            python3 ./experiments/results/Diff_model_train/plots/plot_samples_over_train.py --exp_id dit/test_dit_x0
"""



def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_checkpoint_folders(path_samples):
    pat = re.compile(r"^model_(\d+)_batches$")
    folders = []
    for name in os.listdir(path_samples):
        m = pat.match(name)
        if m:
            num = int(m.group(1))
            folders.append((num, os.path.join(path_samples, name)))
    folders.sort(key=lambda x: x[0])
    return folders


def iter_pt_files(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".pt")]


def collect_first_n_samples(folder, n):
    """Return up to n samples (tensors) loaded from .pt files in this folder."""
    out = []
    for f in iter_pt_files(folder):
        samples = torch.load(f)
        for i in range(samples.shape[0]):
            out.append(samples[i])
            if len(out) >= n:
                return out
    return out


def draw_point_cloud(ax, points, radar_disc):
    if points.size != 0:
        ax.scatter(*points[:, :2].T, c=points[:, 2])
    else:
        ax.scatter(0, 0, alpha=0.0)
    ax.add_patch(
        plt.Rectangle(
            (radar_disc.x_min, radar_disc.y_min),
            radar_disc.x_max - radar_disc.x_min,
            radar_disc.y_max - radar_disc.y_min,
            facecolor="#8905be",
            alpha=0.15,
            zorder=0,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (radar_disc.x_min, radar_disc.y_min),
            radar_disc.x_max - radar_disc.x_min,
            radar_disc.y_max - radar_disc.y_min,
            facecolor="none",
            edgecolor="#FF4208",
            linewidth=2,
            zorder=3,
        )
    )
    ax.set_xlim(radar_disc.x_min, radar_disc.x_max)
    ax.set_ylim(radar_disc.y_min, radar_disc.y_max)
    ax.grid(True)


def draw_grid(ax, sample_tensor, radar_disc):
    img = radar_disc.grid_to_image(
        sample_tensor.to("cpu").numpy().transpose(1, 2, 0),
        swap_xy=True,
        invert_rows=True,
        invert_columns=False,
    )
    ax.matshow(img)
    ax.grid(True)


def make_grid(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def plot_global_overview(folders, radar_disc, valid_threshold, path_plots):
    ensure_dir(path_plots)
    count = len(folders)
    rows, cols = make_grid(count)

    fig_points, ax_points = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), tight_layout=True)
    fig_grids, ax_grids = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), tight_layout=True)

    ax_points = ax_points.flatten() if isinstance(ax_points, (list, tuple)) or hasattr(ax_points, "flatten") else [ax_points]
    ax_grids = ax_grids.flatten() if isinstance(ax_grids, (list, tuple)) or hasattr(ax_grids, "flatten") else [ax_grids]

    for idx, (num, folder) in enumerate(folders):
        pt_files = iter_pt_files(folder)
        if not pt_files:
            ax_points[idx].axis("off")
            ax_grids[idx].axis("off")
            continue

        # first sample for overview
        samples = torch.load(pt_files[0])
        sample = samples[0]

        points = radar_disc.grid_to_points(sample.cpu(), valid_threshold=valid_threshold)
        draw_point_cloud(ax_points[idx], points, radar_disc)
        ax_points[idx].set_title(f"Trained for {num} Batches")

        draw_grid(ax_grids[idx], sample, radar_disc)
        ax_grids[idx].set_title(f"Trained for {num} Batches")

    for j in range(count, len(ax_points)):
        ax_points[j].axis("off")
        ax_grids[j].axis("off")

    fig_points_path = os.path.join(path_plots, "sampled_points.png")
    fig_grids_path = os.path.join(path_plots, "sampled_grids.png")


    fig_points.savefig(fig_points_path, bbox_inches="tight", pad_inches=0.3)
    fig_grids.savefig(fig_grids_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig_points)
    plt.close(fig_grids)

    print(f"[INFO] Plots saved at {fig_points_path} and {fig_grids_path}")


def plot_per_checkpoint(folders, radar_disc, valid_threshold, num_samples_per_checkpoint):
    for num, folder in folders:
        samples_list = collect_first_n_samples(folder, num_samples_per_checkpoint)
        if not samples_list:
            continue

        n = len(samples_list)
        rows, cols = make_grid(n)

        # point clouds
        fig_p, axes_p = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), tight_layout=True)
        axes_p = axes_p.flatten() if hasattr(axes_p, "flatten") else [axes_p]

        for i, sample in enumerate(samples_list):
            points = radar_disc.grid_to_points(sample.cpu(), valid_threshold=valid_threshold)
            draw_point_cloud(axes_p[i], points, radar_disc)
            axes_p[i].set_title(f"#{i+1}")
        for j in range(n, len(axes_p)):
            axes_p[j].axis("off")

        out_p = os.path.join(folder, "samples_point_clouds.png")
        fig_p.savefig(out_p, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig_p)

        # grids
        fig_g, axes_g = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), tight_layout=True)
        axes_g = axes_g.flatten() if hasattr(axes_g, "flatten") else [axes_g]

        for i, sample in enumerate(samples_list):
            draw_grid(axes_g[i], sample, radar_disc)
            axes_g[i].set_title(f"#{i+1}")
        for j in range(n, len(axes_g)):
            axes_g[j].axis("off")

        out_g = os.path.join(folder, "samples_grids.png")
        fig_g.savefig(out_g, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig_g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sample points and grids per checkpoint or globally.")
    parser.add_argument("--exp_id", "-e", required=True, help="Experiment folder name")
    parser.add_argument("--valid_threshold", type=float, default=0.6)
    parser.add_argument("--per_checkpoint", action="store_true", help="Save figures inside each checkpoint folder")
    parser.add_argument("--num_samples_per_checkpoint", type=int, default=9, help="Samples per checkpoint figure if --per_checkpoint")
    args = parser.parse_args()

    base_path = "./experiments/results/Diff_model_train/"
    path_config = os.path.join(base_path, str(args.exp_id), "setting.json")
    path_samples = os.path.join(base_path, str(args.exp_id), "samples_model")
    path_plots = os.path.join(base_path, str(args.exp_id), "plots")

    ensure_dir(path_plots)

    config = load_config(path_config)
    radar_disc = RadarDiscretizer(**config["discretizer_params"])

    folders = find_checkpoint_folders(path_samples)

    if args.per_checkpoint:
        plot_per_checkpoint(
            folders=folders,
            radar_disc=radar_disc,
            valid_threshold=args.valid_threshold,
            num_samples_per_checkpoint=args.num_samples_per_checkpoint,
        )
    else:
        plot_global_overview(
            folders=folders,
            radar_disc=radar_disc,
            valid_threshold=args.valid_threshold,
            path_plots=path_plots,
        )
