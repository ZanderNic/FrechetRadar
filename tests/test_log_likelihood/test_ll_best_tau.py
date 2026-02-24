
# tau_sweep_loglik.py
# -*- coding: utf-8 -*-

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from RadarDataGen.Data_Generator.pseudo_radar_points import pseudo_radar_points
from RadarDataGen.Metrics.log_likelihood import log_likelihood_pseudo_radar_points

lambda_lines_2d: float = 10.0
lambda_points_line_2d: float = 20.0
lambda_clutter: float = 50.0

n_clouds: int = 1000
axis_limits: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
crop_to_limits: bool = False   
seed: Optional[int] = None     

min_inliers: int = 8
num_workers: int = 16
normalize: bool = True        

# τ-Sweep
tau_max: float = 1 # 5e-1          
tau_min: float = 1e-5         
n_steps: int = 20              
use_linspace: bool = False 

# Output
out_path: str = "./tests/test_log_likelihood/tau_vs_neg_loglik.png"
plot_title: str = "Tau-Sweep: Negative Log-Likelihood"

######################################################################################

def generate_point_clouds(
    n_clouds: int,
    config: dict,
    axis_limits: Optional[Tuple[float, float, float, float]] = None,
    crop_to_limits: bool = False,
    seed: Optional[int] = None,
) -> List[np.ndarray]:

    if seed is not None:
        np.random.seed(seed)

    pcs: List[np.ndarray] = []
    for _ in range(n_clouds):
        pts = pseudo_radar_points(
            lambda_lines_2d=config["reference_generators"][0]["lambda_lines_2d"],
            lambda_points_line_2d=config["reference_generators"][0]["lambda_points_line_2d"],
            lambda_clutter=config["reference_generators"][0]["lambda_clutter"],
        )
        if crop_to_limits and axis_limits is not None:
            xmin, xmax, ymin, ymax = axis_limits
            mask = (
                (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
                (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
            )
            pts = pts[mask]
        pcs.append(pts)
    return pcs


def compute_loglikelihood_over_taus(
    taus: np.ndarray,
    config: dict,
    point_clouds: List[np.ndarray],
    min_inliers: int,
    num_workers: int,
    normalize: bool = True,
) -> np.ndarray:
    loglikes = []
    for tau in taus:
        ll = log_likelihood_pseudo_radar_points(
            config,
            point_clouds,
            num_workers=num_workers,
            normalize=normalize,
            num_inliers=min_inliers,
            tau=float(tau),
        )
        loglikes.append(ll)
    return np.array(loglikes, dtype=float)


def make_tau_sweep_plot(
    taus: np.ndarray,
    neg_loglikes: np.ndarray,
    out_path: Path,
    title: str = "Tau-Sweep: Negative Log-Likelihood",
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(taus, neg_loglikes, marker="o", color="#8905be", lw=2, label="−log L(τ)")

    best_idx = int(np.argmin(neg_loglikes))
    best_tau = taus[best_idx]
    best_val = neg_loglikes[best_idx]
    ax.scatter([best_tau], [best_val], color="#FF4208", zorder=5, label=f"Bestes τ={best_tau:.2e}")

    ax.set_title(title)
    ax.set_xlabel("τ")
    ax.set_ylabel("−Log-Likelihood")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


if __name__ == "__main__":
    config = {
        "reference_generators": [{
            "lambda_lines_2d": float(lambda_lines_2d),
            "lambda_points_line_2d": float(lambda_points_line_2d),
            "lambda_clutter": float(lambda_clutter),
        }]
    }

    if use_linspace:
        taus = np.linspace(tau_max, tau_min, num=n_steps)
    else:
        taus = np.geomspace(tau_max, tau_min, num=n_steps)

    t0 = time.perf_counter()
    point_clouds = generate_point_clouds(
        n_clouds=n_clouds,
        config=config,
        axis_limits=axis_limits,
        crop_to_limits=crop_to_limits,
        seed=seed,
    )

    loglikes = compute_loglikelihood_over_taus(
        taus=taus,
        config=config,
        point_clouds=point_clouds,
        min_inliers=min_inliers,
        num_workers=num_workers,
        normalize=normalize,
    )
    neg_loglikes = -loglikes
    t1 = time.perf_counter()

    out_path_path = Path(out_path)
    make_tau_sweep_plot(
        taus=taus,
        neg_loglikes=neg_loglikes,
        out_path=out_path_path,
        title=f"{plot_title} ({n_clouds} Clouds)",
    )

    best_idx = int(np.argmin(neg_loglikes))
    print(f"[INFO] Fertig in {t1 - t0:.3f} s")
    print(f"[INFO] Plot gespeichert unter: {out_path_path.resolve()}")
    print(f"[INFO] Bestes τ: {taus[best_idx]:.6g} -> min(−log L)={neg_loglikes[best_idx]:.6f} "
          f"(log L={loglikes[best_idx]:.6f})")

