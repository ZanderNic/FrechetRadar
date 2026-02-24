# std lib imports
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from functools import partial

# 3 party import
import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

def detect_lines_and_clutter(
    points: np.ndarray,
    tau: float = 0.001,
    min_inliers: int = 8
):  # -> Dict[str, np.ndarray | List[np.ndarray]]:  # Not suported in python 3.9.9
    """
        Fast greedy line detection for nearly perfect line data with small deviations.
        Sequentially tries to form lines starting from each point.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N, 3) with 3D points.
        tau : float
            Distance threshold for inliers (small deviation allowed).
        min_inliers : int
            Minimum number of points to accept a line.

        Returns
        -------
        Dict[str, np.ndarray | List[np.ndarray]]
            {
                "lines": [np.ndarray of inlier points per line],
                "clutter": np.ndarray of remaining points
            }
    """
    pts = np.array(points, copy=True)
    lines: List[np.ndarray] = []
    clutter: List[np.ndarray] = []

    while len(pts) > 0:
        p1 = pts[0]

        if len(pts) < min_inliers:
            clutter.append( pts)
            break

        start_index = 1

        # Pick second point to define line
        for idx in range(1, len(pts) - 1):
            p2 = pts[idx]
            u = p2 - p1
            norm_u = np.linalg.norm(u)
            u /= norm_u  

            # Compute distances of all points to this line
            diffs = pts - p1
            cross = np.cross(diffs, u)
            d = np.linalg.norm(cross, axis=1) / (np.linalg.norm(u) + 1e-12)

            inliers_idx = np.where(d < tau)[0]

            if len(inliers_idx) >= min_inliers:
                # Accept line
                lines.append(pts[inliers_idx])
                mask = np.ones(len(pts), dtype=bool)
                mask[inliers_idx] = False
                pts = pts[mask]
                break
        
        # No line found so continue
        clutter.append(p1)
        pts = pts[1:]

    return {
        "lines": lines,
        "clutter": np.vstack(clutter) if clutter else np.empty((0, 3))
    }

def poisson_entropy(lam: float):
    """
    
    """
    lam = float(lam)
    return (
        0.5 * np.log(2 * np.pi * np.e * lam)
        - 1/(12 * lam)
        - 1/(24 * lam**2)
        - 19/(360 * lam**3)
    )


def poisson_ll(k: np.ndarray, lam: float, normalize : bool = True) -> float:
    """
        Vectorized Poisson log-likelihood for counts array k given mean lam
    """

    k = np.asarray(k, dtype=np.int64)
    n = k.size
    if n == 0:
        return 0.0
    if lam <= 0.0:
        return 0.0 if np.all(k == 0) else -np.inf
    
    if normalize:
        #return (-lam * n + np.sum(k) * np.log(lam) - np.sum(gammaln(k + 1))) / len(k)
        return poisson.logpmf(k, lam).mean()

    return poisson.logpmf(k, lam)


def process_single_cloud(cloud: np.ndarray, tau: float = 0.01, min_inliers: int = 8) -> Dict:
    """
        Worker function: Detect lines and clutter for one point cloud and return counts.
    """
    res = detect_lines_and_clutter(cloud, tau=tau, min_inliers=min_inliers)
    return {
        "num_lines": len(res["lines"]),
        "points_per_line": [len(pts) for pts in res["lines"]],
        "num_clutter": len(res["clutter"])
    }


def log_likelihood_pseudo_radar_points(
    config: dict, 
    point_clouds: List[np.ndarray], 
    num_workers: int = 4, 
    normalize: bool = True,
    num_inliers: int = 8,
    tau: float = 0.01
) -> float:
    """
        Compute total log-likelihood for radar point clouds using parallel processing.
    """
    # Extract config
    data_gen_dict = (config.get("reference_generators", {}) or config.get("data_generator", {}))[0]
    lam_lines = float(data_gen_dict.get("lambda_lines_2d", 0))
    lam_points_per_line = float(data_gen_dict.get("lambda_points_line_2d", 0))
    lam_clutter = float(data_gen_dict.get("lambda_clutter", 0))
    
    worker = partial(process_single_cloud, tau=tau, min_inliers=num_inliers)

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker, point_clouds)

    # Aggregate results
    number_lines = np.array([r["num_lines"] for r in results], dtype=np.int64)
    number_points_per_line = np.array([p for r in results for p in r["points_per_line"]], dtype=np.int64)
    number_clutter = np.array([r["num_clutter"] for r in results], dtype=np.int64)

    # Compute Poisson log-likelihoods
    ll_lines = poisson_ll(number_lines, lam_lines, normalize)
    ll_points = poisson_ll(number_points_per_line, lam_points_per_line, normalize)
    ll_clutter = poisson_ll(number_clutter, lam_clutter, normalize)

    return ll_lines + ll_points + ll_clutter