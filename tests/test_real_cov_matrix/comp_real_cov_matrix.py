
# std lib
import argparse
import json
import os
from pathlib import Path
from typing import Tuple

# 3rd party
import numpy as np
import torch
from tqdm import tqdm

# project
from RadarDataGen.Data_Generator.generator import (
    PseudoRadarGridGenerator,
    StreamingRadarDataset,
)
from RadarDataGen.Metrics.random_projections import RandomProjektions
from RadarDataGen.Statistic.onlineStat import OnlineStats


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_npz(out_path: Path, mean: np.ndarray, cov: np.ndarray, meta: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        mean=mean.astype(np.float64),
        cov=cov.astype(np.float64),
        meta=json.dumps(meta),
    )



if __name__ == "__main__":
    ##### Settings #####################################

    gen_params = {
      "lambda_lines_2d": 10,
      "lambda_points_line_2d": 20,
      "lambda_clutter": 50
    }

    disc_params = {
        "grid_size": 64,
        "x_min": -1,
        "x_max": 1,
        "y_min": -1,
        "y_max": 1,
        "valid_indicator": 1.0
    }

    num_workers = 8
    batch_size = 64

    ##########################################################
   
    parser = argparse.ArgumentParser(description="Compute exact empirical covariance (features) and save to NPZ.")
    parser.add_argument("--feature_dim", "-d", type=int, default=256, help="Feature dimension for RandomProjektions")
    parser.add_argument("--num_samples", "-n", type=int, default=400_000, help="Total number of samples to aggregate")
    parser.add_argument("--batch_size", "-b", type=int, default=256, help="Override batch_size from config")
    parser.add_argument("--ddof", type=int, default=1, help="Degrees of freedom for covariance")
    parser.add_argument("--out_name", type=str, default=None, help="Optional custom output filename (without extension)")
    args = parser.parse_args()

    base_dir = Path("./tests/test_real_cov_matrix/")
    out_dir = base_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "auto"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = args.batch_size or batch_size
    ref_gen = PseudoRadarGridGenerator(gen_params, discretizer_params=disc_params)

    stream = StreamingRadarDataset(
        ref_gen,
        dataset_size=args.num_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    feature_dim = disc_params["grid_size"] ** 2 * 4
    online_stat = OnlineStats(feature_dim=feature_dim, dtype=torch.float64, device=device)

    processed = 0
    with torch.no_grad():
        for batch in tqdm(stream, total=(args.num_samples + batch_size - 1) // batch_size, desc="Streaming & projecting"):
            flatten_batch = batch.reshape(batch.shape[0], -1)

            if processed + flatten_batch.shape[0] > args.num_samples:
                keep = args.num_samples - processed
                if keep <= 0:
                    break
                flatten_batch = flatten_batch[:keep]

            online_stat.update(flatten_batch)
            processed += flatten_batch.shape[0]
            if processed >= args.num_samples:
                break

    mean, cov = online_stat.get_mean_cvar()

    # --- Save results ---
    out_name = args.out_name or f"true_cov_features_d{args.feature_dim}_n{n}"
    out_path = out_dir / f"{out_name}.npz"
    meta = {
        "n": int(n),
        "data_dim": int(data_dim),
        "ddof": int(args.ddof),
        "device": device,
        "generator_type": args.generator,
        "gen_index": int(args.gen_index),
        "generator_params": gen_params,
        "discretizer_params": disc_params,
    }
    save_npz(out_path, mean, cov, meta)

    # (optional) fast separate saves
    np.save(out_dir / f"{out_name}_mean.npy", mean)
    np.save(out_dir / f"{out_name}_cov.npy", cov)

    print(f"[OK] Saved mean & covariance to:\n  {out_path}")
    print(f"     mean.npy: {out_dir / (out_name + '_mean.npy')}")
    print(f"     cov.npy : {out_dir / (out_name + '_cov.npy')}")
