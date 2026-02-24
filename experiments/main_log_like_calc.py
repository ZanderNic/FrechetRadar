# std lib imports
import argparse
import json
import os
import time
from multiprocessing import Pool
from functools import partial

# 3 party import
import torch
import pandas as pd

# projekt imports
from RadarDataGen.Metrics.log_likelihood import log_likelihood_pseudo_radar_points
from RadarDataGen.Data_Generator.pseudo_radar_points import pseudo_radar_points



"""
    This script evaluates the log-likelihood of synthetic pseudo-radar point clouds
    under different generator configurations.

    For each combination of reference and comparison generators defined in a JSON
    configuration file, multiple point clouds are sampled using a parametric
    pseudo-radar data generator. The generated point clouds are spatially filtered
    according to the specified discretization bounds.

    A fixed number of point clouds is generated in parallel using multiprocessing,
    and their average log-likelihood under the reference generator is computed.
    The results, together with the corresponding generator parameters and dataset
    size, are aggregated and continuously written to a CSV file for reproducibility
    and further analysis.

    The script is fully configuration-driven and intended for systematic,
    repeatable evaluation of synthetic radar data generators.

    
    Use: 
        python3 ./experiments/main_log_like_calc.py --config ./experiments/results/fd_different_parameters/exp_div_lines/setting.json

        python3 ./experiments/main_log_like_calc.py --config ./experiments/results/fd_same_parameters/exp_lines/setting.json
"""



## Load Experiment Params #################################

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def generate_cloud(config, _):
    points = pseudo_radar_points(
        lambda_lines_2d=config["lambda_lines_2d"],
        lambda_points_line_2d=config["lambda_points_line_2d"],
        lambda_clutter=config["lambda_clutter"]
    )
    
    mask = (
        (points[:, 0] >= config["discretizer_params"].get("x_min", -1)) &
        (points[:, 0] <= config["discretizer_params"].get("x_max", 1)) &
        (points[:, 1] >= config["discretizer_params"].get("y_min", -1)) &
        (points[:, 1] <= config["discretizer_params"].get("y_max", 1))
    )

    return points[mask]
########################################################################################################################################################################


if __name__ == "__main__":
    time_start = time.perf_counter()
    results_ll = pd.DataFrame()
    
    parser = argparse.ArgumentParser(description="Run RadarDataGen experiments from JSON config.")
    parser.add_argument("--config", "-c", required=True, help="Path to config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = os.path.dirname(args.config)
    
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device = {device}")

    results_ll_path = os.path.join(output_dir, "results_ll.csv")

    for try_idx, _ in enumerate(range(config["num_trys"])):
        for rev_idx, rev_gen_param in enumerate(config["reference_generators"]):
                for comp_idx, comp_gen_param in enumerate(config["comparison_generators"]):
                    log_like_examples = 1_000
                    print(f"[INFO]: Calculating log likelihood for {log_like_examples} point clouds")

                    config_gen = {
                        "discretizer_params" : config["discretizer_params"],
                        "reference_generators" : [rev_gen_param],
                        ** comp_gen_param,
                    }
                  
                    with Pool(processes=config["num_workers"]) as pool:
                        point_clouds = pool.map(partial(generate_cloud, config_gen), range(log_like_examples))

                    ll = log_likelihood_pseudo_radar_points(
                        config=config_gen,
                        point_clouds=point_clouds,
                        num_workers=config["num_workers"]
                    )

                    print(f"[RESULT]: Log-Likelihood for {log_like_examples} clouds = {ll:.4f}")

                    log_like =  pd.DataFrame({
                            "log_like": [ll],
                            "reference_params": [rev_gen_param],
                            "comparison_params": [comp_gen_param],
                            "dataset_size": [log_like_examples]
                        })

                    results_ll = pd.concat([
                        results_ll,
                        log_like
                    ])

                    results_ll.to_csv(results_ll_path, header=True, index=False)

    time_end = time.perf_counter()

    print("Done Done Done!")
    print(f"Took {time_end-time_start:.6f} long")