# std lib imports
import argparse
import json
import os
import time

# 3 party import
import torch
import pandas as pd

# projekt imports
from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer
from RadarDataGen.Data_Generator.generator import PseudoRadarGridGenerator, StreamingRadarDataset, worker_init_fn
from RadarDataGen.Metrics.frechet_distance import frechet_distance_stats
from RadarDataGen.Metrics.random_projections import RandomProjektions
from RadarDataGen.Metrics.log_likelihood import log_likelihood_pseudo_radar_points

# utils import
from experiments.utils_main_model_training_FRD import generate_online_stats


"""
Main cumulative sampling experiment for RadarDataGen.

This script loads an experiment configuration from a JSON file and performs
Frechet Radar Distance (FRD) evaluations between radar data generators defined in the
config. For each reference comparison generator pair, it computes:

- Streaming/online feature statistics for multiple sample sizes
- Random projection embeddings for different feature dimensions
- Frechet Radar Distance between the corresponding online statistics
- Optional log-likelihood estimates

The script supports:
- Automatic checkpointing and resume after interruption
- Continuous saving of partial results to results_cumalative_sampling.csv
- Efficient streaming via PyTorch DataLoader to avoid memory overflow
- GPU/CPU auto-selection

Example usage:
    python3 ./experiments/main_cumulativ_sampling_exp_FRD.py --config ./experiments/results/fd_different_parameters/exp_div_lines/setting.json

    python3 ./experiments/main_cumulativ_sampling_exp_FRD.py --config ./experiments/results/fd_same_parameters/exp_lines/setting.json
"""


## Load Experiment Params #################################

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def save_checkpoint(done_tasks, path):
    with open(path, "w") as f:
        json.dump(done_tasks, f)

def default_checkpoint():
    return {
        "rev_idx": 0,
        "comp_idx": 0,
        "rand_idx": 0,
        "sample_idx": 0,
        "try_idx": 0,
        "completed_samples": 0
    }


def load_checkpoint(path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as f:
            try:
                ckpt = json.load(f)
                base = default_checkpoint()
                base.update(ckpt)
                print(f"[INFO] Loaded checkpoint: {base}")
                return base
            except Exception:
                return default_checkpoint()
    return default_checkpoint()


def compute_data_dim(discretizer_params: dict) -> int:
    g = int(discretizer_params["grid_size"])
    return (g ** 2) * 4


########################################################################################################################################################################


if __name__ == "__main__":

    time_start = time.process_time()
    results = pd.DataFrame()

    parser = argparse.ArgumentParser(description="Run RadarDataGen experiments from JSON config.")
    parser.add_argument("--config", "-c", required=True, help="Path to config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = os.path.dirname(args.config)
    
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device = {device}")

    results_path = os.path.join(output_dir, "results_cumalative_sampling.csv")
    checkpoint_path = os.path.join(output_dir, "checkpoints.json")

    data_dim = compute_data_dim(config["discretizer_params"])

    ### load checkpoint
    checkpoints = load_checkpoint(checkpoint_path)

    if  os.path.isfile(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        results = pd.read_csv(results_path)

    if checkpoints:
        rev_gen_param_to_do = config["reference_generators"][checkpoints["rev_idx"]:]
        comp_gen_param_to_do = config["comparison_generators"][checkpoints["comp_idx"]:]
        random_dim_to_do = config["random_projection_dims"][checkpoints["rand_idx"]:]
        sample_size_to_do = config["sample_sizes"][checkpoints["sample_idx"]:]
        num_trys_to_do = range(config["num_trys"] - checkpoints["try_idx"])

    for sample_size in sample_size_to_do:
        if sample_size % config["batch_size"] != 0:
            raise ValueError(f"sample size: {sample_size} is not / with batch_size: {config['batch_size']} D: !!!")

    radar_disc = RadarDiscretizer(**config["discretizer_params"])

    for try_idx, _ in enumerate(num_trys_to_do):
        
        ramdom_pros = {} # here we save all the random pros that we need to do
        for rand_idx, random_dim in enumerate(random_dim_to_do):
            random_pro = RandomProjektions(data_dim = data_dim, feature_dim=random_dim, device=device)
            ramdom_pros[random_dim] = random_pro

        for rev_idx, rev_gen_param in enumerate(rev_gen_param_to_do):
            reverence_generator = PseudoRadarGridGenerator(rev_gen_param, discretizer_params=config["discretizer_params"])
            
            ref_dataset = torch.utils.data.DataLoader(
                StreamingRadarDataset(sampler=reverence_generator, dtype=torch.float32, base_seed=None),
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],               
                pin_memory=True,              
                persistent_workers=True,      
                prefetch_factor=4,  
                worker_init_fn=worker_init_fn,
            )


            # now we create us len(sample_size_to_do) online stats models 
            rev_statistics = generate_online_stats(
                generator= ref_dataset,
                sample_sizes= sample_size_to_do,
                feature_extractors= ramdom_pros,
                device=device
            )

            for comp_idx, comp_gen_param in enumerate(comp_gen_param_to_do):
                comp_generator = PseudoRadarGridGenerator(comp_gen_param, discretizer_params=config["discretizer_params"])
   
                comp_dataset = torch.utils.data.DataLoader(
                    StreamingRadarDataset(sampler=comp_generator, dtype=torch.float32, base_seed=None),
                    batch_size=config["batch_size"], 
                    num_workers=config["num_workers"],              
                    pin_memory=True,              
                    persistent_workers=True,      
                    prefetch_factor=4,  
                    worker_init_fn=worker_init_fn,
                )
    
                comp_statistics = generate_online_stats(
                    generator= comp_dataset,
                    sample_sizes= sample_size_to_do,
                    feature_extractors= ramdom_pros,
                    log_like_samples=1000,
                    log_liklehood_computation= None,
                    config= config,
                    radar_discretizer= radar_disc,
                    device=device
                )
                
                log_like = 0

                for rand_idx, random_dim in enumerate(random_dim_to_do):
                    for sample_idx, sample_size in enumerate(sample_size_to_do):
                        if random_dim >= sample_size:
                            continue

                        try:
                            dist = frechet_distance_stats(rev_statistics[random_dim][sample_size], comp_statistics[random_dim][sample_size], device=device)

                            distance =  pd.DataFrame({
                                    "frechet_distance": [dist],
                                    "reference_params": [rev_gen_param],
                                    "comparison_params": [comp_gen_param],
                                    "data_dim": [data_dim],
                                    "feature_dim": [random_dim],
                                    "dataset_size": [sample_size],
                                    "log_likelihood": [log_like if log_like else None],
                                    "neg_log_likelihood": [-log_like if log_like else None],
                                    "log_like_samples" : [1000]
                                })

                            results = pd.concat([
                                results,
                                distance
                            ])

                            results.to_csv(results_path, header=True, index=False)
                            
                            checkpoint = {
                                "rev_idx": rev_idx,
                                "comp_idx": comp_idx,
                                "rand_idx": rand_idx,
                                "sample_idx": sample_idx,
                                "try_idx": try_idx
                            }
                            save_checkpoint(checkpoint, checkpoint_path)
                        
                        except Exception as e:
                            print("Error:", e)

    time_end = time.process_time()

    print("Done Done Done!")
    print(f"Took {time_end-time_start:.6f} long")
    print(f"Results: {results_path}")
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Deleted checkpoint file: {checkpoint_path}")