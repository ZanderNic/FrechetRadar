# std lib imports
import argparse
import json
import os

# 3 party import
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# projekt imports
from RadarDataGen.Data_Generator.generator import PseudoRadarGridGenerator, StreamingRadarDataset, pseudo_radar_batch_stream, FixedRadarDataset
from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer
from RadarDataGen.Metrics.frechet_distance import frechet_distance_generator
from RadarDataGen.Metrics.random_projections import RandomProjektions


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
                return base
            except Exception:
                return default_checkpoint()
    return default_checkpoint()


def compute_data_dim(discretizer_params: dict) -> int:
    g = int(discretizer_params["grid_size"])
    return (g ** 2) * 4


def single_experiment(
    reference_generator_params: dict,
    discretizer_params: dict,
    data_dim: int, 
    feature_dim: int,
    reference_generator: PseudoRadarGridGenerator,
    comparison_generator: PseudoRadarGridGenerator,
    random_projection_model: RandomProjektions,
    dataset_size: int,
    batch_size: int,
    device: str,
    num_workers: int = 4
) -> pd.DataFrame:
    
    ref_gen = StreamingRadarDataset(reference_generator, dataset_size=dataset_size, batch_size=batch_size, num_workers=num_workers, device=device)
    comp_gen = StreamingRadarDataset(comparison_generator, dataset_size=dataset_size, batch_size=batch_size, num_workers=num_workers, device=device)

    dist = frechet_distance_generator(ref_gen, comp_gen, feature_extractor = random_projection_model, feature_dim=feature_dim, device=device)

    return pd.DataFrame({
            "frechet_distance": [dist],
            "reference_params": [reference_generator_params],
            "comparison_params": [discretizer_params],
            "data_dim": [data_dim],
            "feature_dim": [feature_dim],
            "dataset_size": [dataset_size]
        })


########################################################################################################################################################################


if __name__ == "__main__":
    
    raise NotImplementedError("Do to refactoring the current main_resampling_exp_FD.py is not working. Please contact the maintainers if you need this functionality.")
    
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

    results_path = os.path.join(output_dir, "result_resampling.csv")
    checkpoint_path = os.path.join(output_dir, "checkpoints.json")

    data_dim = compute_data_dim(config["discretizer_params"])

    # Calculate the total samples we need to generate and compare 
    total_samples = 0
    for random_dim in config["random_projection_dims"]:
        for sample_size in  config["sample_sizes"]:
            if random_dim < sample_size:
                total_samples += len(config["reference_generators"]) * len(config["comparison_generators"]) * 2 * sample_size * config["num_trys"]

    progress_bar = tqdm(total=total_samples, desc="Generated and Compared Samples")

    ### load checkpoint
    checkpoints = load_checkpoint(checkpoint_path)
    progress_bar.update(checkpoints["completed_samples"])

    if checkpoints:
        rev_gen_param_to_do = config["reference_generators"][checkpoints["rev_idx"]:]
        comp_gen_param_to_do = config["comparison_generators"][checkpoints["comp_idx"]:]
        random_dim_to_do = config["random_projection_dims"][checkpoints["rand_idx"]:]
        sample_size_to_do = config["sample_sizes"][checkpoints["sample_idx"]:]
        num_trys_to_do = range(config["num_trys"] - checkpoints["try_idx"])

    for rev_idx, rev_gen_param in enumerate(rev_gen_param_to_do):
        reverence_generator = PseudoRadarGridGenerator(rev_gen_param, discretizer_params=config["discretizer_params"])

        for comp_idx, comp_gen_param in enumerate(comp_gen_param_to_do):
            comp_generator = PseudoRadarGridGenerator(comp_gen_param, discretizer_params=config["discretizer_params"])

            for rand_idx, random_dim in enumerate(random_dim_to_do):
                random_pro = RandomProjektions(data_dim = data_dim, feature_dim=random_dim, device=device)

                for sample_idx, sample_size in enumerate(sample_size_to_do):
                    if random_dim >= sample_size:
                        continue

                    for try_idx, _ in enumerate(num_trys_to_do):
                        try:
                            results = pd.concat([
                                results,
                                single_experiment(
                                    reference_generator_params=rev_gen_param,
                                    discretizer_params=comp_gen_param,
                                    data_dim=data_dim,
                                    feature_dim=random_dim,
                                    reference_generator=reverence_generator,
                                    comparison_generator=comp_generator,
                                    random_projection_model=random_pro,
                                    dataset_size=sample_size,
                                    batch_size=config["batch_size"],
                                    device=device
                                )
                            ])

                            progress_bar.update(sample_size*2)
                            results.to_csv(results_path, header=True, index=False)
                            
                            checkpoint = {
                                "rev_idx": rev_idx,
                                "comp_idx": comp_idx,
                                "rand_idx": rand_idx,
                                "sample_idx": sample_idx,
                                "try_idx": try_idx,
                                "completed_samples": progress_bar.n
                            }
                            save_checkpoint(checkpoint, checkpoint_path)
                        
                        except Exception as e:
                            print("Error:" , e)

    progress_bar.close()
    
    print("Done Done Done!")
    print(f"Results: {results_path}")
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Deleted checkpoint file: {checkpoint_path}")
