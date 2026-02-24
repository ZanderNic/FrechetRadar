# std lib imports
import argparse
import json
import os
import math
import random

# 3 party import
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# projekt imports
from RadarDataGen.Data_Generator.generator import PseudoRadarGridGenerator, StreamingRadarDataset, RadarDataset, worker_init_fn
from RadarDataGen.Models.DiffusionModell.diff_model import DiffusionModel
from RadarDataGen.Statistic.onlineStat import OnlineStats
from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer


###  Utils  ####
def make_iterable_sampler(model, num_baches: int , batch_size: int, model_sampling_args: dict = {}):
    for _ in range(int(num_baches)):
        yield model.sample(int(batch_size), **model_sampling_args)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_datasets(
    dataset_type: str,
    generator: PseudoRadarGridGenerator,
    batch_size: int,
    num_workers: int,
    dataset_size: int = None,
    seed: int = 42,
):
    """
        Create training and reference datasets.

        Parameters
            dataset_type : str
                Either "streaming" or "fixed"
            generator :
                PseudoRadarGridGenerator
            batch_size : int
            num_workers : int
            dataset_size : int | None
                Only used for fixed datasets
            cache_data : bool
                Cache generated samples (time-efficient)
            cache_seeds : bool
                Cache RNG seeds (space-efficient)

        Returns
            dataset :
                Dataset used for training
            dataset_rev :
                Dataset used for reference statistics
    """

    if dataset_type == "streaming":
        dataset = torch.utils.data.DataLoader(
            StreamingRadarDataset(sampler=generator, dtype=torch.float32, base_seed=seed),
            batch_size=batch_size,
            num_workers=num_workers,               
            pin_memory=True,              
            persistent_workers=True,      
            prefetch_factor=4,  
            worker_init_fn=worker_init_fn,
        )
        
        dataset_rev = torch.utils.data.DataLoader(
            StreamingRadarDataset(sampler=generator, dtype=torch.float32, base_seed=seed ** 2 + 42),
            batch_size=batch_size,
            num_workers=num_workers,               
            pin_memory=True,              
            persistent_workers=True,      
            prefetch_factor=4,  
            worker_init_fn=worker_init_fn,
        )

    elif dataset_type == "fixed":
        dataset = torch.utils.data.DataLoader(
            RadarDataset(sampler=generator, dtype=torch.float32, num_samples= dataset_size, base_seed=seed),
            batch_size=batch_size,
            num_workers=num_workers,               
            pin_memory=True,              
            persistent_workers=True,      
            prefetch_factor=4,  
            worker_init_fn=worker_init_fn,
        )

        # Reference dataset should still stream fresh samples
        dataset_rev = torch.utils.data.DataLoader(
            StreamingRadarDataset(sampler=generator, dtype=torch.float32, base_seed=seed ** 2 + 42),
            batch_size=batch_size,
            num_workers=num_workers,               
            pin_memory=True,              
            persistent_workers=True,      
            prefetch_factor=4,  
            worker_init_fn=worker_init_fn,
        )

    else:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. "
            "Use 'streaming' or 'fixed'."
        )

    return dataset, dataset_rev


def generate_online_stats(
    generator,
    sample_sizes: list,
    feature_extractors: dict,
    log_liklehood_computation: callable = None,
    log_like_samples: int = 1000,
    radar_discretizer: RadarDiscretizer = None,
    config: dict = None,
    device : str = "cuda" if torch.cuda.is_available() else "cpu",
):
    online_stats = {feature_dim: {} for feature_dim in feature_extractors.keys()} # here we will save different copys of the Online Stats Class for all the different feature dims and sample sizes
    stat = {feature_dim: OnlineStats(feature_dim, device=device) for feature_dim in feature_extractors.keys()}
    gen_samples: int = 0 
    mean_log_like: float = 0.0 if log_liklehood_computation is not None and config is not None else None
    log_like_comp: int = 0
    progress_bar = tqdm(total=max(sample_sizes), desc="Generating Samples + Calculating Stats")

    for data in generator:
        gen_samples += len(data)
        data = data.to(device, non_blocking=True)

        if log_liklehood_computation is not None and config is not None and radar_discretizer is not None:
            if gen_samples <= log_like_samples:
                log_like_comp += 1

                point_grids = data.detach().cpu()
                point_clouds =  radar_discretizer.grid_to_points(point_grids)

                mean_log_like += log_liklehood_computation(
                    config = config, 
                    point_clouds = point_clouds, 
                    num_workers = 4, 
                    normalize = True,
                    num_inliers = 8,
                    tau= 0.01
                )

        for feature_dim in online_stats.keys():
            ref_latent = feature_extractors[feature_dim](data)
            stat[feature_dim].update(ref_latent)

        if gen_samples in sample_sizes:
            for feature_dim in online_stats.keys():
                online_stats[feature_dim][gen_samples] = stat[feature_dim].deepcopy()

        progress_bar.update(len(data))

        if gen_samples >= max(sample_sizes):
            break
        
    progress_bar.close()

    if mean_log_like is not None:
        return online_stats, mean_log_like / log_like_comp

    return online_stats


def save_samples_current_model(
    diff_model : DiffusionModel,
    path: str,
    radar_discretizer: RadarDiscretizer,
    sampler: str = "ddim",
    num_samples_tensors: int = 25,
    num_samples_point_clouds: int = 4,
    num_samples_grids: int = 4,
    valid_threshold: float = 0.6
):  
    
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    num_samples_to_gen = max(num_samples_tensors, num_samples_point_clouds, num_samples_grids)

    samples = diff_model.sample(num_samples_to_gen, sampler=sampler, sample_time_steps=250, use_ema_model=True).cpu()   # Generate samples

    # save raw tensors
    if num_samples_tensors > 0:
        tensor_save_path = os.path.join(path, "samples_raw_tensor.pt") 
        torch.save(samples[:num_samples_tensors], tensor_save_path)
    
    # save point clouds
    if num_samples_point_clouds > 0:
        cols_p = math.ceil(math.sqrt(num_samples_point_clouds))  
        rows_p = math.ceil(num_samples_point_clouds / cols_p)  
    
        fig_p, ax_p = plt.subplots(rows_p, cols_p, figsize=(cols_p*5, rows_p*5), tight_layout=True)
        ax_p = ax_p.flatten()

        for idx in range(num_samples_point_clouds):
            points = radar_discretizer.grid_to_points(samples[idx].cpu(), valid_threshold=valid_threshold)

            if not points.size == 0:
                ax_p[idx].scatter(*points[:, :2].T, c=points[:, 2])
            else:
                ax_p[idx].scatter(0, 0, alpha=0)                            # without this the plot scale is wrong 

            ax_p[idx].add_patch(
                plt.Rectangle(
                    (radar_discretizer.x_min, radar_discretizer.y_min),
                    radar_discretizer.x_max - radar_discretizer.x_min,
                    radar_discretizer.y_max - radar_discretizer.y_min,
                    facecolor="#8905be",
                    alpha=0.15,                       
                    zorder=0
                )
            )

            ax_p[idx].add_patch(
                plt.Rectangle(
                    (radar_discretizer.x_min, radar_discretizer.y_min),
                    radar_discretizer.x_max - radar_discretizer.x_min,
                    radar_discretizer.y_max - radar_discretizer.y_min,
                    facecolor="none",               
                    edgecolor="#FF4208",      
                    linewidth=2,
                    zorder=3       
                )
            )

            ax_p[idx].set_title(f'Sampled Point Clouds')
            ax_p[idx].grid(True)

        for j in range(num_samples_point_clouds, len(ax_p)):
            ax_p[j].axis('off')

        point_cloud_path = os.path.join(path, "samples_point_clouds.png") 
        fig_p.savefig(point_cloud_path, bbox_inches="tight", pad_inches=0.3)

    # save grids
    if num_samples_grids > 0:
        cols_g = math.ceil(math.sqrt(num_samples_grids))  
        rows_g = math.ceil(num_samples_grids / cols_g)  
    
        fig_g, ax_g = plt.subplots(cols_g, rows_g, figsize=(cols_g*5, rows_g*5), tight_layout=True)
        ax_g = ax_g.flatten()
        
        for idx in range(num_samples_grids):

            image = radar_discretizer.grid_to_image(samples[idx].to("cpu").numpy().transpose(1, 2, 0), swap_xy=True, invert_rows=True, invert_columns=False)  
            
            ax_g[idx].matshow(image)
            ax_g[idx].grid(True)

        for j in range(num_samples_grids, len(ax_g)):
            ax_g[j].axis('off')

        grids_path = os.path.join(path, "samples_grids.png") 
        fig_g.savefig(grids_path, bbox_inches="tight", pad_inches=0.3)


def create_model_summary_file(dif_mod, output_dir: str):
    """
        Create a model summary file if it does not exist yet.

        Parameters
        ----------
        dif_mod : DiffusionModel
            The model to summarize.
        output_dir : str
            Directory where the summary file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "model_summary.txt")

    try:
        with open(save_path, "x", encoding="utf-8") as file:  # 'x' -> only create if not exists
            file.write(dif_mod.summary(print_model_structure=True))
            print(f"[INFO] Model summary saved at {save_path}")
    except FileExistsError:
        print(f"[INFO] Model summary already exists at {save_path}")
    except OSError as e:
        print(f"[INFO] Could not write model summary: {e}")



def append_df_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Append `df` to CSV at `path`. Writes header only if the file doesn't exist or is empty.
    Creates parent directory if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not (os.path.isfile(path) and os.path.getsize(path) > 0)
    df.to_csv(path, mode='a', header=write_header, index=False)
