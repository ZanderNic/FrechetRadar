# std lib imports
import copy
import os
import random

# 3rd party imports
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset

# project imports
from RadarDataGen.Data_Generator.pseudo_radar_points import pseudo_radar_points
from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer



###### Utils ###########################

def set_global_seed(seed: int):
    """
        Set global RNG state for reproducibility.
        This affects numpy, random, and torch.

        Note:
            This is process-local. For correctness with multiprocessing,
            each process should call this with its own seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    os.environ["OMP_NUM_THREADS"] = "1"

######################################

class PseudoRadarGridGenerator:
    """
        Thin wrapper around `pseudo_radar_points` + `RadarDiscretizer`.

        - Generates a single pseudo radar point cloud.
        - Optionally clips to a bounding box.
        - Discretizes to a grid.
    """

    def __init__(
        self,
        params: dict,
        discretizer_params: dict = None,
        discretizer: RadarDiscretizer = None,
    ):
        self.params = copy.deepcopy(params)

        if discretizer is not None:
            self.discretizer = discretizer
            self.discretizer_params = {
                "grid_size": discretizer.grid_size,
                "x_min": discretizer.x_min,
                "x_max": discretizer.x_max,
                "y_min": discretizer.y_min,
                "y_max": discretizer.y_max,
                "valid_indicator": discretizer.valid_indicator,
            }
        elif discretizer_params is not None:
            self.discretizer_params = copy.deepcopy(discretizer_params)
            self.discretizer = RadarDiscretizer(**discretizer_params)
        else:
            raise ValueError("Please provide either a discretizer instance or discretizer_params dict.")

        self.min_x = self.discretizer_params["x_min"]
        self.max_x = self.discretizer_params["x_max"]
        self.min_y = self.discretizer_params["y_min"]
        self.max_y = self.discretizer_params["y_max"]

    def __call__(self, seed = None):
        """
            Generate one pseudo radar example and return it as a grid (H, W, C).
        """
        points = pseudo_radar_points(**self.params, seed = seed)

        if (
            self.max_x is not None
            and self.min_x is not None
            and self.max_y is not None
            and self.min_y is not None
        ):
            mask = (
                (points[:, 0] >= self.min_x) & (points[:, 0] <= self.max_x)
                & (points[:, 1] >= self.min_y) & (points[:, 1] <= self.max_y)
            )
            points = points[mask]

        return self.discretizer.points_to_grid(points) # expected return shape: (H, W, C)



class StreamingRadarDataset(torch.utils.data.IterableDataset):
    """
        
    """
    def __init__(self, sampler, dtype=torch.float32, base_seed=42):
        super().__init__()
        self.sampler = sampler
        self.dtype = dtype
        self.base_seed = base_seed if base_seed is not None else random.randint(0, 2**31 - 1)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id

        seed = self.base_seed + worker_id * 1_000_000
        rng = np.random.default_rng(seed) 

        while True:
            iter_seeds = rng.integers(0, 2**31-1)
            arr = self.sampler(iter_seeds)  # (H, W, C), numpy

            t = torch.from_numpy(arr).to(self.dtype).contiguous()

            if t.ndim == 3:
                t = t.permute(2, 0, 1)

            yield t


class RadarDataset(torch.utils.data.IterableDataset):
    """
        
    """
    def __init__(
            self, 
            sampler, 
            dtype=torch.float32, 
            num_samples: int = 1000, 
            base_seed: int = 42
    ):
        super().__init__()
        self.sampler = sampler
        self.dtype = dtype
        self.num_samples: int = num_samples
        self.base_seed: int = base_seed if base_seed is not None else random.randint(0, 2**31 - 1)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id

        seed = self.base_seed + worker_id * 1_000_000
        rng = np.random.default_rng(seed) 

        while True:
            sample_seed = rng.integers(0, self.num_samples)

            arr = self.sampler(seed = sample_seed)  # (H, W, C), numpy

            t = torch.from_numpy(arr).to(self.dtype).contiguous()

            if t.ndim == 3:
                t = t.permute(2, 0, 1)

            yield t