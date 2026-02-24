# std lib imports
import time

# 3 party import
import numpy as np
import torch

# projekt imports
from RadarDataGen.Metrics.random_projections import RandomProjektions
from RadarDataGen.Statistic.onlineStat import OnlineStats


def frechet_distance_generator(
    reference_generator,
    comparison_generator,
    feature_extractor,
    feature_dim : int, 
    device :str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
        Computes the Fréchet Distance between two data distributions provided two generators that can be iteraded using random projections
        and online statistics.

        Frechet distance is def as || mean_r - mean_c ||2 + TR(cov_r + cov_c - 2 (cov_r @ cov_c)**1/2)
        because TR is linear we can split TR(cov_r + cov_c - 2 (cov_r @ cov_c)**1/2) = TR(cov_r) + TR(cov_c) - 2 * TR((cov_r @ cov_c)**1/2) 
        (cov_r @ cov_c)**1/2 = M **1/2 is the matrix sqare root of M and can be dev by M ** 1/2 = P * (ew **1/2) * P ** -1 so the TR of the 
        Matrix sqare root is eaqual to sum(ew **1/2) because the covarianz is positiv semidefenit all ew >= 0 but because of numeric errors
        ew could be negativ so we just take the sqare root of the ew >= 0  

        Parameters:
        - reference_generator: Callable that generates reference data samples.
        - comparison_generator: Callable that generates comparison data samples.
        - data_dim: Dimensionality of the input data.
        - feature_dim: Dimensionality of the projected feature space.
        - iterations: Number of iterations to accumulate statistics.

        Returns:
        - Fréchet Distance between the two distributions.
    """
    ref_stat = OnlineStats(feature_dim, device=device)
    comp_stat = OnlineStats(feature_dim, device=device)

    for ref_data, comp_data in zip(reference_generator, comparison_generator):
   
        ref_latent = feature_extractor(ref_data)
        comp_latent = feature_extractor(comp_data)

        ref_stat.update(ref_latent)
        comp_stat.update(comp_latent)

    mean_r, cvar_r = ref_stat.get_mean_cvar()
    mean_c, cvar_c = comp_stat.get_mean_cvar()

    if device == "cuda":
        a = torch.sum((mean_r - mean_c)**2)
        b = torch.trace(cvar_r) + torch.trace(cvar_c)
        c = torch.sqrt(torch.clamp(torch.linalg.eigval((cvar_r @ cvar_c)), min=0)).real.sum()
        dist = a + b - 2 * c
        return dist.item()

    else:
        # use numpy 
        mean_r, cvar_r = mean_r.cpu().numpy(), cvar_r.cpu().numpy()
        mean_c, cvar_c = mean_c.cpu().numpy(), cvar_c.cpu().numpy()

        a = np.sum((mean_r - mean_c)**2)
        b = np.trace(cvar_r) + np.trace(cvar_c)
        eigvals = np.clip(np.linalg.eigvals(cvar_r @ cvar_c), a_min=0, a_max=None)  # the eigenvalues theoretically can't be negative but because of numeric errors they can
        c = np.sqrt(eigvals).real.sum()   # eigvals faster than matrix square root but still O(N^3)
        return a + b - 2 * c


def frechet_distance_stats(
    ref_stat: OnlineStats,
    comp_stat: OnlineStats,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
        Computes the Fréchet Distance between two data distributions using random projections
        and online statistics given two Online Stats Models.

        Frechet distance is def as || mean_r - mean_c ||2 + TR(cov_r + cov_c - 2 (cov_r @ cov_c)**1/2)
        because TR is linear we can split TR(cov_r + cov_c - 2 (cov_r @ cov_c)**1/2) = TR(cov_r) + TR(cov_c) - 2 * TR((cov_r @ cov_c)**1/2) 
        (cov_r @ cov_c)**1/2 = M **1/2 is the matrix sqare root of M and can be dev by M ** 1/2 = P * (ew **1/2) * P ** -1 so the TR of the 
        Matrix sqare root is eaqual to sum(ew **1/2) because the covarianz is positiv semidefenit all ew >= 0 but because of numeric errors
        ew could be negativ so we just take the sqare root of the ew >= 0  

        from https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/fid.py#L176-L180 

        Parameters:
        - ref_stat: 
        - comp_stat: Callable that generates comparison data samples.

        Returns:
        - Fréchet Distance between the two distributions.
    """
    mean_r, cvar_r = ref_stat.get_mean_cvar()
    mean_c, cvar_c = comp_stat.get_mean_cvar()

    if device == "torch":
        
        a = torch.sum((mean_r - mean_c)**2)
        b = torch.trace(cvar_r) + torch.trace(cvar_c)
        eigvals = torch.clamp(torch.linalg.eigvals(cvar_r @ cvar_c), min=0)
        c = torch.sqrt(eigvals).real.sum()
        dist = a + b - 2 * c
        return dist.item()

    else:
        mean_r, cvar_r = mean_r.cpu().numpy(), cvar_r.cpu().numpy()
        mean_c, cvar_c = mean_c.cpu().numpy(), cvar_c.cpu().numpy()

        a = np.sum((mean_r - mean_c)**2)
        b = np.trace(cvar_r) + np.trace(cvar_c)
        eigvals = np.clip(np.linalg.eigvals(cvar_r @ cvar_c), a_min=0, a_max=None)  # the eigenvalues theoretically can't be negative but because of numeric errors they can
        c = np.sqrt(eigvals).real.sum()   # eigvals faster than matrix square root but still O(N^3)
        
        return a + b - 2 * c