# std lib imports
import math

# 3 party import
import numpy as np
import torch
import pandas as pd

# projekt imports
from RadarDataGen.Statistic.onlineStat import OnlineStats


# Test Params ########################################

feature_dim = 5
num_samples = 10_000
device = "cpu"
batch_size = 128

true_mean = np.array([1, 2, 3, 4, 5])
true_cov = np.array([
    [1.0, 0.5, 0.3, 0.2, 0.1],
    [0.5, 1.5, 0.4, 0.3, 0.2],
    [0.3, 0.4, 2.0, 0.5, 0.3],
    [0.2, 0.3, 0.5, 1.2, 0.4],
    [0.1, 0.2, 0.3, 0.4, 1.8]
])

######################################################


if __name__ == "__main__":
    # Generate data
    samples = np.random.multivariate_normal(true_mean, true_cov, size=num_samples)
    samples_tensor = torch.tensor(samples, dtype=torch.float32)

    # Online stats
    stats = OnlineStats(feature_dim=feature_dim, device=device)

    # Logging
    log_rows = []

    for i in range(0, num_samples, batch_size):
        batch = samples_tensor[i:i + batch_size]
        stats.update(batch)

        seen = i + batch.shape[0]
        current_samples = samples[:seen]

        np_mean = current_samples.mean(axis=0)
        torch_mean = samples_tensor[:seen].mean(dim=0).numpy()
        online_mean = stats.mean.cpu().numpy().flatten()

        if seen >= 2:
            online_cov = stats.get_mean_cvar()[1].cpu().numpy()

            np_cov = np.cov(current_samples.T, bias=False)

            torch_cov = torch.cov(samples_tensor[:seen].T).numpy()

            err_cov_np = np.linalg.norm(online_cov - np_cov, ord="fro")
            err_cov_torch = np.linalg.norm(online_cov - torch_cov, ord="fro")
            err_cov_real = np.linalg.norm(online_cov - true_cov, ord="fro")

            err_cov_np_real = np.linalg.norm(np_cov - true_cov, ord="fro")
            err_cov_torch_real = np.linalg.norm(torch_cov - true_cov, ord="fro")
        else:
            err_cov_np = err_cov_torch = err_cov_real = np.nan
            err_cov_np_real = err_cov_torch_real = np.nan

        err_np = np.linalg.norm(online_mean - np_mean)
        err_torch = np.linalg.norm(online_mean - torch_mean)
        err_real = np.linalg.norm(online_mean - true_mean)

        err_np_real = np.linalg.norm(np_mean - true_mean)
        err_torch_real = np.linalg.norm(torch_mean - true_mean)

        log_rows.append({
            "num_samples": seen,

            # mean errors
            "err_mean_online_vs_numpy": err_np,
            "err_mean_online_vs_torch": err_torch,
            "err_mean_online_vs_real": err_real,
            "err_mean_numpy_vs_real": err_np_real,
            "err_mean_torch_vs_real": err_torch_real,

            # covariance errors
            "err_cov_online_vs_numpy": err_cov_np,
            "err_cov_online_vs_torch": err_cov_torch,
            "err_cov_online_vs_real": err_cov_real,
            "err_cov_numpy_vs_real": err_cov_np_real,
            "err_cov_torch_vs_real": err_cov_torch_real,
        })


    # Create DataFrame
    df = pd.DataFrame(log_rows)

    online_mean, online_cov = stats.get_mean_cvar()
    online_mean = online_mean.cpu().numpy().flatten()
    online_cov = online_cov.cpu().numpy()

    np_cov = np.cov(samples.T, bias=False)
    torch_cov = torch.cov(samples_tensor.T).numpy()

    print("========== FINAL COMPARISON ==========")
    print(f"Mean error (Online vs NumPy):  {np.linalg.norm(online_mean - np_mean):.6e}")
    print(f"Cov  error (Online vs NumPy):  {np.linalg.norm(online_cov - np_cov, ord='fro'):.6e}")
    print()
    print("Real Mean:", true_mean)
    print("Online Mean:", online_mean)
    print("NumPy  Mean:", np_mean)
    print()
    print("Real Cov:\n", true_cov)
    print("Online Cov:\n", online_cov)
    print("NumPy  Cov:\n", np_cov)

    print("\n========== LOG DATAFRAME (head) ==========")
    print(df.head())

    print("\n========== LOG DATAFRAME (tail) ==========")
    print(df.tail())
