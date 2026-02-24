"""
    Purpose
        Minimal exporter that:
        - Rebuilds the training model (Discretizer + U-Net + DiffusionModel) from a config.
        - Scans `<output_dir>/models/` for checkpoints (e.g., `model_50000_batches.pt`).
        - Samples N tensors per checkpoint and optionally saves them to samples_model/<checkpoint_name>/samples.pt.
        - Optionally computes (pseudo) log-likelihoods and writes a CSV to `results_train_loglike.csv`.

    Requirements
    - Project modules `RadarDataGen.*` must be importable (same environment as training).
    - Checkpoints located under `<output_dir>/models/`.
    - Config JSON with the training parameters.

    Examples:
        1) Sample and save (default):
            python3 ./experiments/results/Diff_model_train/plots/generate_samples_from_checkpoints.py --config ./experiments/results/Diff_model_train/u_net/exp_medium_model_x0/setting.json --num_samples 100
                
        2) With (pseudo) log-likelihood:
            python3 ./experiments/results/Diff_model_train/plots/generate_samples_from_checkpoints.py --config ./experiments/results/Diff_model_train/u_net/exp_medium_model_x0/setting.json --num_samples 1000 --no-save-samples --loglike-calc

        3) Sample and save for DiT model:
            python3 ./experiments/results/Diff_model_train/plots/generate_samples_from_checkpoints.py --config experiments/results/Diff_model_train/dit/test_dit_x0/setting.json --num-samples 100 

        4) With (pseudo) log-likelihood for DiT model:
            python3 ./experiments/results/Diff_model_train/plots/generate_samples_from_checkpoints.py --config ./experiments/results/Diff_model_train/dit/test_dit_x0/setting.json --num_samples 1000 --no-save-samples --loglike-calc    

"""

# stdlib
import argparse
import json
import math
import os
import re
from typing import Optional, List

# third-party
import torch
from tqdm import tqdm
import pandas as pd

# project
from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer
from RadarDataGen.Models.UNet.u_net import U_Net
from RadarDataGen.Models.DIT.dit import DiT
from RadarDataGen.Models.DiffusionModell.diff_model import DiffusionModel

# optional fallback sampler
try:
    from experiments.utils_main_model_training_FRD import make_iterable_sampler
    HAS_UTILS_SAMPLER = True
except Exception:
    HAS_UTILS_SAMPLER = False


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_checkpoints(models_dir: str, pattern: str = r"^model_\d+_batches(?:_crash)?\.pt$") -> List[str]:
    if not os.path.isdir(models_dir):
        print(f"[WARN] models dir not found: {models_dir}")
        return []
    rx = re.compile(pattern)
    files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if rx.match(f)]

    def _key(p: str) -> int:
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    return sorted(files, key=_key)


def build_model_from_config(config: dict, device: str) -> DiffusionModel:
    channels = 4
    grid_size = int(config["discretizer_params"]["grid_size"])
    data_shape = [channels, grid_size, grid_size]

    from RadarDataGen.Models.DiffusionModell.frechet_radar_dif_loss import (
        weighted_mse_loss,
        mixed_ce_mse_loss_x0_pred,
        presence_aware_ce_mse_loss_x0_pred,
        presence_aware_weighted_mse_loss,
    )

    loss_cfg = config.get("diffusion_model", {}).get("loss", {}) or {}
    loss_type = (loss_cfg.get("loss_type", "mse") or "mse").lower()
    loss_args = loss_cfg.get("args", {})

    processed_args = {}
    for k, v in loss_args.items():
        if isinstance(v, (list, tuple)):
            processed_args[k] = torch.tensor(v, device=device, dtype=torch.float32)
        elif isinstance(v, torch.Tensor):
            processed_args[k] = v.to(device=device, dtype=torch.float32)
        else:
            processed_args[k] = v

    if loss_type == "mse":
        loss_fn = weighted_mse_loss
    elif loss_type == "weighted_mse":
        loss_fn = lambda real, pred, batch_weights=None: weighted_mse_loss(
            real=real, pred=pred, batch_weights=batch_weights, **processed_args
        )
    elif loss_type == "cross_mse":
        loss_fn = lambda real, pred, batch_weights=None: mixed_ce_mse_loss_x0_pred(
            real=real, pred=pred, batch_weights=batch_weights, **processed_args
        )
    elif loss_type == "presence_aware_weighted_mse":
        loss_fn = lambda real, pred, batch_weights=None: presence_aware_weighted_mse_loss(
            real=real, pred=pred, batch_weights=batch_weights, **processed_args
        )
    elif loss_type == "presence_aware_cross_mse":
        loss_fn = lambda real, pred, batch_weights=None: presence_aware_ce_mse_loss_x0_pred(
            real=real, pred=pred, batch_weights=batch_weights, **processed_args
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    prediction_type = config.get("diffusion_model", {}).get("prediction_type", "x0")

    function_approximator = None
    if isinstance(config.get("diffusion_model", {}).get("u_net", False), dict):
            function_approximator = U_Net(
                input_chanels=data_shape[0],
                channels_per_level= config["diffusion_model"]["u_net"]["channels_per_level"],
                time_embedding_dim=config["diffusion_model"]["mlp_time_embeding_dim"],
                resnet_blocks_per_depth= config["diffusion_model"]["u_net"]["resnet_blocks_per_depth"],
                attention_levels= config["diffusion_model"]["u_net"]["attention_levels"],
                device=device
            )
            time_embeding_dim = config["diffusion_model"]["mlp_time_embeding_dim"]
    elif isinstance(config.get("diffusion_model", {}).get("dit", False), dict):
        function_approximator = DiT(
            in_channels=data_shape[0],
            input_size = data_shape[1:],
            patch_size = config["diffusion_model"]["dit"]["patch_size"],
            hidden_size = config["diffusion_model"]["dit"]["hidden_size"],
            depth = config["diffusion_model"]["dit"]["depth"],
            num_heads = config["diffusion_model"]["dit"]["num_heads"],
            mlp_ratio = config["diffusion_model"]["dit"]["mlp_ratio"],
            enable_routing = config["diffusion_model"]["dit"]["enable_routing"],
            num_classes = 0,                                                                                    # Because we don't have labels
            cond_mode = None,
            device=device
        )
        time_embeding_dim = config["diffusion_model"]["dit"]["hidden_size"]
    else:
        raise ValueError("Please specify a model in the config (u_net or dit)")

    dif_mod = DiffusionModel(
        function_approximator=function_approximator,
        prediction_type=prediction_type,
        output_data_shape=data_shape,
        time_steps=config["diffusion_model"]["time_steps"],
        sinus_time_embeding_dim=config["diffusion_model"]["sinus_time_embeding_dim"],
        mlp_time_embeding_dim=time_embeding_dim,
        schedule_type=config["diffusion_model"]["schedule_type"],
        loss_func=loss_fn,
        device=device,
    )

    return dif_mod


def sample_tensor(
    dif_mod: DiffusionModel,
    total_samples: int,
    batch_size: int,
    sampling_args: Optional[dict] = None,
) -> torch.Tensor:
    sampling_args = sampling_args or {}
    batches = []
    n_batches = math.ceil(total_samples / batch_size)

    if hasattr(dif_mod, "sample") and callable(getattr(dif_mod, "sample")):
        for b in tqdm(range(n_batches), desc="Sampling"):
            cur = min(batch_size, total_samples - b * batch_size)
            with torch.no_grad():
                out = dif_mod.sample(num_samples=cur, batch_size=cur, **sampling_args)
            batches.append(out.detach().cpu())
    elif HAS_UTILS_SAMPLER:
        for out in tqdm(make_iterable_sampler(dif_mod, n_batches, batch_size=batch_size), desc="Sampling"):
            batches.append(out.detach().cpu())
    else:
        raise RuntimeError("No sampling method available")

    cat = torch.cat(batches, dim=0)
    return cat[:total_samples]


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint samples (torch tensor only)")
    parser.add_argument("--config", "-c", required=True, help="Path to config.json")
    parser.add_argument("--num_samples", "--num-samples", dest="num_samples", type=int, required=True,
                        help="Number of samples to export")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--save-samples", "--save_samples", dest="save_samples", action="store_true",
                   help="Save samples as .pt (default)")
    g.add_argument("--no-save-samples", "--dont_save_samples", dest="save_samples", action="store_false",
                   help="Do not save samples")
    parser.set_defaults(save_samples=True)

    parser.add_argument("--loglike-calc", "--loglike_calc", dest="loglike_calc",
                        action="store_true", default=False,
                        help="Compute and append (pseudo) log-likelihood CSV")

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    output_dir = os.path.dirname(config_path)
    models_dir = os.path.join(output_dir, "models")
    samples_root = os.path.join(output_dir, "samples_model")
    ensure_dir(samples_root)

    config = load_config(config_path)

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device = {device}")

    batch_size = int(config.get("batch_size", 64))
    sampling_args = config.get("diffusion_model", {}).get("sampling_args", {}) or {}
    export_num_samples = int(args.num_samples)

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    radar_disc = RadarDiscretizer(**config["discretizer_params"])
    dif_mod = build_model_from_config(config, device=device)

    ckpts = list_checkpoints(models_dir)
    if not ckpts:
        print(f"[WARN] No checkpoints found under: {models_dir}")
        return

    print(f"[INFO] Found checkpoints: {len(ckpts)}")
    for i, p in enumerate(ckpts, 1):
        print(f"  {i:02d}. {os.path.basename(p)}")

    ll_df, ll_path = None, None
    if args.loglike_calc:
        ll_path = os.path.join(output_dir, "result_train_loglike.csv")
        ll_df = pd.DataFrame(columns=["batches", "log_likelihood", "neg_loglike"])
        try:
            from RadarDataGen.Metrics.log_likelihood import log_likelihood_pseudo_radar_points
        except Exception as e:
            print(f"[WARN] Cannot import log_likelihood_pseudo_radar_points: {e}")
            print("[WARN] Skipping loglike calculation.")
            args.loglike_calc = False

    for ckpt_path in ckpts:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_dir = os.path.join(samples_root, ckpt_name)
        ensure_dir(out_dir)

        if hasattr(dif_mod, "load") and callable(getattr(dif_mod, "load")):
            dif_mod.load(path=ckpt_path)
        else:
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                dif_mod.function_approximator.load_state_dict(state["state_dict"])
            else:
                dif_mod.function_approximator.load_state_dict(state)

        print(f"[INFO] Sampling {export_num_samples} examples from {ckpt_name} (batch_size={batch_size})")
        samples = sample_tensor(
            dif_mod=dif_mod,
            total_samples=export_num_samples,
            batch_size=batch_size,
            sampling_args=sampling_args
        )

        if args.loglike_calc and ll_df is not None:
            try:
                from RadarDataGen.Metrics.log_likelihood import log_likelihood_pseudo_radar_points
                valid_threshold = 0.6
                num_workers = 4

                point_clouds = []
                for i in range(samples.shape[0]):
                    pts = radar_disc.grid_to_points(samples[i], valid_threshold=valid_threshold)
                    point_clouds.append(pts)

                ll = log_likelihood_pseudo_radar_points(
                    config=config,
                    point_clouds=point_clouds,
                    num_workers=num_workers,
                    tau=0.10
                )

                m = re.search(r"(\d+)", ckpt_name)
                batches_int = int(m.group(1)) if m else -1

                ll_df = pd.concat([
                    ll_df,
                    pd.DataFrame({"batches": [batches_int], "log_likelihood": [ll], "neg_log_likelihood": [-ll]})
                ], ignore_index=True)

                print(f"[INFO] Loglike calc for model trained for {batches_int} was {ll}")

            except Exception as e:
                print(f"[WARN] Loglike calc skipped for {ckpt_name}: {e}")

        if args.save_samples:
            save_path = os.path.join(out_dir, "samples.pt")
            torch.save(samples, save_path)
            print(f"[OK] Saved: {save_path} | shape={tuple(samples.shape)} | dtype={samples.dtype}")

    if args.loglike_calc and ll_df is not None and len(ll_df) > 0:
        os.makedirs(os.path.dirname(ll_path), exist_ok=True)
        ll_df.to_csv(ll_path, index=False)
        print(f"[OK] Wrote loglike CSV: {ll_path}")


if __name__ == "__main__":
    main()
