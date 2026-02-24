# std lib imports
import argparse
import json
import os
import math

# 3 party import
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# project imports
from RadarDataGen.Data_Generator.generator import PseudoRadarGridGenerator
from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer

from RadarDataGen.Metrics.frechet_distance import frechet_distance_stats
from RadarDataGen.Metrics.random_projections import RandomProjektions
from RadarDataGen.Metrics.log_likelihood import log_likelihood_pseudo_radar_points

from RadarDataGen.Models.DiffusionModell.diff_model import DiffusionModel
from RadarDataGen.Models.UNet.u_net import U_Net
from RadarDataGen.Models.DIT.dit import DiT
from RadarDataGen.Models.DiffusionModell.frechet_radar_dif_loss import weighted_mse_loss, mixed_ce_mse_loss_x0_pred, presence_aware_ce_mse_loss_x0_pred, presence_aware_weighted_mse_loss

# utils import 
from utils_main_model_training_FRD import *


"""
    This script runs a full experimental pipeline for training and evaluating a
    U-Net-based diffusion model on synthetic pseudo-radar grid data.

    The experiment is fully configuration-driven via a JSON file and includes:
    - Data generation using parametric pseudo-radar generators
    - Online estimation of reference statistics from ground-truth generators
    - Training of a diffusion model in multiple staged intervals
    - Periodic sampling from the trained model
    - Evaluation using Fréchet Radar Distance with random projections
    - Optional log-likelihood estimation of generated samples
    - Checkpointing to allow safe resumption after interruptions

    For each data generator configuration, the diffusion model is trained incrementally.
    After each training stage, model samples are generated and compared against the
    reference data distribution using multiple random projection dimensions and
    dataset sizes. Results are continuously written to CSV files.

    The script supports different diffusion loss variants, prediction types, and
    training schedules, and is designed for long-running, reproducible experiments.

    Command-line arguments
        --config, -c : str (required)
            Path to the JSON configuration file defining data generators, model
            architecture, training schedule, evaluation settings, and output options.

    Outputs
        - result_resampling.csv
            Contains Fréchet distance results for different training stages, projection
            dimensions, and sample sizes.
        - result_train_info.csv
            Contains training and test loss logs over time.
        - checkpoints.json
            Stores experiment progress to allow resuming interrupted runs.
        - models/
            Optional saved model checkpoints.
        - samples_model/
            Optional visual and raw samples generated from the model at different
            training stages.

    Intended use
        This script is intended for systematic evaluation of generative diffusion models
        on synthetic radar-like data, enabling controlled analysis of training dynamics,
        sample quality, and distributional convergence.
"""



###  Load Experiment Params  ##################################################################################################################

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def default_checkpoint():
    return {
        "data_gen_idx": 0,
        "training_stage_idx": 0,
        "fid_calc_idx": 0,
        "completed_samples": 0
    }

def save_experiment_checkpoint(state, path):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

def load_experiment_checkpoint(path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as f:
            try:
                ckpt = json.load(f)
                base = default_checkpoint()
                base.update(ckpt)
                print("[INFO] Resuming from checkpoint:", base)

                return base
            except Exception:
                return default_checkpoint()
    return default_checkpoint()

###  Main  ####################################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RadarDataGen experiments from JSON config.")
    parser.add_argument("--config", "-c", required=True, help="Path to config.json")
    args = parser.parse_args()

    results = pd.DataFrame()

    # load config
    config = load_config(args.config)
    
    # handel device from config
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device = {device}")

    # set all the paths for the stuff we save
    output_dir = os.path.dirname(args.config)
    results_path = os.path.join(output_dir, "result_resampling.csv")
    results_path_model_train = os.path.join(output_dir, "result_train_info.csv")
    checkpoint_path = os.path.join(output_dir, "checkpoints.json")
    checkpoint_model_train_path = os.path.join(output_dir, "models/")
    tensorboard_log_dir = os.path.join(output_dir, "./runs/")

    # Get the Data Sahpe 
    channels = 4
    discretizer_grid_size = int(config["discretizer_params"]["grid_size"])
    data_shape = [channels, discretizer_grid_size, discretizer_grid_size]        # saves the shap of the data 
    data_dim = (int(config["discretizer_params"]["grid_size"]) ** 2) * channels     # saves the dimension ot the data 

    # get the different config things 
    data_generators = config["data_generator"]
    random_projections = config["random_projection_dims"]
    training_evaluation_batches = config["training_evaluation_batches"]
    sample_sizes_to_do = config["sample_sizes"]
    
    # generate a diskretizer
    radar_disc = RadarDiscretizer(**config["discretizer_params"])

    model_sampling_args = config.get("diffusion_model", {}).get("sampling_args" , {})

    # load checkpoint
    ckpt = load_experiment_checkpoint(checkpoint_path)

    os.makedirs(checkpoint_model_train_path, exist_ok=True)
    os.makedirs(output_dir + "/samples_model/", exist_ok=True)


    for data_gen_idx, data_gen in enumerate(data_generators):
        if data_gen_idx < ckpt["data_gen_idx"]:
            continue

        ramdom_pros = {} # here we save all the random pros that we need to do
        for rand_idx, random_dim in enumerate(random_projections):
            random_pro = RandomProjektions(data_dim = data_dim, feature_dim=random_dim, device=device)
            ramdom_pros[random_dim] = random_pro
        
        # get the loss_type from the config. The different loss types are "MSE", weightet mse (more weight on the first dim), cross_mse (cross entropy loss on valid dim and rest mse)
        loss_type = config.get("diffusion_model", {}).get("loss", None).get("loss_type", None)
        loss_args = config.get("diffusion_model", {}).get("loss", None).get("args", None)

        # converting lists to tensors 
        loss_args = {
            k: (torch.tensor(v, device=device, dtype=torch.float32) if isinstance(v, (list, tuple))
                else (v.to(device=device, dtype=torch.float32) if isinstance(v, torch.Tensor) else v))
            for k, v in loss_args.items()
        }

        if loss_type is None or loss_type.lower() == "mse":
            loss_dif = weighted_mse_loss
        elif loss_type.lower() == "weighted_mse":
            loss_dif = lambda real, pred, batch_weights = None: weighted_mse_loss(real=real, pred=pred, batch_weights=batch_weights, **loss_args)
        elif loss_type.lower() == "cross_mse":
            loss_dif =  lambda real, pred, batch_weights = None: mixed_ce_mse_loss_x0_pred(real=real, pred=pred, batch_weights=batch_weights, **loss_args)
        elif loss_type.lower() == "presence_aware_weighted_mse":
            loss_dif = lambda real, pred, batch_weights = None: presence_aware_weighted_mse_loss(real=real, pred=pred, batch_weights=batch_weights, **loss_args)
        elif loss_type.lower() == "presence_aware_cross_mse":
            loss_dif = lambda real, pred, batch_weights = None: presence_aware_ce_mse_loss_x0_pred(real=real, pred=pred, batch_weights=batch_weights, **loss_args)  
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # get the prediction_type of the model
        prediction_type = config.get("diffusion_model", {}).get("prediction_type", "x0")

        # create the function approximater for the diffusion model [U_Net, DIT]
        function_approximator = None
        time_embeding_dim  = 0

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

        # create diffusion modell
        dif_mod = DiffusionModel(
            function_approximator= function_approximator,
            prediction_type = prediction_type,
            output_data_shape= data_shape,
            time_steps= config["diffusion_model"]["time_steps"],
            sinus_time_embeding_dim= config["diffusion_model"]["sinus_time_embeding_dim"],
            mlp_time_embeding_dim= time_embeding_dim,
            schedule_type= config["diffusion_model"]["schedule_type"],
            loss_func = loss_dif,
            device= device
        )

        # create a writer for tensorboard 
        writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # create a model summary txt where one can see num params etc 
        create_model_summary_file(dif_mod, output_dir)

        # now we can create the sample batch function where we sample data from the data generator and that we need to calc the orginal stats
        generator = PseudoRadarGridGenerator(data_gen, discretizer=radar_disc)

        # now we check if we have a dataset that streams the data so resampling new data all the time or if we have a fixed dataset that we go trought multiple times 
        dataset, dataset_rev = get_datasets(
            dataset_type=config["dataset"]["type"],
            generator=generator,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            dataset_size=config["dataset"].get("dataset_size", None),
        )

        stream_dataset = iter(dataset) 
        stream_test_data = iter(dataset_rev) 
        # history df 
        train_history_df = pd.DataFrame(columns=["Batch", "Batch_Size", "Train_Loss", "Test_Loss"])

        # calculate the data stats
        print()
        print(f"[INFO] Generating data from the Pseudo Radar Grid generator and calculating the stats for the generated data")
        rev_statistics = generate_online_stats(
            generator= dataset_rev,
            sample_sizes= config["sample_sizes"],
            feature_extractors= ramdom_pros,
            device=device
        )

        # now we train the model step by step 
        for train_idx, total_batches in enumerate(training_evaluation_batches):
            if train_idx == 0:
                train_batches = training_evaluation_batches[0]
            else:
                train_batches = training_evaluation_batches[train_idx] - training_evaluation_batches[train_idx-1]
            
            if not (data_gen_idx == ckpt["data_gen_idx"] and train_idx < ckpt["training_stage_idx"]):
                print()
                print(f"[INFO] Diff Model was trained for {training_evaluation_batches[train_idx - 1] - 1 if train_idx >= 1 else 0} Batches ")
                print(f"[INFO] Continue Training the Diff Model for {train_batches} Batches with batch size {config['batch_size']} => {train_batches * config['batch_size']} Examples")

                try:
                    history = dif_mod.train_model(
                        sample_batch = lambda _: next(stream_dataset),
                        num_train_batches= train_batches,
                        batch_size= config["batch_size"],
                        log_train_loss_per_batch = config["diffusion_model"].get("log_train_loss_per_batch", 100),
                        sample_test_batch = lambda _: next(stream_test_data), 
                        log_test_loss_per_batch= 5_000,
                        num_test_batches_log = 100,
                        max_grad_norm = config["diffusion_model"].get("max_grad_norm", 1.0),
                        lr = config["diffusion_model"].get("learning_rate", 2e-4),
                        betas = config["diffusion_model"].get("betas", (0.9, 0.999)),
                        weight_decay = config["diffusion_model"].get("weight_decay", 0.01),
                        use_scheduler = config["diffusion_model"].get("use_scheduler", False),
                        tensorboard_writer = writer,
                        loggin_start_batch = training_evaluation_batches[train_idx - 1] if train_idx != 0 else 0
                    )
                except Exception as e:
                    print("[ERROR] Training crashed:", e)
                    crash_checkpoint_path = os.path.join(checkpoint_model_train_path, f"model_{training_evaluation_batches[train_idx]}_batches_crash.pt")
                    dif_mod.save(path=crash_checkpoint_path)
                    raise e
                
                append_df_to_csv(history, results_path_model_train)

                checkpoint_model_path = None
                if config["checkpoint_models"]:
                    checkpoint_model_path = os.path.join(checkpoint_model_train_path, f"model_{training_evaluation_batches[train_idx]}_batches.pt")
                    dif_mod.save(path=checkpoint_model_path)

                # check if we want to save samples in the form of images + raw of the model at the current train stage 
                if config["save_samples"]:
                    path_save_samples = output_dir + f"/samples_model/model_{training_evaluation_batches[train_idx]}_batches/"
                    save_samples_current_model(diff_model=dif_mod, path=path_save_samples, radar_discretizer=radar_disc)

                # udate checkpoint training 
                ckpt.update({
                    "data_gen_idx": data_gen_idx,
                    "training_stage_idx": train_idx + 1,
                    "trained_batches": total_batches,
                    "model_checkpoint_path": checkpoint_model_path,
                })
                save_experiment_checkpoint(ckpt, checkpoint_path)

            # now we need to sample points and calc our statistics for the sampled points 
            if not (data_gen_idx == ckpt["data_gen_idx"] and train_idx < ckpt["fid_calc_idx"]):
                print()
                print(f'[INFO] Generating {max(config["sample_sizes"]) // config["batch_size"]} Batches of each {config["batch_size"]} Samples and calculating ther stats for the fid calculation')
                model_statistics, log_like_data = generate_online_stats(
                    generator= make_iterable_sampler(dif_mod, math.ceil(max(config["sample_sizes"]) / config["batch_size"]), batch_size= config["batch_size"]),
                    sample_sizes= config["sample_sizes"],
                    feature_extractors= ramdom_pros,
                    device=device, 
                    radar_discretizer=radar_disc,
                    log_liklehood_computation = log_likelihood_pseudo_radar_points,
                    log_like_samples = 1000,
                    config = config,
                )
                
                print()
                print(f'[INFO] Calculating the FID')
                progress_bar = tqdm(total=len(sample_sizes_to_do) * len(random_projections), desc="Calculating the FID")  # the calculation time is independend of the dataset size because we already have the stats

                # calculate the frechet distance for the generator and the diffusion model
                for rand_idx, random_dim in enumerate(random_projections):
                    for sample_idx, sample_size in enumerate(sample_sizes_to_do):
                        if random_dim >= sample_size:
                            progress_bar.update(1)
                            continue
                        try:
                            dist = frechet_distance_stats(rev_statistics[random_dim][sample_size], model_statistics[random_dim][sample_size], device=device)

                            distance =  pd.DataFrame({
                                            "frechet_distance": [dist],
                                            "trained_batches": [training_evaluation_batches[train_idx]],
                                            "batch_size": [config["batch_size"]],
                                            "data_dim": [data_dim],
                                            "feature_dim": [random_dim],
                                            "dataset_size": [sample_size], 
                                            "loglike_data": [log_like_data],
                                            "neg_loglike_data": [-log_like_data]
                                        })

                            append_df_to_csv(distance, results_path)
                            progress_bar.update(1)

                            # update checkpoint fid calculation 
                            ckpt["fid_calc_idx"] = train_idx + 1
                            save_experiment_checkpoint(ckpt, checkpoint_path)
                            
                        except Exception as e:
                            print("[Error] in calculating the Distance from two stats objekts. Error:", e) 

                progress_bar.close() 
                writer.close()
    print()
    print(f"[END] Results saved at {results_path}")