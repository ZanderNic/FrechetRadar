# std lib imports
import random
import os

# 3 party import
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow_datasets as tfds
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

# projekt imports
from RadarDataGen.Models.DiffusionModell.diff_model import DiffusionModel
from RadarDataGen.Models.UNet.u_net import U_Net
from RadarDataGen.Models.DIT.dit import DiT

# test utils plot import 
from plot_utils import *

class FlowersTorchDataset(Dataset):
    def __init__(self, split="train", img_size=64):
        self.ds = list(tfds.as_numpy(tfds.load("oxford_flowers102", split=split, as_supervised=True)))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1]
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        img, _ = example
        img = Image.fromarray(img)
        img = self.transform(img)
        return img


def get_dataloader(batch_size=16, img_size=64):
    dataset = FlowersTorchDataset(split="train", img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def sample_batch(dataset, batch_size, device="cuda"):
    indices = random.sample(range(len(dataset)), batch_size)
    batch = torch.stack([dataset[i] for i in indices])
    
    return batch.to(device)


if __name__ == "__main__":

    ###  Setting  #####################
    model_type =  "dit"  # "u-net" or "dit"
    
    output_data_shape=(3, 64, 64)
    time_embedding_dim = 256
    batch_size = 8
    train_objektive = "x0" # eather "x0" or "eps" 
    device= "cuda"
    output_dir = "./tests/test_dif_flowers/"
    model_final_path = output_dir + "/samples_model/model_final_" + model_type + "/"
    train_info_path = output_dir + "/train_info_" + model_type + "/"

 
    ###################################

    os.makedirs(model_final_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_info_path, exist_ok=True)

    train_loader = get_dataloader(batch_size=batch_size)
    test_loader = get_dataloader(batch_size=batch_size)

    if model_type == "u-net":
        function_approximator = U_Net(
            input_chanels = output_data_shape[0],
            time_embedding_dim= time_embedding_dim,
            channels_per_level= [32, 64, 128, 256],
            attention_levels= [0, 0, 1, 1],
            resnet_blocks_per_depth= 1,
            device=device
        )
    elif model_type == "dit":
        function_approximator = DiT(
            in_channels= output_data_shape[0],
            input_size = output_data_shape[1:],
            patch_size = 4,
            hidden_size = 384,
            depth = 9,
            num_heads = 6,
            mlp_ratio = 3,
            enable_routing = False,
            num_classes = 0,                                                                                    # Because we don't have labels
            cond_mode = None,
            device=device
        )
    else:
        raise ValueError("error please use vit or u-net")

    model = DiffusionModel(
        function_approximator=function_approximator,
        prediction_type= train_objektive,
        output_data_shape=output_data_shape,
        time_steps=1_000,
        mlp_time_embeding_dim= time_embedding_dim if model_type == "u-net" else 384,
        sinus_time_embeding_dim=128,
        schedule_type="cosine",
        device=device
    )

    num_train_batches = 2_000
    repeat = 20
    train_history_df = pd.DataFrame()

    model.load("./tests/test_dif_flowers/model_dit/model.pt")

    for i in range(repeat):
        
        i = i + 10

        history = model.train_model(
            sample_batch = lambda _: sample_batch(train_loader.dataset, batch_size=batch_size, device=device),
            num_train_batches= num_train_batches,
            batch_size= batch_size,
            log_train_loss_per_batch= 100,
            sample_test_batch = lambda _: sample_batch(test_loader.dataset, batch_size=batch_size, device=device),
            log_test_loss_per_batch= 500,
        )

        history["Batch"] = history["Batch"] + num_train_batches * i 

        train_history_df =  pd.concat([train_history_df, history])

        path = output_dir + f"/samples_model/model_{model_type}_{num_train_batches * (i + 1)}_batches/"
        save_samples_current_model_images(diff_model=model, path=path)


    # save model and train hist
    model.save(output_dir + f"/model_{model_type}/model.pt")
    train_history_df.to_csv( output_dir + f"/train_info/train_history{model_type}_df.csv")

    samples_ddim = model.sample(4, sampler="ddim", sample_time_steps=25, eta=0, use_ema_model=True) 
    visualize_and_save(samples_ddim, output_path = model_final_path + "sample_ddim.png")
    
    samples_ddpm = model.sample(4, sampler="ddpm", use_ema_model=True) 
    visualize_and_save(samples_ddpm, output_path = model_final_path + "sample_ddpm.png")

    plot_training_progress_images(output_dir + "/samples_model/", train_info_path + "sampled_images.png", model_type=model_type) 
    plot_loss_curve(train_history_df, train_info_path + "loss_plot.png", alpha=0.95)
        
