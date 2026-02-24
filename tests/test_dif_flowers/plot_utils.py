# std lib imports
import os
import math
import re

# 3 party import
import torch
import matplotlib.pyplot as plt

# projekt imports
from RadarDataGen.Models.DiffusionModell.diff_model import DiffusionModel


def ema(values, alpha=0.95):
    ema_values = []
    prev = values.iloc[0]
    for v in values:
        prev = alpha * prev + (1 - alpha) * v
        ema_values.append(prev)
    return ema_values


def plot_loss_curve(train_history_df, output_path, alpha=0.95):
    fig, ax = plt.subplots(figsize=(10, 8))

    train_loss_ema = ema(train_history_df["Train_Loss"], alpha=alpha)
    test_loss_ema = ema(train_history_df[train_history_df["Test_Loss"].notna()]["Test_Loss"], alpha=alpha)

    ax.plot(train_history_df["Batch"], train_loss_ema, label=f'Train Loss (EMA α={alpha})', color="#8905be")
    ax.plot(train_history_df[train_history_df["Test_Loss"].notna()]["Batch"], test_loss_ema, label=f'Test Loss (EMA α={alpha})', color="#FF4208")

    ax.set_title('Train vs Test Loss')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def visualize_and_save(samples, output_path="samples_grid.png"):
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu()

    samples = (samples + 1) / 2
    count_samples = samples.shape[0]

    cols = math.ceil(math.sqrt(count_samples))
    rows = math.ceil(count_samples / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), tight_layout=True)
    ax = ax.flatten()

    for idx in range(count_samples):
        img = samples[idx].permute(1, 2, 0).numpy()
        ax[idx].imshow(img)
        ax[idx].axis("off")

    for j in range(count_samples, len(ax)):
        ax[j].axis("off")

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def save_samples_current_model_images(
    diff_model: DiffusionModel,
    path: str,
    sampler: str = "ddim",
    num_samples_tensors: int = 1,
    num_samples_images: int = 4
):
    os.makedirs(path, exist_ok=True)

    num_samples_to_gen = max(num_samples_tensors, num_samples_images)
    samples = diff_model.sample(num_samples_to_gen, sampler=sampler, sample_time_steps=250, use_ema_model=True).cpu()
    samples = (samples + 1) / 2

    if num_samples_tensors > 0:
        tensor_save_path = os.path.join(path, "samples_raw_tensor.pt")
        torch.save(samples, tensor_save_path)

    if num_samples_images > 0:
        img_save_path = os.path.join(path, "samples_images.png")
        visualize_and_save(samples[:num_samples_images], output_path=img_save_path)



def plot_training_progress_images(path_samples: str, output_path: str, model_type : str ):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    pattern = re.compile(rf"^model_{model_type}_(\d+)_batches$")
    folders = []

    for name in os.listdir(path_samples):
        match = pattern.match(name)
        if match:
            number = int(match.group(1))
            folders.append((number, os.path.join(path_samples, name)))

    folders.sort(key=lambda x: x[0])
    count_samples = len(folders)

    cols = math.ceil(math.sqrt(count_samples))
    rows = math.ceil(count_samples / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), tight_layout=True)
    ax = ax.flatten()

    for idx, (num, folder) in enumerate(folders):
        for file in os.listdir(folder):
            if file.endswith(".pt"):
                samples = torch.load(os.path.join(folder, file))
                samples = (samples + 1) / 2
                img = samples[0].permute(1, 2, 0).numpy()
                ax[idx].imshow(img)
                ax[idx].set_title(f"{num} batches")
                ax[idx].axis("off")
                break

    for j in range(count_samples, len(ax)):
        ax[j].axis("off")

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)