# RadarDataGen

RadarDataGen provides experimental pipelines for synthetic radar data generation,
distribution-level evaluation, and diffusion-model training.
All experiments are fully configuration-driven, reproducible, and designed for
large-scale analysis using the Fréchet Radar Distance (FRD).

---

## Experiments

The repository contains two main experiment classes.

### FRD Generator Analysis

These experiments compare parametric pseudo-radar generators by measuring the
distance between their induced data distributions.

Two sampling strategies are implemented:

- Cumulative Sampling (main_cumulativ_sampling_exp_FD.py)  
  A large dataset is generated once and progressively larger prefixes are used
  for evaluation. This approach is significantly faster and supports checkpointing.

- Resampling (main_resampling_exp_FD.py)  
  New data is generated for each sample size. This approach is statistically clean
  but substantially slower.

Cumulative sampling is recommended for most experiments.

---

### Diffusion Model Training and Evaluation

This experiment trains a U-Net–based diffusion model on discretized radar grids.
The model is evaluated during training using FRD, multiple random projection
dimensions, different dataset sizes, and optional log-likelihood estimation.

Training is performed incrementally, with evaluations at predefined batch counts.

---

## Requirements

Python >= 3.9  
PyTorch (CUDA optional, recommended)  
NumPy, Pandas  
Matplotlib  
tqdm  

GPU acceleration is strongly recommended for diffusion training.
Large RAM is recommended for large sample sizes and projection dimensions.

---

## Running Experiments

FRD Generator Analysis (Cumulative Sampling):
```bash
python3 ./experiments/main_cumulativ_sampling_exp_FD.py --config path/to/config.json
```

FRD Generator Analysis (Resampling):

```bash
python3 ./experiments/main_resampling_exp_FD.py --config path/to/config.json

```

Diffusion Model Training:
```bash
python3 experiments/main_diffusion_training_exp.py --config path/to/config.json

```

All outputs are written to the same directory as the configuration file.

---

## Configuration Files

All experiments are controlled via a single JSON configuration file.

### Common Parameters

```json
device: auto | cpu | cuda  
num_workers: number of parallel workers  
batch_size: batch size for sampling or training  
num_trys: number of independent repetitions  
```
---

### Discretizer Parameters

Define how point clouds are discretized into radar grids.
```json
grid_size  
x_min, x_max  
y_min, y_max  
valid_indicator  
```
---

### FRD Generator Analysis Configuration

reference_generators define the target distribution.  
comparison_generators are evaluated against the reference.
random_projection_dim must be strictly smaller than sample_size.

---

### Diffusion Model Training Configuration

data_generator defines the ground-truth data distribution.  

dataset type:
- streaming: data generated on-the-fly
- fixed: pre-generated dataset (requires dataset_size)

---

### U-Net Architecture

channels_per_level  
resnet_blocks_per_depth  
attention_levels  

---

### Diffusion Process

time_steps  
schedule_type  
prediction_type  

---

### Loss Configuration

Supported loss types:
- weighted_mse
- cross_mse
- presence_aware_weighted_mse
- presence_aware_cross_mse

---

### Training and Evaluation Control

training_evaluation_batches  
checkpoint_models  
save_samples  

---

## Outputs

FRD Experiments:
- results.csv
- checkpoints.json

Diffusion Training:
- result_resampling.csv
- result_train_info.csv
- models/
- samples_model/

---

## Intended Use

RadarDataGen is intended for research-grade experiments on synthetic radar data,
including generator comparison, sample-size analysis, and diffusion-model
convergence evaluation.