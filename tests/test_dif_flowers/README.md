# Oxford Flowers Diffusion Model Test

## Overview
This folder contains a simple test setup for training and sampling a diffusion model on the Oxford Flowers dataset. It is intended for experimentation and demonstration purposes only.

Included files:
- requirements.txt – Python dependencies for this test.
- test_flowers.py – Example script to train and sample images.
- plot_utils.py – Helper functions for plotting training progress and generated samples.

## Requirements
Before running any scripts, ensure:
- Python 3.9+

- Install dependencies:
  pip install -r requirements.txt       (main package)
    +
  pip install -r requirements.txt       (test)

## Usage
1. Run training:
   python test_flowers.py
2. Outputs (samples and plots) will be saved in:
   ./tests/test_dif_flowers/

## Notes
- This is a minimal test
- GPU recommended for faster training.
- Adjust hyperparameters in test_flowers.py as needed.