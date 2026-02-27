# RadarDataGen

RadarDataGen is a research-oriented toolbox for generating synthetic radar point clouds, discretizing them into grid representations, training U-Net diffusion models on those grids, and evaluating distributional similarity using Fréchet Radar Distance (FRD) computed with random projections.

The repository is structured around a clean src implementation (generators, discretizer, metrics, models, statistics) and separate experiment scripts driven by JSON configs.

----------------------------------------------------------------

## Installation

Recommended workflow: create a virtual environment, activate it, and install the package in editable mode.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Notes:
- pip install -e . installs the project in editable mode, so changes in src/ are immediately reflected without reinstalling.
- PyTorch installation depends on your system and CUDA setup. If PyTorch is not installed yet, install it first (CPU or CUDA build) and then run pip install -e .

----------------------------------------------------------------

##  Source Code Overview (src/) 

###  RadarDataGen/Data_Generator/

Contains parametric pseudo-radar generators and dataset wrappers used for experiments and model training.

Key components:
- pseudo_radar_points(...)
  Compositional point-cloud generator combining multiple geometric primitives:
  - 2D/3D lines (Poisson number of lines + Poisson points per line)
  - 2D rectangles (filled) and rectangle outlines
  - 2D circles (filled)
  - uniform clutter
  Output is a shuffled array of points with shape (N, 3) (e.g., x, y, feature/color).

- PseudoRadarGridGenerator
  Thin wrapper that:
  1) samples a point cloud via pseudo_radar_points
  2) optionally clips it to the discretizer bounding box
  3) converts it into a grid via RadarDiscretizer.points_to_grid(...)
  Output shape is (H, W, C) (later converted to (C, H, W) for PyTorch).

- Datasets
  - StreamingRadarDataset: infinite stream of freshly generated samples (worker-seeded for multiprocessing)
  - RadarDataset: stream sampling from a fixed range of seeds (useful when you want bounded “dataset size” behavior)

These datasets are designed to work with PyTorch DataLoader and multiple workers.

----------------------------------------------------------------

### RadarDataGen/Discretizer/

Implements the radar discretization logic that maps continuous point clouds to fixed-size grids.

Key components:
- RadarDiscretizer
  - Converts (N, D) points into a grid of shape (grid_size, grid_size, 1 + D)
  - Encodes each occupied cell as:
    - valid_indicator (presence flag)
    - offsets (dx, dy) relative to the cell center (in grid coordinates)
    - remaining point channels (e.g., color/rcs/etc.)
  - Handles collisions (multiple points mapping to the same cell) by relocating duplicates to nearby empty cells via a linear assignment (Hungarian algorithm) to minimize total squared distance.

- GridNeighbors
  Neighbor caching for expanding search radii (circle or square neighborhoods), used to find candidate empty cells during collision resolution.

----------------------------------------------------------------

### RadarDataGen/Metrics/

Contains evaluation metrics used in experiments:
- Random projections (feature extraction for FRD)
- Fréchet distance computation between Gaussian statistics (mean/cov)
- Log-likelihood utilities for pseudo radar point distributions

----------------------------------------------------------------

### RadarDataGen/Models/

Contains the complete diffusion implementation written in PyTorch:
- U-Net building blocks (ResNet blocks, attention, time embeddings)
- Diffusion forward process and schedulers
- Multiple samplers (e.g., DDIM-style sampling)
- Training objectives (e.g., weighted MSE, presence-aware losses, mixed CE/MSE variants)

----------------------------------------------------------------

### RadarDataGen/Statistics/

Provides streaming statistics for large-scale evaluation:
- OnlineStats implements numerically stable online mean/covariance updates (Chan–Golub–LeVeque / batched Welford).
  This is used to compute Gaussian statistics for FRD without storing all samples in memory.


----------------------------------------------------------------

## Experiments

This project contains several experiment scripts that build on the core modules
in src/ and are fully controlled via JSON configuration files. The experiments
are designed to be reproducible, scalable, and resumable via checkpointing.

The main purpose of the experiments is to evaluate synthetic radar data at the
distribution level and to study the training dynamics of diffusion models using
Fréchet Radar Distance (FRD), random projections, and log-likelihood diagnostics.

For a detailed explanation of experiment configuration, parameters, and example
configs, see the dedicated experiment README in the experiments directory.

----------------------------------------------------------------

### FRD Generator Analysis Experiments

The generator analysis experiments compare parametric pseudo-radar generators by
measuring how close their induced data distributions are.

Two execution modes are provided:

- Cumulative sampling experiments generate a large dataset once and reuse
  increasing prefixes for different sample sizes. This mode is significantly
  faster and is recommended for large-scale experiments.

- Resampling experiments regenerate fresh samples for each sample size. This
  mode is slower but provides strictly independent datasets for each evaluation
  point.

Both experiment types support multiple random projection dimensions and multiple
independent runs (num_trys). Results are written incrementally to CSV files and
can be resumed after interruption using checkpoint files.

----------------------------------------------------------------

### Diffusion Model Training Experiments

The diffusion training experiments train a U-Net–based diffusion model on
discretized radar grids generated from the pseudo-radar generators.

Training proceeds incrementally in predefined stages. After each training stage,
the model is evaluated by:
- Sampling synthetic radar grids from the diffusion model
- Computing FRD against reference generator statistics
- Optionally computing log-likelihood estimates on point-cloud samples

This setup makes it possible to track distributional convergence over training
time instead of relying solely on loss curves.

----------------------------------------------------------------

## Notebooks

The repository also contains Jupyter notebooks intended for exploration and
documentation rather than large-scale experiments.

Typical notebooks include:

- A visualization notebook demonstrating the behavior of different
  pseudo_radar_points configurations and geometric primitives.

- A dataset usage notebook showing how to work with:
  - StreamingRadarDataset for infinite, on-the-fly data generation
  - RadarDataset for reproducible sampling from a fixed seed range

The notebooks are meant as interactive examples and sanity checks and are not
used by the automated experiment pipelines.


----------------------------------------------------------------


## Tests

The repository contains a set of lightweight test scripts that validate the
correctness and numerical behavior of some components. These tests are not
meant as exhaustive unit tests but as focused sanity checks for individual
modules.

### The tests cover the following aspects:

#### Diffusion Model Tests 
- A minimal end-to-end example showing how a diffusion model can be trained on
  simple image data.
- Verification that the forward and reverse diffusion processes run without
  numerical instabilities.
- Basic checks that training loss decreases and sampling produces valid outputs.

#### Log-Likelihood Tests
- Validation of the log-likelihood computation for pseudo radar point clouds.
- Comparison of likelihood values under different generator configurations.
- Sanity checks for numerical stability and reproducibility.

#### Discretizer Tests
- Tests for correct mapping from point clouds to grid representation.
- Verification of collision handling when multiple points fall into the same
  grid cell.
- Round-trip tests converting grid representations back to point clouds.

#### FRD and Random Projection Tests
- Validation of random projection generation and dimensionality handling.
- Tests for Fréchet Radar Distance computation given known statistics.
- Checks for edge cases such as small sample sizes and degenerate covariance
  scenarios.

#### Online Statistics Tests
- Verification of online mean and covariance updates.
- Consistency checks against batch-computed statistics.
- Tests for correct behavior when processing data in multiple batches.

Overall, the test files provide confidence that individual components behave as
expected and can be safely composed in the larger experiment pipelines.

-------------------

## Acknowledgements
This project was developed as part of my dual study program and within the scope of my Bachelor’s thesis at Aumovio. The work of Jörg Reichardt and Jonas Neuhofer, carried out in the context of the nxtAIM project, provided an essential foundation for this repository and significantly supported its development. Their prior implementation and conceptual contributions greatly facilitated the extension and refinement of several components. I would particularly like to thank Jörg Reichardt for supervising this thesis and for his continuous guidance and valuable feedback throughout the project.

The implementation of the DIT (Diffusion Image Transformer) is based on the work provided at
https://github.com/compvis/tread .

I gratefully acknowledge all contributors whose prior work and support made this project possible.
