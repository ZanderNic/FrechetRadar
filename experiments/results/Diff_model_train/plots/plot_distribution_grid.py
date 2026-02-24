
# std lib imports
import argparse
import json
import os
import re
import math

# 3rd party imports
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# project imports
from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer
from RadarDataGen.Data_Generator.generator import (
    PseudoRadarGridGenerator,
    StreamingRadarDataset,
    worker_init_fn,
)


"""


    Examples:

        python3 ./experiments/results/Diff_model_train/plots/plot_distribution_grid.py --exp_id dit/test_dit_x0 -d 0

        python3 ./experiments/results/Diff_model_train/plots/plot_distribution_grid.py --exp_id dit/test_dit_x0 -d valid_indicator

        python3 ./experiments/results/Diff_model_train/plots/plot_distribution_grid.py --exp_id dit/test_dit_x0 -d x

        python3 ./experiments/results/Diff_model_train/plots/plot_distribution_grid.py --exp_id dit/test_dit_x0 -d y

        python3 ./experiments/results/Diff_model_train/plots/plot_distribution_grid.py --exp_id dit/test_dit_x0 -d color -l d

        python3 ./experiments/results/Diff_model_train/plots/plot_distribution_grid.py --exp_id dit/test_dit_x0 -d 0

"""


# ======================================================================
# Configuration & Labels
# ======================================================================

ALIAS_TO_DIM = {
    "valid_indc": 0,
    "valid_indicator": 0,
    "valid": 0,
    "x": 1,
    "y": 2,
    "color": 3,
}

LABELS = {
    "dim": {
        "en": {0: "Validity Indicator", 1: "X Offsets", 2: "Y Offsets", 3: "Color"},
        "de": {0: "Validitätsindikator", 1: "X‑Verschiebung", 2: "Y‑Verschiebung", 3: "Farbwert"},
    },
    "title": {
        "en": "Trained for {batches} batches\n({gen} generated samples, {cells} cells/sample)",
        "de": "Trainiert über {batches} Batches\n({gen} generierte Samples, {cells} Zellen/Sample)",
    },
    "xlabel": {
        "en": "Values — {dim}",
        "de": "Werte — {dim}",
    },
    "ylabel": {
        "en": "Density",
        "de": "Dichte",
    },
    "legend_real": {
        "en": "{dim} (real)",
        "de": "{dim} (real)",
    },
    "legend_gen": {
        "en": "{dim} (generated)",
        "de": "{dim} (generiert)",
    },
    "valid_ticks": {
        "en": {0.0: "not valid", 1.0: "valid"},
        "de": {0.0: "nicht valide", 1.0: "valide"},
    },
}


# ======================================================================
# Utility functions
# ======================================================================

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_dims(dims_str):
    dims = []
    for token in dims_str.split(","):
        token = token.strip().lower()
        if token == "":
            continue

        if token.isdigit():
            dims.append(int(token))
        elif token in ALIAS_TO_DIM:
            dims.append(ALIAS_TO_DIM[token])
        else:
            raise ValueError(
                f"Unknown dimension alias: '{token}'. Allowed are digits or {list(ALIAS_TO_DIM.keys())}"
            )

    # remove duplicates, preserve order
    seen = set()
    ordered = []
    for d in dims:
        if d not in seen:
            ordered.append(d)
            seen.add(d)

    return ordered


def load_generated_samples(folder, requested):
    """Load all .pt files from a folder and combine them."""
    files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
    tensors = [torch.load(os.path.join(folder, f)) for f in files]

    if not tensors:
        return None, 0

    all_samples = torch.cat(tensors, dim=0)
    total = all_samples.shape[0]

    if requested == "all":
        return all_samples, total

    return all_samples[:requested], total


def load_real_samples(stream_iter, n_required):
    """Fetch enough batches from the DataLoader iterator."""
    real_batches = []
    while sum(t.shape[0] for t in real_batches) < n_required:
        try:
            real_batches.append(next(stream_iter))
        except StopIteration:
            stream_iter = iter(stream_iter)
            real_batches.append(next(stream_iter))

    return torch.cat(real_batches, dim=0)[:n_required]


def plot_histogram(axis, gen_vals, real_vals, dim, lang, num_gen_used):
    dim_label = LABELS["dim"][lang].get(dim, f"Dim {dim}")

    # real histogram
    if real_vals is not None:
        axis.hist(
            real_vals,
            bins=120,
            color="#FF4208",
            alpha=0.5,
            density=True,
            label=LABELS["legend_real"][lang].format(dim=dim_label),
        )

    # generated histogram
    axis.hist(
        gen_vals,
        bins=120,
        color="#8905be",
        alpha=0.5,
        density=True,
        label=LABELS["legend_gen"][lang].format(dim=dim_label, n=num_gen_used),
    )

    axis.axvline(0.6, color="red", linestyle="--", linewidth=2)

    axis.set_xlabel(LABELS["xlabel"][lang].format(dim=dim_label))
    axis.set_ylabel(LABELS["ylabel"][lang])
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="upper right")


def apply_validity_ticks(axis, lang, valid_value):
    ticks = LABELS["valid_ticks"][lang]
    combined_ticks = sorted(set(axis.get_xticks().tolist()) | set(ticks.keys()))
    axis.set_xticks(combined_ticks)

    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: ticks.get(x, f"{x:g}"))
    )


def process_model_folder(axis, folder, dims, lang, gen_requested, include_real, stream_iter, n_real, discretizer_valid):
    gen_all, total_gen = load_generated_samples(folder, gen_requested)
    if gen_all is None:
        axis.axis("off")
        return

    num_gen_used = gen_all.shape[0]
    C, H, W = gen_all.shape[1], gen_all.shape[2], gen_all.shape[3]
    real_samples = load_real_samples(stream_iter, n_real) if include_real else None

    return gen_all, num_gen_used, C, H, W, real_samples


def setup_axes_grid(num_models, num_dims, structured):
    if structured:
        rows, cols = num_models, num_dims
    else:
        cols = math.ceil(math.sqrt(num_models * num_dims))
        rows = math.ceil((num_models * num_dims) / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), tight_layout=True)
    if isinstance(ax, plt.Axes):
        ax = [[ax]]
    elif rows == 1:
        ax = [ax]

    return fig, ax, rows, cols


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean multilingual distribution plotter")
    parser.add_argument("--exp_id", "-e", required=True)
    parser.add_argument("--dims", "-d", required=True)
    parser.add_argument("--num_gen_samples", "-ng", default="all")
    parser.add_argument("--num_real_samples", "-nr", type=int, default=128)
    parser.add_argument("--structured_cols", action="store_true")
    parser.add_argument("--outfile", "-o", default=None)
    parser.add_argument("--seed", "-s", default=43, type=int)
    parser.add_argument("--lang", "-l", choices=["de", "en"], default="en")
    parser.add_argument("--not_include_real", action="store_true",help="If set, do not overlay the real distribution (only possible when config['data_generator'] has exactly one entry).")
    args = parser.parse_args()

    # Parse options
    gen_requested = "all" if args.num_gen_samples == "all" else int(args.num_gen_samples)

    # Paths
    base = "./experiments/results/Diff_model_train/"
    path_config = os.path.join(base, args.exp_id, "setting.json")
    path_samples = os.path.join(base, args.exp_id, "samples_model")
    path_plots = os.path.join(base, args.exp_id, "plots")
    os.makedirs(path_plots, exist_ok=True)

    # Config
    config = load_config(path_config)
    discretizer = RadarDiscretizer(
        **config["discretizer_params"]
    )
    valid_value = float(config["discretizer_params"]["valid_indicator"])

    # Determine real overlay capability
    include_real = (not args.not_include_real) and (len(config.get("data_generator", [])) == 1)

    if include_real:
        generator = PseudoRadarGridGenerator(config["data_generator"][0], discretizer=discretizer)
        dataset = torch.utils.data.DataLoader(
            StreamingRadarDataset(sampler=generator, dtype=torch.float32, base_seed=args.seed),
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            worker_init_fn=worker_init_fn,
        )
        real_stream = iter(dataset)
    else:
        real_stream = None

    # Model folders
    pattern = re.compile(r"^model_(\d+)_batches$")
    folders = sorted(
        [
            (int(m.group(1)), os.path.join(path_samples, name))
            for name in os.listdir(path_samples)
            if (m := pattern.match(name))
        ],
        key=lambda x: x[0],
    )

    # Axes setup
    dims = parse_dims(args.dims)
    fig, ax_grid, rows, cols = setup_axes_grid(len(folders), len(dims), args.structured_cols)

    # Main loop
    for r, (batches, folder) in enumerate(folders):
        for c, dim in enumerate(dims):
            if args.structured_cols:
                axis = ax_grid[r][c]
            else:
                index = r * len(dims) + c
                axis = ax_grid[index // cols][index % cols]

            gen_all, num_gen_used, C, H, W, real_samples = process_model_folder(
                axis, folder, dims, args.lang, gen_requested, include_real, real_stream,
                args.num_real_samples, valid_value
            )

            if gen_all is None or dim >= C:
                axis.axis("off")
                continue

            # flatten
            gen_vals = gen_all[:, dim, :, :].reshape(-1).cpu()
            real_vals = real_samples[:, dim, :, :].reshape(-1).cpu() if include_real else None

            # plotting
            plot_histogram(axis, gen_vals, real_vals, dim, args.lang, num_gen_used)

            # titles
            axis.set_title(
                LABELS["title"][args.lang].format(
                    batches=batches, gen=num_gen_used, cells=H * W
                )
            )

            # validity tick labels
            if dim == 0:
                apply_validity_ticks(axis, args.lang, valid_value)

    # filename
    if args.outfile is None:
        dim_names = "_".join(
            LABELS["dim"][args.lang].get(d, f"dim{d}").replace(" ", "_").lower() for d in dims
        )
        outfile = f"distribution_{dim_names}.png"
    else:
        outfile = args.outfile

    output_path = os.path.join(path_plots, outfile)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.3)

    print(f"[INFO] Saved plot: {output_path}")
