"""Plot the detector-plane intensity with labelled red detector regions."""

from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import SimulationConfig
from d2nn.data import build_mnist_dataset
from d2nn.torch_model import TorchD2NN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/mnist_overfit_zero_32/last_checkpoint.pt",
    )
    parser.add_argument("--sample-index", type=int, default=11)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--padded-size", type=int, default=4096)
    parser.add_argument("--layer-spacing-mm", type=float, default=10.0)
    parser.add_argument("--detector-spacing-mm", type=float, default=10.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output",
        default="docs/assets/mnist_zero_32_output_intensity.png",
    )
    return parser.parse_args()


def detector_intensity_image(model, amplitude, device):
    with torch.no_grad():
        output = model(amplitude.to(device=device, dtype=torch.float64).unsqueeze(0))
    intensity = output.detector_inner[0].abs().square().detach().cpu().numpy()
    return intensity, output


def add_region_boxes(ax, config):
    for region in config.detector_regions:
        x0 = (region.center_x_m - region.width_m / 2.0) * 1.0e3
        y0 = (region.center_y_m - region.height_m / 2.0) * 1.0e3
        width = region.width_m * 1.0e3
        height = region.height_m * 1.0e3
        box = Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor="red",
            linewidth=1.25,
            zorder=5,
        )
        ax.add_patch(box)
        label = ax.text(
            region.center_x_m * 1.0e3,
            region.center_y_m * 1.0e3,
            str(region.label),
            color="red",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=2.0, foreground="white")]
        )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    config = SimulationConfig(
        padded_size=args.padded_size,
        layer_spacing_m=args.layer_spacing_mm * 1.0e-3,
        detector_spacing_m=args.detector_spacing_mm * 1.0e-3,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "simulation_config" in checkpoint:
        config = checkpoint["simulation_config"]
    state_dict = checkpoint["model_state_dict"]
    has_trainable_amplitude = any(
        key.startswith("amplitude_logits") for key in state_dict
    )
    model = TorchD2NN(
        config,
        train_amplitude=has_trainable_amplitude,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    dataset = build_mnist_dataset(
        root=args.data_root,
        train=True,
        field_size=config.field_size,
        download=False,
    )
    amplitude, label = dataset[args.sample_index]
    amplitude = amplitude[0]
    intensity, output = detector_intensity_image(model, amplitude, device)
    prediction = int(output.prediction[0].detach().cpu())
    eta = float(output.eta[0].detach().cpu())

    positive = intensity[intensity > 0.0]
    vmin = max(float(np.percentile(positive, 1.0)), float(intensity.max()) * 1.0e-6)
    vmax = float(intensity.max())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.7),
        gridspec_kw={"width_ratios": [0.85, 1.15]},
    )
    axes[0].imshow(amplitude.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Input amplitude\nMNIST label = {int(label)}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    image = axes[1].imshow(
        intensity,
        origin="upper",
        extent=[0.0, 1.0, 1.0, 0.0],
        cmap="inferno",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    add_region_boxes(axes[1], config)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(1.0, 0.0)
    axes[1].set_xlabel("Detector x (mm)")
    axes[1].set_ylabel("Detector y (mm)")
    axes[1].set_title(
        f"Output-plane intensity\nprediction = {prediction}, eta = {eta:.3f}"
    )
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label(r"Intensity $|U|^2$ (a.u.)")

    fig.suptitle(
        f"MNIST sample {args.sample_index}: true {int(label)}, predicted {prediction}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"sample_index={args.sample_index}")
    print(f"true_label={int(label)}")
    print(f"prediction={prediction}")
    print(f"eta={eta:.8f}")
    print(f"saved_png={output_path}")


if __name__ == "__main__":
    main()
