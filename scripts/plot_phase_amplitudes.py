"""Plot trained phase and amplitude distributions from a D2NN checkpoint."""

from pathlib import Path
import argparse
from datetime import datetime
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import SimulationConfig
from d2nn.torch_model import TorchD2NN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/20261012-171952_mnist_5mm_4096pad_phase-amplitude_zero-init_32samples_10epochs/20261012-171952_mnist_5mm_4096pad_phase-amplitude_zero-init_32samples_10epochs_last.pt",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("simulation_config", SimulationConfig(padded_size=4096))
    state_dict = checkpoint["model_state_dict"]
    has_trainable_amplitude = any(
        key.startswith("amplitude_logits") for key in state_dict
    )
    model = TorchD2NN(config, train_amplitude=has_trainable_amplitude)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(torch.device(args.device)).eval()

    phases = model.phases.detach().cpu().numpy()
    amplitudes = (
        None
        if model.amplitude_values() is None
        else model.amplitude_values().detach().cpu().numpy()
    )
    num_layers = phases.shape[0]

    rows = 2 if amplitudes is not None else 1
    fig, axes = plt.subplots(
        rows,
        num_layers,
        figsize=(3.0 * num_layers, 5.6 if rows == 2 else 3.2),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.90,
        bottom=0.08,
        top=0.86,
        wspace=0.18,
        hspace=0.28,
    )

    extent = (-0.5, 0.5, 0.5, -0.5)
    phase_image = None
    amplitude_image = None
    for layer_index in range(num_layers):
        ax = axes[0, layer_index]
        phase_image = ax.imshow(
            phases[layer_index],
            origin="upper",
            extent=extent,
            cmap="twilight",
            vmin=0.0,
            vmax=2.0 * np.pi,
            interpolation="nearest",
        )
        ax.set_title(f"Layer {layer_index}")
        ax.set_xticks((-0.5, 0.0, 0.5))
        ax.set_yticks((-0.5, 0.0, 0.5))
        if layer_index == 0:
            ax.set_ylabel("Phase y (mm)")
        ax.set_xlabel("x (mm)")

    if amplitudes is not None:
        for layer_index in range(num_layers):
            ax = axes[1, layer_index]
            amplitude_image = ax.imshow(
                amplitudes[layer_index],
                origin="upper",
                extent=extent,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax.set_xticks((-0.5, 0.0, 0.5))
            ax.set_yticks((-0.5, 0.0, 0.5))
            if layer_index == 0:
                ax.set_ylabel("Amplitude y (mm)")
            ax.set_xlabel("x (mm)")

    cbar_phase_ax = fig.add_axes([0.915, 0.56, 0.014, 0.27])
    phase_colorbar = fig.colorbar(phase_image, cax=cbar_phase_ax)
    phase_colorbar.set_label("Phase (rad)")
    if amplitude_image is not None:
        cbar_amp_ax = fig.add_axes([0.915, 0.12, 0.014, 0.27])
        amplitude_colorbar = fig.colorbar(amplitude_image, cax=cbar_amp_ax)
        amplitude_colorbar.set_label("Amplitude transmission")

    if amplitudes is None:
        fig.suptitle(
            "Pure-phase method: 5 mm layer spacing\n"
            "Trainable phase distributions",
            fontsize=12,
        )
    else:
        fig.suptitle(
            "Complex-amplitude pretraining: 5 mm layer spacing\n"
            "Top: phase distributions | Bottom: amplitude distributions",
            fontsize=12,
        )

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_stem = Path(args.checkpoint).stem
        optimization_slug = "纯相位法" if amplitudes is None else "复振幅预训练"
        output_path = Path(
            f"docs/assets/{timestamp}_figure_{checkpoint_stem}_{optimization_slug}.png"
        )
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"layers={num_layers}")
    print(
        "phase_rad",
        f"min={phases.min():.6f}",
        f"max={phases.max():.6f}",
        f"mean={phases.mean():.6f}",
    )
    if amplitudes is not None:
        print(
            "amplitude",
            f"min={amplitudes.min():.6f}",
            f"max={amplitudes.max():.6f}",
            f"mean={amplitudes.mean():.6f}",
        )
    print(f"saved_png={output_path}")


if __name__ == "__main__":
    main()
