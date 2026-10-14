"""Training entry point for the pure-phase D2NN branch."""

from pathlib import Path
import argparse
from datetime import datetime
import math
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import (
    COMPLEX_AMPLITUDE_PRETRAIN_METHOD,
    PURE_PHASE_METHOD,
    SimulationConfig,
)
from d2nn.data import SyntheticAmplitudeDataset, build_mnist_dataset, limit_dataset
from d2nn.environment import environment_summary
from d2nn.torch_model import TorchD2NN
from d2nn.training import TrainConfig, fit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("synthetic", "mnist"), default="mnist")
    parser.add_argument("--data-root", default=str(Path("data")))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--padded-size", type=int, default=2048)
    parser.add_argument("--layer-spacing-mm", type=float, default=5.0)
    parser.add_argument("--detector-spacing-mm", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--phase-init", choices=("random", "zero"), default="random")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--phase-only",
        dest="optimize_amplitude",
        action="store_false",
        default=False,
        help="Optimize phases only (default; amplitude transmission is fixed to 1).",
    )
    mode_group.add_argument(
        "--optimize-amplitude",
        dest="optimize_amplitude",
        action="store_true",
        help="Optional complex-amplitude pretraining only; final training is pure phase.",
    )
    parser.add_argument("--amplitude-init", type=float, default=0.8)
    parser.add_argument(
        "--normalize-input-amplitude",
        dest="normalize_input_amplitude",
        action="store_true",
        default=True,
        help="Normalize each input grayscale amplitude image to unit power.",
    )
    parser.add_argument(
        "--no-normalize-input-amplitude",
        dest="normalize_input_amplitude",
        action="store_false",
        help="Use raw grayscale values directly without input-power normalization.",
    )
    parser.set_defaults(optimize_amplitude=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument(
        "--dataset-fraction",
        type=float,
        default=0.01,
        help="Use this fraction of the training and validation datasets (default: 0.01).",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.dataset_fraction is not None and not 0.0 < args.dataset_fraction <= 1.0:
        parser.error("--dataset-fraction must be in (0, 1]")
    return args


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    resume_checkpoint = None
    if args.resume_from is not None:
        resume_checkpoint = torch.load(args.resume_from, map_location="cpu")
        config = resume_checkpoint["simulation_config"]
        args.optimize_amplitude = any(
            key.startswith("amplitude_logits")
            for key in resume_checkpoint["model_state_dict"]
        )
    else:
        config = SimulationConfig(
            padded_size=args.padded_size,
            layer_spacing_m=args.layer_spacing_mm * 1.0e-3,
            detector_spacing_m=args.detector_spacing_mm * 1.0e-3,
            seed=args.seed,
        )
    config.validate()
    optimization_slug = (
        COMPLEX_AMPLITUDE_PRETRAIN_METHOD
        if args.optimize_amplitude
        else PURE_PHASE_METHOD
    )

    if args.dataset == "synthetic":
        train_dataset = SyntheticAmplitudeDataset(
            size=args.limit,
            field_size=config.field_size,
            seed=config.seed,
        )
        validation_dataset = SyntheticAmplitudeDataset(
            size=max(1, args.limit // 5),
            field_size=config.field_size,
            seed=config.seed + 1,
        )
        if args.dataset_fraction is not None:
            train_count = max(1, int(math.floor(len(train_dataset) * args.dataset_fraction)))
            validation_count = max(
                1,
                int(math.floor(len(validation_dataset) * args.dataset_fraction)),
            )
            train_dataset = limit_dataset(train_dataset, train_count)
            validation_dataset = limit_dataset(validation_dataset, validation_count)
    else:
        train_dataset = build_mnist_dataset(
            root=args.data_root,
            train=True,
            field_size=config.field_size,
            download=args.download,
        )
        validation_dataset = build_mnist_dataset(
            root=args.data_root,
            train=False,
            field_size=config.field_size,
            download=args.download,
        )
        if args.dataset_fraction is None:
            train_count = args.limit
            validation_count = max(1, args.limit // 5)
        else:
            train_count = max(
                1,
                int(math.floor(len(train_dataset) * args.dataset_fraction)),
            )
            validation_count = max(
                1,
                int(math.floor(len(validation_dataset) * args.dataset_fraction)),
            )
        train_dataset = limit_dataset(train_dataset, train_count)
        validation_dataset = limit_dataset(validation_dataset, validation_count)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=0,
    )

    train_config = TrainConfig(
        epochs=args.epochs,
        effective_batch_size=args.effective_batch_size,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        max_batches_per_epoch=args.max_batches,
        seed=args.seed,
    )

    print(environment_summary())
    print(f"device={device}")
    print(f"dataset={args.dataset}")
    print(f"train_examples={len(train_dataset)}")
    print(f"validation_examples={len(validation_dataset)}")
    print(f"dataset_fraction={args.dataset_fraction}")
    print(f"accumulation_steps={train_config.accumulation_steps}")
    print(f"phase_init={args.phase_init}")
    print(f"layer_spacing_mm={args.layer_spacing_mm}")
    print(f"detector_spacing_mm={args.detector_spacing_mm}")
    print(f"optimization_branch={optimization_slug}")
    print(f"optimize_amplitude={args.optimize_amplitude}")
    print(f"normalize_input_amplitude={args.normalize_input_amplitude}")

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_mode = "resume" if resume_checkpoint is not None else "fresh"
    run_slug = (
        f"{args.dataset}_{args.layer_spacing_mm:g}mm_"
        f"{args.padded_size}pad_{optimization_slug}_"
        f"{run_mode}_{args.phase_init}-init_{len(train_dataset)}samples_"
        f"{args.epochs}epochs"
    )
    output_dir = args.output_dir or f"outputs/{run_timestamp}_{run_slug}"
    checkpoint_name = f"{run_timestamp}_{run_slug}_last.pt"
    print(f"output_dir={output_dir}")
    print(f"checkpoint_name={checkpoint_name}")

    initial_phases = None
    if args.phase_init == "zero":
        initial_phases = torch.zeros(
            (
                config.layer_count,
                config.field_size,
                config.field_size,
            ),
            dtype=torch.float64,
        )
    if resume_checkpoint is None:
        model = TorchD2NN(
            config,
            phases=initial_phases,
            train_amplitude=args.optimize_amplitude,
            amplitude_init=args.amplitude_init,
            normalize_input=args.normalize_input_amplitude,
        ).to(device)
    else:
        model = TorchD2NN(
            config,
            train_amplitude=args.optimize_amplitude,
            normalize_input=args.normalize_input_amplitude,
        )
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        model = model.to(device)
    fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_config=train_config,
        device=device,
        output_dir=output_dir,
        checkpoint_name=checkpoint_name,
        resume_checkpoint=resume_checkpoint,
    )


if __name__ == "__main__":
    main()
