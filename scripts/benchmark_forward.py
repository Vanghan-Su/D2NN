"""Benchmark one full-size forward/backward pass on CPU or CUDA."""

from pathlib import Path
import argparse
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import SimulationConfig
from d2nn.torch_model import TorchD2NN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--padded-size", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-backward", action="store_true")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    device = torch.device(args.device)
    config = SimulationConfig(padded_size=args.padded_size)
    model = TorchD2NN(config).to(device)
    amplitude = torch.rand(
        args.batch_size,
        config.field_size,
        config.field_size,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)
    start = time.perf_counter()
    output = model(amplitude)
    synchronize(device)
    forward_seconds = time.perf_counter() - start

    print(f"device={device}")
    print(f"batch_size={args.batch_size}")
    print(f"forward_seconds={forward_seconds:.6f}")
    if device.type == "cuda":
        print(
            "peak_forward_MB="
            f"{torch.cuda.max_memory_allocated(device) / 1.0e6:.3f}"
        )

    if not args.no_backward:
        labels = torch.zeros(args.batch_size, dtype=torch.long, device=device)
        loss = model.loss(output.sk, labels)
        synchronize(device)
        start = time.perf_counter()
        loss.backward()
        synchronize(device)
        backward_seconds = time.perf_counter() - start
        print(f"backward_seconds={backward_seconds:.6f}")
        if device.type == "cuda":
            print(
                "peak_backward_MB="
                f"{torch.cuda.max_memory_allocated(device) / 1.0e6:.3f}"
            )
        print(f"loss={float(loss.detach().cpu()):.8f}")
        print(f"gradient_finite={bool(torch.isfinite(model.phases.grad).all())}")


if __name__ == "__main__":
    main()

