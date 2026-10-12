"""Zero-padding convergence scan for the full-size D2NN forward model."""

from pathlib import Path
import argparse
import csv
from datetime import datetime
from dataclasses import replace
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import SimulationConfig
from d2nn.torch_model import TorchD2NN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1024, 1536, 2048, 3072, 4096],
    )
    parser.add_argument("--reference-size", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser.parse_args()


def relative_l2(reference, candidate):
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float(np.linalg.norm(candidate))
    return float(np.linalg.norm(candidate - reference) / denominator)


def relative_vector_error(reference, candidate):
    scale = np.maximum(np.abs(reference), np.finfo(np.float64).eps)
    return float(np.max(np.abs(candidate - reference) / scale))


def run_size(config, phases, amplitude, device):
    model = TorchD2NN(config, phases=phases).to(device)
    model.eval()
    amplitude_tensor = torch.as_tensor(amplitude, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model(amplitude_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1.0e6
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "field": output.detector_inner[0].detach().cpu().numpy(),
        "sk": output.sk[0].detach().cpu().numpy(),
        "eta": float(output.eta[0].detach().cpu()),
        "prediction": int(output.prediction[0].detach().cpu()),
        "seconds": elapsed,
        "peak_mb": peak_mb,
    }
    del model, output, amplitude_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    sizes = list(dict.fromkeys(args.sizes))
    if args.reference_size not in sizes:
        sizes.append(args.reference_size)
    sizes.sort()

    base_config = SimulationConfig(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    amplitude = rng.random(
        (base_config.field_size, base_config.field_size),
        dtype=np.float64,
    )
    phases = rng.uniform(
        base_config.phase_min_rad,
        base_config.phase_max_rad,
        size=(
            base_config.layer_count,
            base_config.field_size,
            base_config.field_size,
        ),
    )

    results = {}
    for size in sizes:
        config = replace(base_config, padded_size=size)
        config.validate()
        print(f"running Npad={size} ...", flush=True)
        result = run_size(config, phases, amplitude, device)
        results[size] = result
        print(
            f"  eta={result['eta']:.8f} prediction={result['prediction']} "
            f"seconds={result['seconds']:.3f} peak_MB={result['peak_mb']:.1f}",
            flush=True,
        )

    reference = results[args.reference_size]
    rows = []
    for size in sizes:
        result = results[size]
        row = {
            "padded_size": size,
            "field_relative_l2": relative_l2(reference["field"], result["field"]),
            "max_sk_relative_error": relative_vector_error(
                reference["sk"], result["sk"]
            ),
            "eta_error": abs(result["eta"] - reference["eta"]),
            "eta": result["eta"],
            "prediction": result["prediction"],
            "seconds": result["seconds"],
            "peak_mb": result["peak_mb"],
        }
        rows.append(row)
        print(
            f"Npad={size} field_rel_l2={row['field_relative_l2']:.6e} "
            f"max_sk_rel={row['max_sk_relative_error']:.6e} "
            f"eta_err={row['eta_error']:.6e}",
            flush=True,
        )

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(
            f"outputs/{timestamp}_padding-convergence_"
            f"{base_config.layer_spacing_m * 1.0e3:g}mm_"
            f"{min(sizes)}-{max(sizes)}pad.csv"
        )
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
