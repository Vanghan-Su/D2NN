"""End-to-end smoke test for NumPy and PyTorch forward paths."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import SimulationConfig
from d2nn.environment import environment_summary
from d2nn.numpy_model import NumpyD2NN
from d2nn.torch_model import TorchD2NN


def make_small_config():
    return SimulationConfig(
        pixel_size_m=10.0e-6,
        field_size=100,
        padded_size=128,
        layer_spacing_m=1.0e-3,
        detector_spacing_m=1.0e-3,
    )


def main() -> None:
    print(environment_summary())
    config = make_small_config()
    config.validate()
    rng = np.random.default_rng(config.seed)
    amplitude = rng.random((config.field_size, config.field_size))
    phases = rng.uniform(
        config.phase_min_rad,
        config.phase_max_rad,
        size=(config.layer_count, config.field_size, config.field_size),
    )

    numpy_model = NumpyD2NN(config, phases=phases, normalize_input=True)
    numpy_result = numpy_model.forward(amplitude, return_fields=False)
    print("numpy_prediction", numpy_result.detector.prediction)
    print("numpy_eta", numpy_result.detector.eta)
    print("numpy_contrast", numpy_result.detector.contrast)

    torch_model = TorchD2NN(config, phases=torch.as_tensor(phases))
    torch_model.eval()
    torch_result = torch_model(torch.as_tensor(amplitude))
    print("torch_prediction", int(torch_result.prediction[0].item()))
    print("torch_eta", float(torch_result.eta[0].item()))

    loss = torch_model.loss(torch_result.sk, torch.tensor([0], dtype=torch.long))
    loss.backward()
    print("torch_loss", float(loss.item()))
    print("grad_mean_abs", float(torch_model.phases.grad.abs().mean().item()))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_model = TorchD2NN(config, phases=torch.as_tensor(phases)).to(device)
        cuda_amplitude = torch.as_tensor(amplitude, device=device)
        with torch.no_grad():
            cuda_result = cuda_model(cuda_amplitude)
        print("cuda_prediction", int(cuda_result.prediction[0].item()))
        print("cuda_eta", float(cuda_result.eta[0].item()))


if __name__ == "__main__":
    main()

