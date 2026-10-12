from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import SimulationConfig
from d2nn.numpy_model import NumpyD2NN
from d2nn.torch_model import TorchD2NN


def small_config():
    return SimulationConfig(
        pixel_size_m=10.0e-6,
        field_size=100,
        padded_size=128,
        layer_spacing_m=1.0e-3,
        detector_spacing_m=1.0e-3,
    )


def random_problem(seed=123):
    config = small_config()
    rng = np.random.default_rng(seed)
    amplitude = rng.random((config.field_size, config.field_size))
    phases = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=(config.layer_count, config.field_size, config.field_size),
    )
    return config, amplitude, phases


class TorchReferenceTests(unittest.TestCase):
    def test_torch_matches_numpy_reference(self):
        config, amplitude, phases = random_problem()
        numpy_model = NumpyD2NN(config, phases=phases, normalize_input=True)
        torch_model = TorchD2NN(config, phases=torch.as_tensor(phases))
        torch_model.eval()

        with torch.no_grad():
            numpy_result = numpy_model.forward(amplitude)
            torch_result = torch_model(torch.as_tensor(amplitude))

        torch_inner = torch_result.detector_inner[0].detach().cpu().numpy()
        torch_sk = torch_result.sk[0].detach().cpu().numpy()
        self.assertLess(
            float(np.max(np.abs(torch_inner - numpy_result.detector_inner))),
            1e-9,
        )
        self.assertLess(
            float(np.max(np.abs(torch_sk - numpy_result.detector.sk))),
            1e-11,
        )

    def test_torch_matches_numpy_with_distinct_detector_spacing(self):
        config = SimulationConfig(
            pixel_size_m=10.0e-6,
            field_size=100,
            padded_size=128,
            layer_spacing_m=1.0e-3,
            detector_spacing_m=2.0e-3,
        )
        rng = np.random.default_rng(321)
        amplitude = rng.random((config.field_size, config.field_size))
        phases = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(config.layer_count, config.field_size, config.field_size),
        )
        numpy_result = NumpyD2NN(config, phases=phases).forward(amplitude)
        torch_result = TorchD2NN(config, phases=torch.as_tensor(phases))(
            torch.as_tensor(amplitude)
        )
        self.assertLess(
            float(
                np.max(
                    np.abs(
                        torch_result.detector_inner[0].detach().cpu().numpy()
                        - numpy_result.detector_inner
                    )
                )
            ),
            1e-9,
        )

    def test_autograd_matches_central_finite_difference(self):
        config, amplitude, phases = random_problem(seed=789)
        model = TorchD2NN(config, phases=torch.as_tensor(phases))
        amplitude_tensor = torch.as_tensor(amplitude)
        labels = torch.tensor([5], dtype=torch.long)
        loss = model.loss(model(amplitude_tensor).sk, labels)
        loss.backward()

        index = (0, 10, 20)
        original = float(model.phases.data[index])
        step = 1.0e-5
        with torch.no_grad():
            model.phases.data[index] = original + step
            loss_plus = float(model.loss(model(amplitude_tensor).sk, labels))
            model.phases.data[index] = original - step
            loss_minus = float(model.loss(model(amplitude_tensor).sk, labels))
            model.phases.data[index] = original

        numerical = (loss_plus - loss_minus) / (2.0 * step)
        analytic = float(model.phases.grad[index])
        np.testing.assert_allclose(
            numerical,
            analytic,
            rtol=1.0e-4,
            atol=1.0e-10,
        )

    def test_gradient_is_finite_and_nonzero(self):
        config, amplitude, _ = random_problem(seed=456)
        model = TorchD2NN(config)
        amplitude_tensor = torch.as_tensor(amplitude)
        labels = torch.tensor([3], dtype=torch.long)
        result = model(amplitude_tensor)
        loss = model.loss(result.sk, labels)
        loss.backward()
        self.assertIsNotNone(model.phases.grad)
        gradient = model.phases.grad
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(float(gradient.abs().mean().item()), 0.0)


if __name__ == "__main__":
    unittest.main()


