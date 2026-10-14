from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

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
    def test_input_amplitude_is_normalized_by_default(self):
        config, amplitude, phases = random_problem(seed=4242)
        model = TorchD2NN(config, phases=torch.as_tensor(phases))
        amplitude_tensor = torch.as_tensor(amplitude)

        result = model(amplitude_tensor)
        self.assertTrue(
            torch.allclose(
                result.input_power,
                torch.ones_like(result.input_power),
                rtol=0.0,
                atol=1e-12,
            )
        )

    def test_input_amplitude_can_disable_normalization(self):
        config, amplitude, phases = random_problem(seed=4243)
        model = TorchD2NN(
            config,
            phases=torch.as_tensor(phases),
            normalize_input=False,
        )
        amplitude_tensor = torch.as_tensor(amplitude)
        result = model(amplitude_tensor)
        expected_power = torch.sum(amplitude_tensor.square(), dim=(-2, -1))
        self.assertTrue(
            torch.allclose(result.input_power, expected_power, rtol=0.0, atol=1e-12)
        )

    def test_loss_combines_cross_entropy_and_mse_for_zero_field(self):
        config = small_config()
        model = TorchD2NN(
            config,
            phases=torch.zeros(
                (config.layer_count, config.field_size, config.field_size),
                dtype=torch.float64,
            ),
        )
        sk = torch.zeros(
            1,
            len(config.detector_regions),
            dtype=torch.float64,
        )
        labels = torch.tensor([3], dtype=torch.long)
        input_power = torch.tensor([2.0], dtype=torch.float64)

        loss = model.loss(sk, labels, input_power)
        expected = float(torch.log(torch.tensor(float(len(config.detector_regions)))) + 1.0 / len(config.detector_regions))

        self.assertAlmostEqual(float(loss.item()), expected, delta=1.0e-7)
    def test_torch_matches_numpy_reference(self):
        config, amplitude, phases = random_problem()
        numpy_model = NumpyD2NN(config, phases=phases)
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
        result = model(amplitude_tensor)
        loss = model.loss(result.sk, labels, result.input_power)
        loss.backward()

        index = (0, 10, 20)
        original = float(model.phases.data[index])
        step = 1.0e-3
        with torch.no_grad():
            model.phases.data[index] = original + step
            result_plus = model(amplitude_tensor)
            loss_plus = float(
                model.loss(result_plus.sk, labels, result_plus.input_power)
            )
            model.phases.data[index] = original - step
            result_minus = model(amplitude_tensor)
            loss_minus = float(
                model.loss(result_minus.sk, labels, result_minus.input_power)
            )
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
        loss = model.loss(result.sk, labels, result.input_power)
        loss.backward()
        self.assertIsNotNone(model.phases.grad)
        gradient = model.phases.grad
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(float(gradient.abs().mean().item()), 0.0)

    def test_torch_matches_numpy_with_amplitudes(self):
        config, amplitude, phases = random_problem(seed=987)
        rng = np.random.default_rng(988)
        transmissions = rng.uniform(
            0.1,
            0.9,
            size=(config.layer_count, config.field_size, config.field_size),
        )
        numpy_result = NumpyD2NN(
            config,
            phases=phases,
            amplitudes=transmissions,
        ).forward(amplitude)
        torch_result = TorchD2NN(
            config,
            phases=torch.as_tensor(phases),
            amplitudes=torch.as_tensor(transmissions),
        )(torch.as_tensor(amplitude))
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

    def test_loss_combines_cross_entropy_and_normalized_mse(self):
        config = small_config()
        model = TorchD2NN(config)
        sk = torch.tensor(
            [[3.0, 1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7, 2.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        labels = torch.tensor([0, 1], dtype=torch.long)
        input_power = torch.tensor([4.0, 5.0], dtype=torch.float64)
        weight = 0.35
        actual = model.loss(sk, labels, input_power, mse_weight=weight)
        targets = F.one_hot(labels, num_classes=10).to(sk.dtype) * input_power[:, None]
        expected_mse = (
            ((sk - targets) ** 2).mean(dim=1) / input_power.square()
        ).mean()
        expected = F.cross_entropy(sk, labels) + weight * expected_mse
        self.assertAlmostEqual(float(actual), float(expected), places=12)

    def test_amplitude_and_phase_gradients_are_finite(self):
        config, amplitude, phases = random_problem(seed=654)
        model = TorchD2NN(
            config,
            phases=torch.as_tensor(phases),
            train_amplitude=True,
            amplitude_init=0.8,
        )
        result = model(torch.as_tensor(amplitude))
        loss = model.loss(
            result.sk,
            torch.tensor([2], dtype=torch.long),
            result.input_power,
        )
        loss.backward()

        self.assertIsNotNone(model.phases.grad)
        self.assertIsNotNone(model.amplitude_logits.grad)
        self.assertTrue(torch.isfinite(model.phases.grad).all().item())
        self.assertTrue(torch.isfinite(model.amplitude_logits.grad).all().item())
        self.assertGreater(float(model.phases.grad.abs().mean().item()), 0.0)
        self.assertGreater(float(model.amplitude_logits.grad.abs().mean().item()), 0.0)
        self.assertTrue(torch.all(model.amplitude_values() > 0.0).item())
        self.assertTrue(torch.all(model.amplitude_values() < 1.0).item())


if __name__ == "__main__":
    unittest.main()





