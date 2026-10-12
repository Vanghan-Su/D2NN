from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import SimulationConfig
from d2nn.grid import (
    angular_spectrum_transfer_function,
    crop_center,
    pad_center,
    propagate_angular_spectrum,
)


def small_config():
    return SimulationConfig(
        pixel_size_m=10.0e-6,
        field_size=100,
        padded_size=128,
        layer_spacing_m=1.0e-3,
        detector_spacing_m=1.0e-3,
    )


class GridAndPropagationTests(unittest.TestCase):
    def test_pad_crop_round_trip(self):
        inner = np.arange(9, dtype=np.float64).reshape(3, 3)
        padded = pad_center(inner, 8)
        restored = crop_center(padded, 3)
        np.testing.assert_array_equal(restored, inner)

    def test_transfer_function_is_unitary_in_propagating_band(self):
        config = small_config()
        transfer = angular_spectrum_transfer_function(config, 1.0e-3)
        self.assertTrue(np.allclose(np.abs(transfer), 1.0, rtol=0.0, atol=1e-15))

    def test_energy_is_conserved_for_propagation(self):
        config = small_config()
        rng = np.random.default_rng(7)
        field = (
            rng.normal(size=(config.padded_size, config.padded_size))
            + 1j * rng.normal(size=(config.padded_size, config.padded_size))
        ).astype(np.complex128)
        propagated = propagate_angular_spectrum(field, 1.0e-3, config)
        input_power = float(np.sum(np.abs(field) ** 2))
        output_power = float(np.sum(np.abs(propagated) ** 2))
        self.assertLess(abs(output_power - input_power) / input_power, 1e-12)

    def test_gaussian_beam_matches_paraxial_analytic_solution(self):
        wavelength = 632.8e-9
        pixel_size = 2.0e-6
        size = 1024
        distance = 5.0e-3
        waist = 50.0e-6
        config = SimulationConfig(
            wavelength_m=wavelength,
            pixel_size_m=pixel_size,
            field_size=256,
            padded_size=size,
            layer_spacing_m=distance,
            detector_spacing_m=distance,
        )
        coordinates = (np.arange(size) - size // 2) * pixel_size
        x_grid, y_grid = np.meshgrid(coordinates, coordinates, indexing="xy")
        radius_sq = x_grid * x_grid + y_grid * y_grid
        initial = np.exp(
            -radius_sq / (waist * waist)
        ).astype(np.complex128)
        propagated = propagate_angular_spectrum(initial, distance, config)

        wave_number = 2.0 * np.pi / wavelength
        rayleigh_range = np.pi * waist * waist / wavelength
        q0 = 1j * rayleigh_range
        q = distance + q0
        analytic = (q0 / q) * np.exp(
            -1j * wave_number * radius_sq / (2.0 * q)
        )

        centre = slice(size // 2 - 200, size // 2 + 201)
        numerical_magnitude = np.abs(propagated[centre, centre])
        analytic_magnitude = np.abs(analytic[centre, centre])
        numerical_magnitude /= numerical_magnitude.max()
        analytic_magnitude /= analytic_magnitude.max()
        correlation = np.corrcoef(
            numerical_magnitude.ravel(),
            analytic_magnitude.ravel(),
        )[0, 1]
        relative_rmse = np.sqrt(
            np.mean(
                (numerical_magnitude - analytic_magnitude) ** 2
            )
        )
        self.assertGreater(correlation, 0.999999)
        self.assertLess(relative_rmse, 1.0e-5)

    def test_zero_distance_is_identity(self):
        config = small_config()
        field = np.ones((config.padded_size, config.padded_size), dtype=np.complex128)
        propagated = propagate_angular_spectrum(field, 0.0, config)
        np.testing.assert_array_equal(propagated, field)



