from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import SimulationConfig
from d2nn.detector import build_detector_masks, integrate_detector_regions


def small_config():
    return SimulationConfig(
        pixel_size_m=10.0e-6,
        field_size=100,
        padded_size=128,
        layer_spacing_m=1.0e-3,
        detector_spacing_m=1.0e-3,
    )


class DetectorTests(unittest.TestCase):
    def test_regions_are_non_overlapping(self):
        config = small_config()
        masks = build_detector_masks(config)
        self.assertEqual(masks.shape[0], len(config.detector_regions))
        self.assertLessEqual(float(np.max(np.sum(masks, axis=0))), 1.0)

    def test_integrated_intensity_is_not_coherent_sum_square(self):
        real = np.array([[1.0, -1.0]], dtype=np.complex128)
        mask = np.ones((1, 2), dtype=np.float64)
        result = integrate_detector_regions(real, real, mask[None, :, :], 2.0)
        self.assertAlmostEqual(float(result.sk[0]), 2.0)
        self.assertAlmostEqual(float(abs(np.sum(real)) ** 2), 0.0)

    def test_eta_for_uniform_intensity_matches_masked_area_fraction(self):
        config = small_config()
        masks = build_detector_masks(config)
        field = np.ones((config.field_size, config.field_size), dtype=np.complex128)
        input_power = float(config.field_size * config.field_size)
        result = integrate_detector_regions(field, field, masks, input_power)
        expected = float(np.sum(masks)) / float(config.field_size * config.field_size)
        self.assertAlmostEqual(result.eta, expected, places=12)
        self.assertLess(result.eta, 1.0)
