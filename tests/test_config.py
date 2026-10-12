from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import SimulationConfig


class ConfigDefaultTests(unittest.TestCase):
    def test_default_layer_spacing_is_five_mm(self):
        config = SimulationConfig()
        self.assertAlmostEqual(config.layer_spacing_m, 5.0e-3)
        self.assertAlmostEqual(config.detector_spacing_m, 10.0e-3)

    def test_default_padding_is_sufficient_for_five_mm_layers(self):
        config = SimulationConfig()
        self.assertGreaterEqual(config.padded_size, config.support_padded_size)


if __name__ == "__main__":
    unittest.main()
