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

    def test_default_padding_matches_pure_phase_document(self):
        config = SimulationConfig()
        self.assertEqual(config.padded_size, 2048)
        self.assertGreaterEqual(config.padded_size, config.minimum_padded_size)

    def test_base_config_matches_pure_phase_defaults(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "base.json"
        config = SimulationConfig.from_json(config_path)
        self.assertEqual(config.padded_size, 2048)
        self.assertEqual(config.amplitude_transmission, 1.0)


if __name__ == "__main__":
    unittest.main()
