from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train


class TrainingCliTests(unittest.TestCase):
    def test_defaults_to_complex_amplitude_2048(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = train.parse_args()
        self.assertTrue(args.optimize_amplitude)
        self.assertEqual(args.padded_size, 2048)
        self.assertEqual(args.effective_batch_size, 32)
        self.assertEqual(args.micro_batch_size, 8)
        self.assertEqual(args.amplitude_init, 0.8)
        self.assertTrue(args.normalize_input_amplitude)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.dataset_fraction, 0.01)

    def test_phase_only_can_be_selected_explicitly(self):
        with patch.object(sys, "argv", ["train.py", "--phase-only"]):
            args = train.parse_args()
        self.assertFalse(args.optimize_amplitude)

    def test_dataset_fraction_is_accepted(self):
        with patch.object(
            sys,
            "argv",
            ["train.py", "--dataset-fraction", "0.01"],
        ):
            args = train.parse_args()
        self.assertEqual(args.dataset_fraction, 0.01)


if __name__ == "__main__":
    unittest.main()
