from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d2nn.config import SimulationConfig
from d2nn.data import SyntheticAmplitudeDataset
from d2nn.torch_model import TorchD2NN
from d2nn.training import TrainConfig, evaluate, train_one_epoch


def small_config():
    return SimulationConfig(
        pixel_size_m=10.0e-6,
        field_size=100,
        padded_size=128,
        layer_spacing_m=1.0e-3,
        detector_spacing_m=1.0e-3,
    )


class TrainingSmokeTests(unittest.TestCase):
    def test_one_training_step_changes_phases(self):
        config = small_config()
        dataset = SyntheticAmplitudeDataset(size=8, field_size=config.field_size)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = TorchD2NN(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
        before = model.phases.detach().clone()
        loss = train_one_epoch(
            model,
            loader,
            optimizer,
            TrainConfig(
                epochs=1,
                effective_batch_size=4,
                micro_batch_size=2,
                max_batches_per_epoch=2,
            ),
            torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertFalse(torch.equal(before, model.phases.detach()))

    def test_evaluate_returns_finite_metrics(self):
        config = small_config()
        dataset = SyntheticAmplitudeDataset(size=4, field_size=config.field_size)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = TorchD2NN(config)
        metrics = evaluate(model, loader, torch.device("cpu"))
        self.assertGreater(metrics["examples"], 0)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["eta"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
