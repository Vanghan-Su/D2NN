"""Dataset adapters for amplitude-encoded D2NN inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticAmplitudeDataset(Dataset):
    """Deterministic synthetic amplitude images for local smoke training.

    The first channel is a smooth random field and the label is generated from
    a weak spatial pattern so that a trainability check has a non-degenerate
    target. This dataset is not a replacement for MNIST evaluation.
    """

    size: int
    field_size: int
    num_classes: int = 10
    seed: int = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        if index < 0 or index >= self.size:
            raise IndexError(index)
        rng = np.random.default_rng(self.seed + index)
        rows = np.arange(self.field_size, dtype=np.float64)[:, None]
        cols = np.arange(self.field_size, dtype=np.float64)[None, :]
        label = int(index % self.num_classes)
        frequency = 1.0 + 0.35 * label
        phase = 0.17 * label
        image = 0.5 + 0.5 * np.sin(
            2.0 * np.pi * frequency * rows / self.field_size
            + 2.0 * np.pi * cols / self.field_size
            + phase
        )
        image = image * (0.75 + 0.25 * rng.random((self.field_size, self.field_size)))
        image = image.astype(np.float32)
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


def build_mnist_dataset(
    root: str,
    train: bool,
    field_size: int,
    download: bool = False,
) -> Dataset:
    """Create a torchvision MNIST dataset resized to the D2NN grid.

    The model normalizes each sample to unit sum of squared amplitude. The
    returned dataset therefore contains amplitudes in [0, 1], not pre-squared
    intensities.
    """

    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Resize(
                (field_size, field_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    return datasets.MNIST(
        root=root,
        train=train,
        transform=transform,
        download=download,
    )


def limit_dataset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit <= 0:
        raise ValueError("limit must be positive")
    return torch.utils.data.Subset(dataset, list(range(limit)))
