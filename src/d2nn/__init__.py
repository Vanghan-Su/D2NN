"""Diffractive optical neural network simulation."""

from .config import (
    COMPLEX_AMPLITUDE_PRETRAIN_METHOD,
    PURE_PHASE_METHOD,
    DetectorRegion,
    SimulationConfig,
)
from .detector import DetectorResult, build_detector_masks, integrate_detector_regions
from .numpy_model import NumpyD2NN, NumpyForwardResult
from .torch_model import TorchD2NN, TorchForwardResult

__all__ = [
    "COMPLEX_AMPLITUDE_PRETRAIN_METHOD",
    "PURE_PHASE_METHOD",
    "DetectorRegion",
    "SimulationConfig",
    "DetectorResult",
    "NumpyD2NN",
    "NumpyForwardResult",
    "TorchD2NN",
    "TorchForwardResult",
    "build_detector_masks",
    "integrate_detector_regions",
]
