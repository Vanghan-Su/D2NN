"""Phase-only diffractive optical neural network simulation."""

from .config import DetectorRegion, SimulationConfig
from .detector import DetectorResult, build_detector_masks, integrate_detector_regions
from .numpy_model import NumpyD2NN, NumpyForwardResult
from .torch_model import TorchD2NN, TorchForwardResult

__all__ = [
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
