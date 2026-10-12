"""Detector geometry and intensity integration."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DetectorResult:
    sk: np.ndarray
    input_power: float
    eta: float
    prediction: int
    contrast: float
    outside_power: float


def build_detector_masks(config) -> np.ndarray:
    """Build binary masks for all detector regions.

    Returns an array with shape (num_regions, field_size, field_size). The
    coordinates follow the detector image convention: x is the column axis and
    y is the row axis, with the origin at the upper-left corner.
    """

    size = config.field_size
    pixel_size = config.pixel_size_m
    x_coordinates = (np.arange(size, dtype=np.float64) + 0.5) * pixel_size
    y_coordinates = (np.arange(size, dtype=np.float64) + 0.5) * pixel_size
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates, indexing="xy")

    masks = np.zeros(
        (len(config.detector_regions), size, size),
        dtype=np.float64,
    )
    for region_index, region in enumerate(config.detector_regions):
        x_min = region.center_x_m - region.width_m / 2.0
        x_max = region.center_x_m + region.width_m / 2.0
        y_min = region.center_y_m - region.height_m / 2.0
        y_max = region.center_y_m + region.height_m / 2.0
        mask = (
            (x_grid >= x_min)
            & (x_grid <= x_max)
            & (y_grid >= y_min)
            & (y_grid <= y_max)
        )
        if not np.any(mask):
            raise ValueError(f"detector region {region.label} contains no pixels")
        masks[region_index] = mask.astype(np.float64)

    overlap = np.sum(masks, axis=0)
    if np.any(overlap > 1.0):
        raise ValueError("detector regions overlap")
    return masks


def integrate_detector_regions(
    detector_field_inner: np.ndarray,
    detector_field_full: np.ndarray,
    masks: np.ndarray,
    input_power: float,
) -> DetectorResult:
    """Integrate pointwise intensity over detector regions.

    The detector signal is:

        sk = sum_{(x,y) in Dk} |U(x,y)|^2

    It is intentionally not |sum U|^2. A physical detector performs an
    incoherent sum of the intensity incident on each detector pixel.
    """

    if detector_field_inner.ndim != 2:
        raise ValueError("detector_field_inner must be two-dimensional")
    if detector_field_full.ndim != 2:
        raise ValueError("detector_field_full must be two-dimensional")
    if masks.ndim != 3:
        raise ValueError("masks must have shape (num_regions, rows, cols)")
    if masks.shape[1:] != detector_field_inner.shape:
        raise ValueError("mask shape does not match detector_field_inner")
    if input_power <= 0.0:
        raise ValueError("input_power must be positive")

    intensity = np.abs(detector_field_inner) ** 2
    sk = np.einsum("kij,ij->k", masks, intensity, optimize=True)
    total_region_power = float(np.sum(sk))
    eta = total_region_power / float(input_power)
    outside_power = float(input_power) - total_region_power
    prediction = int(np.argmax(sk))

    ordered = np.sort(sk)
    largest = float(ordered[-1])
    second_largest = float(ordered[-2]) if ordered.size > 1 else 0.0
    if largest == 0.0:
        contrast = 0.0
    elif second_largest <= np.finfo(np.float64).eps:
        contrast = float("inf")
    else:
        contrast = largest / second_largest

    return DetectorResult(
        sk=sk,
        input_power=float(input_power),
        eta=float(eta),
        prediction=prediction,
        contrast=contrast,
        outside_power=outside_power,
    )
