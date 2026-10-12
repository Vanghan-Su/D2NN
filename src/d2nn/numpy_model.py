"""NumPy reference implementation of the phase-only D2NN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .detector import DetectorResult, build_detector_masks, integrate_detector_regions
from .grid import (
    angular_spectrum_transfer_function,
    crop_center,
    pad_center,
    propagate_angular_spectrum,
)


@dataclass(frozen=True)
class NumpyForwardResult:
    detector_full: np.ndarray
    detector_inner: np.ndarray
    detector: DetectorResult
    layer_fields: Optional[Tuple[np.ndarray, ...]] = None


class NumpyD2NN:
    """Differentiable-compatible NumPy reference model for validation.

    The model does not perform training itself. It is the reference used to
    verify the PyTorch implementation and the physics conventions.
    """

    def __init__(self, config, phases: np.ndarray, normalize_input: bool = True):
        config.validate()
        self.config = config
        self.normalize_input = bool(normalize_input)
        self.phases = np.asarray(phases, dtype=np.float64)
        expected_shape = (
            config.layer_count,
            config.field_size,
            config.field_size,
        )
        if self.phases.shape != expected_shape:
            raise ValueError(
                f"phases must have shape {expected_shape}, got {self.phases.shape}"
            )
        self.detector_masks = build_detector_masks(config)
        self.layer_transfer = angular_spectrum_transfer_function(
            config, config.layer_spacing_m
        )
        self.detector_transfer = angular_spectrum_transfer_function(
            config, config.detector_spacing_m
        )

    def _apply_phase(self, field: np.ndarray, phase: np.ndarray) -> np.ndarray:
        output = np.array(field, dtype=np.complex128, copy=True)
        start = (self.config.padded_size - self.config.field_size) // 2
        stop = start + self.config.field_size
        phase_factor = np.exp(1j * phase)
        if self.config.amplitude_transmission != 1.0:
            phase_factor = (
                self.config.amplitude_transmission * phase_factor
            )
        output[start:stop, start:stop] *= phase_factor
        return output

    def forward(
        self,
        amplitude_inner: np.ndarray,
        return_fields: bool = False,
    ) -> NumpyForwardResult:
        amplitude = np.asarray(amplitude_inner)
        expected_shape = (self.config.field_size, self.config.field_size)
        if amplitude.shape != expected_shape:
            raise ValueError(
                f"amplitude_inner must have shape {expected_shape}, "
                f"got {amplitude.shape}"
            )
        if np.iscomplexobj(amplitude):
            amplitude = np.asarray(amplitude, dtype=np.complex128)
        else:
            amplitude = np.asarray(amplitude, dtype=np.float64)

        field = pad_center(amplitude.astype(np.complex128, copy=False), self.config.padded_size)
        input_power = float(np.sum(np.abs(field) ** 2))
        if input_power <= 0.0:
            raise ValueError("input amplitude has zero power")
        if self.normalize_input:
            field = field * np.sqrt(1.0 / input_power)
            input_power = 1.0

        layer_fields = [] if return_fields else None
        for layer_index in range(self.config.layer_count):
            field = self._apply_phase(field, self.phases[layer_index])
            if layer_fields is not None:
                layer_fields.append(field.copy())
            if layer_index < self.config.layer_count - 1:
                field = propagate_angular_spectrum(
                    field,
                    self.config.layer_spacing_m,
                    self.config,
                    transfer_function=self.layer_transfer,
                )

        field = propagate_angular_spectrum(
            field,
            self.config.detector_spacing_m,
            self.config,
            transfer_function=self.detector_transfer,
        )
        detector_inner = crop_center(field, self.config.field_size)
        detector_result = integrate_detector_regions(
            detector_inner,
            field,
            self.detector_masks,
            input_power,
        )
        return NumpyForwardResult(
            detector_full=field,
            detector_inner=detector_inner,
            detector=detector_result,
            layer_fields=tuple(layer_fields) if layer_fields is not None else None,
        )
