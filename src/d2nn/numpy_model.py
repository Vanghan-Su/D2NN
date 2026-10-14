"""NumPy reference implementation of the phase/amplitude D2NN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .config import COMPLEX_AMPLITUDE_PRETRAIN_METHOD, PURE_PHASE_METHOD
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
    verify the PyTorch implementation and the physics conventions. By default,
    input grayscale values can be used directly as amplitudes or normalized
    to unit power; normalize_input controls this behavior and defaults to True.
    """

    def __init__(
        self,
        config,
        phases: np.ndarray,
        amplitudes: np.ndarray = None,
        normalize_input: bool = True,
    ):
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

        if amplitudes is None:
            self.amplitudes = None
        else:
            self.amplitudes = np.asarray(amplitudes, dtype=np.float64)
            if self.amplitudes.shape != expected_shape:
                raise ValueError(
                    "amplitudes must have shape "
                    f"{expected_shape}, got {self.amplitudes.shape}"
                )
            if np.any(self.amplitudes < 0.0) or np.any(self.amplitudes > 1.0):
                raise ValueError("amplitudes must lie in [0, 1]")
        self.detector_masks = build_detector_masks(config)
        self.layer_transfer = angular_spectrum_transfer_function(
            config, config.layer_spacing_m
        )
        self.detector_transfer = angular_spectrum_transfer_function(
            config, config.detector_spacing_m
        )

    @property
    def optimization_branch(self) -> str:
        """Name the active optimization branch for consistency with PyTorch."""

        if self.amplitudes is None:
            return PURE_PHASE_METHOD
        return COMPLEX_AMPLITUDE_PRETRAIN_METHOD

    def _apply_modulation(
        self,
        field: np.ndarray,
        phase: np.ndarray,
        amplitude: np.ndarray = None,
    ) -> np.ndarray:
        output = np.array(field, dtype=np.complex128, copy=True)
        start = (self.config.padded_size - self.config.field_size) // 2
        stop = start + self.config.field_size
        modulation = np.exp(1j * phase)
        if amplitude is None:
            if self.config.amplitude_transmission != 1.0:
                modulation = self.config.amplitude_transmission * modulation
        else:
            modulation = np.asarray(amplitude, dtype=np.float64) * modulation
        output[start:stop, start:stop] *= modulation
        return output

    def _apply_phase(self, field: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Backward-compatible phase-only helper."""

        return self._apply_modulation(field, phase, amplitude=None)

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
        # The current simulation uses raw grayscale values as amplitudes.
        # The normalized mode is retained only for legacy reference checks.
        if self.normalize_input:
            field = field * np.sqrt(1.0 / input_power)
            input_power = 1.0

        layer_fields = [] if return_fields else None
        for layer_index in range(self.config.layer_count):
            layer_amplitude = (
                None
                if self.amplitudes is None
                else self.amplitudes[layer_index]
            )
            field = self._apply_modulation(
                field,
                self.phases[layer_index],
                layer_amplitude,
            )
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
