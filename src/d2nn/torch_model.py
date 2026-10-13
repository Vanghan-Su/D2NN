"""PyTorch implementation for phase-only or phase-amplitude D2NN models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import COMPLEX_AMPLITUDE_PRETRAIN_METHOD, PURE_PHASE_METHOD
from .detector import build_detector_masks
from .grid import angular_spectrum_transfer_function


@dataclass
class TorchForwardResult:
    detector_full: torch.Tensor
    detector_inner: torch.Tensor
    sk: torch.Tensor
    input_power: torch.Tensor
    eta: torch.Tensor
    prediction: torch.Tensor
    contrast: torch.Tensor


class TorchD2NN(nn.Module):
    """Free-space D2NN with optional trainable amplitude transmission.

    The output detector signal is the pointwise intensity integral

        sk = sum_{p in Dk} |U_p|^2

    not the squared coherent sum of the complex field. With the default
    settings this class is phase-only and compatible with phase-only
    checkpoints. Setting ``train_amplitude=True`` adds a per-layer amplitude
    logit parameter and optimizes phase and amplitude simultaneously.
    """

    def __init__(
        self,
        config,
        phases: Optional[torch.Tensor] = None,
        amplitudes: Optional[torch.Tensor] = None,
        train_amplitude: bool = False,
        amplitude_init: float = 1.0,
    ):
        super().__init__()
        config.validate()
        self.config = config
        expected_shape = (
            config.layer_count,
            config.field_size,
            config.field_size,
        )

        if phases is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed)
            phase_values = (
                torch.rand(expected_shape, generator=generator, dtype=torch.float64)
                * config.phase_span_rad()
                + config.phase_min_rad
            )
        else:
            phase_values = torch.as_tensor(phases, dtype=torch.float64).clone()

        if tuple(phase_values.shape) != expected_shape:
            raise ValueError(
                f"phases must have shape {expected_shape}, got {tuple(phase_values.shape)}"
            )

        self.phases = nn.Parameter(phase_values)
        self.train_amplitude = bool(train_amplitude or amplitudes is not None)

        if self.train_amplitude:
            if amplitudes is None:
                if not 0.0 < amplitude_init <= 1.0:
                    raise ValueError("amplitude_init must be in (0, 1]")
                amplitude_values = torch.full(
                    expected_shape,
                    float(amplitude_init),
                    dtype=torch.float64,
                )
            else:
                amplitude_values = torch.as_tensor(
                    amplitudes, dtype=torch.float64
                ).clone()
                if tuple(amplitude_values.shape) != expected_shape:
                    raise ValueError(
                        "amplitudes must have shape "
                        f"{expected_shape}, got {tuple(amplitude_values.shape)}"
                    )

            amplitude_values = amplitude_values.clamp(1.0e-6, 1.0 - 1.0e-6)
            amplitude_logits = torch.log(
                amplitude_values / (1.0 - amplitude_values)
            )
            self.amplitude_logits = nn.Parameter(amplitude_logits)
        else:
            self.amplitude_logits = None

        detector_masks = build_detector_masks(config)
        transfer = angular_spectrum_transfer_function(
            config, config.layer_spacing_m
        )
        detector_transfer = angular_spectrum_transfer_function(
            config, config.detector_spacing_m
        )

        self.register_buffer(
            "detector_masks",
            torch.as_tensor(detector_masks, dtype=torch.float64),
        )
        self.register_buffer(
            "layer_transfer",
            torch.as_tensor(transfer, dtype=torch.complex128),
        )
        self._shared_transfer = (
            self.config.layer_spacing_m == self.config.detector_spacing_m
        )
        if self._shared_transfer:
            self.detector_transfer = None
        else:
            self.register_buffer(
                "detector_transfer",
                torch.as_tensor(detector_transfer, dtype=torch.complex128),
            )

        self._inner_start = (config.padded_size - config.field_size) // 2
        self._inner_stop = self._inner_start + config.field_size

    @property
    def optimization_branch(self) -> str:
        """Name the active optimization branch for metadata and outputs."""

        if self.train_amplitude:
            return COMPLEX_AMPLITUDE_PRETRAIN_METHOD
        return PURE_PHASE_METHOD

    def amplitude_values(self) -> Optional[torch.Tensor]:
        """Return physical amplitude transmission, or None in phase-only mode."""

        if self.amplitude_logits is None:
            return None
        return torch.sigmoid(self.amplitude_logits)

    def _apply_modulation(
        self,
        field: torch.Tensor,
        phase: torch.Tensor,
        amplitude: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply phase and optional amplitude inside the finite aperture.

        Only the central aperture is modified. The vacuum area outside the
        aperture is concatenated unchanged, avoiding a full padded complex
        phase tensor for every layer.
        """

        start = self._inner_start
        stop = self._inner_stop
        modulation = torch.exp(1j * phase.to(torch.complex128))

        if amplitude is None:
            if self.config.amplitude_transmission != 1.0:
                modulation = (
                    self.config.amplitude_transmission * modulation
                )
        else:
            modulation = amplitude.to(torch.complex128) * modulation

        top = field[..., :start, :]
        middle = field[..., start:stop, :]
        middle_left = middle[..., :start]
        middle_center = middle[..., start:stop] * modulation.unsqueeze(0)
        middle_right = middle[..., stop:]
        middle = torch.cat(
            (middle_left, middle_center, middle_right),
            dim=-1,
        )
        bottom = field[..., stop:, :]
        return torch.cat((top, middle, bottom), dim=-2)

    def _apply_phase(self, field: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Backward-compatible phase-only helper."""

        return self._apply_modulation(field, phase, amplitude=None)

    def _propagate(self, field: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(torch.fft.fft2(field) * transfer)

    def _normalize_input(self, amplitude: torch.Tensor) -> torch.Tensor:
        amplitude = amplitude.to(dtype=torch.float64)
        power = amplitude.square().sum(dim=(-2, -1), keepdim=True)
        if torch.any(power <= 0.0):
            raise ValueError("input amplitude contains a zero-power sample")
        return amplitude / torch.sqrt(power)

    def forward(self, amplitude: torch.Tensor) -> TorchForwardResult:
        if amplitude.ndim == 2:
            amplitude = amplitude.unsqueeze(0)
        if amplitude.ndim != 3:
            raise ValueError("amplitude must have shape (batch, rows, cols)")
        if amplitude.shape[-2:] != (
            self.config.field_size,
            self.config.field_size,
        ):
            raise ValueError(
                "amplitude spatial shape must be "
                f"({self.config.field_size}, {self.config.field_size})"
            )

        amplitude = self._normalize_input(amplitude)
        input_power = torch.ones(
            amplitude.shape[0],
            dtype=torch.float64,
            device=amplitude.device,
        )

        field = amplitude.to(torch.complex128)
        padding_left = self._inner_start
        padding_right = (
            self.config.padded_size
            - self.config.field_size
            - padding_left
        )
        field = F.pad(
            field,
            (padding_left, padding_right, padding_left, padding_right),
        )

        amplitude_values = self.amplitude_values()
        for layer_index in range(self.config.layer_count):
            layer_amplitude = (
                None
                if amplitude_values is None
                else amplitude_values[layer_index]
            )
            field = self._apply_modulation(
                field,
                self.phases[layer_index],
                layer_amplitude,
            )
            if layer_index < self.config.layer_count - 1:
                field = self._propagate(field, self.layer_transfer)

        detector_transfer = (
            self.layer_transfer
            if self._shared_transfer
            else self.detector_transfer
        )
        field = self._propagate(field, detector_transfer)
        detector_inner = field[
            ...,
            self._inner_start:self._inner_stop,
            self._inner_start:self._inner_stop,
        ]
        intensity = detector_inner.abs().square()
        sk = torch.einsum("bij,kij->bk", intensity, self.detector_masks)
        total_region_power = sk.sum(dim=1)
        eta = total_region_power / input_power
        prediction = torch.argmax(sk, dim=1)

        sorted_sk = torch.sort(sk, dim=1).values
        largest = sorted_sk[:, -1]
        second_largest = (
            sorted_sk[:, -2]
            if sk.shape[1] > 1
            else torch.zeros_like(largest)
        )
        eps = torch.finfo(sk.dtype).eps
        contrast = torch.where(
            largest <= 0.0,
            torch.zeros_like(largest),
            torch.where(
                second_largest <= eps,
                torch.full_like(largest, float("inf")),
                largest / second_largest,
            ),
        )

        return TorchForwardResult(
            detector_full=field,
            detector_inner=detector_inner,
            sk=sk,
            input_power=input_power,
            eta=eta,
            prediction=prediction,
            contrast=contrast,
        )

    def loss(self, sk: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean squared loss with Pin=1 per sample in this implementation."""

        labels = labels.to(dtype=torch.long, device=sk.device)
        targets = F.one_hot(labels, num_classes=len(self.config.detector_regions))
        targets = targets.to(dtype=sk.dtype)
        return ((sk - targets) ** 2).mean(dim=1).mean()

    def wrap_phases_(self) -> None:
        """Wrap trainable phases into the configured interval."""

        with torch.no_grad():
            self.phases.remainder_(self.config.phase_span_rad())
