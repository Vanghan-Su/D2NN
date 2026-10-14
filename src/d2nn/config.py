"""Simulation configuration and physical constants."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Tuple


PURE_PHASE_METHOD = "纯相位法"
COMPLEX_AMPLITUDE_METHOD = "复振幅调制"
COMPLEX_AMPLITUDE_PRETRAIN_METHOD = COMPLEX_AMPLITUDE_METHOD


@dataclass(frozen=True)
class DetectorRegion:
    """One rectangular detector region in top-left detector coordinates."""

    label: int
    center_x_m: float
    center_y_m: float
    width_m: float
    height_m: float

    def validate(self) -> None:
        if self.label < 0:
            raise ValueError("detector label must be non-negative")
        if self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("detector region dimensions must be positive")
        if self.center_x_m < 0.0 or self.center_y_m < 0.0:
            raise ValueError("detector region center must use top-left coordinates")


DEFAULT_DETECTOR_REGIONS = (
    DetectorRegion(0, 0.487e-3, 0.143e-3, 0.170e-3, 0.180e-3),
    DetectorRegion(1, 0.225e-3, 0.370e-3, 0.170e-3, 0.180e-3),
    DetectorRegion(2, 0.470e-3, 0.370e-3, 0.170e-3, 0.180e-3),
    DetectorRegion(3, 0.735e-3, 0.370e-3, 0.170e-3, 0.180e-3),
    DetectorRegion(4, 0.225e-3, 0.615e-3, 0.170e-3, 0.180e-3),
    DetectorRegion(5, 0.470e-3, 0.615e-3, 0.170e-3, 0.180e-3),
    DetectorRegion(6, 0.735e-3, 0.615e-3, 0.170e-3, 0.180e-3),
    DetectorRegion(7, 0.225e-3, 0.835e-3, 0.170e-3, 0.180e-3),
    DetectorRegion(8, 0.470e-3, 0.835e-3, 0.170e-3, 0.180e-3),
    DetectorRegion(9, 0.735e-3, 0.835e-3, 0.170e-3, 0.180e-3),
)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the free-space D2NN.

    All physical lengths are stored in metres. The detector coordinate system
    uses its upper-left corner as the origin, with x increasing to the right
    and y increasing downwards.
    """

    wavelength_m: float = 632.8e-9
    pixel_size_m: float = 2.5e-6
    field_size: int = 400
    layer_count: int = 5
    layer_spacing_m: float = 5.0e-3
    detector_spacing_m: float = 10.0e-3
    padded_size: int = 2048
    amplitude_transmission: float = 1.0
    phase_min_rad: float = 0.0
    phase_max_rad: float = 2.0 * math.pi
    seed: int = 20261012
    detector_regions: Tuple[DetectorRegion, ...] = field(
        default_factory=lambda: DEFAULT_DETECTOR_REGIONS
    )

    @property
    def field_extent_m(self) -> float:
        return self.field_size * self.pixel_size_m

    @property
    def padded_extent_m(self) -> float:
        return self.padded_size * self.pixel_size_m

    @property
    def layer_positions_m(self) -> Tuple[float, ...]:
        return tuple(
            layer_index * self.layer_spacing_m
            for layer_index in range(self.layer_count)
        )

    @property
    def detector_position_m(self) -> float:
        return (
            (self.layer_count - 1) * self.layer_spacing_m
            + self.detector_spacing_m
        )

    @property
    def normalized_frequency_limit(self) -> float:
        return self.wavelength_m / (2.0 * self.pixel_size_m)

    @property
    def minimum_padded_size(self) -> int:
        """Minimum size for transfer-function phase sampling.

        This is the criterion from the simulation document. It is necessary
        but not sufficient for a finite-aperture, multi-layer propagation
        because it does not bound the transverse spread of the field.
        """

        max_distance = max(self.layer_spacing_m, self.detector_spacing_m)
        if max_distance <= 0.0:
            return 1
        return int(
            math.ceil(
                max_distance
                * self.wavelength_m
                / (self.pixel_size_m * self.pixel_size_m)
            )
        )

    @property
    def total_propagation_m(self) -> float:
        return (
            (self.layer_count - 1) * self.layer_spacing_m
            + self.detector_spacing_m
        )

    @property
    def support_padded_size(self) -> int:
        """Finite-aperture support size over the complete optical path.

        The maximum in-plane spatial frequency is 1/(2*dx). The corresponding
        ray angle over the complete path gives a transverse displacement. To
        avoid periodic wrap-around, the padded window must contain the input
        aperture plus both positive and negative displacements.
        """

        sine = min(self.normalized_frequency_limit, 1.0 - 1.0e-15)
        tangent = sine / math.sqrt(max(1.0 - sine * sine, 1.0e-30))
        transverse_spread = self.total_propagation_m * tangent
        required_width = self.field_extent_m + 2.0 * transverse_spread
        return int(math.ceil(required_width / self.pixel_size_m))

    def validate(self) -> None:
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive")
        if self.pixel_size_m <= 0.0:
            raise ValueError("pixel_size_m must be positive")
        if self.field_size <= 0:
            raise ValueError("field_size must be positive")
        if self.layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if self.layer_spacing_m <= 0.0 or self.detector_spacing_m <= 0.0:
            raise ValueError("propagation distances must be positive")
        if self.padded_size < self.field_size:
            raise ValueError("padded_size must be >= field_size")
        if self.amplitude_transmission < 0.0:
            raise ValueError("amplitude_transmission must be non-negative")
        if self.phase_max_rad <= self.phase_min_rad:
            raise ValueError("phase_max_rad must exceed phase_min_rad")
        if self.normalized_frequency_limit >= 1.0:
            raise ValueError(
                "sampling is not sufficient: wavelength/(2*pixel_size) must be < 1"
            )
        if self.padded_size < self.minimum_padded_size:
            raise ValueError(
                "padded_size is too small for the configured propagation: "
                "padded_size >= "
                f"{self.minimum_padded_size}, got {self.padded_size}"
            )
        if not self.detector_regions:
            raise ValueError("at least one detector region is required")

        labels = [region.label for region in self.detector_regions]
        if labels != list(range(len(labels))):
            raise ValueError("detector labels must be contiguous starting at zero")

        detector_extent = self.field_extent_m
        for region in self.detector_regions:
            region.validate()
            x_min = region.center_x_m - region.width_m / 2.0
            x_max = region.center_x_m + region.width_m / 2.0
            y_min = region.center_y_m - region.height_m / 2.0
            y_max = region.center_y_m + region.height_m / 2.0
            if x_min < 0.0 or x_max > detector_extent:
                raise ValueError(
                    f"detector region {region.label} exceeds detector x extent"
                )
            if y_min < 0.0 or y_max > detector_extent:
                raise ValueError(
                    f"detector region {region.label} exceeds detector y extent"
                )

    @classmethod
    def from_json(cls, path):
        """Load a configuration from a JSON file.

        A JSON file may either omit ``detector_regions`` to use the default
        geometry, or provide a list of objects with the keys ``label``,
        ``center_x_m``, ``center_y_m``, ``width_m`` and ``height_m``.
        """

        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        region_payload = payload.pop("detector_regions", None)
        if region_payload is not None:
            payload["detector_regions"] = tuple(
                DetectorRegion(**item) for item in region_payload
            )
        return cls(**payload)

    def phase_span_rad(self) -> float:
        return self.phase_max_rad - self.phase_min_rad



