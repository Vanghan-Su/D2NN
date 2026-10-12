"""Sampling grids, zero-padding, cropping, and angular-spectrum propagation."""

from __future__ import annotations

import numpy as np


def frequency_grid_1d(size: int, pixel_size_m: float, dtype=np.float64) -> np.ndarray:
    """Return signed spatial frequencies using NumPy's native FFT ordering."""

    if size <= 0:
        raise ValueError("size must be positive")
    if pixel_size_m <= 0.0:
        raise ValueError("pixel_size_m must be positive")
    return np.fft.fftfreq(size, d=pixel_size_m).astype(dtype, copy=False)


def frequency_grid_2d(
    size: int,
    pixel_size_m: float,
    dtype=np.float64,
) -> tuple:
    """Return (fx, fy) in cycles/metre with shape (size, size)."""

    fx = frequency_grid_1d(size, pixel_size_m, dtype=dtype)
    fy = frequency_grid_1d(size, pixel_size_m, dtype=dtype)
    return np.meshgrid(fx, fy, indexing="ij")


def angular_spectrum_transfer_function(config, distance_m: float) -> np.ndarray:
    """Return the sampled angular-spectrum transfer function.

    The returned array follows NumPy's unshifted FFT frequency ordering. It
    uses the propagating convention:

    H = exp(j * 2*pi*d*sqrt(1/lambda^2 - fx^2 - fy^2))

    and sets evanescent frequencies to zero.
    """

    if distance_m < 0.0:
        raise ValueError("distance_m must be non-negative")

    size = config.padded_size
    if distance_m == 0.0:
        return np.ones((size, size), dtype=np.complex128)

    fx, fy = frequency_grid_2d(size, config.pixel_size_m)
    inverse_wavelength_sq = 1.0 / (config.wavelength_m * config.wavelength_m)
    radial_sq = fx * fx + fy * fy
    kz_sq = inverse_wavelength_sq - radial_sq
    propagating = kz_sq >= 0.0
    kz = np.sqrt(np.maximum(kz_sq, 0.0))
    phase = 2.0 * np.pi * distance_m * kz
    transfer = np.exp(1j * phase).astype(np.complex128, copy=False)
    transfer[~propagating] = 0.0
    return transfer


def propagate_angular_spectrum(
    field: np.ndarray,
    distance_m: float,
    config,
    transfer_function: np.ndarray = None,
) -> np.ndarray:
    """Propagate a complex field with the sampled angular-spectrum method."""

    if field.ndim != 2:
        raise ValueError("field must be two-dimensional")
    if field.shape != (config.padded_size, config.padded_size):
        raise ValueError(
            "field shape must equal (padded_size, padded_size), got "
            f"{field.shape}"
        )

    if distance_m == 0.0:
        return np.asarray(field, dtype=np.complex128).copy()

    transfer = (
        angular_spectrum_transfer_function(config, distance_m)
        if transfer_function is None
        else transfer_function
    )
    if transfer.shape != field.shape:
        raise ValueError("transfer_function shape must match field shape")

    field_fft = np.fft.fft2(field)
    propagated_fft = field_fft * transfer
    return np.fft.ifft2(propagated_fft).astype(np.complex128, copy=False)


def pad_center(inner_field: np.ndarray, padded_size: int) -> np.ndarray:
    """Place an inner field at the centre of a larger zero-filled array."""

    if inner_field.ndim != 2:
        raise ValueError("inner_field must be two-dimensional")
    inner_rows, inner_cols = inner_field.shape
    if inner_rows > padded_size or inner_cols > padded_size:
        raise ValueError("inner_field is larger than padded_size")

    padded = np.zeros(
        (padded_size, padded_size),
        dtype=np.asarray(inner_field).dtype,
    )
    row_start = (padded_size - inner_rows) // 2
    col_start = (padded_size - inner_cols) // 2
    padded[
        row_start:row_start + inner_rows,
        col_start:col_start + inner_cols,
    ] = inner_field
    return padded


def crop_center(field: np.ndarray, inner_size: int) -> np.ndarray:
    """Return the central inner_size x inner_size subarray."""

    if field.ndim != 2:
        raise ValueError("field must be two-dimensional")
    rows, cols = field.shape
    if inner_size > rows or inner_size > cols:
        raise ValueError("inner_size exceeds field dimensions")
    row_start = (rows - inner_size) // 2
    col_start = (cols - inner_size) // 2
    return field[
        row_start:row_start + inner_size,
        col_start:col_start + inner_size,
    ]
