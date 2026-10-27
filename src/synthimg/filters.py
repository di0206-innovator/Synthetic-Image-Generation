"""
Vectorized 2D spatial convolution and post-processing filters implemented
exclusively with NumPy (`sliding_window_view` and `np.einsum`).
Zero Python loops over spatial coordinates.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


def convolve2d_vectorized(
    img: NDArray[np.uint8 | np.floating],
    kernel: NDArray[np.floating],
    clip_uint8: bool = True,
) -> NDArray[np.uint8 | np.float32]:
    """
    Applies a 2D spatial convolution kernel to a 2D grayscale `(H, W)` or 3D RGB `(H, W, C)`
    image using zero-copy `sliding_window_view` and Einstein summation (`np.einsum`).

    Parameters
    ----------
    img : NDArray
        Input image of shape `(H, W)` or `(H, W, C)`.
    kernel : NDArray
        2D filter kernel of odd dimensions `(kh, kw)`.
    clip_uint8 : bool
        If True, clamps result to `[0, 255]` and casts back to `np.uint8`.
        If False, returns raw `np.float32` activation map (useful for signed gradients like Sobel).
    """
    k = np.asarray(kernel, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError(f"Kernel must be 2D with odd dimensions, got shape {k.shape}.")

    kh, kw = k.shape
    pad_h, pad_w = kh // 2, kw // 2
    arr = np.asarray(img, dtype=np.float32)

    if arr.ndim == 2:
        padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        windows = sliding_window_view(padded, window_shape=(kh, kw))  # (H, W, kh, kw)
        out = np.einsum("hwij,ij->hw", windows, k, optimize=True)
    elif arr.ndim == 3:
        padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="reflect")
        windows = sliding_window_view(padded, window_shape=(kh, kw), axis=(0, 1))  # (H, W, C, kh, kw)
        out = np.einsum("hwcij,ij->hwc", windows, k, optimize=True)
    else:
        raise ValueError(f"Expected 2D or 3D image array, got {arr.ndim}D.")

    if clip_uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out.astype(np.float32)


def gaussian_kernel_2d(size: int = 5, sigma: float = 1.2) -> NDArray[np.float32]:
    """Constructs a normalized 2D isotropic Gaussian kernel via outer product."""
    if size % 2 == 0 or size < 3:
        raise ValueError("Gaussian kernel size must be an odd integer >= 3.")
    ax = np.linspace(-(size // 2), size // 2, size, dtype=np.float32)
    gauss_1d = np.exp(-0.5 * (ax / max(sigma, 1e-5)) ** 2)
    kernel = np.outer(gauss_1d, gauss_1d)
    return (kernel / np.sum(kernel)).astype(np.float32)


def gaussian_blur(
    img: NDArray[np.uint8],
    size: int = 5,
    sigma: float = 1.2,
) -> NDArray[np.uint8]:
    """Applies vectorized Gaussian smoothing to an image."""
    kernel = gaussian_kernel_2d(size=size, sigma=sigma)
    return convolve2d_vectorized(img, kernel, clip_uint8=True)


def sobel_edges(
    img: NDArray[np.uint8],
) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """
    Computes vectorized Sobel edge magnitude and gradient orientation (radians).
    Converts RGB input to ITU-R BT.601 luminance before computing spatial derivatives.

    Returns
    -------
    magnitude_u8 : NDArray[np.uint8]
        Normalized `[0, 255]` edge magnitude map of shape `(H, W)`.
    orientation_rad : NDArray[np.float32]
        Gradient angle `arctan2(Gy, Gx)` in `[-pi, pi]` of shape `(H, W)`.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        lum = arr

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = convolve2d_vectorized(lum, kx, clip_uint8=False)
    gy = convolve2d_vectorized(lum, ky, clip_uint8=False)

    mag = np.hypot(gx, gy)
    max_val = float(mag.max())
    if max_val > 1e-6:
        mag_norm = (mag / max_val) * 255.0
    else:
        mag_norm = mag

    angle = np.arctan2(gy, gx).astype(np.float32)
    return np.clip(np.rint(mag_norm), 0, 255).astype(np.uint8), angle


def sharpen(img: NDArray[np.uint8], strength: float = 1.0) -> NDArray[np.uint8]:
    """Applies an unsharp-mask / Laplacian sharpening kernel."""
    s = float(strength)
    kernel = np.array(
        [
            [0.0, -s, 0.0],
            [-s, 1.0 + 4.0 * s, -s],
            [0.0, -s, 0.0],
        ],
        dtype=np.float32,
    )
    return convolve2d_vectorized(img, kernel, clip_uint8=True)


def emboss(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Applies a directional relief emboss filter biased around mid-gray (128)."""
    kernel = np.array(
        [
            [-2.0, -1.0, 0.0],
            [-1.0, 1.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    raw = convolve2d_vectorized(img, kernel, clip_uint8=False) + 128.0
    return np.clip(np.rint(raw), 0, 255).astype(np.uint8)


def chromatic_aberration(img: NDArray[np.uint8], shift_px: int = 3) -> NDArray[np.uint8]:
    """Simulates optical lens dispersion by horizontally shifting Red and Blue channels."""
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Chromatic aberration requires an RGB image of shape (H, W, 3).")
    s = int(shift_px)
    if s == 0:
        return img.copy()
    out = img.copy()
    out[:, :, 0] = np.roll(img[:, :, 0], shift=-s, axis=1)
    out[:, :, 2] = np.roll(img[:, :, 2], shift=s, axis=1)
    return out


def vignette(img: NDArray[np.uint8], strength: float = 0.55) -> NDArray[np.uint8]:
    """Applies smooth radial lens vignetting toward the corners of the frame."""
    h, w = img.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    radius_sq = x**2 + y**2
    mask = np.clip(1.0 - float(strength) * (radius_sq / 1.5), 0.15, 1.0)
    if img.ndim == 3:
        mask = mask[:, :, None]
    out = img.astype(np.float32) * mask
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
