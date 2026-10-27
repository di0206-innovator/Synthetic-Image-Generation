"""
Memory layout inspection (strides, zero-copy views vs. copies) and
vectorization speedup benchmarks (naive Python loops vs. NumPy broadcasting).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def inspect_memory_layout(
    base_img: NDArray[np.uint8],
) -> List[Dict[str, Any]]:
    """
    Demonstrates how NumPy handles memory buffers, strides, and zero-copy views
    across slicing, negative-stride flipping, channel extraction, and boolean masking.
    """
    cropped = base_img[50:150, 50:150]
    flipped_h = base_img[:, ::-1, :]
    flipped_v = base_img[::-1, :, :]
    red_2d = base_img[:, :, 0]
    mask = base_img[:, :, 0] > 128
    masked_selection = base_img[mask]
    float_cast = base_img.astype(np.float32)

    operations = [
        ("Original Canvas (img)", base_img),
        ("Cropped ROI (img[50:150, 50:150])", cropped),
        ("Horizontal Flip (img[:, ::-1, :])", flipped_h),
        ("Vertical Flip (img[::-1, :, :])", flipped_v),
        ("Red Channel Slice (img[:, :, 0])", red_2d),
        ("Boolean Mask Indexing (img[mask])", masked_selection),
        ("Float32 Type Cast (img.astype(f32))", float_cast),
    ]

    records: List[Dict[str, Any]] = []
    for label, arr in operations:
        shares_mem = (
            True
            if arr is base_img
            else bool(np.shares_memory(base_img, arr))
        )
        records.append(
            {
                "operation": label,
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "strides": arr.strides,
                "nbytes": int(arr.nbytes),
                "shares_memory": shares_mem,
                "allocation": "Zero-Copy View (0 B new)" if shares_mem else f"New Buffer ({arr.nbytes:,} B)",
                "c_contiguous": bool(arr.flags["C_CONTIGUOUS"]),
            }
        )
    return records


def _python_loop_radial_scene(height: int, width: int) -> NDArray[np.uint8]:
    """Baseline pure-Python nested loop implementation of a gradient + circle scene."""
    out = np.empty((height, width, 3), dtype=np.uint8)
    cy, cx = height * 0.5, width * 0.5
    r_sq = (min(height, width) * 0.25) ** 2
    inv_h = 1.0 / max(height - 1, 1)

    for y in range(height):
        t = y * inv_h
        bg_r = int(20 * (1.0 - t) + 245 * t)
        bg_g = int(10 * (1.0 - t) + 95 * t)
        bg_b = int(48 * (1.0 - t) + 75 * t)
        dy_sq = (y - cy) ** 2
        for x in range(width):
            if dy_sq + (x - cx) ** 2 <= r_sq:
                out[y, x, 0] = 255
                out[y, x, 1] = 220
                out[y, x, 2] = 60
            else:
                out[y, x, 0] = bg_r
                out[y, x, 1] = bg_g
                out[y, x, 2] = bg_b
    return out


def _numpy_vectorized_radial_scene(height: int, width: int) -> NDArray[np.uint8]:
    """Vectorized NumPy implementation of the exact same gradient + circle scene."""
    top = np.array([20.0, 10.0, 48.0], dtype=np.float32).reshape(1, 1, 3)
    bot = np.array([245.0, 95.0, 75.0], dtype=np.float32).reshape(1, 1, 3)
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]

    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = ((1.0 - t) * top + t * bot).astype(np.uint8)

    y, x = np.ogrid[:height, :width]
    cy, cx = height * 0.5, width * 0.5
    r_sq = (min(height, width) * 0.25) ** 2
    mask = (y - cy) ** 2 + (x - cx) ** 2 <= r_sq
    canvas[mask] = (255, 220, 60)
    return canvas


def run_vectorization_benchmark(
    resolutions: Sequence[Tuple[str, int, int]] = (
        ("128x128 (Thumbnail)", 128, 128),
        ("256x256 (Standard)", 256, 256),
        ("512x512 (HD Patch)", 512, 512),
        ("1024x1024 (1MP)", 1024, 1024),
    ),
) -> List[Dict[str, Any]]:
    """
    Benchmarks pure-Python pixel loops against NumPy vectorized broadcasting & masking.
    Returns timing metrics in milliseconds and relative speedup factors.
    """
    results: List[Dict[str, Any]] = []
    for label, h, w in resolutions:
        # Time Python nested loop (1 pass for large sizes, 3 passes for small)
        py_runs = 3 if (h * w) <= 65536 else 1
        t0 = time.perf_counter()
        for _ in range(py_runs):
            _python_loop_radial_scene(h, w)
        py_ms = ((time.perf_counter() - t0) / py_runs) * 1000.0

        # Time NumPy vectorized pipeline (warmup + 10 runs)
        _numpy_vectorized_radial_scene(h, w)
        np_runs = 15 if (h * w) <= 262144 else 5
        t1 = time.perf_counter()
        for _ in range(np_runs):
            _numpy_vectorized_radial_scene(h, w)
        np_ms = ((time.perf_counter() - t1) / np_runs) * 1000.0

        speedup = py_ms / max(np_ms, 1e-6)
        results.append(
            {
                "resolution": label,
                "height": h,
                "width": w,
                "pixels": h * w,
                "python_ms": round(py_ms, 2),
                "numpy_ms": round(np_ms, 3),
                "speedup": round(speedup, 1),
            }
        )
    return results
