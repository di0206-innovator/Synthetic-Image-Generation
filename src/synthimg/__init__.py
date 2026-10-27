"""
synthimg: Pure-NumPy Procedural Rendering, Vectorized Spatial Filtering & Synthetic CV Dataset Engine.
"""

from .engine import BoundingBox, ObjectAnnotation, SyntheticImageGenerator
from .filters import (
    chromatic_aberration,
    convolve2d_vectorized,
    emboss,
    gaussian_blur,
    sharpen,
    sobel_edges,
    vignette,
)
from .benchmarks import inspect_memory_layout, run_vectorization_benchmark

__version__ = "1.0.0"
__all__ = [
    "BoundingBox",
    "ObjectAnnotation",
    "SyntheticImageGenerator",
    "convolve2d_vectorized",
    "gaussian_blur",
    "sobel_edges",
    "sharpen",
    "emboss",
    "chromatic_aberration",
    "vignette",
    "inspect_memory_layout",
    "run_vectorization_benchmark",
]
