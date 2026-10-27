"""
Unit test suite for the `synthimg` package.
Validates tensor shapes, `uint8` overflow safety, deterministic RNG seeding,
zero-copy memory view invariants, vectorized 2D convolutions, and COCO/YOLO exports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from synthimg import (  # noqa: E402
    SyntheticImageGenerator,
    chromatic_aberration,
    convolve2d_vectorized,
    emboss,
    gaussian_blur,
    inspect_memory_layout,
    run_vectorization_benchmark,
    sharpen,
    sobel_edges,
    vignette,
)
from synthimg.cli import build_preset, main as cli_main  # noqa: E402


def test_generator_initialization_and_bounds() -> None:
    gen = SyntheticImageGenerator(height=120, width=160, bg_color=(10, 20, 30), seed=99)
    img = gen.get_image()
    assert img.shape == (120, 160, 3)
    assert img.dtype == np.uint8
    assert np.all(img[0, 0] == np.array([10, 20, 30], dtype=np.uint8))

    with pytest.raises(ValueError):
        SyntheticImageGenerator(height=0, width=100)


def test_deterministic_seeding_reproducibility() -> None:
    gen1 = build_preset("synthwave", height=128, width=128, seed=12345)
    gen2 = build_preset("synthwave", height=128, width=128, seed=12345)
    gen3 = build_preset("synthwave", height=128, width=128, seed=54321)

    assert np.array_equal(gen1.get_image(), gen2.get_image())
    assert not np.array_equal(gen1.get_image(), gen3.get_image())


def test_sdf_shapes_and_ground_truth_annotations() -> None:
    gen = SyntheticImageGenerator(height=200, width=200, seed=7)
    gen.add_rectangle(20, 60, 30, 90, color=(255, 0, 0))
    gen.add_circle(120, 120, 25, color=(0, 255, 0), anti_alias=True)
    gen.add_ring(60, 140, outer_radius=20, inner_radius=10, color=(0, 180, 255))
    gen.add_regular_polygon(150, 50, radius=22, num_sides=6, color=(255, 200, 0))

    assert len(gen.annotations) == 4
    coco = gen.export_coco_annotations()
    assert len(coco["annotations"]) == 4
    first_bbox = coco["annotations"][0]["bbox"]
    assert first_bbox == [30, 20, 60, 40]

    yolo_lines = gen.export_yolo_annotations()
    assert len(yolo_lines) == 4
    parts = yolo_lines[0].split()
    assert len(parts) == 5
    assert int(parts[0]) == 0  # rectangle class index

    seg = gen.get_segmentation_mask()
    assert seg.shape == (200, 200)
    assert set(np.unique(seg)).issubset({0, 1, 2, 3, 4})

    overlay = gen.render_annotations_overlay()
    assert overlay.shape == (200, 200, 3)
    assert overlay.dtype == np.uint8


def test_vectorized_convolution_identity_and_filters() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    # Identity 3x3 kernel must reproduce the original image exactly
    identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    out_id = convolve2d_vectorized(img, identity_kernel, clip_uint8=True)
    assert np.array_equal(img, out_id)

    blurred = gaussian_blur(img, size=5, sigma=1.2)
    assert blurred.shape == img.shape and blurred.dtype == np.uint8
    # Gaussian smoothing reduces high-frequency spatial variance
    assert float(blurred.std()) < float(img.std())

    mag, angle = sobel_edges(img)
    assert mag.shape == (64, 64) and mag.dtype == np.uint8
    assert angle.shape == (64, 64) and angle.dtype == np.float32

    assert sharpen(img).shape == img.shape
    assert emboss(img).shape == img.shape
    assert chromatic_aberration(img, shift_px=2).shape == img.shape
    assert vignette(img, strength=0.5).shape == img.shape


def test_zero_copy_memory_layout_diagnostics() -> None:
    base = np.zeros((200, 200, 3), dtype=np.uint8)
    diagnostics = inspect_memory_layout(base)
    by_name = {d["operation"]: d for d in diagnostics}

    assert by_name["Cropped ROI (img[50:150, 50:150])"]["shares_memory"] is True
    assert by_name["Horizontal Flip (img[:, ::-1, :])"]["shares_memory"] is True
    assert by_name["Horizontal Flip (img[:, ::-1, :])"]["strides"][1] == -3
    assert by_name["Boolean Mask Indexing (img[mask])"]["shares_memory"] is False
    assert by_name["Float32 Type Cast (img.astype(f32))"]["shares_memory"] is False


def test_benchmark_runner_and_cli_execution(tmp_path: Path) -> None:
    bench = run_vectorization_benchmark(resolutions=[("64x64", 64, 64)])
    assert len(bench) == 1
    assert bench[0]["speedup"] > 1.0

    out_png = tmp_path / "out.png"
    out_json = tmp_path / "coco.json"
    rc = cli_main(
        [
            "--preset",
            "cv_dataset",
            "--height",
            "128",
            "--width",
            "128",
            "--out",
            str(out_png),
            "--export-coco",
            str(out_json),
            "--lens-fx",
        ]
    )
    assert rc == 0
    assert out_png.exists()
    assert out_json.exists()
