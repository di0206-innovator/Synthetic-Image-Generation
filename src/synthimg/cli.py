"""
Command-Line Interface (CLI) for `synthimg`.
Usage:
    python -m synthimg.cli --preset synthwave --height 512 --width 512 --out output.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from .engine import SyntheticImageGenerator
from .filters import chromatic_aberration, vignette


def build_preset(
    preset: str,
    height: int = 400,
    width: int = 400,
    seed: int = 42,
) -> SyntheticImageGenerator:
    """Constructs a pre-configured procedural scene using `SyntheticImageGenerator`."""
    gen = SyntheticImageGenerator(height=height, width=width, seed=seed)

    if preset == "synthwave":
        gen.add_vertical_gradient(top_color=(16, 8, 42), bottom_color=(245, 90, 75))
        gen.add_fractal_noise(amplitude=10.0, octaves=3)
        cy, cx = int(height * 0.52), int(width * 0.50)
        sun_r = int(min(height, width) * 0.24)
        gen.add_radial_gradient(cy, cx, sun_r * 1.6, inner_color=(255, 140, 80), alpha=0.45)
        gen.add_circle(cy, cx, sun_r, color=(255, 220, 65), anti_alias=True)
        gen.add_stripes(
            start_y=int(cy - sun_r * 0.2),
            end_y=int(cy + sun_r),
            step=max(4, height // 25),
            thickness=max(2, height // 90),
            color=(235, 85, 80),
        )
        gen.add_perspective_grid(
            horizon_y=int(height * 0.70),
            floor_color=(14, 10, 30),
            line_color_h=(255, 0, 128),
            line_color_v=(0, 230, 255),
            num_rays=15,
        )
        gen.add_gaussian_noise(sigma=4.5)
    elif preset == "cv_dataset":
        gen.add_vertical_gradient(top_color=(28, 34, 48), bottom_color=(18, 22, 30))
        gen.add_fractal_noise(amplitude=12.0, octaves=3)
        # Generate randomized geometric primitives with non-overlapping or semi-occluded positions
        rng = gen.rng
        for _ in range(3):
            ry = int(rng.integers(int(height * 0.12), int(height * 0.65)))
            rx = int(rng.integers(int(width * 0.12), int(width * 0.65)))
            rh = int(rng.integers(int(height * 0.14), int(height * 0.26)))
            rw = int(rng.integers(int(width * 0.14), int(width * 0.26)))
            col = tuple(int(c) for c in rng.integers(80, 250, size=3))
            gen.add_rectangle(ry, ry + rh, rx, rx + rw, color=col, alpha=0.95)

        for _ in range(3):
            cy = int(rng.integers(int(height * 0.2), int(height * 0.8)))
            cx = int(rng.integers(int(width * 0.2), int(width * 0.8)))
            rad = int(rng.integers(int(min(height, width) * 0.08), int(min(height, width) * 0.16)))
            col = tuple(int(c) for c in rng.integers(90, 255, size=3))
            gen.add_circle(cy, cx, rad, color=col, anti_alias=True)

        gen.add_ring(
            center_y=int(height * 0.35),
            center_x=int(width * 0.72),
            outer_radius=int(min(height, width) * 0.14),
            inner_radius=int(min(height, width) * 0.08),
            color=(0, 220, 255),
            anti_alias=True,
        )
        gen.add_regular_polygon(
            center_y=int(height * 0.72),
            center_x=int(width * 0.28),
            radius=int(min(height, width) * 0.14),
            num_sides=6,
            color=(255, 110, 90),
            rotation_deg=15.0,
            anti_alias=True,
        )
        gen.add_gaussian_noise(sigma=6.0)
    else:
        raise ValueError(f"Unknown preset '{preset}'. Choose from ['synthwave', 'cv_dataset'].")

    return gen


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="synthimg: Pure-NumPy Procedural Image & CV Ground-Truth Generator"
    )
    parser.add_argument(
        "--preset",
        choices=["synthwave", "cv_dataset"],
        default="synthwave",
        help="Procedural scene preset to generate.",
    )
    parser.add_argument("--height", type=int, default=400, help="Canvas height in pixels.")
    parser.add_argument("--width", type=int, default=400, help="Canvas width in pixels.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument("--out", type=str, default="synthetic_output.png", help="Output PNG path.")
    parser.add_argument(
        "--export-coco",
        type=str,
        default=None,
        help="Optional file path to save COCO ground-truth bounding box annotations JSON.",
    )
    parser.add_argument(
        "--lens-fx",
        action="store_true",
        help="Apply optical chromatic aberration and radial vignette post-processing.",
    )

    args = parser.parse_args(argv)
    gen = build_preset(args.preset, height=args.height, width=args.width, seed=args.seed)
    img = gen.get_image()
    if args.lens_fx:
        img = vignette(chromatic_aberration(img, shift_px=2), strength=0.45)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"Saved synthetic image ({img.shape[1]}x{img.shape[0]}) -> {out_path}")

    if args.export_coco:
        coco_path = Path(args.export_coco)
        coco_path.parent.mkdir(parents=True, exist_ok=True)
        coco_data = gen.export_coco_annotations(file_name=out_path.name)
        coco_path.write_text(json.dumps(coco_data, indent=2), encoding="utf-8")
        print(f"Exported {len(gen.annotations)} COCO annotations -> {coco_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
