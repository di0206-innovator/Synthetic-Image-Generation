"""
Core procedural image synthesis and automatic Computer Vision ground-truth annotation engine.
Implemented exclusively in vectorized NumPy using Signed Distance Fields (SDFs),
zero-copy broadcasting, and alpha-blended compositing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


ColorRGB = Tuple[int, int, int] | Sequence[int]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned 2D bounding box in pixel coordinates."""

    x_min: int
    y_min: int
    width: int
    height: int

    @property
    def x_max(self) -> int:
        return self.x_min + self.width

    @property
    def y_max(self) -> int:
        return self.y_min + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def to_coco(self) -> List[int]:
        return [int(self.x_min), int(self.y_min), int(self.width), int(self.height)]

    def to_yolo(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        x_center = (self.x_min + self.width / 2.0) / img_width
        y_center = (self.y_min + self.height / 2.0) / img_height
        norm_w = self.width / img_width
        norm_h = self.height / img_height
        return (
            round(float(x_center), 6),
            round(float(y_center), 6),
            round(float(norm_w), 6),
            round(float(norm_h), 6),
        )


@dataclass(frozen=True)
class ObjectAnnotation:
    """Ground-truth object metadata for synthetic CV training datasets."""

    instance_id: int
    category_id: int
    category_name: str
    bbox: BoundingBox
    pixel_count: int
    color: Tuple[int, int, int]


class SyntheticImageGenerator:
    """
    A modular, deterministic synthetic image and CV annotation generator built exclusively with NumPy.

    Features:
    - Zero-copy broadcast linear and radial gradients
    - Sub-pixel anti-aliased Signed Distance Field (SDF) primitives (circles, rings, rectangles, regular polygons)
    - Alpha-blended layer compositing (`out = src * alpha + dst * (1 - alpha)`)
    - Vanishing-point 3D perspective grids & multi-octave procedural noise
    - Automatic semantic segmentation mask & COCO/YOLO bounding-box generation
    """

    CATEGORY_MAP: Dict[str, int] = {
        "background": 0,
        "rectangle": 1,
        "circle": 2,
        "ring": 3,
        "polygon": 4,
    }

    def __init__(
        self,
        height: int = 200,
        width: int = 200,
        bg_color: ColorRGB = (20, 20, 30),
        seed: Optional[int] = 42,
    ) -> None:
        if height <= 0 or width <= 0:
            raise ValueError(f"Height and width must be positive integers, got ({height}, {width}).")
        self.height = int(height)
        self.width = int(width)
        self.rng = np.random.default_rng(seed)
        self.img: NDArray[np.uint8] = np.full(
            (self.height, self.width, 3), fill_value=bg_color, dtype=np.uint8
        )
        # Pixel-level semantic segmentation mask (0 = background, 1..K = category_id)
        self.segmentation_mask: NDArray[np.uint8] = np.zeros(
            (self.height, self.width), dtype=np.uint8
        )
        # Pixel-level instance mask (0 = background, 1..N = unique object instance ID)
        self.instance_mask: NDArray[np.uint16] = np.zeros(
            (self.height, self.width), dtype=np.uint16
        )
        self.annotations: List[ObjectAnnotation] = []

    def _alpha_composite(
        self,
        color: ColorRGB,
        alpha_map: NDArray[np.float32],
    ) -> None:
        """Vectorized alpha blending onto `self.img` using `out = src * a + dst * (1 - a)`."""
        alpha_3d = np.clip(alpha_map, 0.0, 1.0)[:, :, None]
        src_rgb = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        dst_rgb = self.img.astype(np.float32)
        blended = src_rgb * alpha_3d + dst_rgb * (1.0 - alpha_3d)
        self.img = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    def _record_annotation(
        self,
        category_name: str,
        binary_mask: NDArray[np.bool_],
        color: ColorRGB,
    ) -> None:
        """Records pixel-accurate segmentation masks and bounding box coordinates."""
        pixel_count = int(np.count_nonzero(binary_mask))
        if pixel_count == 0:
            return
        cat_id = self.CATEGORY_MAP.get(category_name, 1)
        instance_id = len(self.annotations) + 1
        self.segmentation_mask[binary_mask] = cat_id
        self.instance_mask[binary_mask] = instance_id

        ys, xs = np.nonzero(binary_mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox = BoundingBox(
            x_min=x_min,
            y_min=y_min,
            width=x_max - x_min + 1,
            height=y_max - y_min + 1,
        )
        rgb_tuple = (int(color[0]), int(color[1]), int(color[2]))
        self.annotations.append(
            ObjectAnnotation(
                instance_id=instance_id,
                category_id=cat_id,
                category_name=category_name,
                bbox=bbox,
                pixel_count=pixel_count,
                color=rgb_tuple,
            )
        )

    def add_vertical_gradient(
        self,
        top_color: ColorRGB,
        bottom_color: ColorRGB,
        alpha: float = 1.0,
    ) -> "SyntheticImageGenerator":
        """
        Creates a smooth vertical gradient using zero-copy NumPy broadcasting
        without redundant intermediate `np.repeat` allocations.
        """
        top = np.asarray(top_color, dtype=np.float32).reshape(1, 1, 3)
        bottom = np.asarray(bottom_color, dtype=np.float32).reshape(1, 1, 3)
        t = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None, None]
        gradient_1d = (1.0 - t) * top + t * bottom  # Shape (H, 1, 3)

        if alpha >= 1.0:
            self.img[:] = np.clip(np.rint(gradient_1d), 0, 255).astype(np.uint8)
        else:
            a = float(np.clip(alpha, 0.0, 1.0))
            blended = gradient_1d * a + self.img.astype(np.float32) * (1.0 - a)
            self.img[:] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        return self

    def add_radial_gradient(
        self,
        center_y: float,
        center_x: float,
        radius: float,
        inner_color: ColorRGB,
        outer_color: Optional[ColorRGB] = None,
        alpha: float = 1.0,
    ) -> "SyntheticImageGenerator":
        """Adds a smooth radial glow or gradient around (center_y, center_x)."""
        y, x = np.ogrid[: self.height, : self.width]
        dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2).astype(np.float32)
        t = np.clip(dist / max(radius, 1e-5), 0.0, 1.0)
        # Smoothstep falloff: 1 - (3t^2 - 2t^3)
        falloff = 1.0 - (3.0 * t**2 - 2.0 * t**3)

        if outer_color is None:
            self._alpha_composite(inner_color, falloff * float(alpha))
        else:
            inner = np.asarray(inner_color, dtype=np.float32).reshape(1, 1, 3)
            outer = np.asarray(outer_color, dtype=np.float32).reshape(1, 1, 3)
            grad = falloff[:, :, None] * inner + (1.0 - falloff[:, :, None]) * outer
            a = float(np.clip(alpha, 0.0, 1.0))
            blended = grad * a + self.img.astype(np.float32) * (1.0 - a)
            self.img = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        return self

    def add_rectangle(
        self,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
        color: ColorRGB,
        alpha: float = 1.0,
        track_annotation: bool = True,
    ) -> "SyntheticImageGenerator":
        """Draws a rectangle with optional alpha blending and ground-truth tracking."""
        y_start, y_end = max(0, int(y1)), min(self.height, int(y2))
        x_start, x_end = max(0, int(x1)), min(self.width, int(x2))
        if y_start >= y_end or x_start >= x_end:
            return self

        if alpha >= 1.0:
            self.img[y_start:y_end, x_start:x_end] = color
        else:
            mask_f = np.zeros((self.height, self.width), dtype=np.float32)
            mask_f[y_start:y_end, x_start:x_end] = float(np.clip(alpha, 0.0, 1.0))
            self._alpha_composite(color, mask_f)

        if track_annotation:
            binary_mask = np.zeros((self.height, self.width), dtype=bool)
            binary_mask[y_start:y_end, x_start:x_end] = True
            self._record_annotation("rectangle", binary_mask, color)
        return self

    def add_circle(
        self,
        center_y: float,
        center_x: float,
        radius: float,
        color: ColorRGB,
        anti_alias: bool = True,
        alpha: float = 1.0,
        track_annotation: bool = True,
    ) -> "SyntheticImageGenerator":
        """
        Draws a circle using Signed Distance Field (SDF) evaluation.
        When `anti_alias=True`, sub-pixel boundary distances [r - 0.5, r + 0.5] are
        smoothly feathered via alpha compositing to eliminate staircase aliasing.
        """
        y, x = np.ogrid[: self.height, : self.width]
        dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2).astype(np.float32)
        binary_mask = dist <= radius

        if anti_alias:
            sdf_alpha = np.clip(radius + 0.5 - dist, 0.0, 1.0) * float(alpha)
            self._alpha_composite(color, sdf_alpha)
        else:
            if alpha >= 1.0:
                self.img[binary_mask] = color
            else:
                self._alpha_composite(color, binary_mask.astype(np.float32) * float(alpha))

        if track_annotation:
            self._record_annotation("circle", binary_mask, color)
        return self

    def add_ring(
        self,
        center_y: float,
        center_x: float,
        outer_radius: float,
        inner_radius: float,
        color: ColorRGB,
        anti_alias: bool = True,
        alpha: float = 1.0,
        track_annotation: bool = True,
    ) -> "SyntheticImageGenerator":
        """Draws an anti-aliased annulus/ring using Signed Distance Field math."""
        y, x = np.ogrid[: self.height, : self.width]
        dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2).astype(np.float32)
        half_width = 0.5 * (outer_radius - inner_radius)
        mid_radius = 0.5 * (outer_radius + inner_radius)
        sdf = np.abs(dist - mid_radius) - half_width
        binary_mask = sdf <= 0.0

        if anti_alias:
            ring_alpha = np.clip(0.5 - sdf, 0.0, 1.0) * float(alpha)
            self._alpha_composite(color, ring_alpha)
        else:
            self.img[binary_mask] = color

        if track_annotation:
            self._record_annotation("ring", binary_mask, color)
        return self

    def add_regular_polygon(
        self,
        center_y: float,
        center_x: float,
        radius: float,
        num_sides: int,
        color: ColorRGB,
        rotation_deg: float = 0.0,
        anti_alias: bool = True,
        alpha: float = 1.0,
        track_annotation: bool = True,
    ) -> "SyntheticImageGenerator":
        """
        Draws a regular N-sided polygon (triangle, diamond, hexagon, etc.)
        using vectorized half-plane Signed Distance Field intersection.
        """
        if num_sides < 3:
            raise ValueError("num_sides must be >= 3.")
        y, x = np.mgrid[: self.height, : self.width].astype(np.float32)
        dy = y - float(center_y)
        dx = x - float(center_x)

        angles = np.deg2rad(rotation_deg) + np.linspace(
            0.0, 2.0 * np.pi, num_sides, endpoint=False, dtype=np.float32
        )
        normals_x = np.cos(angles)[:, None, None]
        normals_y = np.sin(angles)[:, None, None]

        apothem = float(radius) * np.cos(np.pi / num_sides)
        # Signed distance to the furthest half-plane boundary
        projections = dx[None, :, :] * normals_x + dy[None, :, :] * normals_y
        sdf = np.max(projections, axis=0) - apothem
        binary_mask = sdf <= 0.0

        if anti_alias:
            poly_alpha = np.clip(0.5 - sdf, 0.0, 1.0) * float(alpha)
            self._alpha_composite(color, poly_alpha)
        else:
            self.img[binary_mask] = color

        if track_annotation:
            self._record_annotation("polygon", binary_mask, color)
        return self

    def add_stripes(
        self,
        start_y: int,
        end_y: int,
        step: int = 8,
        thickness: int = 2,
        color: ColorRGB = (255, 255, 255),
    ) -> "SyntheticImageGenerator":
        """Draws horizontal stripes safely within canvas bounds using slice strides."""
        y0 = max(0, int(start_y))
        y1 = min(self.height, int(end_y))
        for offset in range(max(1, int(thickness))):
            if y0 + offset < y1:
                self.img[y0 + offset : y1 : max(1, int(step)), :] = color
        return self

    def add_perspective_grid(
        self,
        horizon_y: int,
        floor_color: ColorRGB = (16, 12, 32),
        line_color_h: ColorRGB = (255, 0, 128),
        line_color_v: ColorRGB = (0, 220, 255),
        num_rays: int = 13,
        alpha: float = 0.85,
    ) -> "SyntheticImageGenerator":
        """
        Renders a 3D vanishing-point perspective floor grid below `horizon_y`
        using purely vectorized coordinate projection.
        """
        hy = int(np.clip(horizon_y, 0, self.height - 1))
        self.add_rectangle(hy, self.height, 0, self.width, floor_color, track_annotation=False)

        floor_h = self.height - hy
        if floor_h <= 2:
            return self

        y_local = np.arange(floor_h, dtype=np.float32)[:, None]
        x_coords = np.arange(self.width, dtype=np.float32)[None, :]
        center_x = self.width / 2.0

        # Perspective depth parameter z in (0, 1]
        z = np.maximum(y_local / float(floor_h), 0.04)

        # 1. Converging radial rays from vanishing point (center_x, horizon_y)
        bottom_targets = np.linspace(-self.width * 0.8, self.width * 1.8, num_rays)
        ray_alpha = np.zeros((floor_h, self.width), dtype=np.float32)
        for target_x in bottom_targets:
            line_x = center_x + (target_x - center_x) * z
            dist_x = np.abs(x_coords - line_x)
            ray_alpha = np.maximum(ray_alpha, np.clip(1.2 - dist_x, 0.0, 1.0))

        full_ray_mask = np.zeros((self.height, self.width), dtype=np.float32)
        full_ray_mask[hy:, :] = ray_alpha * float(alpha)
        self._alpha_composite(line_color_v, full_ray_mask)

        # 2. Foreshortened horizontal scanlines (quadratic spacing for depth illusion)
        t_steps = (np.linspace(0.12, 1.0, 7) ** 1.8) * (floor_h - 1)
        h_mask = np.zeros((self.height, self.width), dtype=np.float32)
        for dy in t_steps:
            row_idx = hy + int(round(float(dy)))
            if 0 <= row_idx < self.height:
                h_mask[row_idx, :] = float(alpha)
        self._alpha_composite(line_color_h, h_mask)
        return self

    def add_fractal_noise(
        self,
        amplitude: float = 18.0,
        octaves: int = 3,
    ) -> "SyntheticImageGenerator":
        """
        Synthesizes smooth multi-octave procedural value noise (clouds/texture)
        using vectorized bilinear grid interpolation in pure NumPy.
        """
        accum = np.zeros((self.height, self.width), dtype=np.float32)
        weight = 1.0
        total_weight = 0.0

        y_idx = np.linspace(0.0, 1.0, self.height, endpoint=False, dtype=np.float32)
        x_idx = np.linspace(0.0, 1.0, self.width, endpoint=False, dtype=np.float32)

        for octave in range(max(1, int(octaves))):
            freq = 4 * (2**octave)
            grid = self.rng.uniform(-1.0, 1.0, size=(freq + 1, freq + 1)).astype(np.float32)

            gy = y_idx * freq
            gx = x_idx * freq
            y0 = gy.astype(np.int32)
            x0 = gx.astype(np.int32)
            fy = (gy - y0)[:, None]
            fx = (gx - x0)[None, :]
            # Smoothstep weights
            uy = fy * fy * (3.0 - 2.0 * fy)
            ux = fx * fx * (3.0 - 2.0 * fx)

            c00 = grid[y0[:, None], x0[None, :]]
            c10 = grid[y0[:, None], x0[None, :] + 1]
            c01 = grid[y0[:, None] + 1, x0[None, :]]
            c11 = grid[y0[:, None] + 1, x0[None, :] + 1]

            top = c00 * (1.0 - ux) + c10 * ux
            bot = c01 * (1.0 - ux) + c11 * ux
            layer = top * (1.0 - uy) + bot * uy

            accum += layer * weight
            total_weight += weight
            weight *= 0.5

        normalized = (accum / max(total_weight, 1e-6)) * float(amplitude)
        img_f = self.img.astype(np.float32) + normalized[:, :, None]
        self.img = np.clip(np.rint(img_f), 0, 255).astype(np.uint8)
        return self

    def add_gaussian_noise(self, sigma: float = 8.0) -> "SyntheticImageGenerator":
        """Injects zero-mean Gaussian sensor noise using the instance's isolated RNG."""
        if sigma <= 0:
            return self
        noise = self.rng.normal(0.0, float(sigma), size=(self.height, self.width, 3))
        img_f = self.img.astype(np.float32) + noise
        self.img = np.clip(np.rint(img_f), 0, 255).astype(np.uint8)
        return self

    def get_image(self) -> NDArray[np.uint8]:
        """Returns a copy of the generated `(H, W, 3)` `uint8` image array."""
        return self.img.copy()

    def get_segmentation_mask(self) -> NDArray[np.uint8]:
        """Returns the `(H, W)` `uint8` class segmentation mask (0=bg, 1..K=classes)."""
        return self.segmentation_mask.copy()

    def get_colorized_mask(self) -> NDArray[np.uint8]:
        """Maps the semantic segmentation mask to a high-contrast RGB palette for visualization."""
        palette = np.array(
            [
                [18, 22, 32],     # 0: background
                [46, 204, 113],   # 1: rectangle (emerald)
                [255, 195, 0],    # 2: circle (amber)
                [52, 152, 219],   # 3: ring (cyan-blue)
                [231, 76, 60],    # 4: polygon (coral)
            ],
            dtype=np.uint8,
        )
        safe_indices = np.clip(self.segmentation_mask, 0, len(palette) - 1)
        return palette[safe_indices]

    def render_annotations_overlay(self, thickness: int = 2) -> NDArray[np.uint8]:
        """Returns a copy of the image with ground-truth bounding boxes drawn over each object."""
        canvas = self.img.copy()
        t = max(1, int(thickness))
        box_colors = {
            1: (46, 255, 130),
            2: (255, 230, 0),
            3: (0, 220, 255),
            4: (255, 90, 90),
        }
        for ann in self.annotations:
            b = ann.bbox
            c = box_colors.get(ann.category_id, (255, 255, 255))
            # Top & bottom edges
            canvas[b.y_min : min(self.height, b.y_min + t), b.x_min : b.x_max] = c
            canvas[max(0, b.y_max - t) : b.y_max, b.x_min : b.x_max] = c
            # Left & right edges
            canvas[b.y_min : b.y_max, b.x_min : min(self.width, b.x_min + t)] = c
            canvas[b.y_min : b.y_max, max(0, b.x_max - t) : b.x_max] = c
        return canvas

    def export_coco_annotations(self, image_id: int = 1, file_name: str = "synthetic.png") -> Dict[str, Any]:
        """Exports ground-truth bounding boxes and metadata in standard MS-COCO JSON schema."""
        categories = [
            {"id": cid, "name": cname}
            for cname, cid in self.CATEGORY_MAP.items()
            if cid != 0
        ]
        ann_list = []
        for ann in self.annotations:
            ann_list.append(
                {
                    "id": ann.instance_id,
                    "image_id": image_id,
                    "category_id": ann.category_id,
                    "category_name": ann.category_name,
                    "bbox": ann.bbox.to_coco(),
                    "area": ann.bbox.area,
                    "pixel_count": ann.pixel_count,
                    "iscrowd": 0,
                }
            )
        return {
            "images": [
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": self.width,
                    "height": self.height,
                }
            ],
            "categories": categories,
            "annotations": ann_list,
        }

    def export_yolo_annotations(self) -> List[str]:
        """Exports normalized `<class_id> <x_center> <y_center> <width> <height>` YOLO lines."""
        lines: List[str] = []
        for ann in self.annotations:
            xc, yc, w, h = ann.bbox.to_yolo(self.width, self.height)
            # Zero-index class IDs for YOLO (category_id - 1)
            lines.append(f"{ann.category_id - 1} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        return lines
