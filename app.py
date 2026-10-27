"""
Interactive Streamlit Web Studio for `synthimg`.
Run locally via:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure local package is importable without installation
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from synthimg import (  # noqa: E402
    SyntheticImageGenerator,
    chromatic_aberration,
    emboss,
    gaussian_blur,
    inspect_memory_layout,
    sharpen,
    sobel_edges,
    vignette,
)

st.set_page_config(
    page_title="SynthImg Studio | Pure-NumPy Rendering & CV Engine",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 SynthImg Studio: Pure-NumPy Procedural Rendering & CV Dataset Engine")
st.markdown(
    "Generate anti-aliased Signed Distance Field (SDF) scenes, apply zero-loop `np.einsum` "
    "spatial convolutions, and export automatic **COCO / YOLO** ground-truth annotations in real time."
)

with st.sidebar:
    st.header("1. Scene & Canvas Controls")
    preset = st.selectbox("Scene Mode", ["Retro Synthwave", "Synthetic CV Multi-Object Dataset"])
    resolution = st.select_slider("Resolution (H x W)", options=[128, 200, 256, 400, 512], value=256)
    seed = st.number_input("Deterministic RNG Seed", min_value=0, max_value=99999, value=42, step=1)
    noise_sigma = st.slider("Gaussian Sensor Noise (σ)", 0.0, 25.0, 5.0, 0.5)
    fractal_amp = st.slider("Multi-Octave Fractal Cloud Amplitude", 0.0, 35.0, 10.0, 1.0)
    anti_alias = st.checkbox("Sub-Pixel SDF Anti-Aliasing", value=True)

    st.header("2. Vectorized 2D Convolution Filter")
    active_filter = st.selectbox(
        "Post-Processing Kernel (`sliding_window_view` + `einsum`)",
        [
            "None (Raw Render)",
            "Gaussian Blur (5x5)",
            "Sobel Edge Magnitude",
            "Laplacian Sharpen",
            "Directional Emboss",
            "Chromatic Aberration + Vignette",
        ],
    )

gen = SyntheticImageGenerator(height=resolution, width=resolution, seed=int(seed))

if preset == "Retro Synthwave":
    gen.add_vertical_gradient(top_color=(16, 8, 42), bottom_color=(245, 90, 75))
    if fractal_amp > 0:
        gen.add_fractal_noise(amplitude=fractal_amp, octaves=3)
    cy, cx = int(resolution * 0.52), int(resolution * 0.50)
    sun_r = int(resolution * 0.24)
    gen.add_radial_gradient(cy, cx, sun_r * 1.6, inner_color=(255, 140, 80), alpha=0.45)
    gen.add_circle(cy, cx, sun_r, color=(255, 220, 65), anti_alias=anti_alias)
    gen.add_stripes(
        start_y=int(cy - sun_r * 0.2),
        end_y=int(cy + sun_r),
        step=max(4, resolution // 25),
        thickness=max(2, resolution // 90),
        color=(235, 85, 80),
    )
    gen.add_perspective_grid(horizon_y=int(resolution * 0.70), num_rays=15)
else:
    gen.add_vertical_gradient(top_color=(24, 30, 44), bottom_color=(14, 18, 26))
    if fractal_amp > 0:
        gen.add_fractal_noise(amplitude=fractal_amp, octaves=3)
    rng = gen.rng
    for _ in range(3):
        ry = int(rng.integers(int(resolution * 0.10), int(resolution * 0.62)))
        rx = int(rng.integers(int(resolution * 0.10), int(resolution * 0.62)))
        rh = int(rng.integers(int(resolution * 0.15), int(resolution * 0.26)))
        rw = int(rng.integers(int(resolution * 0.15), int(resolution * 0.26)))
        col = tuple(int(c) for c in rng.integers(80, 250, size=3))
        gen.add_rectangle(ry, ry + rh, rx, rx + rw, color=col, alpha=0.92)
    for _ in range(2):
        cy = int(rng.integers(int(resolution * 0.20), int(resolution * 0.80)))
        cx = int(rng.integers(int(resolution * 0.20), int(resolution * 0.80)))
        rad = int(rng.integers(int(resolution * 0.09), int(resolution * 0.16)))
        col = tuple(int(c) for c in rng.integers(100, 255, size=3))
        gen.add_circle(cy, cx, rad, color=col, anti_alias=anti_alias)
    gen.add_ring(
        center_y=int(resolution * 0.32),
        center_x=int(resolution * 0.72),
        outer_radius=int(resolution * 0.15),
        inner_radius=int(resolution * 0.08),
        color=(0, 220, 255),
        anti_alias=anti_alias,
    )
    gen.add_regular_polygon(
        center_y=int(resolution * 0.72),
        center_x=int(resolution * 0.28),
        radius=int(resolution * 0.14),
        num_sides=6,
        color=(255, 105, 90),
        rotation_deg=15.0,
        anti_alias=anti_alias,
    )

if noise_sigma > 0:
    gen.add_gaussian_noise(sigma=noise_sigma)

raw_img = gen.get_image()
if active_filter == "Gaussian Blur (5x5)":
    display_img = gaussian_blur(raw_img, size=5, sigma=1.4)
elif active_filter == "Sobel Edge Magnitude":
    edges, _ = sobel_edges(raw_img)
    display_img = np.stack([edges, edges, edges], axis=-1)
elif active_filter == "Laplacian Sharpen":
    display_img = sharpen(raw_img, strength=0.8)
elif active_filter == "Directional Emboss":
    display_img = emboss(raw_img)
elif active_filter == "Chromatic Aberration + Vignette":
    display_img = vignette(chromatic_aberration(raw_img, shift_px=3), strength=0.5)
else:
    display_img = raw_img

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Rendered Output")
    st.image(display_img, use_container_width=True, caption=f"Shape: {display_img.shape} | dtype: {display_img.dtype}")
with col2:
    st.subheader("2. Semantic Segmentation Mask")
    st.image(gen.get_colorized_mask(), use_container_width=True, caption="Pixel-Accurate Class Labels")
with col3:
    st.subheader("3. Ground-Truth Bounding Boxes")
    st.image(gen.render_annotations_overlay(thickness=2), use_container_width=True, caption=f"{len(gen.annotations)} Tracked Objects")

st.subheader("Exported COCO & YOLO Ground-Truth Annotations")
tab_coco, tab_yolo, tab_mem = st.tabs(["MS-COCO JSON", "Ultralytics YOLO Format", "NumPy Stride & View Diagnostics"])
with tab_coco:
    st.json(gen.export_coco_annotations())
with tab_yolo:
    st.code("\n".join(gen.export_yolo_annotations()) or "# No annotations tracked", language="text")
with tab_mem:
    st.dataframe(inspect_memory_layout(raw_img), use_container_width=True)
