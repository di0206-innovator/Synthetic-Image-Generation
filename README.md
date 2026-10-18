# Image Pixel Manipulator (Synthetic Image Generator)

## Overview

This project implements an image generation and pixel manipulation toolkit using pure NumPy and Matplotlib. In digital image processing and computer vision, images are treated as multidimensional numerical arrays (tensors) of shape `(Height, Width, Channels)`. This project demonstrates how core array operations—such as multi-axis slicing, data type casting, memory management, and boolean masking—can perform complex image processing tasks without external computer vision frameworks.

## Key Objectives

1. Represent color images as 3D NumPy arrays using the `uint8` data format.
2. Generate synthetic images and color fields from scratch using array initialization methods.
3. Apply 3D array slicing and stride slicing to draw geometric patterns, stripes, and color blocks.
4. Perform spatial manipulations including cropping, horizontal/vertical flipping, and color channel isolation.
5. Safely adjust brightness and darkness while preventing unsigned integer arithmetic overflow.
6. Identify and modify regions of interest using vectorized boolean masking.
7. Compare memory footprints between integer and floating-point representations.

## Core Concepts Covered

- **Array Initialization:** `np.zeros`, `np.ones`, `np.full`, and procedural generation with `np.random`.
- **Tensor Slicing:** 3D coordinate slicing `[y1:y2, x1:x2, channel]`, multi-pixel strides `[::step]`, and negative-step flipping `[::-1]`.
- **Channel Extraction:** Isolating individual Red, Green, and Blue color channels as 2D intensity maps and masked 3D representations.
- **Arithmetic Safety:** Avoiding `uint8` modulo wrap-around overflow bugs by casting to `float32`, clipping with `np.clip`, and casting back to `uint8`.
- **Boolean Indexing:** Constructing threshold masks based on perceived luminance and recoloring masked pixels in place.
- **Memory Profiling:** Evaluating array memory footprint via `nbytes` across `uint8`, `float32`, and `float64` data types.
- **Display Normalization:** Adhering to Matplotlib `imshow` display conventions for integer ranges `[0, 255]` versus floating-point ranges `[0.0, 1.0]`.

## Project Structure

```
AIML/
|-- ImageGenerator.ipynb   # Main interactive notebook with code and outputs
|-- Sample_128x128.png     # Sample input image used for manipulation tests
`-- README.md              # Project documentation
```

### Notebook Workflow

1. **Canvas Initialization:** Creation of blank canvases (black, white) and multi-region landscapes using `np.zeros`, `np.ones`, and `np.full`.
2. **Block and Stripe Drawing:** Manipulation of specific channels (e.g., `img[10:60, 20:90, 0] = 255`) and procedural stripe generation.
3. **Spatial Transformations:** Sub-region cropping, axis reversals (horizontal/vertical mirrors), and single-channel isolation.
4. **Brightness and Masking:** Demonstration of integer overflow mechanics, safe floating-point brightness adjustments, and luminance-based boolean filtering.
5. **Memory and Datatype Analysis:** Comparative memory benchmarks showing 1x (`uint8`), 4x (`float32`), and 8x (`float64`) allocations, alongside Matplotlib floating-point normalization rules.
6. **Synthetic Image Generator Class:** A reusable object-oriented generator (`SyntheticImageGenerator`) that builds structured synthetic artwork using linear gradients, Euclidean distance circle masks, stripes, and Gaussian noise.
7. **Real Image Processing:** Application of the array manipulation toolkit to `Sample_128x128.png`.

## Requirements and Installation

The notebook requires Python 3.8+ with the following standard scientific computing libraries:

```bash
pip install numpy matplotlib pillow
```

## Running the Notebook

You can run the notebook locally using Jupyter Notebook, JupyterLab, Visual Studio Code, or Google Colab:

```bash
jupyter notebook ImageGenerator.ipynb
```
