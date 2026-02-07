#!/usr/bin/env python3
"""
Vinyl Playmat Digital Restoration Script — New Colour Regime

Removes wrinkles, glare, and texture from scanned vinyl playmat images
while preserving logos, text, stars, and silhouettes with accurate colors.

Only the 7 Master Digital Cleanup colours are permitted in output.
Uses GPU acceleration (CUDA) where available for faster processing.

Usage:
    Place scanned images in the 'scans/' folder and run this script.
    Cleaned images will be saved to 'scans/output/'.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# ============================================================================
# FIXED PATHS — resolved relative to this script so it works regardless of CWD
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "scans"
OUTPUT_DIR = SCRIPT_DIR / "scans" / "output"

# ============================================================================
# GPU DETECTION — use CUDA when available, fall back to CPU transparently
# ============================================================================
USE_GPU = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        USE_GPU = True
except AttributeError:
    pass

# ============================================================================
# MASTER COLOUR SPECIFICATION (HSL)
# OpenCV HLS channel order: H 0-180, L 0-255, S 0-255
# ============================================================================

def _h(deg):
    """Convert hue degrees (0-360) to OpenCV H (0-180)."""
    return deg / 2.0

def _sl(pct):
    """Convert saturation/lightness percent (0-100) to OpenCV S/L (0-255)."""
    return pct * 2.55

COLOUR_SPEC = {
    'BG_SKY_BLUE': {
        'target_hls': (_h(206), _sl(71), _sl(64)),
        'range_h': (_h(198), _h(214)),
        'range_s': (_sl(55), _sl(72)),
        'range_l': (_sl(64), _sl(78)),
    },
    'PRIMARY_YELLOW': {
        'target_hls': (_h(59), _sl(61), _sl(98)),
        'range_h': (_h(55), _h(61)),
        'range_s': (_sl(92), _sl(100)),
        'range_l': (_sl(55), _sl(66)),
    },
    'HOT_PINK': {
        'target_hls': (_h(338), _sl(55), _sl(96)),
        'range_h': (_h(330), _h(346)),
        'range_s': (_sl(90), _sl(100)),
        'range_l': (_sl(48), _sl(62)),
    },
    'PURE_WHITE': {
        'target_hls': (_h(0), _sl(99), _sl(0)),
        'range_h': (0, 180),
        'range_s': (0, _sl(4)),
        'range_l': (_sl(96), 255),
    },
    'STEP_RED_OUTLINE': {
        'target_hls': (_h(345), _sl(52), _sl(94)),
        'range_h': (_h(338), _h(352)),
        'range_s': (_sl(88), _sl(98)),
        'range_l': (_sl(46), _sl(58)),
    },
    'LIME_ACCENT': {
        'target_hls': (_h(89), _sl(55), _sl(92)),
        'range_h': (_h(82), _h(96)),
        'range_s': (_sl(85), _sl(96)),
        'range_l': (_sl(48), _sl(62)),
    },
    'DEAD_BLACK': {
        'target_hls': (_h(0), _sl(2), _sl(0)),
        'range_h': (0, 180),
        'range_s': (0, _sl(6)),
        'range_l': (0, _sl(6)),
    },
}


def hls_to_bgr(hls_pixel):
    """Convert a single HLS pixel to BGR."""
    pixel = np.uint8([[hls_pixel]])
    return cv2.cvtColor(pixel, cv2.COLOR_HLS2BGR)[0][0]


# Pre-calculate BGR targets for the 7 permitted colours
BGR_TARGETS = {k: hls_to_bgr(v['target_hls']) for k, v in COLOUR_SPEC.items()}


# ============================================================================
# GPU / CPU HELPERS — each function tries CUDA first, then falls back to CPU
# ============================================================================

def gpu_cvt_color(src, code):
    """Colour-space conversion with GPU fallback."""
    if USE_GPU:
        try:
            g = cv2.cuda_GpuMat()
            g.upload(src)
            return cv2.cuda.cvtColor(g, code).download()
        except Exception:
            pass
    return cv2.cvtColor(src, code)


def _gpu_morph_apply(src, op, kernel, iterations):
    """Run a CUDA morphology filter, looping for multiple iterations."""
    g = cv2.cuda_GpuMat()
    g.upload(src)
    f = cv2.cuda.createMorphologyFilter(op, cv2.CV_8UC1, kernel)
    for _ in range(iterations):
        g = f.apply(g)
    return g.download()


def gpu_morphology(src, op, kernel, iterations=1):
    """Morphological operation (close / open) with GPU fallback."""
    if USE_GPU:
        try:
            return _gpu_morph_apply(src, op, kernel, iterations)
        except Exception:
            pass
    return cv2.morphologyEx(src, op, kernel, iterations=iterations)


def gpu_erode(src, kernel, iterations=1):
    """Erosion with GPU fallback."""
    if USE_GPU:
        try:
            return _gpu_morph_apply(src, cv2.MORPH_ERODE, kernel, iterations)
        except Exception:
            pass
    return cv2.erode(src, kernel, iterations=iterations)


def gpu_dilate(src, kernel, iterations=1):
    """Dilation with GPU fallback."""
    if USE_GPU:
        try:
            return _gpu_morph_apply(src, cv2.MORPH_DILATE, kernel, iterations)
        except Exception:
            pass
    return cv2.dilate(src, kernel, iterations=iterations)


# ============================================================================
# MASK CREATION
# ============================================================================

def get_mask(hls_img, spec):
    """Create a binary mask for pixels matching a colour specification."""
    h, l, s = cv2.split(hls_img)

    h_min, h_max = spec['range_h']
    l_min, l_max = spec['range_l']
    s_min, s_max = spec['range_s']

    if h_min > h_max:
        h_mask = (h >= h_min) | (h <= h_max)
    else:
        h_mask = (h >= h_min) & (h <= h_max)

    return h_mask & (l >= l_min) & (l <= l_max) & (s >= s_min) & (s <= s_max)


# ============================================================================
# IMAGE PROCESSING PIPELINE
# ============================================================================

def process_image(image_path):
    """Run the full 7-colour cleanup pipeline on a single image."""
    print(f"Processing: {image_path.name}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Error: could not load {image_path}")
        return

    # Step 1 — Convert BGR → HLS
    hls = gpu_cvt_color(img, cv2.COLOR_BGR2HLS)

    # Step 2 — Cluster pixels using HSL ranges.
    # Matched pixels are snapped to the clean target, which also normalises
    # lightness deviations caused by glare or wrinkles (Step 3 Texture Removal).
    raw_masks = {}
    for name, spec in COLOUR_SPEC.items():
        raw_masks[name] = get_mask(hls, spec).astype(np.uint8) * 255

    # Step 4 — Edge Preservation: morphological close then open on each mask
    # to fill small holes and remove speckle while keeping outlines crisp.
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masks = {}
    for name, m in raw_masks.items():
        m = gpu_morphology(m, cv2.MORPH_CLOSE, edge_kernel)
        m = gpu_morphology(m, cv2.MORPH_OPEN, edge_kernel)
        masks[name] = m

    # Step 5 — Stroke Priority Rule
    # Where HOT_PINK and STEP_RED_OUTLINE overlap, prefer STEP_RED_OUTLINE
    # for thin strokes and regions adjacent to yellow STEP text.
    overlap = cv2.bitwise_and(masks['HOT_PINK'], masks['STEP_RED_OUTLINE'])
    if np.any(overlap):
        stroke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = gpu_erode(overlap, stroke_kernel, iterations=1)
        thin_pixels = (overlap > 0) & (eroded == 0)

        yellow_dilated = gpu_dilate(
            masks['PRIMARY_YELLOW'], stroke_kernel, iterations=2)
        adjacent_to_yellow = (overlap > 0) & (yellow_dilated > 0)

        step_red_wins = thin_pixels | adjacent_to_yellow
        hot_pink_wins = (overlap > 0) & ~step_red_wins

        masks['STEP_RED_OUTLINE'] = np.where(
            hot_pink_wins, 0, masks['STEP_RED_OUTLINE']).astype(np.uint8)
        masks['HOT_PINK'] = np.where(
            step_red_wins, 0, masks['HOT_PINK']).astype(np.uint8)

    # Assign colours using priority order (highest first)
    order = [
        'PURE_WHITE', 'DEAD_BLACK', 'PRIMARY_YELLOW',
        'STEP_RED_OUTLINE', 'HOT_PINK', 'LIME_ACCENT', 'BG_SKY_BLUE',
    ]

    result = np.zeros_like(img)
    assigned = np.zeros(img.shape[:2], dtype=bool)

    for name in order:
        m = (masks[name] > 0) & ~assigned
        result[m] = BGR_TARGETS[name]
        assigned |= m

    # Step 6 — Dead-Space Handling: unclassified pixels default to background
    result[~assigned] = BGR_TARGETS['BG_SKY_BLUE']

    # Output contains only the 7 permitted colours.
    # Save as PNG (lossless) to guarantee no compression artifacts.
    output_path = OUTPUT_DIR / (image_path.stem + ".png")
    success = cv2.imwrite(str(output_path), result)
    if not success or not output_path.exists():
        print(f"  ERROR: failed to write {output_path}")
    else:
        print(f"  Saved: {output_path} ({output_path.stat().st_size} bytes)")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    print("=" * 60)
    print("  Vinyl Playmat Restoration — New Colour Regime")
    print("=" * 60)
    print(f"  GPU acceleration: {'ENABLED' if USE_GPU else 'not available (CPU mode)'}")
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    if not INPUT_DIR.exists():
        print(f"Error: input directory '{INPUT_DIR}' not found.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = [f for f in INPUT_DIR.iterdir()
              if f.suffix.lower() in image_extensions]

    if not images:
        print("No images found — nothing to process.")
        sys.exit(0)

    print(f"Found {len(images)} image(s)\n")

    for img_p in images:
        process_image(img_p)

    print(f"\nDone — all output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
