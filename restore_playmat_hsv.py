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
    # Ranges widened from the ideal spec to accommodate real-world scan
    # variation: scanner colour shift, glare, wrinkles, lighting.
    # Targets remain at the clean spec values.
    # Hue ranges use exclusive bands to avoid misclassification in dense
    # regions; the nearest-colour fallback handles edge-case pixels.
    'BG_SKY_BLUE': {
        'target_hls': (_h(206), _sl(71), _sl(64)),
        'range_h': (_h(190), _h(228)),
        'range_s': (_sl(45), _sl(90)),
        'range_l': (_sl(50), _sl(90)),
    },
    'PRIMARY_YELLOW': {
        'target_hls': (_h(59), _sl(61), _sl(98)),
        'range_h': (_h(38), _h(60)),
        'range_s': (_sl(70), _sl(100)),
        'range_l': (_sl(35), _sl(75)),
    },
    'HOT_PINK': {
        'target_hls': (_h(338), _sl(55), _sl(96)),
        'range_h': (_h(280), _h(340)),
        'range_s': (_sl(45), _sl(100)),
        'range_l': (_sl(20), _sl(82)),
    },
    'PURE_WHITE': {
        'target_hls': (_h(0), _sl(99), _sl(0)),
        'range_h': (0, 180),
        'range_s': (0, _sl(18)),
        'range_l': (_sl(82), 255),
    },
    'STEP_RED_OUTLINE': {
        'target_hls': (_h(345), _sl(52), _sl(94)),
        'range_h': (_h(340), _h(20)),
        'range_s': (_sl(45), _sl(100)),
        'range_l': (_sl(20), _sl(82)),
    },
    'LIME_ACCENT': {
        'target_hls': (_h(89), _sl(55), _sl(92)),
        'range_h': (_h(60), _h(140)),
        'range_s': (_sl(30), _sl(100)),
        'range_l': (_sl(20), _sl(82)),
    },
    'DEAD_BLACK': {
        'target_hls': (_h(0), _sl(2), _sl(0)),
        'range_h': (0, 180),
        'range_s': (0, _sl(12)),
        'range_l': (0, _sl(12)),
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
    # Text-carrying colours (PURE_WHITE, LIME_ACCENT) skip OPEN to avoid
    # eroding thin strokes — only CLOSE is applied to fill tiny gaps.
    TEXT_COLOURS = {'PURE_WHITE', 'LIME_ACCENT'}
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masks = {}
    for name, m in raw_masks.items():
        m = gpu_morphology(m, cv2.MORPH_CLOSE, edge_kernel)
        if name not in TEXT_COLOURS:
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

    # Step 6 — Nearest-colour assignment for unclassified pixels.
    # Real scans have wider S/L variation (glare, wrinkles, lighting) than
    # the tight spec ranges.  Instead of dumping everything to background,
    # map each unmatched pixel to the closest permitted colour in HLS space.
    # Processed in chunks to keep memory bounded for large images.
    unmapped = ~assigned
    if np.any(unmapped):
        um_indices = np.where(unmapped)
        um_hls = hls[unmapped].astype(np.float32)  # (N, 3) H, L, S

        targets = np.array([
            list(COLOUR_SPEC[n]['target_hls']) for n in order
        ], dtype=np.float32)  # (7, 3) H, L, S

        bgr_lut = np.array([BGR_TARGETS[n] for n in order], dtype=np.uint8)

        CHUNK = 500_000
        for start in range(0, len(um_hls), CHUNK):
            end = min(start + CHUNK, len(um_hls))
            chunk = um_hls[start:end]

            # Circular hue distance (H range 0-180 in OpenCV)
            dh = np.abs(chunk[:, 0:1] - targets[:, 0])  # (C, 7)
            dh = np.minimum(dh, 180.0 - dh)
            dh *= (255.0 / 180.0)  # normalise to 0-255 scale

            dl = np.abs(chunk[:, 1:2] - targets[:, 1])  # (C, 7)
            ds = np.abs(chunk[:, 2:3] - targets[:, 2])  # (C, 7)

            dist = dh ** 2 + dl ** 2 + ds ** 2
            nearest_idx = np.argmin(dist, axis=1)

            rows = um_indices[0][start:end]
            cols = um_indices[1][start:end]
            result[rows, cols] = bgr_lut[nearest_idx]

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
