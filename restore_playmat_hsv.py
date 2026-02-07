#!/usr/bin/env python3
"""
Vinyl Playmat Digital Restoration Script - HSL-Based Implementation
Removes wrinkles, glare, and texture from scanned vinyl playmat images
while preserving logos, text, stars, and silhouettes with accurate colors.

This version is strictly aligned with the Master Digital Cleanup Colour Specification.
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# ============================================================================
# MASTER COLOUR SPECIFICATION (HSL)
# OpenCV HSL (HLS): H: 0-180, L: 0-255, S: 0-255
# To convert degrees to OpenCV H: H_cv = degrees / 2
# To convert % to OpenCV S/L: S_cv = % * 2.55
# ============================================================================

def deg_to_cv_h(h_deg): return h_deg / 2.0
def pct_to_cv_sl(pct): return pct * 2.55

COLOUR_SPEC = {
    'BG_SKY_BLUE': {
        'target_hls': (deg_to_cv_h(206), pct_to_cv_sl(71), pct_to_cv_sl(64)),
        'range_h': (deg_to_cv_h(198), deg_to_cv_h(214)),
        'range_s': (pct_to_cv_sl(55), pct_to_cv_sl(72)),
        'range_l': (pct_to_cv_sl(64), pct_to_cv_sl(78)),
    },
    'PRIMARY_YELLOW': {
        'target_hls': (deg_to_cv_h(59), pct_to_cv_sl(61), pct_to_cv_sl(98)),
        'range_h': (deg_to_cv_h(55), deg_to_cv_h(61)),
        'range_s': (pct_to_cv_sl(92), pct_to_cv_sl(100)),
        'range_l': (pct_to_cv_sl(55), pct_to_cv_sl(66)),
    },
    'HOT_PINK': {
        'target_hls': (deg_to_cv_h(338), pct_to_cv_sl(55), pct_to_cv_sl(96)),
        'range_h': (deg_to_cv_h(330), deg_to_cv_h(346)),
        'range_s': (pct_to_cv_sl(90), pct_to_cv_sl(100)),
        'range_l': (pct_to_cv_sl(48), pct_to_cv_sl(62)),
    },
    'PURE_WHITE': {
        'target_hls': (deg_to_cv_h(0), pct_to_cv_sl(99), pct_to_cv_sl(0)),
        'range_h': (0, 180),
        'range_s': (0, pct_to_cv_sl(4)),
        'range_l': (pct_to_cv_sl(96), 255),
    },
    'STEP_RED_OUTLINE': {
        'target_hls': (deg_to_cv_h(345), pct_to_cv_sl(52), pct_to_cv_sl(94)),
        'range_h': (deg_to_cv_h(338), deg_to_cv_h(352)),
        'range_s': (pct_to_cv_sl(88), pct_to_cv_sl(98)),
        'range_l': (pct_to_cv_sl(46), pct_to_cv_sl(58)),
    },
    'LIME_ACCENT': {
        'target_hls': (deg_to_cv_h(89), pct_to_cv_sl(55), pct_to_cv_sl(92)),
        'range_h': (deg_to_cv_h(82), deg_to_cv_h(96)),
        'range_s': (pct_to_cv_sl(85), pct_to_cv_sl(96)),
        'range_l': (pct_to_cv_sl(48), pct_to_cv_sl(62)),
    },
    'DEAD_BLACK': {
        'target_hls': (deg_to_cv_h(0), pct_to_cv_sl(2), pct_to_cv_sl(0)),
        'range_h': (0, 180),
        'range_s': (0, pct_to_cv_sl(6)),
        'range_l': (0, pct_to_cv_sl(6)),
    }
}

def hls_to_bgr(hls_pixel):
    pixel = np.uint8([[hls_pixel]])
    return cv2.cvtColor(pixel, cv2.COLOR_HLS2BGR)[0][0]

# Pre-calculate BGR targets
BGR_TARGETS = {k: hls_to_bgr(v['target_hls']) for k, v in COLOUR_SPEC.items()}

def get_mask(hls_img, spec):
    h, l, s = cv2.split(hls_img)
    
    h_min, h_max = spec['range_h']
    l_min, l_max = spec['range_l']
    s_min, s_max = spec['range_s']
    
    # Handle hue wrap-around for reds
    if h_min > h_max:
        h_mask = (h >= h_min) | (h <= h_max)
    else:
        h_mask = (h >= h_min) & (h <= h_max)
        
    mask = h_mask & (l >= l_min) & (l <= l_max) & (s >= s_min) & (s <= s_max)
    return mask

def process_image(image_path, output_dir):
    print(f"Processing: {image_path.name}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading {image_path}")
        return

    # Step 1: Convert BGR -> HLS (OpenCV's HSL implementation)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Step 2: Cluster pixels using HSL ranges — each pixel is snapped to
    # its Selected clean HSL target, flattening glare/wrinkle lightness
    # deviations within each band (Texture Removal).
    raw_masks = {}
    for name, spec in COLOUR_SPEC.items():
        raw_masks[name] = get_mask(hls, spec).astype(np.uint8) * 255

    # Step 4: Edge Preservation — apply erosion/dilation (morphological
    # close then open) *after* recolouring masks to preserve STEP outlines,
    # avoid outline bleed, and maintain crisp silhouettes.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masks = {}
    for name in raw_masks:
        m = raw_masks[name]
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        masks[name] = m

    # Step 5: Stroke Priority Rule — if a pixel qualifies for both
    # HOT_PINK and STEP_RED_OUTLINE, choose STEP_RED_OUTLINE when the
    # region is thin or adjacent to yellow text (STEP label context).
    overlap = cv2.bitwise_and(masks['HOT_PINK'], masks['STEP_RED_OUTLINE'])
    if np.any(overlap):
        thick_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(overlap, thick_kernel, iterations=1)
        thin_pixels = (overlap > 0) & (eroded == 0)

        yellow_dilated = cv2.dilate(masks['PRIMARY_YELLOW'], thick_kernel, iterations=2)
        adjacent_to_yellow = (overlap > 0) & (yellow_dilated > 0)

        step_red_wins = thin_pixels | adjacent_to_yellow
        hot_pink_wins = (overlap > 0) & ~step_red_wins

        masks['STEP_RED_OUTLINE'] = np.where(
            hot_pink_wins, 0, masks['STEP_RED_OUTLINE']).astype(np.uint8)
        masks['HOT_PINK'] = np.where(
            step_red_wins, 0, masks['HOT_PINK']).astype(np.uint8)

    # Priority order for remaining overlaps
    order = ['PURE_WHITE', 'DEAD_BLACK', 'PRIMARY_YELLOW',
             'STEP_RED_OUTLINE', 'HOT_PINK', 'LIME_ACCENT', 'BG_SKY_BLUE']

    result = np.zeros_like(img)
    assigned = np.zeros(img.shape[:2], dtype=bool)

    for name in order:
        m = (masks[name] > 0) & ~assigned
        result[m] = BGR_TARGETS[name]
        assigned |= m

    # Step 6: Dead-Space Handling — unclassified pixels default to background
    unmapped = ~assigned
    result[unmapped] = BGR_TARGETS['BG_SKY_BLUE']

    # Output is guaranteed to contain only the 7 permitted colours
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), result)
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Vinyl Playmat Restoration - 2026 Production Spec")
    parser.add_argument("input_dir", type=str, help="Directory containing scans")
    parser.add_argument("--output_dir", type=str, default="scans/output", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

    print(f"Found {len(images)} images in {input_path}")
    
    # Use sequential processing for stability on Replit free tier resources
    for img_p in images:
        process_image(img_p, output_path)

if __name__ == "__main__":
    main()
