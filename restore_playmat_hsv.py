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

    # 1. Convert RGB -> HLS (OpenCV's HSL implementation)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # Create blank output
    result = np.zeros_like(img)
    final_mask = np.zeros(img.shape[:2], dtype=bool)
    
    # Priority order for masking (to handle overlapping ranges)
    # 1. White (stars/text)
    # 2. Black (borders)
    # 3. Yellow (main shapes)
    # 4. Red Outline (special rule: prioritize over Pink if thin)
    # 5. Hot Pink
    # 6. Lime Accent
    # 7. Blue Background (fill remaining)
    
    order = ['PURE_WHITE', 'DEAD_BLACK', 'PRIMARY_YELLOW', 'STEP_RED_OUTLINE', 'HOT_PINK', 'LIME_ACCENT', 'BG_SKY_BLUE']
    
    masks = {}
    for name in order:
        masks[name] = get_mask(hls, COLOUR_SPEC[name])
        
    # Apply priority logic
    current_processed = np.zeros(img.shape[:2], dtype=bool)
    
    # Special rule: STEP_RED_OUTLINE vs HOT_PINK handled by order for now
    # as order already prioritizes thin red outlines if they are processed first.
    
    for name in order:
        m = masks[name] & ~current_processed
        result[m] = BGR_TARGETS[name]
        current_processed |= m
        
    # Handle any unclassified pixels (Texture Removal/Normalization)
    # Map them to the nearest color in HLS space or background
    unmapped = ~current_processed
    if np.any(unmapped):
        result[unmapped] = BGR_TARGETS['BG_SKY_BLUE']
        
    # 2. Texture Removal & Smoothing
    # Apply a light median blur to remove scan noise and solidify inks
    result = cv2.medianBlur(result, 3)
    
    # 3. Edge Preservation
    # Bilateral filter to smooth colors without blurring edges
    result = cv2.bilateralFilter(result, 5, 75, 75)

    # Save output
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
