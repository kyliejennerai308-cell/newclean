#!/usr/bin/env python3
"""
Vinyl Playmat Digital Restoration Script — New Colour Regime

Removes wrinkles, glare, and texture from scanned vinyl playmat images
while preserving logos, text, stars, and silhouettes with accurate colors.

Only the 8 Master Digital Cleanup colours are permitted in output.
Uses GPU acceleration (CUDA) where available for faster processing.

Usage:
    Place scanned images in the 'scans/' folder and run this script.
    Cleaned images will be saved to 'scans/output/'.
    NOTE: No command-line flags are implemented.
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

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# ============================================================================
# GPU DETECTION — use CUDA when available, fall back to CPU transparently
# ============================================================================
USE_GPU = False
GPU_AVAILABLE = False
GPU_BACKEND = None
GPU_DETECTION_MESSAGE = ""

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def _detect_gpu():
    """Detect GPU/CUDA availability and provide diagnostic information."""
    has_cuda_module = hasattr(cv2, 'cuda')
    if has_cuda_module and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count > 0:
                try:
                    device_info = cv2.cuda.getDevice()
                    return True, 'opencv', (
                        f"OpenCV CUDA enabled with {device_count} device(s). "
                        f"Active device: {device_info}")
                except (cv2.error, AttributeError):
                    return True, 'opencv', (
                        f"OpenCV CUDA enabled with {device_count} device(s).")
        except cv2.error:
            pass

    if CUPY_AVAILABLE:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                device_name = (
                    device_props['name'].decode('utf-8')
                    if isinstance(device_props['name'], bytes)
                    else device_props['name']
                )
                return True, 'cupy', (
                    f"CuPy CUDA enabled with {device_count} device(s). "
                    f"GPU: {device_name}")
        except Exception:
            pass

    if CUPY_AVAILABLE:
        return False, None, (
            "CuPy installed but no CUDA GPU detected. "
            "Check your NVIDIA drivers.")
    return False, None, (
        "No GPU backend available. Install CuPy for GPU acceleration: "
        "pip install cupy-cuda12x")


try:
    GPU_AVAILABLE, GPU_BACKEND, GPU_DETECTION_MESSAGE = _detect_gpu()
except Exception as exc:
    GPU_AVAILABLE = False
    GPU_BACKEND = None
    GPU_DETECTION_MESSAGE = f"GPU detection failed: {exc}"

USE_GPU = GPU_AVAILABLE


def _use_opencv_cuda():
    return USE_GPU and GPU_BACKEND == 'opencv'

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
        'range_h': (_h(295), _h(340)),
        'range_s': (_sl(45), _sl(100)),
        'range_l': (_sl(20), _sl(82)),
    },
    'DARK_PURPLE': {
        'target_hls': (_h(275), _sl(35), _sl(80)),
        'range_h': (_h(240), _h(295)),
        'range_s': (_sl(25), _sl(100)),
        'range_l': (_sl(10), _sl(82)),
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


# Pre-calculate BGR targets for the 8 permitted colours
BGR_TARGETS = {k: hls_to_bgr(v['target_hls']) for k, v in COLOUR_SPEC.items()}

# Fill colours = large uniform regions that should not bleed over fine borders.
# Detail colours = outlines, text, small features that must be preserved.
FILL_COLOURS = {'BG_SKY_BLUE', 'PRIMARY_YELLOW', 'HOT_PINK'}
OUTLINE_COLOURS = {'STEP_RED_OUTLINE', 'DARK_PURPLE'}


# ============================================================================
# GPU / CPU HELPERS — each function tries CUDA first, then falls back to CPU
# ============================================================================

def gpu_cvt_color(src, code):
    """Colour-space conversion with GPU fallback."""
    if _use_opencv_cuda():
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
    if _use_opencv_cuda():
        try:
            return _gpu_morph_apply(src, op, kernel, iterations)
        except Exception:
            pass
    return cv2.morphologyEx(src, op, kernel, iterations=iterations)


def gpu_erode(src, kernel, iterations=1):
    """Erosion with GPU fallback."""
    if _use_opencv_cuda():
        try:
            return _gpu_morph_apply(src, cv2.MORPH_ERODE, kernel, iterations)
        except Exception:
            pass
    return cv2.erode(src, kernel, iterations=iterations)


def gpu_dilate(src, kernel, iterations=1):
    """Dilation with GPU fallback."""
    if _use_opencv_cuda():
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

def _guided_filter(I, p, radius, eps):
    """Self-guided filter: edge-aware smoothing without ximgproc.

    Smooths flat regions aggressively while stopping at strong edges.
    Equivalent to cv2.ximgproc.guidedFilter(I, p, radius, eps) when I == p.
    """
    I = I.astype(np.float32) / 255.0
    p = p.astype(np.float32) / 255.0
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(I, -1, ksize)
    mean_p = cv2.boxFilter(p, -1, ksize)
    corr_Ip = cv2.boxFilter(I * p, -1, ksize)
    corr_II = cv2.boxFilter(I * I, -1, ksize)

    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    return np.clip((mean_a * I + mean_b) * 255, 0, 255).astype(np.uint8)


def _auto_gamma(l_channel):
    """Auto-gamma correction: push mean brightness toward mid-grey.

    Compensates for under/over-exposure before CLAHE to ensure the
    adaptive equalisation starts from a balanced baseline.
    """
    mean = np.mean(l_channel)
    if mean < 1:
        return l_channel
    gamma = np.log(127.0) / np.log(mean)
    gamma = np.clip(gamma, 0.5, 2.0)
    table = np.array([
        np.clip(((i / 255.0) ** gamma) * 255, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(l_channel, table)


def _unsharp_mask(img, sigma=1.0, strength=0.5):
    """Unsharp masking: sharpen edges before quantisation.

    Creates a high-frequency detail layer (original − blur) and adds it
    back at controlled strength, making logo/text boundaries crisper
    without amplifying vinyl grain noise.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)


def prep_image(img):
    """Pre-processing: flatten vinyl texture / glare while keeping edges.

    1. Bilateral filter — smooths scan grain but preserves sharp edges at
       logo boundaries (unlike Gaussian blur which smears them).
    2. Guided filter — edge-aware smoothing that further flattens vinyl
       texture while hard-stopping at logo/text boundaries.
    3. Auto-gamma — normalise exposure before CLAHE to avoid bias.
    4. CLAHE on L channel — local contrast enhancement makes small white
       text pop against the blue background, even in shadowed regions.
    5. Unsharp mask — sharpen logo/text edges before colour quantisation.
    """
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Guided filter for additional edge-aware smoothing
    gray_guide = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    for c in range(3):
        smoothed[:, :, c] = _guided_filter(
            gray_guide, smoothed[:, :, c], radius=4, eps=0.01)

    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Auto-gamma to normalise exposure baseline
    l_ch = _auto_gamma(l_ch)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.merge((l_ch, a_ch, b_ch))
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Unsharp mask to crisp up edges before quantisation
    return _unsharp_mask(result, sigma=1.0, strength=0.5)


def detect_edges_keepout(gray):
    """Canny edge detection → dilated "keep-out" zone.

    Creates a thin mask of strong edges that prevents fill-colour masks
    from bleeding across fine borders (purple/red outlines, badge contours).
    """
    edges = cv2.Canny(gray, 50, 150)
    keep_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(edges, keep_kernel, iterations=1)


def detect_white_text(gray):
    """Isolate small bright features (text, stars) from large glare patches.

    1. White top-hat — extracts only small bright elements on dark background
       (catches stars and text) while ignoring wide glare bands.
    2. Adaptive threshold — rescues white text that has degraded to grey
       in scanner shadows where a global lightness threshold fails.
    Both masks are combined for maximum text recovery.
    """
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel)
    _, tophat_mask = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=-8)

    return cv2.bitwise_or(tophat_mask, adaptive)


def detect_dark_outlines(gray):
    """Invert-L trick: dark outlines become bright wires for easy detection.

    Inverting lightness makes thin dark borders (purple/red strokes) appear
    as bright lines that can be thresholded regardless of their hue.
    """
    inverted = cv2.bitwise_not(gray)
    _, dark_mask = cv2.threshold(inverted, 180, 255, cv2.THRESH_BINARY)
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(dark_mask, thin_kernel, iterations=1)
    return cv2.subtract(dark_mask, eroded)


def _area_open(mask, min_area=64):
    """Remove connected components smaller than min_area pixels.

    Superior to morphological OPEN for noise removal because it removes
    tiny blobs regardless of shape without eroding the edges of larger
    regions.  Uses cv2.connectedComponentsWithStats for efficiency.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _conditional_dilate(mask, original, kernel, iterations=1):
    """Dilate mask but only where pixels existed in the original.

    Restores thin strokes that morphological close may have slightly
    displaced, without allowing the mask to grow into new territory.
    This prevents outline masks from bleeding into adjacent fill regions.
    """
    dilated = gpu_dilate(mask, kernel, iterations=iterations)
    return cv2.bitwise_and(dilated, original)


def process_image(image_path):
    """Run the full 8-colour cleanup pipeline on a single image."""
    print(f"Processing: {image_path.name}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Error: could not load {image_path}")
        return

    # ---- PRE-PROCESSING ----
    # Flatten vinyl texture and boost local contrast for text recovery.
    prepped = prep_image(img)

    # Step 1 — Convert BGR → HLS on the pre-processed image.
    hls = gpu_cvt_color(prepped, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(prepped, cv2.COLOR_BGR2GRAY)

    # Detect fine edges, white text, and dark outlines.
    edge_keepout = detect_edges_keepout(gray)
    white_text_mask = detect_white_text(gray)
    dark_outline_mask = detect_dark_outlines(gray)

    # Step 2 — Cluster pixels using HSL ranges.
    # Matched pixels are snapped to the clean target, which also normalises
    # lightness deviations caused by glare or wrinkles (Step 3 Texture Removal).
    raw_masks = {}
    for name, spec in COLOUR_SPEC.items():
        raw_masks[name] = get_mask(hls, spec).astype(np.uint8) * 255

    # Boost PURE_WHITE with the top-hat + adaptive text mask.
    raw_masks['PURE_WHITE'] = cv2.bitwise_or(
        raw_masks['PURE_WHITE'], white_text_mask)

    # Boost outline colours with the dark-outline mask: assign thin dark
    # pixels to whichever outline colour already claims the most neighbours.
    for outline_name in OUTLINE_COLOURS:
        dilated_outline = cv2.dilate(
            raw_masks[outline_name],
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1)
        boost = cv2.bitwise_and(dark_outline_mask, dilated_outline)
        raw_masks[outline_name] = cv2.bitwise_or(
            raw_masks[outline_name], boost)

    # Step 4 — Edge Preservation.
    # Morphological close fills small holes; area-open removes tiny noise
    # blobs based on connected-component size (superior to morph OPEN which
    # erodes edges uniformly).  Conditional dilation restores thin strokes
    # that close may have slightly thickened — it only grows pixels that
    # existed in the original mask, preventing bleed.
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masks = {}
    for name, m in raw_masks.items():
        original = m.copy()
        m = gpu_morphology(m, cv2.MORPH_CLOSE, edge_kernel)
        if name == 'BG_SKY_BLUE':
            # Area-open: remove connected components < 64px.  This threshold
            # corresponds to roughly an 8×8 pixel blob — small enough to be
            # scan noise but large enough to preserve intentional detail.
            m = _area_open(m, min_area=64)
        if name in OUTLINE_COLOURS or name in ('PURE_WHITE', 'LIME_ACCENT'):
            # Conditional dilation: restore thin strokes only where they
            # existed before close, preventing bleed into adjacent colours.
            m = _conditional_dilate(m, original, edge_kernel)
        masks[name] = m

    # Apply Canny keep-out: subtract edge zone from fill colours to prevent
    # them from bleeding over fine borders.  Text and outline colours are
    # not restricted.
    for name in FILL_COLOURS:
        masks[name] = cv2.subtract(masks[name], edge_keepout)

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

    # Assign colours using priority order (highest first).
    # DARK_PURPLE is above HOT_PINK so that purple outlines around pink
    # fills are preserved — thin purple boundaries win over adjacent pink.
    order = [
        'PURE_WHITE', 'DEAD_BLACK', 'PRIMARY_YELLOW',
        'STEP_RED_OUTLINE', 'DARK_PURPLE', 'HOT_PINK',
        'LIME_ACCENT', 'BG_SKY_BLUE',
    ]

    result = np.zeros_like(img)
    assigned = np.zeros(img.shape[:2], dtype=bool)

    for name in order:
        m = (masks[name] > 0) & ~assigned
        result[m] = BGR_TARGETS[name]
        assigned |= m

    # Step 6 — Nearest-colour assignment for unclassified pixels.
    # Uses the pre-processed HLS for better classification (texture/glare
    # has been flattened by bilateral + CLAHE).
    # Processed in chunks to keep memory bounded for large images.
    unmapped = ~assigned
    if np.any(unmapped):
        um_indices = np.where(unmapped)
        um_hls = hls[unmapped].astype(np.float32)  # (N, 3) H, L, S

        targets = np.array([
            list(COLOUR_SPEC[n]['target_hls']) for n in order
        ], dtype=np.float32)  # (8, 3) H, L, S

        bgr_lut = np.array([BGR_TARGETS[n] for n in order], dtype=np.uint8)

        CHUNK = 500_000
        use_cupy = USE_GPU and GPU_BACKEND == 'cupy' and CUPY_AVAILABLE
        targets_gpu = cp.asarray(targets) if use_cupy else None
        bgr_lut_gpu = cp.asarray(bgr_lut) if use_cupy else None
        for start in range(0, len(um_hls), CHUNK):
            end = min(start + CHUNK, len(um_hls))
            rows = um_indices[0][start:end]
            cols = um_indices[1][start:end]
            if use_cupy:
                chunk = cp.asarray(um_hls[start:end])
                dh = cp.abs(chunk[:, 0:1] - targets_gpu[:, 0])
                dh = cp.minimum(dh, 180.0 - dh)
                dh *= (255.0 / 180.0)
                dl = cp.abs(chunk[:, 1:2] - targets_gpu[:, 1])
                ds = cp.abs(chunk[:, 2:3] - targets_gpu[:, 2])
                dist = dh ** 2 + dl ** 2 + ds ** 2
                nearest_idx = cp.argmin(dist, axis=1)
                result[rows, cols] = cp.asnumpy(bgr_lut_gpu[nearest_idx])
            else:
                chunk = um_hls[start:end]
                # Circular hue distance (H range 0-180 in OpenCV)
                dh = np.abs(chunk[:, 0:1] - targets[:, 0])  # (C, 8)
                dh = np.minimum(dh, 180.0 - dh)
                dh *= (255.0 / 180.0)  # normalise to 0-255 scale

                dl = np.abs(chunk[:, 1:2] - targets[:, 1])  # (C, 8)
                ds = np.abs(chunk[:, 2:3] - targets[:, 2])  # (C, 8)

                dist = dh ** 2 + dl ** 2 + ds ** 2
                nearest_idx = np.argmin(dist, axis=1)

                result[rows, cols] = bgr_lut[nearest_idx]

    # Output contains only the 8 permitted colours.
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
    print("  Get ready.")
    gpu_status = (
        "ENABLED"
        if USE_GPU
        else "not available (CPU mode)"
    )
    print(f"  GPU acceleration: {gpu_status}")
    if not GPU_AVAILABLE:
        print(f"  GPU note: {GPU_DETECTION_MESSAGE}")
    elif USE_GPU and GPU_BACKEND:
        print(f"  GPU backend: {GPU_BACKEND}")
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
