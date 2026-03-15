#!/usr/bin/env python3
"""
HalfTone — Print-ready halftone art generator.
Author : Orvlyn
GitHub : https://github.com/Orvlyn/Halftone
"""

import sys
import os
import json
import math
import ctypes
import urllib.request
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFileDialog, QComboBox,
    QSlider, QScrollArea, QColorDialog, QProgressBar, QCheckBox,
    QSizePolicy, QMessageBox, QLineEdit, QFrame,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QSettings, QUrl, QSize
from PySide6.QtGui import QPixmap, QIcon, QImage, QColor, QDesktopServices

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

APP_VERSION      = "1.0.0"
UPDATE_CHECK_URL = "https://raw.githubusercontent.com/Orvlyn/Halftone/main/version.json"
ICON_GITHUB_URL  = "https://raw.githubusercontent.com/Orvlyn/Halftone/main/halftone.ico"
ICON_CACHE_PATH  = os.path.join(os.path.dirname(__file__), ".halftone_icon_cache.ico")
ICON_PATH        = os.path.join(os.path.dirname(__file__), "halftone.ico")

try:
    from ctypes import wintypes  # noqa: F401
    HAS_WINDOWS_TITLEBAR = True
except ImportError:
    HAS_WINDOWS_TITLEBAR = False


def get_cached_icon() -> QIcon:
    for candidate in (ICON_PATH, ICON_CACHE_PATH):
        if candidate and os.path.exists(candidate):
            icon = QIcon(candidate)
            if not icon.isNull():
                return icon
    try:
        urllib.request.urlretrieve(ICON_GITHUB_URL, ICON_CACHE_PATH)
        icon = QIcon(ICON_CACHE_PATH)
        if not icon.isNull():
            return icon
    except Exception:
        pass
    return QIcon()


# ──────────────────────────────────────────────────────────────────────────────
#  HALFTONE ENGINE  (vectorised numpy — faithful port of render-core.js)
# ──────────────────────────────────────────────────────────────────────────────

_BAYER_4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
], dtype=np.float32)


def _hex_to_rgb(hex_str: str, fallback=(24, 24, 28)) -> tuple:
    h = (hex_str or "").strip().lstrip("#")
    if len(h) == 6:
        try:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        except ValueError:
            pass
    return fallback


def _mix_hex(a: str, b: str, t: float) -> str:
    ra, ga, ba = _hex_to_rgb(a, (0, 0, 0))
    rb, gb, bb = _hex_to_rgb(b, (0, 0, 0))
    r  = int(ra + (rb - ra) * t)
    g  = int(ga + (gb - ga) * t)
    bv = int(ba + (bb - ba) * t)
    return f"#{r:02x}{g:02x}{bv:02x}"


def _h2d(x: np.ndarray, y: np.ndarray, seed) -> np.ndarray:
    """Vectorised hash2D matching the JS sin-based version."""
    n = np.sin(x * 127.1 + y * 311.7 + seed * 17.77) * 43758.5453123
    return n - np.floor(n)


def _sample_bilinear(src: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Bilinear sample of (H,W,3) float32 array at fractional coords sx/sy."""
    H, W = src.shape[:2]
    sx = np.clip(sx, 0, W - 1)
    sy = np.clip(sy, 0, H - 1)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    tx = (sx - x0)[..., np.newaxis]
    ty = (sy - y0)[..., np.newaxis]
    d00 = src[y0, x0]
    d10 = src[y0, x1]
    d01 = src[y1, x0]
    d11 = src[y1, x1]
    return d00 * (1 - tx) * (1 - ty) + d10 * tx * (1 - ty) + d01 * (1 - tx) * ty + d11 * tx * ty


def _channel_value(rgb: np.ndarray, key: str) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    k = 1.0 - np.maximum(r, np.maximum(g, b))
    if key == "k":
        return np.clip(k, 0, 1)
    den = np.maximum(1.0 - k, 1e-6)
    if key == "c":
        return np.clip((1.0 - r - k) / den, 0, 1)
    if key == "m":
        return np.clip((1.0 - g - k) / den, 0, 1)
    if key == "y":
        return np.clip((1.0 - b - k) / den, 0, 1)
    if key == "r":    return np.clip(r - (g + b) * 0.3, 0, 1)
    if key == "g":    return np.clip(g - (r + b) * 0.3, 0, 1)
    if key == "b":    return np.clip(b - (r + g) * 0.3, 0, 1)
    if key == "o":    return np.clip((r + g) * 0.6 - b * 0.5, 0, 1)
    if key == "v":    return np.clip((r + b) * 0.55 - g * 0.5, 0, 1)
    # mono: luminance-based
    return 1.0 - (r * 0.2126 + g * 0.7152 + b * 0.0722)


def _adjust_tone(val: np.ndarray, s: dict, rv: np.ndarray) -> np.ndarray:
    t = np.clip(val, 0.0, 1.0)
    g = max(float(s.get("gamma", 1.0)), 0.05)
    t = np.power(np.maximum(t, 1e-9), 1.0 / g)
    t = (t - 0.5) * (1.0 + float(s.get("contrast", 0.18))) + 0.5
    t = np.clip(t, 0.0, 1.0)
    hi = float(s.get("highlights", 0.28))
    sh = float(s.get("shadows",    0.16))
    t += (np.power(np.maximum(t, 1e-9), 0.62) - t) * hi * 0.8
    t += (np.power(np.maximum(t, 1e-9), 1.65) - t) * sh * 0.8
    t  = np.clip(t + (rv - 0.5) * float(s.get("grain", 0.14)) * 0.35, 0.0, 1.0)
    t  = np.clip(t * (0.45 + float(s.get("fill", 0.78)) * 0.95), 0.0, 1.0)
    return t


def _apply_dither(
    tone: np.ndarray, mode: str,
    row_idx: np.ndarray, col_idx: np.ndarray,
    seed: np.ndarray, s: dict,
) -> np.ndarray:
    if mode == "none":
        return tone
    if mode == "bayer":
        thr = (_BAYER_4[row_idx % 4, col_idx % 4] + 0.5) / 16.0
        return np.clip(tone + (thr - 0.5) * 0.32, 0.0, 1.0)
    # floyd / stipple — implemented as noise-based ordered dither
    noise = _h2d(
        col_idx.astype(np.float32) * 1.27,
        row_idx.astype(np.float32) * 1.91,
        seed + 99,
    )
    return np.clip(tone + (noise - 0.5) * (0.45 + float(s.get("stipple", 0.25)) * 0.35), 0.0, 1.0)


def _shape_coverage(
    style: str,
    nx: np.ndarray, ny: np.ndarray,
    tone: np.ndarray, seed: np.ndarray,
) -> np.ndarray:
    ax = np.abs(nx)
    ay = np.abs(ny)
    R  = np.sqrt(nx * nx + ny * ny)
    z  = np.zeros_like(R)

    if style == "dot":
        core = R <= 0.72
        feather = (R > 0.72) & (R <= 1.0)
        edge = 1.0 - ((R - 0.72) / 0.28)
        edge = np.clip(edge, 0.0, 1.0)
        edge = edge * edge * (3.0 - 2.0 * edge)
        return np.where(core, 1.0, np.where(feather, edge, z))

    if style == "line":
        thr = 0.12 + tone * 0.3
        return np.where(ay <= thr, 1.0 - ay / np.maximum(thr, 1e-9), z)

    if style == "square":
        mx = np.maximum(ax, ay)
        return np.where((ax <= 1) & (ay <= 1), 1.0 - mx * 0.55, z)

    if style == "diamond":
        sv = ax + ay
        return np.where(sv <= 1.1, 1.0 - (sv / 1.1) * 0.72, z)

    if style == "cross":
        bar  = 0.15 + tone * 0.22
        in_b = (ax <= bar) | (ay <= bar)
        val  = 1.0 - np.minimum(
            np.minimum(ax / np.maximum(bar, 1e-9), 1.0),
            np.minimum(ay / np.maximum(bar, 1e-9), 1.0),
        ) * 0.5
        return np.where(in_b, val, z)

    if style == "ring":
        th    = 0.12 + (1.0 - tone) * 0.18
        inner = 1.0 - np.abs(R - (1.0 - th * 0.5)) / np.maximum(th, 1e-9)
        return np.where((R <= 1) & (R >= (1.0 - th)), inner, z)

    if style == "spiral":
        ang  = np.arctan2(ny, nx)
        wave = np.abs(np.sin(ang * 3 + R * 8))
        return np.where(R <= 1, np.clip((wave - 0.35) * 1.7, 0, 1) * (1.0 - R * 0.4), z)

    if style == "wave":
        wave = np.cos(nx * 5 + ny * 3.5)
        return np.where(R <= 1.1, np.clip((wave + tone) * 0.65, 0, 1), z)

    if style == "noise":
        noise = _h2d(nx * 17.2, ny * 11.4, seed)
        return np.where(R <= 1.12, np.clip((noise - (0.45 - tone * 0.28)) * 2.8, 0, 1), z)

    if style == "stipple":
        noise = _h2d(nx * 23.5, ny * 29.7, seed + 47)
        return np.where((R <= 1.08) & (noise > (0.7 - tone * 0.42)), 1.0 - R * 0.55, z)

    if style == "hex":
        q = 0.8660254 * ax + 0.5 * ay
        return np.where(q <= 1, 1.0 - q * 0.6, z)

    if style == "ellipse":
        e = np.sqrt(nx * nx * 0.5 + ny * ny * 1.6)
        return np.where(e <= 1, 1.0 - e * 0.7, z)

    # checker
    check = np.where(np.sign(nx) == np.sign(ny), 1.0, 0.45)
    return np.where(R <= 1.06, check * (1.0 - R * 0.45), z)


def _get_plates(s: dict) -> list:
    base = float(s.get("rotation", 18)) * (math.pi / 180)
    pc   = s.get("plateColors") or {}

    def col(key, fb):
        return _hex_to_rgb(pc.get(key, ""), fb)

    mono = col("mono", (24,  24,  28))
    cv   = col("c",    (0,   181, 212))
    mv   = col("m",    (226, 65,  137))
    yv   = col("y",    (246, 211, 60))
    kv   = col("k",    (24,  24,  28))
    ov   = col("o",    (255, 130, 47))
    rv   = col("r",    (235, 83,  78))
    gv   = col("g",    (48,  176, 111))
    bv   = col("b",    (77,  123, 255))
    vv   = col("v",    (136, 81,  255))

    D   = math.pi / 180
    sep = s.get("separation", "mono")

    if sep == "mono":
        return [{"key": "mono", "color": mono, "angle": base,          "tint": 1.00}]
    if sep == "4":
        return [
            {"key": "c", "color": cv, "angle": base + 15 * D, "tint": 0.94},
            {"key": "m", "color": mv, "angle": base + 75 * D, "tint": 0.94},
            {"key": "y", "color": yv, "angle": base,           "tint": 0.88},
            {"key": "k", "color": kv, "angle": base + 45 * D, "tint": 1.00},
        ]
    if sep == "6":
        return [
            {"key": "c", "color": cv, "angle": base + 15  * D, "tint": 0.90},
            {"key": "m", "color": mv, "angle": base + 75  * D, "tint": 0.90},
            {"key": "y", "color": yv, "angle": base,            "tint": 0.84},
            {"key": "k", "color": kv, "angle": base + 45  * D, "tint": 1.00},
            {"key": "o", "color": ov, "angle": base + 105 * D, "tint": 0.82},
            {"key": "g", "color": gv, "angle": base + 135 * D, "tint": 0.82},
        ]
    # 8-channel
    return [
        {"key": "c", "color": cv, "angle": base + 15  * D, "tint": 0.88},
        {"key": "m", "color": mv, "angle": base + 75  * D, "tint": 0.88},
        {"key": "y", "color": yv, "angle": base,            "tint": 0.80},
        {"key": "k", "color": kv, "angle": base + 45  * D, "tint": 1.00},
        {"key": "r", "color": rv, "angle": base + 95  * D, "tint": 0.74},
        {"key": "g", "color": gv, "angle": base + 125 * D, "tint": 0.74},
        {"key": "b", "color": bv, "angle": base + 155 * D, "tint": 0.74},
        {"key": "v", "color": vv, "angle": base + 175 * D, "tint": 0.72},
    ]


def _blend_layer(base: np.ndarray, ink_rgb: np.ndarray, alpha: np.ndarray, mode: str) -> np.ndarray:
    ink = ink_rgb.reshape((1, 1, 3)) if ink_rgb.ndim == 1 else ink_rgb
    if mode == "multiply":
        blended = base * ink
    elif mode == "screen":
        blended = 1.0 - (1.0 - base) * (1.0 - ink)
    elif mode == "overlay":
        blended = np.where(base <= 0.5, 2.0 * base * ink, 1.0 - 2.0 * (1.0 - base) * (1.0 - ink))
    elif mode == "softlight":
        blended = (1.0 - 2.0 * ink) * base * base + 2.0 * ink * base
    else:
        blended = np.broadcast_to(ink, base.shape)
    return np.clip(base * (1.0 - alpha) + blended * alpha, 0.0, 1.0)


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _build_gradient_map(width: int, height: int, settings: dict) -> np.ndarray | None:
    mode = str(settings.get("gradientMode", "none")).lower()
    strength = min(max(float(settings.get("gradientStrength", 0.0)), 0.0), 1.0)
    if mode == "none" or strength <= 0:
        return None

    start = np.array(_hex_to_rgb(settings.get("gradientColor1", "#00b5d4"), (0, 181, 212)), dtype=np.float32) / 255.0
    end = np.array(_hex_to_rgb(settings.get("gradientColor2", "#f6d33c"), (246, 211, 60)), dtype=np.float32) / 255.0

    py_g, px_g = np.mgrid[0:height, 0:width]
    px = px_g.astype(np.float32) / max(width - 1, 1)
    py = py_g.astype(np.float32) / max(height - 1, 1)

    if mode == "horizontal":
        factor = px
    elif mode == "radial":
        fx = px * 2.0 - 1.0
        fy = py * 2.0 - 1.0
        factor = np.clip(np.sqrt(fx * fx + fy * fy), 0.0, 1.0)
    elif mode == "diagonal":
        factor = (px + py) * 0.5
    else:
        factor = py

    factor = _smoothstep01(factor)[..., np.newaxis]
    return start * (1.0 - factor) + end * factor


def _tone_to_shape_scale(style: str, tone: np.ndarray, min_dot: float, flow: float) -> np.ndarray:
    """Map tone to area scale more like an AM halftone: tone primarily changes area, not opacity."""
    t = np.clip(tone, 0.0, 1.0)
    # Lower flow means more aggressive shadow growth; higher flow is tighter and cleaner.
    gamma = 1.35 - np.clip(flow, 0.0, 1.0) * 0.95
    area_tone = np.power(np.maximum(t, 1e-6), gamma)

    if style in ("dot", "ring", "hex", "ellipse", "diamond", "square", "checker"):
        # For area-based shapes, radius should roughly follow sqrt(area).
        scale = np.sqrt(area_tone)
    elif style in ("line", "cross"):
        scale = np.power(area_tone, 0.72)
    else:
        scale = np.power(area_tone, 0.58)

    return min_dot + (1.0 - min_dot) * np.clip(scale, 0.0, 1.0)


def render_halftone(src_pil: Image.Image, out_size: tuple, settings: dict) -> Image.Image:
    """
    Vectorised numpy halftone renderer — faithful port of render-core.js.
    Returns an RGB PIL Image at out_size = (width, height).
    """
    OW, OH = out_size
    SW, SH = src_pil.size

    src_np = np.array(src_pil.convert("RGB"), dtype=np.float32) / 255.0

    paper_rgb = _hex_to_rgb(settings.get("paperColor", "#f7f4ee"), (247, 244, 238))
    paper_f   = [v / 255.0 for v in paper_rgb]
    out       = np.full((OH, OW, 3), paper_f, dtype=np.float32)

    plates    = _get_plates(settings)
    cell_size = float(settings.get("size", 14))
    layout    = settings.get("layout", "square")
    style     = settings.get("style",  "dot")
    blend_mode = str(settings.get("blendMode", "multiply")).lower()
    ink_strength = min(max(float(settings.get("inkStrength", 1.0)), 0.0), 2.0)
    color_boost = min(max(float(settings.get("colorBoost", 0.35)), 0.0), 1.5)
    gradient_strength = min(max(float(settings.get("gradientStrength", 0.0)), 0.0), 1.0)

    # pixel-centre coordinate grids
    py_g, px_g = np.mgrid[0:OH, 0:OW]
    px_g = px_g.astype(np.float32) + 0.5
    py_g = py_g.astype(np.float32) + 0.5
    cx, cy = float(OW) * 0.5, float(OH) * 0.5
    gradient_map = _build_gradient_map(OW, OH, settings)

    for plate_idx, plate in enumerate(plates):
        angle       = float(plate["angle"])
        cos_a       = math.cos(angle)
        sin_a       = math.sin(angle)
        plate_color = np.array(plate["color"], dtype=np.float32) / 255.0
        tint        = float(plate["tint"])
        ink_field = plate_color
        if gradient_map is not None and gradient_strength > 0:
            ink_field = plate_color.reshape((1, 1, 3)) * (1.0 - gradient_strength) + gradient_map * gradient_strength

        # ── Cell assignment ───────────────────────────────────────────────
        if layout == "radial":
            dx_r     = px_g - cx
            dy_r     = py_g - cy
            r_dist   = np.sqrt(dx_r ** 2 + dy_r ** 2)
            ring_idx = np.maximum(
                np.round((r_dist - cell_size * 0.5) / (cell_size * 0.9)).astype(np.int32),
                0,
            )
            ring_rad = cell_size * 0.5 + ring_idx.astype(np.float32) * (cell_size * 0.9)
            circ     = np.maximum(cell_size * 6.0, 2.0 * math.pi * ring_rad)
            count    = np.maximum(8, np.round(circ / cell_size).astype(np.int32))
            theta    = np.arctan2(dy_r, dx_r)
            col_idx  = (
                np.round(theta / (2.0 * math.pi) * count.astype(np.float32)).astype(np.int32)
                % np.maximum(count, 1)
            )
            ca       = col_idx.astype(np.float32) / np.maximum(count.astype(np.float32), 1.0) * 2.0 * math.pi
            cell_x   = cx + np.cos(ca) * ring_rad
            cell_y   = cy + np.sin(ca) * ring_rad
            row_idx  = ring_idx

        else:
            dx       = px_g - cx
            dy       = py_g - cy
            rx       = dx * cos_a + dy * sin_a
            ry       = -dx * sin_a + dy * cos_a
            row_step = cell_size * 0.8660254 if layout == "hex" else cell_size
            row_idx  = np.round(ry / row_step).astype(np.int32)

            if layout in ("hex", "diagonal"):
                row_odd  = (row_idx % 2 != 0)
                col_even = np.round(rx / cell_size).astype(np.int32)
                col_odd  = np.round((rx - cell_size * 0.5) / cell_size).astype(np.int32)
                col_idx  = np.where(row_odd, col_odd, col_even)
                gx       = np.where(
                    row_odd,
                    col_idx.astype(np.float32) * cell_size + cell_size * 0.5,
                    col_idx.astype(np.float32) * cell_size,
                )
            else:
                col_idx  = np.round(rx / cell_size).astype(np.int32)
                gx       = col_idx.astype(np.float32) * cell_size

            gy     = row_idx.astype(np.float32) * row_step
            cell_x = gx * cos_a - gy * sin_a + cx
            cell_y = gx * sin_a + gy * cos_a + cy

        # Cell seed (per-cell, same for all pixels in a cell)
        seed = (
            np.float32((plate_idx + 1) * 109)
            + row_idx.astype(np.float32) * 17.0
            + col_idx.astype(np.float32) * 31.0
        )

        # ── Jitter + distortion ───────────────────────────────────────────
        dist  = min(max(float(settings.get("distortion", 0.08)), 0.0), 1.0)
        da    = dist * cell_size * 0.55
        wx    = (_h2d(cell_x * 0.11, cell_y * 0.07, seed + 211) - 0.5) * 2.0 * da
        wy    = (_h2d(cell_x * 0.09, cell_y * 0.13, seed + 311) - 0.5) * 2.0 * da
        stip  = float(settings.get("stipple", 0.25))
        jx    = (_h2d(col_idx.astype(np.float32), row_idx.astype(np.float32), seed)      - 0.5) * stip * cell_size * 0.18
        jy    = (_h2d(row_idx.astype(np.float32), col_idx.astype(np.float32), seed + 7)  - 0.5) * stip * cell_size * 0.18
        fcx   = cell_x + jx + wx
        fcy   = cell_y + jy + wy

        # ── Source sample with chromatic aberration ───────────────────────
        ab    = float(settings.get("aberration", 0.0))
        ssx   = (fcx / OW) * SW + cos_a * ab
        ssy   = (fcy / OH) * SH + sin_a * ab
        samp  = _sample_bilinear(src_np, ssx, ssy)

        # ── Tone ──────────────────────────────────────────────────────────
        base_v = _channel_value(samp, plate["key"])
        rand_v = _h2d(col_idx.astype(np.float32) * 1.31, row_idx.astype(np.float32) * 1.17, seed + 3)
        tone   = _adjust_tone(base_v, settings, rand_v)
        tone   = _apply_dither(tone, settings.get("dither", "none"), row_idx, col_idx, seed, settings)
        if plate["key"] not in ("mono", "k") and color_boost > 0:
            sat = np.max(samp, axis=-1) - np.min(samp, axis=-1)
            tone = np.clip(tone * (1.0 + sat * color_boost * 0.6) + sat * color_boost * 0.15, 0.0, 1.0)
        tone   = np.clip(tone, 0.0, 1.0)

        # ── Shape scale ───────────────────────────────────────────────────
        mn_d   = min(max(float(settings.get("minDot", 0.03)), 0.0), 0.35)
        flow   = min(max(float(settings.get("flow",   0.62)), 0.0), 1.0)
        ss     = _tone_to_shape_scale(style, tone, mn_d, flow)

        # Spot fade creates a stronger size/tone ramp.
        # Tonal mode is the practical default: darker areas get larger/heavier dots,
        # highlights taper down cleanly.
        sf = min(max(float(settings.get("spotFade", 0.0)), 0.0), 1.0)
        if sf > 0:
            fade_mode = settings.get("spotFadeMode", "none")
            if fade_mode == "none":
                fade_mode = "tonal"

            if fade_mode == "tonal":
                fade_driver = np.clip(tone, 0.0, 1.0)
            elif fade_mode == "radial":
                fx = (fcx - cx) / max(cx, 1.0)
                fy = (fcy - cy) / max(cy, 1.0)
                fade_driver = 1.0 - np.clip(np.sqrt(fx * fx + fy * fy), 0.0, 1.0)
            elif fade_mode == "vertical":
                fade_driver = 1.0 - np.clip(fcy / max(float(OH), 1.0), 0.0, 1.0)
            elif fade_mode == "horizontal":
                fade_driver = 1.0 - np.clip(fcx / max(float(OW), 1.0), 0.0, 1.0)
            else:
                fade_driver = np.clip(tone, 0.0, 1.0)

            if bool(settings.get("spotFadeReverse", False)):
                fade_driver = 1.0 - fade_driver

            # fade_driver near 1 -> larger/heavier; near 0 -> smaller/lighter
            size_map = 0.08 + fade_driver * (1.15 + sf * 2.5)
            tone_map = 0.82 + fade_driver * (sf * 0.55)
            ss *= np.clip(size_map, 0.08, 3.4)
            tone = np.clip(tone * tone_map, 0.0, 1.0)

        bleed  = min(max(float(settings.get("bleed", 0.1)), 0.0), 1.0)
        bs     = 1.0 + bleed * 0.65
        ex_m   = 1.65 if style == "line" else (1.3  if style == "ellipse" else 1.0)
        ey_m   = 0.52 if style == "line" else (0.7  if style == "ellipse" else 1.0)
        ext_x  = cell_size * 0.55 * ss * bs * ex_m
        ext_y  = cell_size * 0.55 * ss * bs * ey_m

        # ── Local normalised coords ───────────────────────────────────────
        dlx  = px_g - fcx
        dly  = py_g - fcy
        nx   = (dlx * cos_a  + dly * sin_a)  / np.maximum(ext_x, 1e-9)
        ny   = (-dlx * sin_a + dly * cos_a)  / np.maximum(ext_y, 1e-9)

        # ── Coverage + opacity ────────────────────────────────────────────
        cov     = _shape_coverage(style, nx, ny, tone, seed)
        fill    = float(settings.get("fill", 0.78))
        rb      = 0.08 if style == "ring" else 0.0
        base_opacity = tint * (0.72 + fill * 0.24) + rb
        opacity = np.clip((base_opacity * (0.9 + tone * 0.1)) * ink_strength, 0.0, 1.0)
        alpha   = (cov * opacity)[..., np.newaxis]

        # ── Blend/composite ───────────────────────────────────────────────
        out = _blend_layer(out, ink_field, alpha, blend_mode)

    # ── Grain pass ────────────────────────────────────────────────────────
    grain = float(settings.get("grain", 0.14))
    if grain > 0:
        gs   = max(0.6, float(settings.get("grainSize", 3.0)))
        gxf  = 1.0 / gs
        gyf  = 1.35 / gs
        py2, px2 = np.mgrid[0:OH, 0:OW]
        noise = (
            _h2d(px2.astype(np.float32) * gxf, py2.astype(np.float32) * gyf, 913.0) - 0.5
        ) * grain * 18.0 / 255.0
        out = np.clip(out + noise[..., np.newaxis], 0.0, 1.0)

    return Image.fromarray((out * 255).astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

class NoWheelSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class PreviewScrollArea(QScrollArea):
    zoomRequested = Signal(int)

    def wheelEvent(self, event):
        self.zoomRequested.emit(event.angleDelta().y())
        event.accept()


class CardPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("Card")

    def update_theme_colors(self):
        pass


class Worker(QThread):
    finished = Signal(object)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn   = fn
        self._args = args
        self._kw   = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kw)
            self.finished.emit(result)
        except Exception:
            self.finished.emit(None)


# ──────────────────────────────────────────────────────────────────────────────
#  HOME PAGE
# ──────────────────────────────────────────────────────────────────────────────

class HomePage(CardPage):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 40, 48, 40)
        root.setSpacing(28)

        # Hero
        hero = QLabel("HalfTone")
        hero.setAlignment(Qt.AlignCenter)
        hero.setStyleSheet("font-size: 56px; font-weight: 800; letter-spacing: 5px;")
        sub = QLabel("Print-Ready Halftone Art Generator")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("font-size: 16px;")
        root.addWidget(hero)
        root.addWidget(sub)

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(18)
        for value, label in [
            ("13",  "Styles"),
            ("4",   "Layouts"),
            ("8",   "Color Modes"),
            ("∞",   "Preset Slots"),
        ]:
            card = QWidget()
            card.setObjectName("StatCard")
            cl   = QVBoxLayout(card)
            cl.setContentsMargins(20, 20, 20, 20)
            cl.setSpacing(4)
            n = QLabel(value)
            n.setAlignment(Qt.AlignCenter)
            n.setStyleSheet("font-size: 38px; font-weight: 700;")
            t = QLabel(label)
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet("font-size: 12px;")
            cl.addWidget(n)
            cl.addWidget(t)
            stats_row.addWidget(card)
        root.addLayout(stats_row)

        # Features card
        feat_w = QWidget()
        feat_w.setObjectName("FeatCard")
        fl = QVBoxLayout(feat_w)
        fl.setContentsMargins(28, 18, 28, 18)
        fl.setSpacing(6)
        fl_title = QLabel("What's inside")
        fl_title.setStyleSheet("font-size: 15px; font-weight: 700;")
        fl.addWidget(fl_title)
        features = [
            "13 halftone styles  —  Dot · Line · Square · Diamond · Cross · Ring · Spiral · Wave · Noise · Stipple · Hex · Ellipse · Checker",
            "4 grid layouts  —  Square · Diagonal · Hex · Radial",
            "Colour separation  —  Mono · 4-channel CMYK · 6-channel · 8-channel",
            "Tone controls  —  Size · Contrast · Fill · Flow · Gamma · Rotation · Highlights · Shadows",
            "FX  —  Grain · Grain Size · Chromatic Aberration · Bleed · Distortion · Stipple · Minimum Dot",
            "Full colour control  —  Paper + per-channel CMYK/extended inks + gradients",
            "Preset system  —  Named saves with JSON export / import",
            "Real-time low-res preview  +  background hi-res full-size render",
        ]
        for line in features:
            lbl = QLabel(f"  {line}")
            lbl.setStyleSheet("font-size: 13px; padding: 2px 0;")
            lbl.setWordWrap(True)
            fl.addWidget(lbl)
        root.addWidget(feat_w)

        # Theme + update row
        bot = QHBoxLayout()
        bot.setSpacing(12)
        tlbl = QLabel("Theme:")
        tlbl.setStyleSheet("font-size: 13px;")
        self.theme_combo = NoWheelComboBox()
        self.theme_combo.setFixedWidth(200)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        chk_btn = QPushButton("Check for Updates")
        chk_btn.setFixedWidth(180)
        chk_btn.clicked.connect(self._check_updates)
        gh_btn = QPushButton("GitHub")
        gh_btn.setFixedWidth(100)
        gh_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/Orvlyn/Halftone"))
        )
        bot.addWidget(tlbl)
        bot.addWidget(self.theme_combo)
        bot.addStretch()
        bot.addWidget(chk_btn)
        bot.addWidget(gh_btn)
        root.addLayout(bot)

        root.addStretch()
        footer = QLabel("Made with care by Orvlyn")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("font-size: 11px; padding-bottom: 4px;")
        root.addWidget(footer)

    def _on_theme_changed(self, name: str):
        win = self.window()
        if win and hasattr(win, "apply_theme"):
            win.apply_theme(name)

    def _check_updates(self):
        win = self.window()
        if win and hasattr(win, "check_for_updates"):
            win.check_for_updates()

    def populate_themes(self, theme_names: list, current: str):
        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        for t in theme_names:
            self.theme_combo.addItem(t)
        idx = self.theme_combo.findText(current)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        self.theme_combo.blockSignals(False)

    def update_theme_colors(self):
        win = self.window()
        if win and hasattr(win, "current_theme_name"):
            idx = self.theme_combo.findText(win.current_theme_name)
            if idx >= 0:
                self.theme_combo.blockSignals(True)
                self.theme_combo.setCurrentIndex(idx)
                self.theme_combo.blockSignals(False)


# ──────────────────────────────────────────────────────────────────────────────
#  HALFTONE PAGE
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SETTINGS = {
    "style":           "dot",
    "layout":          "square",
    "dither":          "none",
    "separation":      "mono",
    "size":            14,
    "contrast":        0.18,
    "fill":            0.88,
    "flow":            0.38,
    "gamma":           1.0,
    "rotation":        18,
    "highlights":      0.28,
    "shadows":         0.16,
    "grain":           0.04,
    "grainSize":       3.0,
    "aberration":      0.0,
    "bleed":           0.02,
    "distortion":      0.0,
    "stipple":         0.0,
    "minDot":          0.03,
    "paperColor":      "#f7f4ee",
    "foregroundColor": "#18181c",
    "platePreset":     "Halftone Classic",
    "plateMono":       "#18181c",
    "plateC":          "#00b5d4",
    "plateM":          "#e24189",
    "plateY":          "#f6d33c",
    "plateK":          "#18181c",
    "plateO":          "#ff8c2f",
    "plateR":          "#eb534e",
    "plateG":          "#30b06f",
    "plateB":          "#4d7bff",
    "plateV":          "#8851ff",
    "inputExposure":   1.0,
    "inputSaturation": 1.0,
    "inputSharpen":    0.0,
    "inputVignette":   0.0,
    "inputInvert":     0.0,
    "spotFade":        0.35,
    "spotFadeMode":    "tonal",
    "spotFadeReverse": False,
    "blendMode":       "normal",
    "inkStrength":     0.95,
    "colorBoost":      0.12,
    "gradientMode":    "none",
    "gradientStrength": 0.0,
    "gradientColor1":  "#00b5d4",
    "gradientColor2":  "#f6d33c",
}

PLATE_PRESETS = {
    "Halftone Classic": {
        "paperColor": "#f7f4ee", "foregroundColor": "#18181c",
        "plateMono": "#18181c", "plateC": "#00b5d4", "plateM": "#e24189", "plateY": "#f6d33c", "plateK": "#18181c",
        "plateO": "#ff8c2f", "plateR": "#eb534e", "plateG": "#30b06f", "plateB": "#4d7bff", "plateV": "#8851ff",
        "blendMode": "normal", "fill": 0.9, "flow": 0.36, "spotFade": 0.38, "inkStrength": 0.98,
    },
    "Print Shop CMYK": {
        "paperColor": "#f2efe7", "foregroundColor": "#101015",
        "plateMono": "#101015", "plateC": "#00a8cc", "plateM": "#d92f89", "plateY": "#f4cf2f", "plateK": "#101015",
        "plateO": "#f08b31", "plateR": "#d14a3e", "plateG": "#2f9e63", "plateB": "#3d6bdb", "plateV": "#7b4de0",
        "blendMode": "multiply", "fill": 0.94, "flow": 0.28, "spotFade": 0.52, "inkStrength": 1.04,
    },
    "Riso Burst": {
        "paperColor": "#f4eed8", "foregroundColor": "#10343d",
        "plateMono": "#10343d", "plateC": "#005f73", "plateM": "#bb3e03", "plateY": "#e9d8a6", "plateK": "#10343d",
        "plateO": "#f18701", "plateR": "#d62828", "plateG": "#2a9d8f", "plateB": "#277da1", "plateV": "#9d4edd",
        "blendMode": "normal", "fill": 0.9, "flow": 0.34, "spotFade": 0.42,
    },
    "Neon Club": {
        "paperColor": "#0d0b17", "foregroundColor": "#f4f2ff",
        "plateMono": "#f4f2ff", "plateC": "#08f7fe", "plateM": "#fe53bb", "plateY": "#f5d300", "plateK": "#f4f2ff",
        "plateO": "#ff9f1c", "plateR": "#ff4365", "plateG": "#00f5a0", "plateB": "#3a86ff", "plateV": "#8338ec",
        "blendMode": "screen", "fill": 0.8, "flow": 0.45, "spotFade": 0.28, "colorBoost": 0.28,
    },
    "Blueprint": {
        "paperColor": "#eef7ff", "foregroundColor": "#0c2e4f",
        "plateMono": "#0c2e4f", "plateC": "#0d3b66", "plateM": "#3a86ff", "plateY": "#8ecae6", "plateK": "#0c2e4f",
        "plateO": "#f77f00", "plateR": "#d62828", "plateG": "#2a9d8f", "plateB": "#4361ee", "plateV": "#7209b7",
        "blendMode": "normal", "fill": 0.88, "flow": 0.34, "spotFade": 0.44,
    },
    "Grit Noir": {
        "paperColor": "#ddd8ce", "foregroundColor": "#111111",
        "plateMono": "#111111", "plateC": "#444444", "plateM": "#7b4f2c", "plateY": "#cf9f67", "plateK": "#111111",
        "plateO": "#b86b34", "plateR": "#8d3b2e", "plateG": "#50634a", "plateB": "#3c4a5b", "plateV": "#5b4a63",
        "blendMode": "multiply", "fill": 0.98, "flow": 0.22, "spotFade": 0.58, "inkStrength": 1.08,
    },
}

PREVIEW_MAX = 512


class ColorSwatch(QPushButton):
    """Button that displays a colour and opens QColorDialog on click."""
    colorChanged = Signal(str)

    def __init__(self, hex_color: str = "#ffffff", parent=None):
        super().__init__(parent)
        self.hex_color = hex_color
        self.setFixedSize(36, 26)
        self._refresh_style()
        self.clicked.connect(self._pick)

    def _refresh_style(self):
        self.setStyleSheet(
            f"QPushButton {{ background: {self.hex_color}; border: 2px solid rgba(255,255,255,0.2); "
            f"border-radius: 5px; }}"
            f"QPushButton:hover {{ border: 2px solid rgba(255,255,255,0.7); }}"
        )

    def _pick(self):
        qc  = QColor(self.hex_color)
        new = QColorDialog.getColor(qc, self, "Pick colour")
        if new.isValid():
            self.hex_color = new.name()
            self._refresh_style()
            self.colorChanged.emit(self.hex_color)

    def set_color(self, hex_str: str):
        self.hex_color = hex_str
        self._refresh_style()


class HalftonePage(CardPage):
    def __init__(self):
        super().__init__()
        self._pil_src  = None
        self._settings = dict(DEFAULT_SETTINGS)
        self._hires    = None
        self._preview_image = None
        self._preview_zoom  = 1.0
        self._base_pix = None
        self._live_hires = False
        self._worker   = None
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(380)
        self._debounce.timeout.connect(self._run_debounced_render)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: scrollable settings panel ──────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(328)
        scroll.setFrameShape(QFrame.NoFrame)

        settings_w = QWidget()
        self._sl   = QVBoxLayout(settings_w)
        self._sl.setContentsMargins(14, 16, 14, 20)
        self._sl.setSpacing(8)
        scroll.setWidget(settings_w)

        sl = self._sl

        # Image
        self._section(sl, "Image")
        load_btn = QPushButton("Load Image…")
        load_btn.clicked.connect(self._load_image)
        sl.addWidget(load_btn)
        self.img_info = QLabel("No image loaded")
        self.img_info.setStyleSheet("font-size: 11px;")
        sl.addWidget(self.img_info)

        # Style
        self._section(sl, "Style")
        self._sliders = {}
        self._combos  = {}
        self._swatches = {}
        self._style_cb  = self._combo(sl, "Style",      "style",
            [("dot","Dot"),("line","Line"),("square","Square"),("diamond","Diamond"),
             ("cross","Cross"),("ring","Ring"),("spiral","Spiral"),("wave","Wave"),
             ("noise","Noise"),("stipple","Stipple"),("hex","Hex"),
             ("ellipse","Ellipse"),("checker","Checker")])
        self._layout_cb = self._combo(sl, "Layout",     "layout",
            [("square","Square"),("diagonal","Diagonal"),("hex","Hex"),("radial","Radial")])
        self._dither_cb = self._combo(sl, "Dither",     "dither",
            [("none","None"),("bayer","Bayer"),("floyd","Floyd"),("stipple","Stipple")])
        self._sep_cb    = self._combo(sl, "Separation", "separation",
            [("mono","Mono"),("4","4 Channel"),("6","6 Channel"),("8","8 Channel")])

        # Tone
        self._section(sl, "Tone")
        self._slider(sl, "Size",       "size",       1,    80,   DEFAULT_SETTINGS["size"],       1)
        self._slider(sl, "Contrast",   "contrast",   -100, 200,  DEFAULT_SETTINGS["contrast"],   0.01)
        self._slider(sl, "Fill",       "fill",       0,    200,  DEFAULT_SETTINGS["fill"],       0.01)
        self._slider(sl, "Flow",       "flow",       0,    100,  DEFAULT_SETTINGS["flow"],       0.01)
        self._slider(sl, "Gamma",      "gamma",      5,    300,  DEFAULT_SETTINGS["gamma"],      0.01)
        self._slider(sl, "Rotation",   "rotation",   0,    360,  DEFAULT_SETTINGS["rotation"],   1)

        # Tonemap
        self._section(sl, "Tonal Range")
        self._slider(sl, "Highlights", "highlights", 0,    100,  DEFAULT_SETTINGS["highlights"], 0.01)
        self._slider(sl, "Shadows",    "shadows",    0,    100,  DEFAULT_SETTINGS["shadows"],    0.01)

        # FX
        self._section(sl, "FX")
        self._slider(sl, "Grain",       "grain",       0,    100,  DEFAULT_SETTINGS["grain"],       0.01)
        self._slider(sl, "Grain Size",  "grainSize",   5,    800,  DEFAULT_SETTINGS["grainSize"],   0.01)
        self._slider(sl, "Aberration",  "aberration",  0,    3000, DEFAULT_SETTINGS["aberration"],  0.01)
        self._slider(sl, "Bleed",       "bleed",       0,    100,  DEFAULT_SETTINGS["bleed"],       0.01)
        self._slider(sl, "Distortion",  "distortion",  0,    100,  DEFAULT_SETTINGS["distortion"],  0.01)
        self._slider(sl, "Stipple",     "stipple",     0,    100,  DEFAULT_SETTINGS["stipple"],     0.01)
        self._slider(sl, "Min Dot",     "minDot",      0,    30,   DEFAULT_SETTINGS["minDot"],      0.01)
        self._slider(sl, "Spot Fade",   "spotFade",    0,    100,  DEFAULT_SETTINGS["spotFade"],     0.01)
        self._slider(sl, "Ink Strength", "inkStrength", 0,   200,  DEFAULT_SETTINGS["inkStrength"],  0.01)
        self._slider(sl, "Color Boost",  "colorBoost",  0,   150,  DEFAULT_SETTINGS["colorBoost"],   0.01)
        self._combo(
            sl,
            "Fade Mode",
            "spotFadeMode",
            [("tonal", "Tonal"), ("radial", "Radial"), ("vertical", "Vertical"), ("horizontal", "Horizontal"), ("none", "None")],
        )
        self._combo(
            sl,
            "Blend",
            "blendMode",
            [("multiply", "Multiply"), ("normal", "Normal"), ("overlay", "Overlay"), ("softlight", "Soft Light"), ("screen", "Screen")],
        )
        fade_reverse_row = QHBoxLayout()
        fade_reverse_lbl = QLabel("Reverse Fade:")
        fade_reverse_lbl.setFixedWidth(92)
        self._fade_reverse_chk = QCheckBox("Enabled")
        self._fade_reverse_chk.stateChanged.connect(
            lambda state: self._on_toggle("spotFadeReverse", state == Qt.Checked)
        )
        fade_reverse_row.addWidget(fade_reverse_lbl)
        fade_reverse_row.addWidget(self._fade_reverse_chk)
        fade_reverse_row.addStretch()
        sl.addLayout(fade_reverse_row)

        # Input FX
        self._section(sl, "Input FX")
        self._slider(sl, "Exposure",    "inputExposure",   20,   300, DEFAULT_SETTINGS["inputExposure"],   0.01)
        self._slider(sl, "Saturation",  "inputSaturation", 0,    300, DEFAULT_SETTINGS["inputSaturation"], 0.01)
        self._slider(sl, "Sharpen",     "inputSharpen",    0,    250, DEFAULT_SETTINGS["inputSharpen"],    0.01)
        self._slider(sl, "Vignette",    "inputVignette",   0,    100, DEFAULT_SETTINGS["inputVignette"],   0.01)
        self._slider(sl, "Invert",      "inputInvert",     0,    100, DEFAULT_SETTINGS["inputInvert"],     0.01)

        # Colours
        self._section(sl, "Colours")
        self._swatch_row(sl, "Paper",  "paperColor")
        self._swatch_row(sl, "Ink",    "foregroundColor")

        self._section(sl, "Gradient")
        self._combo(
            sl,
            "Mode",
            "gradientMode",
            [("none", "None"), ("vertical", "Vertical"), ("horizontal", "Horizontal"), ("radial", "Radial"), ("diagonal", "Diagonal")],
        )
        self._slider(sl, "Strength", "gradientStrength", 0, 100, DEFAULT_SETTINGS["gradientStrength"], 0.01)
        self._swatch_row(sl, "Grad A", "gradientColor1")
        self._swatch_row(sl, "Grad B", "gradientColor2")

        auto_row = QHBoxLayout()
        auto_btn = QPushButton("Auto Palette From Image")
        auto_btn.clicked.connect(self._auto_palette_from_image)
        auto_row.addWidget(auto_btn)
        sl.addLayout(auto_row)

        plate_row = QHBoxLayout()
        plate_lbl = QLabel("Palette:")
        plate_lbl.setFixedWidth(88)
        self._plate_cb = NoWheelComboBox()
        for k in PLATE_PRESETS:
            self._plate_cb.addItem(k)
        self._plate_cb.currentTextChanged.connect(self._apply_plate_preset)
        plate_row.addWidget(plate_lbl)
        plate_row.addWidget(self._plate_cb)
        sl.addLayout(plate_row)

        self._swatch_row(sl, "Mono", "plateMono")
        self._swatch_row(sl, "Cyan", "plateC")
        self._swatch_row(sl, "Magenta", "plateM")
        self._swatch_row(sl, "Yellow", "plateY")
        self._swatch_row(sl, "Black", "plateK")
        self._swatch_row(sl, "Orange", "plateO")
        self._swatch_row(sl, "Red", "plateR")
        self._swatch_row(sl, "Green", "plateG")
        self._swatch_row(sl, "Blue", "plateB")
        self._swatch_row(sl, "Violet", "plateV")

        # Presets
        self._section(sl, "Presets")
        self._preset_name = QLineEdit()
        self._preset_name.setPlaceholderText("Preset name…")
        sl.addWidget(self._preset_name)

        p_btn_row = QHBoxLayout()
        save_btn = QPushButton("Save");   save_btn.clicked.connect(self._save_preset)
        del_btn  = QPushButton("Delete"); del_btn.clicked.connect(self._delete_preset)
        p_btn_row.addWidget(save_btn)
        p_btn_row.addWidget(del_btn)
        sl.addLayout(p_btn_row)

        p_load_row = QHBoxLayout()
        self._preset_cb = NoWheelComboBox()
        self._preset_cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        load_p = QPushButton("Load"); load_p.setFixedWidth(54)
        load_p.clicked.connect(self._load_preset)
        p_load_row.addWidget(self._preset_cb)
        p_load_row.addWidget(load_p)
        sl.addLayout(p_load_row)

        io_row = QHBoxLayout()
        exp_j = QPushButton("Export JSON"); exp_j.clicked.connect(self._export_presets)
        imp_j = QPushButton("Import JSON"); imp_j.clicked.connect(self._import_presets)
        io_row.addWidget(exp_j)
        io_row.addWidget(imp_j)
        sl.addLayout(io_row)
        sl.addStretch()

        # ── Right: preview area ───────────────────────────────────────────
        right = QVBoxLayout()
        right.setContentsMargins(16, 14, 16, 14)
        right.setSpacing(10)

        # toolbar
        tb = QHBoxLayout()
        tb.setSpacing(8)
        self.render_btn = QPushButton("Render Hi-Res")
        self.render_btn.setStyleSheet("font-weight: 600; padding: 8px 20px;")
        self.render_btn.clicked.connect(self._run_hires)
        self.export_btn = QPushButton("Export Image…")
        self.export_btn.clicked.connect(self._export_image)
        self.export_btn.setEnabled(False)
        reset_btn = QPushButton("Reset Settings")
        reset_btn.clicked.connect(self._reset_settings)
        self.status_lbl = QLabel("Load an image to start")
        self.status_lbl.setStyleSheet("font-size: 12px;")
        tb.addWidget(self.render_btn)
        tb.addWidget(self.export_btn)
        tb.addWidget(reset_btn)
        tb.addStretch()
        tb.addWidget(self.status_lbl)
        right.addLayout(tb)

        # preview scroll + label
        self.prev_scroll = PreviewScrollArea()
        self.prev_scroll.setAlignment(Qt.AlignCenter)
        self.prev_scroll.setFrameShape(QFrame.StyledPanel)
        self.prev_scroll.zoomRequested.connect(self._on_preview_wheel)
        self.prev_lbl = QLabel("← Load an image and adjust settings")
        self.prev_lbl.setAlignment(Qt.AlignCenter)
        self.prev_lbl.setStyleSheet("font-size: 13px;")
        self.prev_scroll.setWidget(self.prev_lbl)
        self.prev_scroll.setWidgetResizable(False)
        right.addWidget(self.prev_scroll, 1)

        zoom_row = QHBoxLayout()
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedWidth(28)
        self.zoom_out_btn.clicked.connect(lambda: self._set_preview_zoom(self._preview_zoom * 0.9))
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(28)
        self.zoom_in_btn.clicked.connect(lambda: self._set_preview_zoom(self._preview_zoom * 1.1))
        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.setFixedWidth(52)
        self.zoom_fit_btn.clicked.connect(self._fit_preview_to_view)
        self.zoom_lbl = QLabel("100%")
        self.zoom_lbl.setFixedWidth(56)
        self.zoom_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        zoom_row.addWidget(self.zoom_out_btn)
        zoom_row.addWidget(self.zoom_in_btn)
        zoom_row.addWidget(self.zoom_fit_btn)
        zoom_row.addStretch()
        zoom_row.addWidget(QLabel("Zoom:"))
        zoom_row.addWidget(self.zoom_lbl)
        right.addLayout(zoom_row)

        # indeterminate progress bar (4px strip)
        self.prog = QProgressBar()
        self.prog.setRange(0, 0)
        self.prog.setFixedHeight(4)
        self.prog.setTextVisible(False)
        self.prog.setVisible(False)
        right.addWidget(self.prog)

        root.addWidget(scroll)
        root.addLayout(right, 1)

        self._refresh_channel_color_visibility()
        self._refresh_presets()

    def _on_slider_auto_fade_mode(self):
        sf = float(self._settings.get("spotFade", 0.0))
        if sf > 0 and self._settings.get("spotFadeMode", "none") == "none":
            self._settings["spotFadeMode"] = "tonal"
            cb = self._combos.get("spotFadeMode")
            if cb is not None:
                idx = next((i for i in range(cb.count()) if cb.itemData(i) == "tonal"), -1)
                if idx >= 0:
                    cb.blockSignals(True)
                    cb.setCurrentIndex(idx)
                    cb.blockSignals(False)

    # ── Widget builder helpers ────────────────────────────────────────────

    def _section(self, layout: QVBoxLayout, title: str):
        lbl = QLabel(title.upper())
        lbl.setStyleSheet(
            "font-size: 10px; font-weight: 700; letter-spacing: 2px; padding-top: 10px;"
        )
        layout.addWidget(lbl)

    def _combo(self, layout, label: str, key: str, options: list) -> NoWheelComboBox:
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setFixedWidth(92)
        cb  = NoWheelComboBox()
        for val, text in options:
            cb.addItem(text, val)
        dv  = DEFAULT_SETTINGS.get(key)
        idx = next((i for i in range(cb.count()) if cb.itemData(i) == dv), 0)
        cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(lambda _, k=key, c=cb: self._on_combo(k, c))
        row.addWidget(lbl)
        row.addWidget(cb)
        layout.addLayout(row)
        self._combos[key] = cb
        return cb

    def _slider(self, layout, label: str, key: str, mn: int, mx: int, default, scale):
        row     = QHBoxLayout()
        lbl     = QLabel(f"{label}:")
        lbl.setFixedWidth(92)
        sl      = NoWheelSlider(Qt.Horizontal)
        sl.setRange(mn, mx)
        init    = int(round(float(default) / scale)) if scale != 1 else int(default)
        sl.setValue(init)
        disp_v  = (init * scale) if scale != 1 else init
        val_lbl = QLabel(f"{disp_v:.2f}" if isinstance(disp_v, float) else str(disp_v))
        val_lbl.setFixedWidth(44)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val_lbl.setStyleSheet("font-size: 11px;")

        def on_change(v, k=key, s=scale, vl=val_lbl):
            real = v * s if s != 1 else v
            vl.setText(f"{real:.2f}" if isinstance(real, float) else str(real))
            self._settings[k] = real
            if k == "spotFade":
                self._on_slider_auto_fade_mode()
            self._debounce.start()

        sl.valueChanged.connect(on_change)
        row.addWidget(lbl)
        row.addWidget(sl)
        row.addWidget(val_lbl)
        layout.addLayout(row)
        self._sliders[key] = (sl, scale)

    def _swatch_row(self, layout, label: str, key: str):
        row_w = QWidget()
        row = QHBoxLayout(row_w)
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"{label}:")
        lbl.setFixedWidth(92)
        sw  = ColorSwatch(DEFAULT_SETTINGS.get(key, "#ffffff"))
        sw.colorChanged.connect(lambda c, k=key: self._on_color(k, c))
        row.addWidget(lbl)
        row.addWidget(sw)
        row.addStretch()
        layout.addWidget(row_w)
        self._swatches[key] = sw
        if not hasattr(self, "_swatch_rows"):
            self._swatch_rows = {}
        self._swatch_rows[key] = row_w

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_combo(self, key: str, cb: NoWheelComboBox):
        self._settings[key] = cb.currentData()
        if key == "separation":
            self._refresh_channel_color_visibility()
        self._debounce.start()

    def _on_color(self, key: str, hex_color: str):
        self._settings[key] = hex_color
        self._debounce.start()

    def _on_toggle(self, key: str, value: bool):
        self._settings[key] = bool(value)
        self._debounce.start()

    def _apply_settings_payload(self, payload: dict):
        for key, value in payload.items():
            self._settings[key] = value
            if key in self._swatches:
                self._swatches[key].set_color(value)

    def _apply_plate_preset(self, name: str):
        preset = PLATE_PRESETS.get(name)
        if not preset:
            return
        self._settings["platePreset"] = name
        self._apply_settings_payload(preset)
        self._sync_controls()
        self._debounce.start()

    def _refresh_channel_color_visibility(self):
        sep = self._settings.get("separation", "mono")
        mapping = {
            "mono": {"plateMono"},
            "4": {"plateC", "plateM", "plateY", "plateK"},
            "6": {"plateC", "plateM", "plateY", "plateK", "plateO", "plateG"},
            "8": {"plateC", "plateM", "plateY", "plateK", "plateR", "plateG", "plateB", "plateV"},
        }
        active = mapping.get(sep, {"plateMono"})
        for key in ("plateMono", "plateC", "plateM", "plateY", "plateK", "plateO", "plateR", "plateG", "plateB", "plateV"):
            row = self._swatch_rows.get(key)
            if row:
                row.setVisible(key in active)

    # ── Image loading ─────────────────────────────────────────────────────

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.gif)",
        )
        if not path:
            return
        try:
            self._pil_src = Image.open(path).convert("RGB")
            w, h = self._pil_src.size
            self.img_info.setText(f"{os.path.basename(path)}  {w}×{h}")
            self._hires = None
            self.export_btn.setEnabled(False)
            self._debounce.start()
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", str(exc))

    # ── Rendering ─────────────────────────────────────────────────────────

    def _build_plate_colors(self) -> dict:
        s  = self._settings
        fg = s.get("foregroundColor", "#18181c")
        c_ink = s.get("plateC", "#00b5d4")
        m_ink = s.get("plateM", "#e24189")
        y_ink = s.get("plateY", "#f6d33c")
        return {
            "mono": s.get("plateMono", fg),
            "c":  c_ink,
            "m":  m_ink,
            "y":  y_ink,
            "k":  s.get("plateK", fg),
            "o":  s.get("plateO", _mix_hex(c_ink, "#ff8c00", 0.5)),
            "r":  s.get("plateR", _mix_hex(c_ink, "#ff3300", 0.35)),
            "g":  s.get("plateG", _mix_hex(m_ink, "#00cc44", 0.35)),
            "b":  s.get("plateB", _mix_hex(y_ink, "#3388ff", 0.35)),
            "v":  s.get("plateV", _mix_hex(m_ink, "#8800ff", 0.45)),
        }

    def _prepare_source_image(self, src: Image.Image) -> Image.Image:
        s = self._settings
        out = src.convert("RGB")

        exposure = float(s.get("inputExposure", 1.0))
        if abs(exposure - 1.0) > 0.01:
            out = ImageEnhance.Brightness(out).enhance(exposure)

        saturation = float(s.get("inputSaturation", 1.0))
        if abs(saturation - 1.0) > 0.01:
            out = ImageEnhance.Color(out).enhance(saturation)

        sharpen = float(s.get("inputSharpen", 0.0))
        if sharpen > 0.01:
            radius = 1.2 + sharpen * 1.8
            percent = int(100 + sharpen * 180)
            out = out.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))

        invert = float(s.get("inputInvert", 0.0))
        vignette = float(s.get("inputVignette", 0.0))
        if invert > 0.001 or vignette > 0.001:
            arr = np.array(out, dtype=np.float32) / 255.0
            if invert > 0.001:
                arr = arr * (1.0 - invert) + (1.0 - arr) * invert
            if vignette > 0.001:
                h, w = arr.shape[:2]
                yy, xx = np.mgrid[0:h, 0:w]
                cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
                rx = (xx - cx) / max(cx, 1.0)
                ry = (yy - cy) / max(cy, 1.0)
                rr = np.sqrt(rx * rx + ry * ry)
                mask = np.clip(1.0 - np.clip(rr, 0.0, 1.0) * vignette * 0.9, 0.15, 1.0)
                arr *= mask[..., np.newaxis]
            out = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), "RGB")

        return out

    def _rgb_to_hex(self, rgb: np.ndarray) -> str:
        r = int(np.clip(rgb[0], 0, 255))
        g = int(np.clip(rgb[1], 0, 255))
        b = int(np.clip(rgb[2], 0, 255))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _auto_palette_from_image(self):
        if self._pil_src is None:
            QMessageBox.information(self, "Auto Palette", "Load an image first.")
            return

        img = self._pil_src.copy()
        img.thumbnail((280, 280), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        flat = arr.reshape(-1, 3)
        if flat.size == 0:
            return

        rgb01 = flat / 255.0
        lum = rgb01[:, 0] * 0.2126 + rgb01[:, 1] * 0.7152 + rgb01[:, 2] * 0.0722
        sat = np.max(rgb01, axis=1) - np.min(rgb01, axis=1)

        # Paper from highlights, ink from shadows.
        paper = np.mean(flat[lum >= np.quantile(lum, 0.88)], axis=0) if np.any(lum >= np.quantile(lum, 0.88)) else np.mean(flat, axis=0)
        ink = np.mean(flat[lum <= np.quantile(lum, 0.15)], axis=0) if np.any(lum <= np.quantile(lum, 0.15)) else np.mean(flat, axis=0)

        candidates = flat[sat > 0.16]
        cand_sat = sat[sat > 0.16]
        if candidates.shape[0] < 30:
            candidates = flat
            cand_sat = sat

        w = cand_sat + 0.05
        i1 = int(np.argmax(w))
        c1 = candidates[i1]
        d1 = np.linalg.norm(candidates - c1, axis=1)
        i2 = int(np.argmax(d1 * w))
        c2 = candidates[i2]
        d2 = np.minimum(d1, np.linalg.norm(candidates - c2, axis=1))
        i3 = int(np.argmax(d2 * w))
        c3 = candidates[i3]

        self._settings["paperColor"] = self._rgb_to_hex(paper * 0.85 + np.array([255, 255, 255]) * 0.15)
        self._settings["foregroundColor"] = self._rgb_to_hex(ink * 0.85)
        c1_hex = self._rgb_to_hex(c1)
        c2_hex = self._rgb_to_hex(c2)
        c3_hex = self._rgb_to_hex(c3)

        self._settings["plateC"] = c1_hex
        self._settings["plateM"] = c2_hex
        self._settings["plateY"] = c3_hex
        self._settings["plateK"] = self._settings["foregroundColor"]
        self._settings["plateMono"] = self._settings["foregroundColor"]
        self._settings["plateO"] = _mix_hex(c1_hex, "#ff8c00", 0.55)
        self._settings["plateR"] = _mix_hex(c2_hex, "#ff3300", 0.45)
        self._settings["plateG"] = _mix_hex(c3_hex, "#00cc44", 0.45)
        self._settings["plateB"] = _mix_hex(c1_hex, "#3388ff", 0.45)
        self._settings["plateV"] = _mix_hex(c2_hex, "#8800ff", 0.48)
        self._settings["platePreset"] = "Halftone Classic"

        self._sync_controls()
        self._refresh_channel_color_visibility()
        self._debounce.start()

    def _build_render_settings(self) -> dict:
        s = dict(self._settings)
        s["plateColors"] = self._build_plate_colors()
        return s

    def _fit_size(self, src: Image.Image, max_dim: int):
        w, h = src.size
        if max(w, h) <= max_dim:
            return w, h
        if w >= h:
            return max_dim, max(1, int(h * max_dim / w))
        return max(1, int(w * max_dim / h)), max_dim

    def _run_debounced_render(self):
        if self._live_hires:
            self._run_hires(show_message=False)
        else:
            self._run_preview()

    def _run_preview(self):
        if self._pil_src is None:
            return
        if self._worker and self._worker.isRunning():
            self._debounce.start()
            return
        s   = self._build_render_settings()
        src = self._pil_src.copy()
        ow, oh = self._fit_size(src, PREVIEW_MAX)

        def do_render():
            thumb = src.copy()
            thumb.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
            thumb = self._prepare_source_image(thumb)
            return render_halftone(thumb, thumb.size, s)

        self.status_lbl.setText("Rendering preview…")
        self._worker = Worker(do_render)
        self._worker.finished.connect(self._on_preview_done)
        self._worker.start()

    def _on_preview_done(self, img: Image.Image):
        if img is None:
            self.status_lbl.setText("Preview error")
            return
        self._preview_image = img
        self._show_image(img)
        self.status_lbl.setText(f"Preview  {img.width}×{img.height}")

    def _run_hires(self, show_message=True):
        if self._pil_src is None:
            if show_message:
                QMessageBox.information(self, "No Image", "Load an image first.")
            return
        if self._worker and self._worker.isRunning():
            self._debounce.start()
            return
        self._live_hires = True
        s   = self._build_render_settings()
        src = self._pil_src.copy()
        self.prog.setVisible(True)
        self.render_btn.setEnabled(False)
        self.status_lbl.setText("Rendering hi-res…")

        def do_render():
            return render_halftone(self._prepare_source_image(src), src.size, s)

        self._worker = Worker(do_render)
        self._worker.finished.connect(self._on_hires_done)
        self._worker.start()

    def _on_hires_done(self, img: Image.Image):
        self.prog.setVisible(False)
        self.render_btn.setEnabled(True)
        if img is None:
            self.status_lbl.setText("Render failed")
            return
        self._hires = img
        self._preview_image = img
        self._show_image(img)
        self.export_btn.setEnabled(True)
        self.status_lbl.setText(f"Ready  {img.width}×{img.height}")

    def _show_image(self, img: Image.Image):
        data  = img.tobytes("raw", "RGB")
        qimg  = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
        self._base_pix = QPixmap.fromImage(qimg)
        self._apply_preview_zoom()

    def _on_preview_wheel(self, delta: int):
        if self._base_pix is None:
            return
        if delta > 0:
            self._set_preview_zoom(self._preview_zoom * 1.12)
        elif delta < 0:
            self._set_preview_zoom(self._preview_zoom / 1.12)

    def _set_preview_zoom(self, zoom: float):
        self._preview_zoom = max(0.1, min(8.0, zoom))
        self._apply_preview_zoom()

    def _fit_preview_to_view(self):
        if self._base_pix is None:
            return
        avail = self.prev_scroll.viewport().size()
        if self._base_pix.width() <= 0 or self._base_pix.height() <= 0:
            return
        zw = avail.width() / self._base_pix.width()
        zh = avail.height() / self._base_pix.height()
        self._set_preview_zoom(max(0.1, min(8.0, min(zw, zh))))

    def _apply_preview_zoom(self):
        if self._base_pix is None:
            return
        w = max(1, int(self._base_pix.width() * self._preview_zoom))
        h = max(1, int(self._base_pix.height() * self._preview_zoom))
        pix = self._base_pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.prev_lbl.setPixmap(pix)
        self.prev_lbl.resize(pix.size())
        self.zoom_lbl.setText(f"{int(self._preview_zoom * 100)}%")

    def _export_image(self):
        if self._hires is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "halftone_output.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff)",
        )
        if path:
            self._hires.save(path)
            self.status_lbl.setText(f"Saved → {os.path.basename(path)}")

    def _reset_settings(self):
        self._settings = dict(DEFAULT_SETTINGS)
        self._sync_controls()
        self._refresh_channel_color_visibility()
        self._plate_cb.blockSignals(True)
        self._plate_cb.setCurrentText(self._settings.get("platePreset", "Halftone Classic"))
        self._plate_cb.blockSignals(False)
        self._fade_reverse_chk.blockSignals(True)
        self._fade_reverse_chk.setChecked(bool(self._settings.get("spotFadeReverse", False)))
        self._fade_reverse_chk.blockSignals(False)
        self._debounce.start()

    def _sync_controls(self):
        """Push self._settings back into all UI controls (no signal spam)."""
        for key, (sl, scale) in self._sliders.items():
            val  = float(self._settings.get(key, DEFAULT_SETTINGS.get(key, 0)))
            tick = int(round(val / scale)) if scale != 1 else int(val)
            sl.blockSignals(True)
            sl.setValue(tick)
            sl.blockSignals(False)

        for key, sw in self._swatches.items():
            sw.set_color(self._settings.get(key, "#ffffff"))

        for key, cb in self._combos.items():
            val = self._settings.get(key, "")
            idx = next((i for i in range(cb.count()) if cb.itemData(i) == val), 0)
            cb.blockSignals(True)
            cb.setCurrentIndex(idx)
            cb.blockSignals(False)

        self._plate_cb.blockSignals(True)
        plate_name = self._settings.get("platePreset", "Halftone Classic")
        idx = self._plate_cb.findText(plate_name)
        if idx >= 0:
            self._plate_cb.setCurrentIndex(idx)
        self._plate_cb.blockSignals(False)

        self._fade_reverse_chk.blockSignals(True)
        self._fade_reverse_chk.setChecked(bool(self._settings.get("spotFadeReverse", False)))
        self._fade_reverse_chk.blockSignals(False)

    # ── Preset system ──────────────────────────────────────────────────────

    def _qs(self) -> QSettings:
        return QSettings("Orvlyn", "HalfTone")

    def _load_all_presets(self) -> dict:
        try:
            return json.loads(self._qs().value("presets", "{}"))
        except Exception:
            return {}

    def _save_all_presets(self, presets: dict):
        self._qs().setValue("presets", json.dumps(presets))

    def _refresh_presets(self):
        presets = self._load_all_presets()
        self._preset_cb.clear()
        for name in presets:
            self._preset_cb.addItem(name)

    def _save_preset(self):
        name = self._preset_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Preset", "Enter a preset name first.")
            return
        presets      = self._load_all_presets()
        presets[name] = self._settings.copy()
        self._save_all_presets(presets)
        self._refresh_presets()
        idx = self._preset_cb.findText(name)
        if idx >= 0:
            self._preset_cb.setCurrentIndex(idx)

    def _load_preset(self):
        name    = self._preset_cb.currentText()
        presets = self._load_all_presets()
        data    = presets.get(name)
        if not data:
            return
        self._settings = {**DEFAULT_SETTINGS, **data}
        self._sync_controls()
        self._debounce.start()

    def _delete_preset(self):
        name    = self._preset_cb.currentText()
        presets = self._load_all_presets()
        if name in presets:
            del presets[name]
            self._save_all_presets(presets)
        self._refresh_presets()

    def _export_presets(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Presets", "halftone_presets.json", "JSON (*.json)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._load_all_presets(), f, indent=2)

    def _import_presets(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Presets", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
        except Exception as exc:
            QMessageBox.warning(self, "Import Error", str(exc))
            return
        presets = self._load_all_presets()
        presets.update(data)
        self._save_all_presets(presets)
        self._refresh_presets()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ──────────────────────────────────────────────────────────────────────────────

THEMES = {
    "Halftone Default": {
        "accent":       "#e24189",
        "primary_bg":   "#0e0a12",
        "secondary_bg": "#180f1e",
        "tertiary_bg":  "#2a1735",
        "text":         "#f4e6fc",
    },
    "Ink on Paper": {
        "accent":       "#18181c",
        "primary_bg":   "#f7f4ee",
        "secondary_bg": "#ede8df",
        "tertiary_bg":  "#d4cdc2",
        "text":         "#1a1a1a",
        "is_light":     True,
    },
    "Newsprint": {
        "accent":       "#946f45",
        "primary_bg":   "#1a1410",
        "secondary_bg": "#2a2018",
        "tertiary_bg":  "#3e2f1e",
        "text":         "#e8d8c0",
    },
    "Synthwave": {
        "accent":       "#fe53bb",
        "primary_bg":   "#0b0820",
        "secondary_bg": "#160f3a",
        "tertiary_bg":  "#30196e",
        "text":         "#f3d9ff",
    },
    "Riso": {
        "accent":       "#00b5d4",
        "primary_bg":   "#08100f",
        "secondary_bg": "#0d1c1a",
        "tertiary_bg":  "#0d3030",
        "text":         "#cef5ff",
    },
    "CMYK Lab": {
        "accent":       "#f6d33c",
        "primary_bg":   "#0a0a08",
        "secondary_bg": "#181610",
        "tertiary_bg":  "#2e2914",
        "text":         "#fffff0",
    },
    "Dark (Original)": {
        "accent":       "#00FFC6",
        "primary_bg":   "#070A0E",
        "secondary_bg": "#0B0F15",
        "tertiary_bg":  "#141A22",
        "text":         "#E6EAF0",
    },
    "Ocean Blue": {
        "accent":       "#42a5f5",
        "primary_bg":   "#0a1929",
        "secondary_bg": "#0d2136",
        "tertiary_bg":  "#1565c0",
        "text":         "#e3f2fd",
    },
    "Blood Moon": {
        "accent":       "#ff0033",
        "primary_bg":   "#1a0000",
        "secondary_bg": "#2d0000",
        "tertiary_bg":  "#4d0000",
        "text":         "#ffcccc",
    },
    "Nord": {
        "accent":       "#88c0d0",
        "primary_bg":   "#1a2332",
        "secondary_bg": "#243447",
        "tertiary_bg":  "#2e4060",
        "text":         "#d8dfe9",
    },
    "Tokyo Night": {
        "accent":       "#ff8c00",
        "primary_bg":   "#0d1424",
        "secondary_bg": "#152038",
        "tertiary_bg":  "#1f2f52",
        "text":         "#ffd8a6",
    },
    "Forest Ink": {
        "accent":       "#2a9d8f",
        "primary_bg":   "#08100c",
        "secondary_bg": "#10201a",
        "tertiary_bg":  "#183428",
        "text":         "#d0f0e4",
    },
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._qs   = QSettings("Orvlyn", "HalfTone")
        theme      = self._qs.value("theme", "Halftone Default")
        self.setWindowTitle("HalfTone")
        self.setMinimumSize(1200, 720)

        icon = get_cached_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)

        central = QWidget()
        self.setCentralWidget(central)
        root    = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────
        self._sidebar_l = QVBoxLayout()
        self._sidebar_l.setSpacing(4)
        self._sidebar_l.setContentsMargins(12, 14, 12, 12)

        self.stack    = QStackedWidget()
        self.pages    = {
            "Home":     HomePage(),
            "HalfTone": HalftonePage(),
        }
        for p in self.pages.values():
            self.stack.addWidget(p)

        self.sidebar_buttons = {}
        self._submenus       = {}

        def add_category(name: str, items: list):
            cat_btn = QPushButton(name)
            cat_btn.setCheckable(True)
            self._sidebar_l.addWidget(cat_btn)

            sub   = QWidget()
            sub_l = QVBoxLayout(sub)
            sub_l.setContentsMargins(18, 0, 0, 0)
            sub_l.setSpacing(4)
            sub.setVisible(False)
            self._sidebar_l.addWidget(sub)
            self._submenus[cat_btn] = sub

            def toggle(checked, b=cat_btn, s=sub):
                vis = not s.isVisible()
                for b2, s2 in self._submenus.items():
                    s2.setVisible(b2 == b and vis)
                    b2.setChecked(b2 == b and vis)

            cat_btn.clicked.connect(toggle)

            for item in items:
                btn = QPushButton(item)
                btn.setCheckable(True)
                btn.clicked.connect(lambda _, n=item: self.switch_page(n))
                sub_l.addWidget(btn)
                self.sidebar_buttons[item] = btn

        add_category("Welcome",   ["Home"])
        add_category("HalfTone", ["HalfTone"])

        self._sidebar_l.addStretch()
        self.made_label = QLabel("Made by Orvlyn")
        self.made_label.setStyleSheet("font-size: 11px; padding: 8px;")
        self._sidebar_l.addWidget(self.made_label)

        sidebar_w = QWidget()
        sidebar_w.setLayout(self._sidebar_l)
        sidebar_w.setFixedWidth(215)
        sidebar_w.setObjectName("Sidebar")

        root.addWidget(sidebar_w)
        root.addWidget(self.stack, 1)

        self.apply_theme(theme)
        self.switch_page("Home")

    # ── Version check ──────────────────────────────────────────────────────

    def _ver(self, v: str):
        parts = []
        for t in str(v or "").replace("-", ".").split("."):
            if t.isdigit():
                parts.append(int(t))
        return tuple(parts) or (0,)

    def check_for_updates(self):
        try:
            with urllib.request.urlopen(UPDATE_CHECK_URL, timeout=8) as resp:
                payload  = json.loads(resp.read().decode())
            latest   = str(payload.get("version",      "")).strip()
            download = str(payload.get("download_url", "")).strip()
            notes    = str(payload.get("notes",        "")).strip()
        except Exception as exc:
            QMessageBox.warning(self, "Update Check", f"Could not reach update server.\n\n{exc}")
            return

        if latest and self._ver(latest) > self._ver(APP_VERSION):
            msg = [f"Current: {APP_VERSION}", f"Latest:  {latest}"]
            if notes:    msg += ["", "Release notes:", notes]
            if download: msg += ["", f"Download: {download}"]
            QMessageBox.information(self, "Update Available", "\n".join(msg))
        else:
            QMessageBox.information(self, "Up To Date", f"You are on the latest version ({APP_VERSION}).")

    # ── Theme ──────────────────────────────────────────────────────────────

    def apply_theme(self, name: str):
        theme = THEMES.get(name, THEMES["Halftone Default"])

        self.current_theme_name   = name
        self.current_theme_colors = theme

        a  = theme.get("accent",       "#e24189")
        p  = theme.get("primary_bg",   "#0e0a12")
        s  = theme.get("secondary_bg", "#180f1e")
        t  = theme.get("tertiary_bg",  "#2a1735")
        tx = theme.get("text",         "#f4e6fc")
        il = theme.get("is_light",      False)
        fc = "#111111" if il else p   # foreground-on-accent

        css = f"""
            QMainWindow         {{ background: {p}; }}
            QWidget             {{ background: {p}; color: {tx}; font: 13px 'Segoe UI'; }}
            #Sidebar QPushButton {{
                background: {s}; border: none; padding: 10px 14px;
                border-radius: 6px; text-align: left; color: {tx};
            }}
            #Sidebar QPushButton:checked {{ background: {a}; color: {fc}; font-weight: bold; }}
            #Sidebar QPushButton:hover   {{ background: {t}; }}
            QWidget#Card    {{ background: {s}; border-radius: 12px; }}
            QWidget#StatCard {{ background: {t}; border-radius: 10px; }}
            QWidget#FeatCard {{ background: {t}; border-radius: 10px; }}
            QPushButton {{
                background: {s}; border: 1px solid {t}; border-radius: 7px;
                padding: 7px 12px; color: {tx}; font-weight: 500;
            }}
            QPushButton:hover   {{ border: 1px solid {a}; background: {t}; }}
            QPushButton:pressed {{ background: {a}; color: {fc}; }}
            QLineEdit, QComboBox, QSpinBox {{
                background: {s}; border: 1px solid {t}; border-radius: 6px;
                padding: 5px; color: {tx}; selection-background-color: {a};
            }}
            QLineEdit:focus, QComboBox:focus {{ border: 2px solid {a}; }}
            QComboBox::drop-down  {{ border-left: 1px solid {t}; width: 18px; }}
            QComboBox QAbstractItemView {{ background: {s}; color: {tx}; selection-background-color: {a}; }}
            QSlider             {{ background: {p}; }}
            QSlider::groove:horizontal {{
                background: {t}; height: 5px; border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {a}; width: 14px; margin: -5px 0; border-radius: 7px;
            }}
            QProgressBar  {{ background: {s}; border-radius: 3px; border: none; }}
            QProgressBar::chunk {{ background: {a}; border-radius: 3px; }}
            QLabel        {{ color: {tx}; }}
            QScrollArea   {{ border: none; background: {s}; }}
            QScrollBar:vertical {{
                background: {s}; width: 10px; border-radius: 5px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {a}; border-radius: 5px; min-height: 24px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar:horizontal {{
                background: {s}; height: 10px; border-radius: 5px; margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background: {a}; border-radius: 5px; min-width: 24px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        """
        self.setStyleSheet(css)

        if hasattr(self, "made_label"):
            self.made_label.setStyleSheet(f"font-size: 11px; color: {a}; padding: 8px;")

        home = self.pages.get("Home")
        if home and hasattr(home, "populate_themes"):
            home.populate_themes(list(THEMES.keys()), name)

        for page in self.pages.values():
            page.update_theme_colors()

        self._apply_titlebar(il)
        self._qs.setValue("theme", name)

    def _apply_titlebar(self, is_light: bool = False):
        if sys.platform == "win32" and HAS_WINDOWS_TITLEBAR:
            try:
                hwnd  = int(self.winId())
                DWMWA = 20
                val   = ctypes.c_int(0 if is_light else 1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA, ctypes.byref(val), ctypes.sizeof(val)
                )
            except Exception:
                pass

    # ── Navigation ────────────────────────────────────────────────────────

    def switch_page(self, name: str):
        for btn in self.sidebar_buttons.values():
            btn.setChecked(False)
        if name in self.sidebar_buttons:
            self.sidebar_buttons[name].setChecked(True)
        if name in self.pages:
            self.stack.setCurrentWidget(self.pages[name])


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
