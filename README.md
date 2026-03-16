# HalfTone

Print-ready halftone art generator - a full Python/PySide6 desktop app.

## Features

- **13 halftone styles** - Dot, Line, Square, Diamond, Cross, Ring, Spiral, Wave, Noise, Stipple, Hex, Ellipse, Checker
- **4 grid layouts** - Square, Diagonal, Hex, Radial
- **Colour separation** - Mono, 4-channel CMYK, 6-channel, 8-channel
- **Tone controls** - Size, Contrast, Fill, Flow, Gamma, Rotation, Highlights, Shadows
- **FX** - Grain, Grain Size, Chromatic Aberration, Bleed, Distortion, Stipple, Minimum Dot
- **Full colour control** - Paper, Foreground, 3 Tri-ink channels with 6 built-in presets (Riso Classic, Newsprint Warm, Synthwave, Forest Ink, Comic Pop)
- **Preset system** - Named saves with JSON export / import
- **Real-time low-res preview** + background hi-res full-size render
- **12 UI themes** - including halftone-specific palettes (Riso, Newsprint, Synthwave, CMYK Lab, Forest Ink…)
- **Version checker** via GitHub

## Requirements

```
pip install -r requirements.txt
```

- Python 3.10+
- PySide6 >= 6.4
- Pillow >= 9.0
- numpy >= 1.23

## Usage

```
python main.py
```

1. Click **Load Image** in the HalfTone page
2. Adjust style, layout, tone, FX and colour settings - preview updates automatically
3. Click **Render Hi-Res** to render at full source resolution
4. Click **Export Image** to save as PNG / JPEG / TIFF

## GitHub

<https://github.com/Orvlyn/Halftone>
