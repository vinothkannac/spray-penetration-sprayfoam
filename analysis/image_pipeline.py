"""Spray image processing: penetration length and spray angle from back-lit frames.

Pipeline per frame
  1. flat-field:  signal = 1 - frame / background   (the spray absorbs light, so signal > 0)
  2. Gaussian smoothing
  3. threshold = max(n_sigma * robust noise sigma, min_contrast)
  4. morphological opening, keep the connected component that touches the nozzle edge
  5. penetration = furthest axial extent of that component
  6. spray angle = 2*atan(slope/2) of a linear fit of the spray width over 20-80 % of the penetration

Works on any image sequence with the nozzle at the left edge and the spray travelling left to right
(rotate or flip your images first otherwise). Command line for your own frames:

    python analysis/image_pipeline.py --images DIR --background bg.png --pixel-size 1e-4

Design notes (why I chose these steps and what could replace them)
- Flat-field instead of plain background subtraction. Dividing by the background removes the
  uneven illumination multiplicatively, which is the right model for light attenuation
  (Beer-Lambert). Plain subtraction would leave a position-dependent contrast.
- Threshold from the noise, not a fixed grey level. I estimate sigma from the median absolute
  deviation of the whole frame (robust against the spray itself, which occupies a minority of the
  pixels) and use 6 sigma, with a floor of 2 % contrast so that a very clean frame does not
  produce a threshold of zero. Alternatives: Otsu (assumes a bimodal histogram, which is
  not true for a thin sparse tip), a fixed fraction of the peak extinction (a common
  convention in the spray literature), or a local/adaptive threshold.
  The threshold is the dominant uncertainty: the spray tip is optically thin, so a higher
  threshold shortens the penetration and narrows the angle. This is why the README quotes the
  image-based angle with a caveat.
- Connected component touching the nozzle edge. This rejects isolated noise and detached
  droplets, but it will also cut off a tip that has separated from the spray. Alternative: keep
  all components larger than a minimum area.
- Spray width instead of a cone fit. I fit the width (top edge to bottom edge) over 20-80 % of
  the length, mirroring the parcel-based definition in spray_metrics.py so that the two angles
  are comparable. Alternative: the Naber-Siebers cone over the first 60 % of the penetration.
- Pure NumPy/SciPy/OpenCV, no machine learning. The task is a classical segmentation of a
  high-contrast object, and a deterministic method can be explained and checked. A neural
  segmentation (for example U-Net) would only be worth it for noisy images with strong
  vapour or background structure.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)


def flat_field(frame: np.ndarray, bg: np.ndarray, smooth_px: float = 1.5) -> np.ndarray:
    """Steps 1 and 2: smoothed attenuation signal (0 = background, 1 = fully opaque)."""
    return ndimage.gaussian_filter(1.0 - frame / np.maximum(bg, 1.0), smooth_px)


def segment(frame: np.ndarray, bg: np.ndarray, smooth_px: float = 1.5, n_sigma: float = 6.0,
            min_contrast: float = 0.02, nozzle_margin_px: int = 5) -> tuple[np.ndarray, dict]:
    """Steps 1-4. Returns the binary spray mask and the threshold and noise level that were used."""
    signal = flat_field(frame, bg, smooth_px)
    med = np.median(signal)
    sigma = 1.4826 * np.median(np.abs(signal - med))  # MAD -> Gaussian sigma
    thr = max(n_sigma * sigma, min_contrast)
    mask = (signal - med) > thr
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    labels, n = ndimage.label(mask)
    keep = np.zeros_like(mask)
    for lab in range(1, n + 1):
        cols = np.where((labels == lab).any(axis=0))[0]
        if cols.min() <= nozzle_margin_px:
            keep |= labels == lab
    return keep, {"threshold": float(thr), "noise_sigma": float(sigma)}


def width_profile(mask: np.ndarray, pixel_size: float, frac=(0.2, 0.8)) -> tuple[np.ndarray, np.ndarray]:
    """Spray width (m) at each axial position (m) between frac[0] and frac[1] of the penetration."""
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return np.empty(0), np.empty(0)
    lo, hi = int(frac[0] * (cols.max() + 1)), int(frac[1] * (cols.max() + 1))
    xs, widths = [], []
    for c in range(lo, hi):
        rows = np.where(mask[:, c])[0]
        if len(rows):
            xs.append((c + 0.5) * pixel_size)
            widths.append((rows.max() - rows.min() + 1) * pixel_size)
    return np.array(xs), np.array(widths)


def penetration_and_angle(mask: np.ndarray, pixel_size: float, frac=(0.2, 0.8)) -> tuple[float, float]:
    """Steps 5 and 6. Returns (penetration in m, full spray angle in degrees)."""
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return float("nan"), float("nan")
    s_tip = (cols.max() + 1) * pixel_size
    xs, widths = width_profile(mask, pixel_size, frac)
    if len(xs) < 10:
        return s_tip, float("nan")
    slope = np.polyfit(xs, widths, 1)[0]  # d(width)/ds = 2*tan(theta/2)
    return s_tip, float(np.degrees(2.0 * np.arctan(slope / 2.0)))


def analyse_sequence(images: list[np.ndarray], bg: np.ndarray, times: list[float], pixel_size: float
                     ) -> tuple[pd.DataFrame, list[np.ndarray]]:
    rows, masks = [], []
    for img, t in zip(images, times):
        mask, info = segment(img.astype(np.float64), bg)
        s_tip, theta = penetration_and_angle(mask, pixel_size)
        rows.append({"t": t, "S_image": s_tip, "theta_image_deg": theta, **info})
        masks.append(mask)
    return pd.DataFrame(rows), masks


def _time_from_name(p: Path) -> float:
    """frame_012_t0000250.0us.png -> 250 us (the naming used by render_images.py)."""
    m = re.search(r"_t([0-9.]+)us", p.name)
    return float(m.group(1)) * 1e-6 if m else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory of frames (png/tif), sorted by name")
    ap.add_argument("--background", required=True, help="background image (mean of spray-free frames)")
    ap.add_argument("--pixel-size", type=float, required=True, help="m per pixel")
    ap.add_argument("--fps", type=float, default=None, help="frame rate; used for time if names have no _t<us>")
    ap.add_argument("--out", default="image_results.csv")
    a = ap.parse_args()

    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg"}
                   and p.name != Path(a.background).name)
    imgs = [load_gray(p) for p in files]
    times = [_time_from_name(p) for p in files]
    if np.isnan(times).any():
        if a.fps is None:
            raise SystemExit("File names carry no time stamp; pass --fps.")
        times = [i / a.fps for i in range(len(files))]
    df, _ = analyse_sequence(imgs, load_gray(Path(a.background)), times, a.pixel_size)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
