"""Render Lagrangian parcels as synthetic back-lit (shadowgraph-style) frames.

This is NOT a model of the optics of a real experiment. Its only purpose is to give the image
pipeline (image_pipeline.py) frames that look like raw high-speed camera frames, so that I can
test it against a known answer (the parcel-based penetration).

Design notes
- Projection weight. Each parcel contributes nParticle * d^2, its geometric cross-section,
  because light extinction by droplets scales with projected area. Alternatives: Mie
  scattering efficiency (matters for droplets near the wavelength), or a ray-traced render.
- Attenuation. Beer-Lambert, I = I0 * exp(-k * A_proj), with k chosen so that the densest region
  (99th percentile of the projected area over all frames) transmits 5 %. This keeps every
  frame in the same, fixed contrast scale, as with a fixed camera exposure. I chose 5 % because
  a back-lit dense spray core is nearly black but not clipped.
- Imperfections. I added an uneven background (vignetting plus a ripple) and 1.5 % Gaussian
  sensor noise, because these are exactly what the flat-field and threshold steps of the
  pipeline have to cope with. Not modelled: beam steering, vapour shadowgraphy, lens blur
  (apart from a small Gaussian splat), motion blur, and a saturating or non-linear sensor.
- Projection plane. The spray axis is horizontal and the injector is at the left edge; the
  view direction is perpendicular to `view_lateral_axis` in params.json. Because the cloud is
  nearly axisymmetric, the choice of that axis changes the image only through statistical noise.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from spray_metrics import unit

FULL_SCALE = 60000.0  # background level in 16-bit counts (about 90 % of full scale, no clipping)


def lateral_axis(direction, view_lateral) -> np.ndarray:
    d = unit(direction)
    v = np.asarray(view_lateral, dtype=float)
    v = v - (v @ d) * d  # force it perpendicular to the spray axis
    if np.linalg.norm(v) < 1e-9:
        raise ValueError("view_lateral_axis is parallel to the spray axis")
    return unit(v)


def projection_maps(frames: list[dict], origin, direction, img_params: dict) -> list[np.ndarray]:
    """Projected droplet cross-section per pixel (rows = lateral, columns = axial)."""
    px = img_params["pixel_size_m"]
    la, lw = img_params["field_of_view_axial_m"], img_params["field_of_view_lateral_m"]
    nx, ny = int(round(la / px)), int(round(lw / px))
    d = unit(direction)
    lat_axis = lateral_axis(direction, img_params["view_lateral_axis"])
    maps = []
    for fr in frames:
        if len(fr["pos"]) == 0:
            maps.append(np.zeros((ny, nx)))
            continue
        rel = fr["pos"] - np.asarray(origin, dtype=float)
        s, lat = rel @ d, rel @ lat_axis
        dia = np.where(np.isnan(fr["d"]), 1.0, fr["d"])
        w = fr["nParticle"] * dia ** 2
        h, _, _ = np.histogram2d(s, lat, bins=[nx, ny], range=[[0, la], [-lw / 2, lw / 2]], weights=w)
        # The small Gaussian splat stands in for the point-spread function of a lens and also
        # fills the gaps between the sparse parcels.
        maps.append(gaussian_filter(h.T, img_params["splat_sigma_px"]))
    return maps


def background(shape: tuple[int, int], rng: np.random.Generator, noise_fraction: float,
               clean: bool = False) -> np.ndarray:
    """Uneven bright background: vignetting, a mild ripple, and optionally sensor noise."""
    ny, nx = shape
    x = np.linspace(0, 1, nx)[None, :]
    y = np.linspace(-1, 1, ny)[:, None]
    field = 1.0 - 0.18 * x - 0.12 * y ** 2 + 0.03 * np.sin(6 * x + 2 * y)
    field = field * FULL_SCALE * 0.9
    if clean:
        return field
    return field * (1 + rng.normal(0, noise_fraction, shape))


def render_sequence(frames: list[dict], origin, direction, img_params: dict
                    ) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (list of uint16 frames, background = mean of 8 spray-free noisy frames)."""
    rng = np.random.default_rng(img_params["seed"])  # fixed seed so the figures are reproducible
    maps = projection_maps(frames, origin, direction, img_params)
    ref = max(np.percentile(m[m > 0], 99) if np.any(m > 0) else 0.0 for m in maps)
    k = -np.log(0.05) / ref if ref > 0 else 0.0
    shape = maps[0].shape
    clean = background(shape, rng, img_params["noise_fraction"], clean=True)
    images = []
    for m in maps:
        trans = np.exp(-k * m)
        noisy = clean * trans * (1 + rng.normal(0, img_params["noise_fraction"], shape))
        images.append(np.clip(noisy, 0, 65535).astype(np.uint16))
    # Averaging 8 empty frames, as is usual in experiments, keeps the background noise low
    # enough that it does not dominate the flat-fielded signal.
    bg_stack = [clean * (1 + rng.normal(0, img_params["noise_fraction"], shape)) for _ in range(8)]
    return images, np.mean(bg_stack, axis=0)


def save_sequence(images: list[np.ndarray], bg: np.ndarray, times: list[float], out_dir: Path) -> None:
    """16-bit PNGs. The file name carries the time stamp read back by image_pipeline.py."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "background.png"), np.clip(bg, 0, 65535).astype(np.uint16))
    for i, (img, t) in enumerate(zip(images, times)):
        cv2.imwrite(str(out_dir / f"frame_{i:03d}_t{t * 1e6:07.1f}us.png"), img)
