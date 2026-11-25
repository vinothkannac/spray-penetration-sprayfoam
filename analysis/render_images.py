"""Turn the parcel data into synthetic back-lit (shadowgraph style) frames.

This is NOT a model of the optics of a real experiment. I only use it to make frames that look
like raw camera frames, so that I can test image_pipeline.py against a known answer (the
penetration from the parcels).

Why I render them this way
- Each parcel counts with nParticle * d^2, its cross-section area, because droplets block
  light roughly in proportion to their area. A more detailed model would use Mie scattering
  or ray tracing.
- I use Beer-Lambert, I = I0 * exp(-k * area). I choose k so that the densest region (99th
  percentile over all frames) lets 5 % of the light through. A dense spray core is then
  nearly black but not clipped. All frames use the same k, like a fixed camera exposure.
- I add an uneven background (darker at the edges, plus a small ripple) and 1.5 % sensor
  noise, because the flat-field and threshold steps have to deal with these. I do not model
  beam steering, vapour, motion blur or a non-linear sensor, and the only lens blur is a
  small Gaussian.
- The spray axis is horizontal and the injector is at the left edge. The spray is almost
  round, so the chosen view direction (view_lateral_axis in params.json) mostly changes only
  the random noise.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from spray_metrics import unit

FULL_SCALE = 60000.0  # I use about 90 % of the 16-bit range for the light level, to avoid clipping


def lateral_axis(direction, view_lateral):
    axis = unit(direction)
    side = np.asarray(view_lateral, dtype=float)
    side = side - (side @ axis) * axis  # I make it exactly perpendicular to the spray axis
    if np.linalg.norm(side) < 1e-9:
        raise ValueError("view_lateral_axis is parallel to the spray axis")
    return unit(side)


def projection_maps(frames, origin, direction, img_params):
    """Droplet area in every pixel, one image per time (rows = lateral, columns = axial)."""
    pixel = img_params["pixel_size_m"]
    length = img_params["field_of_view_axial_m"]
    width = img_params["field_of_view_lateral_m"]
    nx = int(round(length / pixel))
    ny = int(round(width / pixel))
    axis = unit(direction)
    side = lateral_axis(direction, img_params["view_lateral_axis"])

    maps = []
    for frame in frames:
        if len(frame["pos"]) == 0:
            maps.append(np.zeros((ny, nx)))
            continue
        relative = frame["pos"] - np.asarray(origin, dtype=float)
        axial = relative @ axis
        lateral = relative @ side
        # If the diameter is missing, every parcel gets the same size
        diameter = np.where(np.isnan(frame["d"]), 1.0, frame["d"])
        area = frame["nParticle"] * diameter ** 2
        hist, _, _ = np.histogram2d(axial, lateral, bins=[nx, ny],
                                    range=[[0, length], [-width / 2, width / 2]], weights=area)
        # I blur a little. This acts like a lens and fills the gaps between the parcels.
        maps.append(gaussian_filter(hist.T, img_params["splat_sigma_px"]))
    return maps


def background(shape, rng, noise_fraction, clean=False):
    """Bright but uneven background. With clean=True no sensor noise is added."""
    ny, nx = shape
    x = np.linspace(0, 1, nx)[None, :]
    y = np.linspace(-1, 1, ny)[:, None]
    light = 1.0 - 0.18 * x - 0.12 * y ** 2 + 0.03 * np.sin(6 * x + 2 * y)
    light = light * FULL_SCALE * 0.9
    if clean:
        return light
    return light * (1 + rng.normal(0, noise_fraction, shape))


def render_sequence(frames, origin, direction, img_params):
    """Return the list of 16-bit frames and the background image."""
    rng = np.random.default_rng(img_params["seed"])  # fixed seed, so I get the same figures again
    maps = projection_maps(frames, origin, direction, img_params)

    # Reference area: the largest 99th percentile over all frames
    reference = 0.0
    for area_map in maps:
        if np.any(area_map > 0):
            reference = max(reference, np.percentile(area_map[area_map > 0], 99))
    k = -np.log(0.05) / reference if reference > 0 else 0.0

    noise = img_params["noise_fraction"]
    shape = maps[0].shape
    clean = background(shape, rng, noise, clean=True)

    images = []
    for area_map in maps:
        transmission = np.exp(-k * area_map)
        noisy = clean * transmission * (1 + rng.normal(0, noise, shape))
        images.append(np.clip(noisy, 0, 65535).astype(np.uint16))

    # I average 8 empty frames, as is common in experiments, so that the noise of the
    # background stays small
    empty_frames = []
    for _ in range(8):
        empty_frames.append(clean * (1 + rng.normal(0, noise, shape)))
    return images, np.mean(empty_frames, axis=0)


def save_sequence(images, bg, times, out_dir):
    """Save 16-bit PNG files. I put the time in the file name so that image_pipeline.py
    can read it again."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "background.png"), np.clip(bg, 0, 65535).astype(np.uint16))
    for i, (image, t) in enumerate(zip(images, times)):
        cv2.imwrite(str(out_dir / f"frame_{i:03d}_t{t * 1e6:07.1f}us.png"), image)
