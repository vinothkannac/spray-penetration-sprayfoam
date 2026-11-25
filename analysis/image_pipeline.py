"""Find the penetration length and spray angle in back-lit spray images.

Steps for every frame
  1. flat-field:  signal = 1 - frame / background   (the spray blocks light, so signal > 0)
  2. Gaussian smoothing
  3. threshold = max(n_sigma * noise sigma, min_contrast)
     (optional: --correct-background, see segment)
  4. morphological opening, then keep the region that touches the nozzle edge
  5. penetration = the furthest axial point of that region
  6. spray angle = 2*atan(slope/2) from a straight line fit of the spray width between
     20 % and 80 % of the penetration

It works on any image sequence with the nozzle at the left edge and the spray going to the
right (rotate or flip your images first if needed). To run it on your own frames:

    python analysis/image_pipeline.py --images DIR --background bg.png --pixel-size 1e-4

Why I chose these steps
- Flat-field and not simple subtraction: dividing by the background corrects the uneven light
  as a factor, which fits how light is blocked (Beer-Lambert). With subtraction, the contrast
  would depend on the position in the image.
- Threshold from the noise and not a fixed grey level: I estimate sigma from the median
  absolute deviation (MAD) of the whole frame. The spray covers only a small part of the
  pixels, so it has little effect on the MAD. I use 6 sigma and at least 2 % contrast, so
  that a very clean frame does not get a threshold of almost zero.
- This has a weakness: uneven light that changes after the background was recorded also
  counts as noise and raises the threshold. I added an optional correction that estimates the
  noise from pixel differences and removes a fitted plane. I did not make it the default,
  because with some curved backgrounds it fails badly without any warning.
- Other threshold options are Otsu (it expects two clear grey level groups, which a thin
  spray tip does not give), a fixed fraction of the peak value, or a local threshold.
- I think the threshold is the biggest source of uncertainty. The spray tip is thin, so a
  higher threshold can make the penetration shorter and the angle narrower. This is why the
  README quotes the image angle with a warning.
- Region touching the nozzle: this removes single noise spots and separate droplets. It also
  removes a tip that has broken off. Another option is to keep all regions above a minimum
  size.
- Width and not a cone fit: I fit the width (top edge to bottom edge) between 20 % and 80 %
  of the length, similar to spray_metrics.py, so that I can compare the two angles.
- I use classical methods with NumPy, SciPy and OpenCV, and no machine learning. I can
  explain and check every step. A neural network such as U-Net would only be worth it for
  noisy images with strong vapour or background structures.
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float64)


def flat_field(frame, bg, smooth_px=1.5):
    """Steps 1 and 2. The result is 0 for background and 1 where all light is blocked."""
    safe_bg = np.maximum(bg, 1.0)  # I do this to avoid dividing by zero
    return ndimage.gaussian_filter(1.0 - frame / safe_bg, smooth_px)


def pixel_noise(signal, smooth_px):
    """Noise sigma from the difference between pixels that are a few pixels apart.

    A slow change in the background is almost the same at both pixels, so it mostly cancels.
    """
    # Smoothing makes neighbouring pixels similar, so I compare pixels further apart
    lag = int(np.ceil(4 * smooth_px)) + 1
    horizontal = signal[:, lag:] - signal[:, :-lag]
    vertical = signal[lag:] - signal[:-lag]
    difference = np.concatenate([horizontal.ravel(), vertical.ravel()])
    mad = np.median(np.abs(difference - np.median(difference)))
    # 1.4826 turns the MAD into a standard deviation. I divide by sqrt(2) because the
    # difference of two noisy pixels has more noise than one pixel.
    return float(1.4826 * mad / np.sqrt(2.0))


def fit_background_plane(signal, clip, n_iter=10, step=7):
    """Fit a plane to the background pixels.

    I fit the plane several times. Each time I leave out the pixels that are more than clip
    away from the typical difference, which are mostly the spray. To make it faster, I only
    use every step-th pixel.
    """
    ny, nx = signal.shape
    y, x = np.mgrid[0:ny, 0:nx]
    # plane: value = c0 + c1 * x/nx + c2 * y/ny
    matrix = np.stack([np.ones(signal.size), x.ravel() / nx, y.ravel() / ny], axis=1)[::step]
    values = signal.ravel()[::step]

    coef = np.array([np.median(values), 0.0, 0.0])  # start with a flat plane
    for _ in range(n_iter):
        residual = values - matrix @ coef
        keep = np.abs(residual - np.median(residual)) < clip
        if keep.sum() < 3:  # a plane needs at least 3 points
            break
        coef = np.linalg.lstsq(matrix[keep], values[keep], rcond=None)[0]
    return coef[0] + coef[1] * x / nx + coef[2] * y / ny


def lowest_block_median(residual, block=40):
    """Split the image into square blocks and return the lowest median of all blocks.

    I use this to check the fitted plane. The spray only makes the signal higher, so a clearly
    negative block means that the fitted background is wrong somewhere.
    """
    size = min(block, *residual.shape)
    ny, nx = residual.shape
    medians = []
    for top in range(0, ny - size + 1, size):
        for left in range(0, nx - size + 1, size):
            medians.append(np.median(residual[top:top + size, left:left + size]))
    return float(min(medians))


def keep_nozzle_regions(mask, nozzle_margin_px):
    """Keep only the connected regions that start close to the left (nozzle) edge."""
    labels, n_regions = ndimage.label(mask)
    keep = np.zeros_like(mask)
    for label in range(1, n_regions + 1):
        region = labels == label
        columns = np.where(region.any(axis=0))[0]
        if columns.min() <= nozzle_margin_px:
            keep |= region
    return keep


def segment(frame, bg, smooth_px=1.5, n_sigma=6.0, min_contrast=0.02, nozzle_margin_px=5,
            open_px=3, correct_background=False):
    """Steps 1 to 4. Returns the spray mask (True = spray) and a dictionary with the
    threshold, the noise level and two flags.

    threshold_inflated: the threshold is more than 1.5 times the value that the noise between
    pixels would give. Uneven light can cause this, but a large spray can too. This flag does
    not change the mask.

    correct_background (optional): I take the noise from pixel differences and subtract a
    fitted plane. If one block of the result is lower than -min_contrast / 4, I go back to
    the default method. I chose a quarter because a plane fitted to a bowl-shaped background
    leaves the edges about twice as far above the plane as the centre is below it.
    background_corrected tells which method was used.
    WARNING: this check misses some backgrounds, for example a local dark patch or a bowl with
    a flat centre. The angle can then be wrong by more than 100 degrees. Check the masks.
    """
    signal = flat_field(frame, bg, smooth_px)
    local_sigma = pixel_noise(signal, smooth_px)

    corrected = False
    if correct_background:
        sigma = local_sigma
        clip = max(3.0 * sigma, min_contrast / 2.0)
        base = fit_background_plane(signal, clip)
        corrected = lowest_block_median(signal - base) >= -min_contrast / 4.0

    if not corrected:
        # Default: noise from the MAD of the whole frame
        base = np.median(signal)
        sigma = 1.4826 * np.median(np.abs(signal - base))

    threshold = max(n_sigma * sigma, min_contrast)
    expected_threshold = max(n_sigma * local_sigma, min_contrast)
    inflated = threshold > 1.5 * expected_threshold

    mask = (signal - base) > threshold
    # The opening removes very small spots. With open_px = 1 nothing is removed.
    mask = ndimage.binary_opening(mask, structure=np.ones((open_px, open_px)))
    mask = keep_nozzle_regions(mask, nozzle_margin_px)

    info = {"threshold": float(threshold), "noise_sigma": float(sigma),
            "background_corrected": bool(corrected), "threshold_inflated": bool(inflated)}
    return mask, info


def width_profile(mask, pixel_size, frac=(0.2, 0.8)):
    """Axial positions and spray widths (both in m) between frac[0] and frac[1] of the penetration."""
    spray_columns = np.where(mask.any(axis=0))[0]
    if len(spray_columns) == 0:
        return np.empty(0), np.empty(0)
    length_px = spray_columns.max() + 1
    first = int(frac[0] * length_px)
    last = int(frac[1] * length_px)

    positions = []
    widths = []
    for column in range(first, last):
        rows = np.where(mask[:, column])[0]
        if len(rows):
            positions.append((column + 0.5) * pixel_size)  # centre of the pixel
            widths.append((rows.max() - rows.min() + 1) * pixel_size)
    return np.array(positions), np.array(widths)


def penetration_and_angle(mask, pixel_size, frac=(0.2, 0.8)):
    """Steps 5 and 6. Returns the penetration in m and the full spray angle in degrees."""
    spray_columns = np.where(mask.any(axis=0))[0]
    if len(spray_columns) == 0:
        return float("nan"), float("nan")
    penetration = (spray_columns.max() + 1) * pixel_size

    positions, widths = width_profile(mask, pixel_size, frac)
    if len(positions) < 10:  # too few points for a useful line fit
        return penetration, float("nan")
    slope = np.polyfit(positions, widths, 1)[0]  # the width grows by 2*tan(angle/2) per metre
    angle = np.degrees(2.0 * np.arctan(slope / 2.0))
    return penetration, float(angle)


def analyse_sequence(images, bg, times, pixel_size, correct_background=False):
    """Analyse all frames. Returns a table with the results and the list of masks."""
    rows = []
    masks = []
    for image, t in zip(images, times):
        mask, info = segment(image.astype(np.float64), bg, correct_background=correct_background)
        penetration, angle = penetration_and_angle(mask, pixel_size)
        row = {"t": t, "S_image": penetration, "theta_image_deg": angle}
        row.update(info)
        rows.append(row)
        masks.append(mask)
    return pd.DataFrame(rows), masks


def time_from_name(path):
    """Time in s from a name like frame_012_t0000250.0us.png (the names from render_images.py)."""
    match = re.search(r"_t([0-9.]+)us", path.name)
    if match:
        return float(match.group(1)) * 1e-6
    return float("nan")


def list_frames(folder, background_name):
    """All image files in the folder except the background, sorted by name."""
    image_types = (".png", ".tif", ".tiff", ".jpg")
    files = []
    for path in Path(folder).iterdir():
        if path.suffix.lower() in image_types and path.name != background_name:
            files.append(path)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="directory of frames (png/tif), sorted by name")
    parser.add_argument("--background", required=True, help="background image (mean of spray-free frames)")
    parser.add_argument("--pixel-size", type=float, required=True, help="m per pixel")
    parser.add_argument("--fps", type=float, default=None, help="frame rate; used for time if names have no _t<us>")
    parser.add_argument("--out", default="image_results.csv")
    parser.add_argument("--correct-background", action="store_true",
                        help="experimental: remove a linear illumination gradient; inspect the results")
    args = parser.parse_args()

    files = list_frames(args.images, Path(args.background).name)
    images = [load_gray(file) for file in files]
    times = [time_from_name(file) for file in files]

    # If a file name has no time, I calculate the times from the frame rate
    if np.isnan(times).any():
        if args.fps is None:
            raise SystemExit("File names carry no time stamp; pass --fps.")
        times = [i / args.fps for i in range(len(files))]

    bg = load_gray(Path(args.background))
    results, _ = analyse_sequence(images, bg, times, args.pixel_size, args.correct_background)
    results.to_csv(args.out, index=False)
    print(results.to_string(index=False))

    n_inflated = int(results["threshold_inflated"].sum())
    if n_inflated > 0:
        print(f"WARNING: threshold_inflated in {n_inflated} of {len(results)} frames; "
              "check the background and the results.")


if __name__ == "__main__":
    main()
