#!/usr/bin/env python3
"""Test how robust image_pipeline.py is, using drawn frames where I know the right answer.

    python3 analysis/image_study.py [--out results/image_study] [--seeds 10]

It does not need OpenFOAM or parcel data. I wanted to answer three questions:
  1. Sensitivity: how much do the length and angle change with the pipeline settings?
  2. Stress: which image defects make the default settings fail?
  3. Lighting: for different kinds of uneven light, how does the optional background
     correction compare with the default method?

How I set up the study
- I draw wedges, so I know the true length and angle exactly. They are NOT spray images, so
  these errors say nothing about the accuracy on camera frames.
- "sharp" has the same opacity up to the tip. "fading" gets weaker along its second half,
  down to TIP_OPACITY at the tip, so the measured length can depend on the threshold.
- I take the end of the drawn wedge as the true length. For a fading tip this is only one
  possible definition, and an experiment has no such exact value.
- I change only one setting or one defect at a time. I did not test combinations.
- Every number is a mean over several noise seeds and two spray lengths.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_pipeline as ip
import render_images as ri

SHAPE = (400, 800)          # (lateral, axial) pixels, the field of view in params.json
PIXEL = 1e-4                # m per pixel
ANGLE_DEG = 20.0            # true full angle
LENGTHS_PX = (150, 350)     # true penetration
NOZZLE_HALF_WIDTH_PX = 3.0  # so that the wedge stays connected to the edge after the opening
CORE_OPACITY = 0.8
TIP_OPACITY = 0.05          # opacity at the tip of the "fading" wedge
NOISE = 0.015               # same sensor noise as in params.json
DEFAULTS = {"min_contrast": 0.02, "n_sigma": 6.0, "smooth_px": 1.5, "open_px": 3}
SCENES = {"sharp": None, "fading": TIP_OPACITY}  # scene name -> opacity at the tip

# Values that I try for each setting
SWEEPS = {
    "min_contrast": [0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12],
    "n_sigma": [3.0, 4.0, 6.0, 8.0, 10.0, 14.0],
    "smooth_px": [0.0, 0.75, 1.5, 2.5, 4.0],
    "open_px": [1, 3, 5, 7],
}
SWEEP_LABELS = {"min_contrast": "contrast floor", "n_sigma": "noise multiplier (n_sigma)",
                "smooth_px": "Gaussian smoothing (px)", "open_px": "opening size (px)"}

# defect name -> (what the level means, levels). The faint tail replaces the tip, so I only
# run it for the sharp wedge.
STRESS = {
    "blur": ("Gaussian sigma (px)", [1.0, 2.0, 4.0]),
    "noise": ("noise fraction", [0.03, 0.06, 0.10]),
    "flicker": ("uniform loss of illumination", [0.01, 0.03, 0.10]),
    "bg_drift": ("illumination loss at far edge", [0.02, 0.05, 0.10]),
    "bad_pixels": ("fraction of stuck pixels", [0.0005, 0.005, 0.02]),
    "droplets": ("gap ahead of the tip (px)", [2, 5, 15]),
    "faint_tail": ("opacity of the last 20 % of the length", [0.04, 0.025, 0.015]),
    "vapour": ("opacity of a shadow beyond the liquid", [0.01, 0.03, 0.06]),
}

# Colour, line style and marker of each scene in the figures
STYLE = {"sharp": ("#0072B2", "-", "o"), "fading": ("#D55E00", "--", "s")}


def wedge_opacity(length_px, angle_deg=ANGLE_DEG, tip_opacity=None, shape=SHAPE):
    """Opacity image of a wedge that starts at the left edge. The side edges are smooth.

    With tip_opacity=None the opacity is the same everywhere. Otherwise it drops linearly from
    the core value at half length to tip_opacity at the tip.
    """
    ny, nx = shape
    x = np.arange(nx) + 0.5
    half_width = NOZZLE_HALF_WIDTH_PX + x * np.tan(np.radians(angle_deg / 2))

    # For every pixel I calculate which fraction of it lies inside the wedge (0 to 1)
    pixel_bottom = np.arange(ny)[:, None] - ny / 2
    coverage = np.clip(np.minimum(pixel_bottom + 1, half_width) - np.maximum(pixel_bottom, -half_width), 0, 1)

    along_axis = np.full(nx, CORE_OPACITY)
    if tip_opacity is not None:
        along_axis = np.interp(x, [0.5 * length_px, length_px], [CORE_OPACITY, tip_opacity])
    along_axis[length_px:] = 0.0  # nothing after the tip
    return coverage * along_axis


def make_frame(opacity, length_px, seed, defect=None, level=0.0):
    """Return (frame, background) in camera counts.

    The background never contains the defect, only sensor noise, because in an experiment the
    background is recorded before the injection.
    """
    rng = np.random.default_rng(seed)
    ny, nx = opacity.shape
    noise = level if defect == "noise" else NOISE
    clean = ri.background(opacity.shape, rng, noise, clean=True)
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    alpha = opacity.copy()

    # Defects that change the spray itself
    if defect == "droplets":
        # three separate droplets with a radius of 3 px, the first one level px after the tip
        for k, dy in enumerate((0, 12, -12)):
            centre_x = length_px + level + 3 + 10 * k
            inside = (xx - centre_x) ** 2 + (yy - ny / 2 - dy) ** 2 <= 9
            alpha[inside] = 0.5
    if defect == "vapour":
        # a wider wedge that is 30 % longer than the liquid
        shadow = wedge_opacity(int(1.3 * length_px), 1.5 * ANGLE_DEG, shape=opacity.shape)
        alpha = np.maximum(alpha, gaussian_filter(shadow / CORE_OPACITY * level, 5.0))

    light = clean * (1 - alpha)

    # Defects that change the light
    if defect == "blur":
        light = clean * (1 - gaussian_filter(alpha, level))
    if defect == "flicker":
        light = light * (1 - level)
    if defect == "bg_drift":
        light = light * (1 - level * xx / nx)

    frame = light * (1 + rng.normal(0, noise, opacity.shape))

    if defect == "bad_pixels":  # half of them stay dark and half stay bright
        random_values = rng.random(opacity.shape)
        frame[random_values < level / 2] = 0
        frame[random_values > 1 - level / 2] = 65535

    empty_frames = []
    for _ in range(8):
        empty_frames.append(clean * (1 + rng.normal(0, noise, opacity.shape)))
    bg = np.mean(empty_frames, axis=0)
    return np.clip(frame, 0, 65535), bg


def measure(frame, bg, length_px, **settings):
    """Run the pipeline and return the errors (with sign). NaN means nothing was detected."""
    mask, info = ip.segment(frame, bg, **settings)
    penetration, angle = ip.penetration_and_angle(mask, PIXEL)
    true_length = length_px * PIXEL
    return {"length_err_mm": 1e3 * (penetration - true_length),
            "length_err_pct": 100 * (penetration / true_length - 1),
            "angle_err_deg": angle - ANGLE_DEG,
            "background_corrected": info["background_corrected"]}


def summarise(rows, keys):
    """Mean errors for every group. detected_pct is the share of frames with a detection."""
    groups = pd.DataFrame(rows).groupby(keys, sort=False)
    result = groups[["length_err_mm", "length_err_pct", "angle_err_deg"]].mean()
    result["length_err_pct_worst_abs"] = groups["length_err_pct"].agg(lambda errors: errors.abs().max())
    # count() skips NaN values, size() counts all frames
    result["detected_pct"] = 100 * groups["length_err_pct"].count() / groups.size()
    result["n_frames"] = groups.size()
    return result.reset_index()


def sensitivity(seeds):
    rows = []
    for scene, tip in SCENES.items():
        for length in LENGTHS_PX:
            opacity = wedge_opacity(length, tip_opacity=tip)
            for seed in range(seeds):
                frame, bg = make_frame(opacity, length, seed)
                for name, values in SWEEPS.items():
                    for value in values:
                        settings = DEFAULTS.copy()
                        settings[name] = value
                        row = {"scene": scene, "setting": name, "value": value}
                        row.update(measure(frame, bg, length, **settings))
                        rows.append(row)
    return summarise(rows, ["scene", "setting", "value"])


def stress_cases():
    """List of (defect, level) pairs, starting with the case without a defect."""
    cases = [("none", 0.0)]
    for defect, (_, levels) in STRESS.items():
        for level in levels:
            cases.append((defect, level))
    return cases


def stress(seeds):
    rows = []
    for scene, tip in SCENES.items():
        for defect, level in stress_cases():
            if defect == "faint_tail" and scene == "fading":
                continue
            for length in LENGTHS_PX:
                opacity = wedge_opacity(length, tip_opacity=tip)
                if defect == "faint_tail":
                    start = int(0.8 * length)
                    opacity[:, start:length] *= level / CORE_OPACITY
                for seed in range(seeds):
                    frame, bg = make_frame(opacity, length, seed, defect, level)
                    row = {"scene": scene, "defect": defect, "level": level}
                    row.update(measure(frame, bg, length, **DEFAULTS))
                    rows.append(row)
    return summarise(rows, ["scene", "defect", "level"])


def lighting_shapes(shape=SHAPE):
    """Patterns of lost light (0 to 1) that are not in the background."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    u = xx / nx  # 0 at the left edge, 1 at the right edge
    v = yy / ny
    return {"dark_far_end": u,
            "dark_nozzle_end": 1 - u,
            "dark_one_side": v,
            "bowl": (2 * u - 1) ** 2,
            "flat_centred_bowl": (2 * u - 1) ** 4,
            "local_patch": np.exp(-((u - 0.6) ** 2 + (v - 0.3) ** 2) / 0.04)}


def lighting(seeds, amplitudes=(0.03, 0.05, 0.1)):
    """Compare the default method with the background correction on the fading wedge."""
    methods = {"whole_frame": False, "corrected": True}  # name -> value of correct_background
    rows = []
    for shape_name, loss in lighting_shapes().items():
        for amplitude in amplitudes:
            for length in LENGTHS_PX:
                opacity = wedge_opacity(length, tip_opacity=TIP_OPACITY)
                for seed in range(seeds):
                    frame, bg = make_frame(opacity, length, seed)
                    frame = frame * (1 - amplitude * loss)
                    for method, use_correction in methods.items():
                        row = {"shape": shape_name, "amplitude": amplitude, "method": method}
                        row.update(measure(frame, bg, length, correct_background=use_correction))
                        row["angle_err_abs_deg"] = abs(row["angle_err_deg"])
                        row["length_err_abs_pct"] = abs(row["length_err_pct"])
                        rows.append(row)

    groups = pd.DataFrame(rows).groupby(["shape", "amplitude", "method"], sort=False)
    result = groups[["length_err_abs_pct", "angle_err_abs_deg"]].mean()
    result["fallback_pct"] = 100 * (1 - groups["background_corrected"].mean())
    # A fallback only makes sense for the corrected method
    is_default = result.index.get_level_values("method") == "whole_frame"
    result.loc[is_default, "fallback_pct"] = np.nan
    result["n_frames"] = groups.size()
    return result.reset_index()


# --------------------------------------------------------------------------------------- figures
def add_legend_and_title(fig, title):
    from matplotlib.lines import Line2D
    handles = []
    for scene, (colour, line_style, marker) in STYLE.items():
        handles.append(Line2D([0], [0], color=colour, ls=line_style, marker=marker, label=f"{scene} tip"))
    fig.legend(handles=handles, loc="upper right", ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")


def plot_sensitivity(table, out_file):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(SWEEPS), figsize=(13, 6), sharey="row")
    # (column in the table, axis label) for the two rows of plots
    errors = [("length_err_pct", "length error (%)"), ("angle_err_deg", "angle error (°)")]
    for j, name in enumerate(SWEEPS):
        for i, (column, label) in enumerate(errors):
            ax = axes[i, j]
            ax.axhline(0, color="#333333", lw=0.8)
            ax.axvline(DEFAULTS[name], color="#7F7F7F", ls=":", lw=1.2)
            for scene, (colour, line_style, marker) in STYLE.items():
                data = table[(table["setting"] == name) & (table["scene"] == scene)]
                ax.plot(data["value"], data[column], color=colour, ls=line_style, marker=marker, ms=4)
            if j == 0:
                ax.set_ylabel(label)
            if i == 1:
                ax.set_xlabel(SWEEP_LABELS[name])
    add_legend_and_title(fig, "Sensitivity to pipeline settings (drawn wedges; dotted line = default)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_file, dpi=200, facecolor="white")
    plt.close(fig)


def plot_stress(table, out_file):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), sharey=True)
    for ax, (defect, (meaning, levels)) in zip(axes.flat, STRESS.items()):
        ax.axhline(0, color="#333333", lw=0.8)
        for scene, (colour, line_style, marker) in STYLE.items():
            data = table[(table["scene"] == scene) & (table["defect"] == defect)]
            data = data.set_index("level").reindex(levels)
            if data["length_err_pct"].notna().any():
                ax.plot(range(len(levels)), data["length_err_pct"], color=colour, ls=line_style,
                        marker=marker, ms=5)
        ax.set_xticks(range(len(levels)), [f"{level:g}" for level in levels])
        ax.set_title(defect.replace("_", " "), loc="left")
        ax.set_xlabel(meaning, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("length error (%)")
    add_legend_and_title(fig, "Length error with image defects, default settings (drawn wedges)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_file, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/image_study")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import plotting as pl
    pl.apply_style()  # same figure style as the rest of the study

    sensitivity_table = sensitivity(args.seeds)
    stress_table = stress(args.seeds)
    lighting_table = lighting(args.seeds)

    lighting_table.to_csv(out_dir / "lighting.csv", index=False, float_format="%.4g")
    sensitivity_table.to_csv(out_dir / "sensitivity.csv", index=False, float_format="%.4g")
    stress_table.to_csv(out_dir / "stress.csv", index=False, float_format="%.4g")
    plot_sensitivity(sensitivity_table, out_dir / "sensitivity.png")
    plot_stress(stress_table, out_dir / "stress.png")

    pd.set_option("display.width", 200)
    print(sensitivity_table.round(2).to_string(index=False))
    print(stress_table.round(3).to_string(index=False))
    print(lighting_table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
