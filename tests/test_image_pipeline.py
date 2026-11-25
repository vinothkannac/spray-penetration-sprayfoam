"""Tests of image_pipeline.py on drawn wedges where I know the answer. Run:  python3 -m pytest tests

I only test cases that the pipeline should handle. For the cases where it fails, I measure the
error in analysis/image_study.py and do not test for it here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import image_pipeline as ip  # noqa: E402
import image_study as study  # noqa: E402

PIXEL = study.PIXEL


def make_ramps():
    """Images that rise from 0 to 1 in three directions, used to make uneven light."""
    ny, nx = study.SHAPE
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    return {"far": xx / nx, "near": 1 - xx / nx, "side": yy / ny}


@pytest.mark.parametrize("angle", [12.0, 20.0, 35.0])
def test_measurement_on_exact_mask(angle):
    # I test steps 5 and 6 alone, without noise or segmentation
    mask = study.wedge_opacity(300, angle) > 0.5 * study.CORE_OPACITY
    penetration, measured_angle = ip.penetration_and_angle(mask, PIXEL)
    assert penetration == pytest.approx(300 * PIXEL)
    assert measured_angle == pytest.approx(angle, abs=0.2)


@pytest.mark.parametrize("length", [100, 250, 500])
@pytest.mark.parametrize("angle", [12.0, 20.0, 35.0])
def test_full_pipeline_on_noisy_wedge(length, angle):
    frame, bg = study.make_frame(study.wedge_opacity(length, angle), length, seed=3)
    mask, _ = ip.segment(frame, bg)
    penetration, measured_angle = ip.penetration_and_angle(mask, PIXEL)
    # Smoothing moves a strong edge outwards by a few pixels, so I allow up to 5 px longer
    assert -1 * PIXEL <= penetration - length * PIXEL <= 5 * PIXEL
    assert measured_angle == pytest.approx(angle, abs=1.0)


def test_empty_frame_gives_nan():
    frame, bg = study.make_frame(np.zeros(study.SHAPE), 0, seed=0)
    mask, _ = ip.segment(frame, bg)
    assert not mask.any()
    penetration, angle = ip.penetration_and_angle(mask, PIXEL)
    assert np.isnan(penetration)
    assert np.isnan(angle)


def test_blob_away_from_nozzle_is_rejected():
    opacity = np.zeros(study.SHAPE)
    opacity[190:210, 300:320] = 0.8
    frame, bg = study.make_frame(opacity, 0, seed=0)
    mask, _ = ip.segment(frame, bg)
    assert not mask.any()


def test_well_separated_droplets_do_not_extend_the_tip():
    length = 250
    frame, bg = study.make_frame(study.wedge_opacity(length), length, 1, "droplets", 15)
    errors = study.measure(frame, bg, length)
    assert abs(errors["length_err_mm"]) <= 0.5


def test_uniform_flicker_is_removed_by_median_subtraction():
    length = 250
    opacity = study.wedge_opacity(length)
    frame, bg = study.make_frame(opacity, length, 1)
    normal = study.measure(frame, bg, length)
    frame, bg = study.make_frame(opacity, length, 1, "flicker", 0.10)
    dimmed = study.measure(frame, bg, length)
    assert abs(dimmed["length_err_mm"] - normal["length_err_mm"]) <= 0.2


def test_opening_size_one_keeps_thin_structures():
    opacity = np.zeros(study.SHAPE)
    opacity[199:201, :200] = 0.8  # a line 2 px thick, which the default 3 x 3 opening removes
    frame, bg = study.make_frame(opacity, 200, seed=0)
    default_mask, _ = ip.segment(frame, bg, smooth_px=0.0, min_contrast=0.2)
    no_opening_mask, _ = ip.segment(frame, bg, smooth_px=0.0, min_contrast=0.2, open_px=1)
    assert not default_mask.any()
    assert no_opening_mask.any()


def test_pixel_noise_ignores_a_gradient():
    noise = np.random.default_rng(0).normal(0, 0.01, study.SHAPE)
    ramp = 0.2 * make_ramps()["far"]
    assert ip.pixel_noise(noise + ramp, 0.0) == pytest.approx(0.01, rel=0.05)


@pytest.mark.parametrize("side", ["far", "near", "side"])
@pytest.mark.parametrize("length", [150, 350])
def test_linear_gradient_is_removed(side, length):
    # A better noise estimate alone is not enough: with the "near" and "side" gradients the
    # background would be detected as spray if it were not corrected too
    frame, bg = study.make_frame(study.wedge_opacity(length, tip_opacity=0.05), length, seed=2)
    frame = frame * (1 - 0.1 * make_ramps()[side])
    _, info = ip.segment(frame, bg, correct_background=True)
    errors = study.measure(frame, bg, length, correct_background=True)
    assert info["background_corrected"]
    assert info["threshold"] < 0.03
    assert abs(errors["length_err_pct"]) < 3
    assert abs(errors["angle_err_deg"]) < 2


def test_gradient_without_spray_gives_no_detection():
    frame, bg = study.make_frame(np.zeros(study.SHAPE), 0, seed=0)
    frame = frame * (1 - 0.1 * make_ramps()["near"])
    mask, _ = ip.segment(frame, bg, correct_background=True)
    assert not mask.any()


@pytest.mark.parametrize("amplitude", [0.03, 0.05, 0.1])
def test_parabolic_background_falls_back_to_default(amplitude):
    frame, bg = study.make_frame(study.wedge_opacity(150, tip_opacity=0.05), 150, seed=2)
    frame = frame * (1 - amplitude * (2 * make_ramps()["far"] - 1) ** 2)
    mask, info = ip.segment(frame, bg, correct_background=True)
    default_mask, default_info = ip.segment(frame, bg)
    assert not info["background_corrected"]
    assert np.array_equal(mask, default_mask)
    assert info == default_info


def test_threshold_inflated_flags_a_gradient():
    frame, bg = study.make_frame(study.wedge_opacity(250), 250, seed=2)
    _, info = ip.segment(frame, bg)
    assert not info["threshold_inflated"]

    _, info = ip.segment(frame * (1 - 0.1 * make_ramps()["far"]), bg)
    assert info["threshold_inflated"]
