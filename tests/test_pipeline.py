#!/usr/bin/env python3
"""Tests for the parcel analysis. Run:  python3 tests/make_synthetic_cases.py && python3 -m pytest tests

I test the parts where I know the right answer, either by hand or from a published example.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
import cloudio  # noqa: E402
import correlations as corr  # noqa: E402
import spray_metrics as sm  # noqa: E402


def test_hiroyasu_arai_is_continuous_at_break_up():
    # The two formulas should give almost the same value at the break-up time
    d0, u0, rho_l, rho_a = 1e-4, 400.0, 684.0, 59.5
    t_break = corr.break_up_time(d0, u0, rho_l, rho_a)
    times = np.array([t_break * 0.999, t_break * 1.001])
    before, after = corr.hiroyasu_arai(times, d0, u0, rho_l, rho_a)
    assert abs(after - before) / before < 0.02, (before, after)


def test_gci_reproduces_celik_2008_example():
    # I use the 2-D example of Celik et al., J. Fluids Eng. 130 (2008): p = 1.53, GCI_fine = 2.2 %
    result = corr.gci(6.063, 5.972, 5.863, 18000, 8000, 4500, dim=2)
    assert result["monotone"]
    assert abs(result["order"] - 1.53) < 0.05, result
    assert abs(result["gci_fine_pct"] - 2.2) < 0.2, result


def test_gci_flags_non_monotone():
    result = corr.gci(10.0, 10.5, 10.2, 3, 2, 1)
    assert result["monotone"] is False
    assert result["order"] is None


def test_mass_percentile():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    weight = np.array([1.0, 1.0, 1.0, 1.0])
    assert sm.mass_percentile_axial(s, weight, 50) == 2.0
    assert sm.mass_percentile_axial(s, weight, 100) == 4.0


def test_axial_radial_decomposition():
    # 10 mm along the axis and (3, 4) mm to the side, so r must be 5 mm
    pos = np.array([[0.003, 0.0995 - 0.01, 0.004]])
    s, r = sm.axial_radial(pos, [0, 0.0995, 0], [0, -1, 0])
    assert abs(s[0] - 0.01) < 1e-12
    assert abs(r[0] - 0.005) < 1e-12


def test_reader_and_known_penetration():
    case = HERE / "synthetic" / "fine"
    frames = cloudio.read_cloud(case)
    assert len(frames) == 20
    assert frames[0]["pos"].shape[1] == 3

    params = json.loads((HERE / "synthetic" / "params_test.json").read_text())
    table = sm.parcel_timeseries(frames, params["injector"]["origin_m"], params["injector"]["direction"],
                                 params["metrics"])
    rho_a = corr.gas_density(5e6, 293.0)
    reference = corr.hiroyasu_arai(table["t"].values, 1e-4, 400.0, 684.0, rho_a)
    error = np.abs(table["S99"].values / reference - 1)
    # In the synthetic data the tip follows the correlation, and S99 is about 0.7 % shorter
    assert error.max() < 0.03, error.max()


def test_reader_rejects_barycentric():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "positions_bary"
        path.write_text("FoamFile\n{\n version 2.0;\n}\n1\n(\n(0.25 0.25 0.25 0.25) 3 1 2\n)\n")
        try:
            cloudio.read_positions(path)
        except ValueError:
            return  # this is what should happen
    raise AssertionError("barycentric file should be rejected")


if __name__ == "__main__":
    # So that I can also run the tests without pytest
    failures = 0
    for name, test in list(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            test()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {error!r}")
    sys.exit(1 if failures else 0)
