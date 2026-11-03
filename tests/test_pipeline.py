#!/usr/bin/env python3
"""Unit and integration tests. Run:  python3 tests/make_synthetic_cases.py && python3 tests/test_pipeline.py"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
import cloudio  # noqa: E402
import correlations as corr  # noqa: E402
import spray_metrics as sm  # noqa: E402


def test_hiroyasu_arai_is_continuous_at_break_up():
    d0, u0, rl, ra = 1e-4, 400.0, 684.0, 59.5
    tb = corr.break_up_time(d0, u0, rl, ra)
    lo, hi = corr.hiroyasu_arai(np.array([tb * 0.999, tb * 1.001]), d0, u0, rl, ra)
    assert abs(hi - lo) / lo < 0.02, (lo, hi)


def test_gci_reproduces_celik_2008_example():
    # Celik et al., J. Fluids Eng. 130 (2008), 2-D example: p = 1.53, GCI_fine = 2.2 %
    g = corr.gci(6.063, 5.972, 5.863, 18000, 8000, 4500, dim=2)
    assert g["monotone"] and abs(g["order"] - 1.53) < 0.05 and abs(g["gci_fine_pct"] - 2.2) < 0.2, g


def test_gci_flags_non_monotone():
    g = corr.gci(10.0, 10.5, 10.2, 3, 2, 1)
    assert g["monotone"] is False and g["order"] is None


def test_mass_percentile():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert sm.mass_percentile_axial(s, w, 50) == 2.0
    assert sm.mass_percentile_axial(s, w, 100) == 4.0


def test_axial_radial_decomposition():
    pos = np.array([[0.003, 0.0995 - 0.01, 0.004]])
    s, r = sm.axial_radial(pos, [0, 0.0995, 0], [0, -1, 0])
    assert abs(s[0] - 0.01) < 1e-12 and abs(r[0] - 0.005) < 1e-12


def test_reader_and_known_penetration():
    case = HERE / "synthetic" / "fine"
    frames = cloudio.read_cloud(case)
    assert len(frames) == 20 and frames[0]["pos"].shape[1] == 3
    params = json.loads((HERE / "synthetic" / "params_test.json").read_text())
    ts = sm.parcel_timeseries(frames, params["injector"]["origin_m"], params["injector"]["direction"], params["metrics"])
    rho_a = corr.gas_density(5e6, 293.0)
    ref = corr.hiroyasu_arai(ts["t"].values, 1e-4, 400.0, 684.0, rho_a)
    err = np.abs(ts["S99"].values / ref - 1)
    assert err.max() < 0.03, err.max()   # synthetic data: tip = correlation, S99 ~0.7 % below tip


def test_reader_rejects_barycentric(tmp_path=Path("/tmp")):
    p = tmp_path / "positions_bary"
    p.write_text("FoamFile\n{\n version 2.0;\n}\n1\n(\n(0.25 0.25 0.25 0.25) 3 1 2\n)\n")
    try:
        cloudio.read_positions(p)
    except ValueError:
        return
    raise AssertionError("barycentric file should be rejected")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
