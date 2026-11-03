#!/usr/bin/env python3
"""Generate SYNTHETIC parcel data in the OpenFOAM lagrangian list layout, for testing only.

Penetration follows the Hiroyasu-Arai correlation with a small mesh-dependent bias, so the
analysis chain (reader -> metrics -> rendered images -> image pipeline -> correlation error ->
GCI) can be tested without running the CFD. Results from this data say NOTHING about sprayFoam
and must never be quoted. Output goes to tests/synthetic/ (not to cases/ or results/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
import correlations as corr  # noqa: E402

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
"""

BIAS = {"coarse": 1.06, "medium": 1.02, "fine": 1.00}   # tip over-prediction on coarse meshes
PER_DIR = {"coarse": [32, 32, 77], "medium": [41, 41, 100], "fine": [53, 53, 130]}
CELLS = {k: int(np.prod(v)) for k, v in PER_DIR.items()}

# Minimal injector dictionary, only what analysis/cloudio.py:read_injection needs for the figures.
CLOUD_PROPERTIES = """massTotal 6.0e-6;
duration 1.25e-3;
outerDiameter 1.0e-4;
Cd constant 0.9;
thetaOuter constant 6.0;
flowRateProfile table ((0 0.1) (1e-4 9) (1.25e-3 4) (1.251e-3 0));
"""

TEST_PARAMS = {
    "injector": {"origin_m": [0.0, 0.0995, 0.0], "direction": [0.0, -1.0, 0.0],
                 "nozzle_diameter_m": 1.0e-4, "injection_velocity_m_s": 400.0, "liquid_density_kg_m3": 684.0},
    "ambient": {"pressure_Pa": 5.0e6, "temperature_K": 293.0, "gas_constant_J_per_kgK": 287.0},
}


def write_list(path: Path, name: str, rows: list[str], cls: str, tdir: str) -> None:
    body = HEADER.format(cls=cls, loc=f"{tdir}/lagrangian/sprayCloud", obj=name)
    body += f"\n\n{len(rows)}\n(\n" + "\n".join(rows) + "\n)\n"
    path.write_text(body)


def main() -> None:
    rng = np.random.default_rng(42)
    root = HERE / "synthetic"
    p = TEST_PARAMS
    rho_a = corr.gas_density(p["ambient"]["pressure_Pa"], p["ambient"]["temperature_K"])
    origin = np.array(p["injector"]["origin_m"])
    axis = np.array(p["injector"]["direction"])
    half = np.radians(6.0)
    for level, bias in BIAS.items():
        case = root / level
        times = np.arange(1, 21) * 5e-5
        for t in times:
            tip = float(corr.hiroyasu_arai(t, p["injector"]["nozzle_diameter_m"],
                                           p["injector"]["injection_velocity_m_s"],
                                           p["injector"]["liquid_density_kg_m3"], rho_a)) * bias
            n = int(2500 + 4.0e6 * t)
            s = tip * rng.random(n) ** 0.7
            r = s * np.tan(half) * np.sqrt(rng.random(n))
            phi = rng.uniform(0, 2 * np.pi, n)
            pos = origin + np.outer(s, axis) + np.outer(r * np.cos(phi), [1, 0, 0]) + np.outer(r * np.sin(phi), [0, 0, 1])
            d = 1.0e-5 * np.exp(rng.normal(0, 0.4, n))
            tdir = f"{t:g}"
            cdir = case / tdir / "lagrangian" / "sprayCloud"
            cdir.mkdir(parents=True, exist_ok=True)
            write_list(cdir / "positions", "positions",
                       [f"({x:.8g} {y:.8g} {z:.8g}) {i % 5000}" for i, (x, y, z) in enumerate(pos)],
                       "Cloud<passiveParticle>", tdir)
            write_list(cdir / "d", "d", [f"{v:.8g}" for v in d], "scalarField", tdir)
            write_list(cdir / "nParticle", "nParticle", ["100000"] * n, "scalarField", tdir)
        (case / "constant").mkdir(exist_ok=True)
        (case / "constant" / "sprayCloudProperties").write_text(CLOUD_PROPERTIES)
        (case / "mesh_info.json").write_text(json.dumps(
            {"level": level, "n_cells": CELLS[level], "cells_per_direction": PER_DIR[level]}))
    params = json.loads((HERE.parent / "params.json").read_text())
    params["injector"].update(p["injector"])
    params["ambient"].update(p["ambient"])
    (root / "params_test.json").write_text(json.dumps(params, indent=2))
    print(f"synthetic cases written to {root}")


if __name__ == "__main__":
    main()
