#!/usr/bin/env python3
"""Make SYNTHETIC parcel data in the OpenFOAM file layout. It is only for testing.

I let the penetration follow the Hiroyasu-Arai correlation, with a small error that depends
on the mesh. With this I can test the whole analysis (reader, measurements, rendered images,
image pipeline, correlation error and GCI) without running the simulation. These results say
NOTHING about sprayFoam and must never be quoted. I write them to tests/synthetic/, so they
cannot get mixed up with cases/ or results/.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
import correlations as corr  # noqa: E402

# Header of an OpenFOAM file. The names in {} are filled in by write_list.
HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{location}";
    object      {name};
}}
"""

BIAS = {"coarse": 1.06, "medium": 1.02, "fine": 1.00}  # coarser meshes get a slightly longer spray
CELLS_PER_DIRECTION = {"coarse": [32, 32, 77], "medium": [41, 41, 100], "fine": [53, 53, 130]}

# A short injector file with only the values that cloudio.read_injection needs for the figures
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


def write_list(path, name, rows, cls, time_name):
    """Write one OpenFOAM list file: header, number of entries, then the entries."""
    text = HEADER.format(cls=cls, location=f"{time_name}/lagrangian/sprayCloud", name=name)
    text += f"\n\n{len(rows)}\n(\n" + "\n".join(rows) + "\n)\n"
    path.write_text(text)


def make_parcels(rng, tip, n, origin, axis, half_angle):
    """Random parcel positions and diameters inside a cone with length tip."""
    s = tip * rng.random(n) ** 0.7                       # distance along the axis
    r = s * np.tan(half_angle) * np.sqrt(rng.random(n))  # distance from the axis
    phi = rng.uniform(0, 2 * np.pi, n)                   # angle around the axis
    pos = (origin + np.outer(s, axis)
           + np.outer(r * np.cos(phi), [1, 0, 0])
           + np.outer(r * np.sin(phi), [0, 0, 1]))
    diameter = 1.0e-5 * np.exp(rng.normal(0, 0.4, n))
    return pos, diameter


def write_time_folder(case, time_name, pos, diameter):
    cloud_dir = case / time_name / "lagrangian" / "sprayCloud"
    cloud_dir.mkdir(parents=True, exist_ok=True)

    position_rows = []
    for i, (x, y, z) in enumerate(pos):
        # the last number is the cell index in the real files; here it is just a placeholder
        position_rows.append(f"({x:.8g} {y:.8g} {z:.8g}) {i % 5000}")
    diameter_rows = [f"{value:.8g}" for value in diameter]
    count_rows = ["100000"] * len(pos)

    write_list(cloud_dir / "positions", "positions", position_rows, "Cloud<passiveParticle>", time_name)
    write_list(cloud_dir / "d", "d", diameter_rows, "scalarField", time_name)
    write_list(cloud_dir / "nParticle", "nParticle", count_rows, "scalarField", time_name)


def main():
    rng = np.random.default_rng(42)  # fixed seed, so the test data is always the same
    root = HERE / "synthetic"
    injector = TEST_PARAMS["injector"]
    ambient = TEST_PARAMS["ambient"]
    rho_a = corr.gas_density(ambient["pressure_Pa"], ambient["temperature_K"])
    origin = np.array(injector["origin_m"])
    axis = np.array(injector["direction"])
    half_angle = np.radians(6.0)

    for level, bias in BIAS.items():
        case = root / level
        for t in np.arange(1, 21) * 5e-5:  # 20 times, every 50 microseconds
            tip = bias * float(corr.hiroyasu_arai(t, injector["nozzle_diameter_m"],
                                                  injector["injection_velocity_m_s"],
                                                  injector["liquid_density_kg_m3"], rho_a))
            n = int(2500 + 4.0e6 * t)  # more parcels at later times
            pos, diameter = make_parcels(rng, tip, n, origin, axis, half_angle)
            write_time_folder(case, f"{t:g}", pos, diameter)

        (case / "constant").mkdir(exist_ok=True)
        (case / "constant" / "sprayCloudProperties").write_text(CLOUD_PROPERTIES)
        cells = CELLS_PER_DIRECTION[level]
        mesh_info = {"level": level, "n_cells": int(np.prod(cells)), "cells_per_direction": cells}
        (case / "mesh_info.json").write_text(json.dumps(mesh_info))

    # I start from the real params.json and only replace the injector and ambient values
    params = json.loads((HERE.parent / "params.json").read_text())
    params["injector"].update(injector)
    params["ambient"].update(ambient)
    (root / "params_test.json").write_text(json.dumps(params, indent=2))
    print(f"synthetic cases written to {root}")


if __name__ == "__main__":
    main()
