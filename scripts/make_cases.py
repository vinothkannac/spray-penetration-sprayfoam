#!/usr/bin/env python3
"""Create the three mesh-level cases from the OpenFOAM sprayFoam aachenBomb tutorial.

The tutorial is copied (never edited in place) and patched with foamDictionary:
end time, write interval, ASCII output, and (mode "cold") the initial chamber temperature.
The number of cells of the first hex block in system/blockMeshDict is scaled per mesh level.

Usage (OpenFOAM environment sourced):
    python3 scripts/make_cases.py [--params params.json] [--tutorial PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> None:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"Command failed in {cwd}: {' '.join(cmd)}\n{res.stderr or res.stdout}")


def find_tutorial(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
    else:
        tut = os.environ.get("FOAM_TUTORIALS")
        if not tut:
            sys.exit("FOAM_TUTORIALS is not set. Source your OpenFOAM environment first "
                     "(e.g. `source /usr/lib/openfoam/openfoam2406/etc/bashrc`) or pass --tutorial.")
        p = Path(tut) / "lagrangian" / "sprayFoam" / "aachenBomb"
    if not p.is_dir():
        sys.exit(f"Tutorial directory not found: {p}\n"
                 "Use OpenFOAM (ESI, openfoam.com) v2306 or newer, or pass --tutorial.")
    return p


HEX_RE = re.compile(r"(hex\s*\([^)]*\)\s*(?:[A-Za-z_]\w*\s*)?)\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)")


def scale_blockmesh(path: Path, factor: float) -> tuple[int, int, int]:
    text = path.read_text()
    m = HEX_RE.search(text)
    if not m:
        sys.exit(f"Could not find a 'hex (...) (nx ny nz)' entry in {path}.\n"
                 "The tutorial mesh definition differs from what this script expects; "
                 "scale the cell counts by hand and write mesh_info.json yourself.")
    nx, ny, nz = (max(4, round(int(g) * factor)) for g in m.groups()[1:])
    new = f"{m.group(1)}({nx} {ny} {nz})"
    path.write_text(text[: m.start()] + new + text[m.end():])
    return nx, ny, nz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(ROOT / "params.json"))
    ap.add_argument("--tutorial", default=None)
    args = ap.parse_args()

    params = json.loads(Path(args.params).read_text())
    tut = find_tutorial(args.tutorial)
    cases_dir = ROOT / "cases"
    cases_dir.mkdir(exist_ok=True)

    for level, factor in params["mesh_levels"].items():
        case = cases_dir / level
        if case.exists():
            shutil.rmtree(case)
        shutil.copytree(tut, case)
        for stale in ("Allrun", "Allclean"):  # tutorial scripts would overwrite our settings
            (case / stale).unlink(missing_ok=True)
        # Newer ESI releases ship the initial fields as 0.orig (the tutorial Allrun restores it).
        if not (case / "0").is_dir() and (case / "0.orig").is_dir():
            shutil.copytree(case / "0.orig", case / "0")

        cd = "system/controlDict"
        run(["foamDictionary", "-entry", "endTime", "-set", str(params["end_time_s"]), cd], case)
        run(["foamDictionary", "-entry", "writeControl", "-set", "adjustableRunTime", cd], case)
        run(["foamDictionary", "-entry", "writeInterval", "-set", str(params["write_interval_s"]), cd], case)
        run(["foamDictionary", "-entry", "writeFormat", "-set", "ascii", cd], case)

        if params["mode"] == "cold":
            t_file = case / "0" / "T"
            text = t_file.read_text()
            if "uniform 800" not in text:
                sys.exit("Expected 'uniform 800' in 0/T of the tutorial; "
                         "set the chamber temperature by hand in this case.")
            t_file.write_text(text.replace("uniform 800", f"uniform {params['cold_temperature_K']:g}"))

        nx, ny, nz = scale_blockmesh(case / "system" / "blockMeshDict", factor)
        info = {"level": level, "scale": factor, "cells_per_direction": [nx, ny, nz], "n_cells": nx * ny * nz}
        (case / "mesh_info.json").write_text(json.dumps(info, indent=2))
        print(f"{level:7s}: scale {factor:4.2f} -> {nx} x {ny} x {nz} = {nx * ny * nz:,} cells")

    print("\nNext: copy these values from cases/medium/constant/sprayCloudProperties into params.json")
    print("      (nozzle_diameter_m, injection_velocity_m_s, liquid_density_kg_m3, origin_m, direction):\n")
    props = list((cases_dir / "medium" / "constant").glob("*CloudProperties"))
    pat = re.compile(r"(Umag|outerDiameter|innerDiameter|position|direction|duration|massTotal|"
                     r"parcelsPerSecond|thetaOuter|thetaInner|SOI|rho0?|liquids|flowRateProfile)")
    for f in props:
        print(f"--- {f.relative_to(ROOT)}")
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if pat.search(line):
                print(f"{i:5d}: {line.rstrip()}")


if __name__ == "__main__":
    main()
