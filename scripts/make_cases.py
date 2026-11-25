#!/usr/bin/env python3
"""Create the coarse, medium and fine cases from the aachenBomb tutorial.

I copy the tutorial and only edit the copy, so the original stays untouched. I change the
controlDict settings with foamDictionary. The chamber temperature and the cell counts I
change by editing the text of the files.

Run it with the OpenFOAM environment sourced:
    python3 scripts/make_cases.py [--params params.json] [--tutorial PATH]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(command, folder):
    """Run a command inside a folder and stop the script if it fails."""
    result = subprocess.run(command, cwd=folder, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr or result.stdout
        sys.exit(f"Command failed in {folder}: {' '.join(command)}\n{message}")


def find_tutorial(user_path):
    if user_path:
        tutorial = Path(user_path)
    else:
        foam_tutorials = os.environ.get("FOAM_TUTORIALS")
        if not foam_tutorials:
            sys.exit("FOAM_TUTORIALS is not set. Source your OpenFOAM environment first "
                     "(e.g. `source /usr/lib/openfoam/openfoam2406/etc/bashrc`) or pass --tutorial.")
        tutorial = Path(foam_tutorials) / "lagrangian" / "sprayFoam" / "aachenBomb"
    if not tutorial.is_dir():
        sys.exit(f"Tutorial directory not found: {tutorial}\n"
                 "Use OpenFOAM (ESI, openfoam.com) v2306 or newer, or pass --tutorial.")
    return tutorial


def scale_blockmesh(path, factor):
    """Multiply the number of cells per direction of the first hex block by factor."""
    text = path.read_text()
    # I use a regular expression to find "hex (8 vertices) (nx ny nz)". The part in the first
    # brackets is kept as it is, and the next three groups are nx, ny and nz. A zone name
    # between the two brackets is allowed.
    pattern = r"(hex\s*\([^)]*\)\s*(?:[A-Za-z_]\w*\s*)?)\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)"
    match = re.search(pattern, text)
    if not match:
        sys.exit(f"Could not find a 'hex (...) (nx ny nz)' entry in {path}.\n"
                 "The tutorial mesh definition differs from what this script expects; "
                 "scale the cell counts by hand and write mesh_info.json yourself.")

    # I keep at least 4 cells in each direction
    nx = max(4, round(int(match.group(2)) * factor))
    ny = max(4, round(int(match.group(3)) * factor))
    nz = max(4, round(int(match.group(4)) * factor))

    new_block = f"{match.group(1)}({nx} {ny} {nz})"
    path.write_text(text[:match.start()] + new_block + text[match.end():])
    return nx, ny, nz


def set_control_dict(case, entry, value):
    run_command(["foamDictionary", "-entry", entry, "-set", value, "system/controlDict"], case)


def set_chamber_temperature(case, temperature):
    t_file = case / "0" / "T"
    text = t_file.read_text()
    if "uniform 800" not in text:
        sys.exit("Expected 'uniform 800' in 0/T of the tutorial; "
                 "set the chamber temperature by hand in this case.")
    t_file.write_text(text.replace("uniform 800", f"uniform {temperature:g}"))


def print_injector_lines(cases_dir):
    """Print the injector settings so that I can copy them into params.json."""
    print("\nNext: copy these values from cases/medium/constant/sprayCloudProperties into params.json")
    print("      (nozzle_diameter_m, injection_velocity_m_s, liquid_density_kg_m3, origin_m, direction):\n")
    keywords = ("Umag|outerDiameter|innerDiameter|position|direction|duration|massTotal|"
                "parcelsPerSecond|thetaOuter|thetaInner|SOI|rho0?|liquids|flowRateProfile")
    for file in (cases_dir / "medium" / "constant").glob("*CloudProperties"):
        print(f"--- {file.relative_to(ROOT)}")
        for number, line in enumerate(file.read_text().splitlines(), 1):
            if re.search(keywords, line):
                print(f"{number:5d}: {line.rstrip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=str(ROOT / "params.json"))
    parser.add_argument("--tutorial", default=None)
    args = parser.parse_args()

    params = json.loads(Path(args.params).read_text())
    tutorial = find_tutorial(args.tutorial)
    cases_dir = ROOT / "cases"
    cases_dir.mkdir(exist_ok=True)

    for level, factor in params["mesh_levels"].items():
        case = cases_dir / level
        if case.exists():
            shutil.rmtree(case)
        shutil.copytree(tutorial, case)

        # I delete the tutorial's own run scripts because they would overwrite my settings
        for script in ("Allrun", "Allclean"):
            (case / script).unlink(missing_ok=True)

        # Newer ESI releases ship the initial fields as 0.orig, so I copy them to 0
        if not (case / "0").is_dir() and (case / "0.orig").is_dir():
            shutil.copytree(case / "0.orig", case / "0")

        set_control_dict(case, "endTime", str(params["end_time_s"]))
        set_control_dict(case, "writeControl", "adjustableRunTime")
        set_control_dict(case, "writeInterval", str(params["write_interval_s"]))
        # ASCII output, because my parcel reader only reads text files
        set_control_dict(case, "writeFormat", "ascii")

        if params["mode"] == "cold":
            set_chamber_temperature(case, params["cold_temperature_K"])

        nx, ny, nz = scale_blockmesh(case / "system" / "blockMeshDict", factor)
        n_cells = nx * ny * nz
        info = {"level": level, "scale": factor, "cells_per_direction": [nx, ny, nz], "n_cells": n_cells}
        (case / "mesh_info.json").write_text(json.dumps(info, indent=2))
        print(f"{level:7s}: scale {factor:4.2f} -> {nx} x {ny} x {nz} = {n_cells:,} cells")

    print_injector_lines(cases_dir)


if __name__ == "__main__":
    main()
