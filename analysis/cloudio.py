"""Read the OpenFOAM files that I need: the spray parcels and the injector table.

Why I wrote my own reader
- The files I need (positions, d, nParticle) are simple lists, so a few regular expressions
  are enough. I did not want to add a package such as fluidfoam or PyFoam, and a small reader
  is easier for me to check. It stops with an error when something is not as expected.
- I could also let OpenFOAM do this, with a function object or foamToVTK. I did not, because
  I want all the measurement definitions in Python (spray_metrics.py), so that I can change
  them without running the simulation again.
- I only read ASCII files. make_cases.py sets writeFormat to ascii. Binary files are smaller
  and faster, but the parcel files here are small, so I accepted that.
- Foundation OpenFOAM 7 and newer write barycentric positions. I raise an error for them and
  do not try to convert them, because a wrong conversion would silently spoil every result.
"""
import re
from pathlib import Path

import numpy as np

# One number, for example 12, -0.5 or 1.2e-05
NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def remove_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # /* block comments */
    return re.sub(r"//[^\n]*", "", text)               # // line comments


def read_body(path):
    """File text without the comments and without the FoamFile header."""
    text = remove_comments(Path(path).read_text())
    return re.sub(r"FoamFile\s*\{.*?\}", "", text, flags=re.S)


def read_positions(path):
    """Parcel positions in m as an array with one row (x, y, z) per parcel."""
    body = read_body(path)
    is_empty_list = re.match(r"^\s*0\s*\(\s*\)", body)
    if not body.strip() or is_empty_list:
        return np.empty((0, 3))

    # Each position is written as (x y z)
    triple = rf"\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)"
    values = re.findall(triple, body)
    if not values:
        raise ValueError(f"No (x y z) entries found in {path}. The file may hold barycentric "
                         "coordinates (Foundation OpenFOAM >= 7); use ESI OpenFOAM or a version "
                         "that writes 'positions'.")
    return np.array(values, dtype=float)


def read_scalars(path):
    """Read an OpenFOAM list of numbers, written as 'N ( v1 v2 ... )'."""
    body = read_body(path)
    start = body.find("(")
    end = body.rfind(")")
    if start < 0 or end < start:
        return np.empty(0)
    return np.array(body[start + 1:end].split(), dtype=float)


def list_times(case):
    """List of (time, folder) for all time folders of a case, sorted by time."""
    times = []
    for folder in case.iterdir():
        if not folder.is_dir():
            continue
        try:
            times.append((float(folder.name), folder))
        except ValueError:
            pass  # the folder name is not a number, e.g. "constant"
    return sorted(times)


def find_cloud(case):
    """Name of the parcel cloud (the folder inside <time>/lagrangian)."""
    names = []
    for _, time_dir in list_times(case):
        lagrangian = time_dir / "lagrangian"
        if lagrangian.is_dir():
            for folder in lagrangian.iterdir():
                if folder.is_dir():
                    names.append(folder.name)
    if not names:
        raise FileNotFoundError(f"No lagrangian/<cloud> directories under {case}. "
                                "Did sprayFoam run and write parcels?")
    return sorted(names)[0]


def read_cloud(case, cloud=None):
    """Read all parcel data. Returns one dictionary for every written time with the time t,
    the positions pos (m), the diameters d (m) and nParticle."""
    if cloud is None:
        cloud = find_cloud(case)

    frames = []
    for t, time_dir in list_times(case):
        cloud_dir = time_dir / "lagrangian" / cloud
        if not (cloud_dir / "positions").is_file():
            continue
        pos = read_positions(cloud_dir / "positions")
        n = len(pos)

        # If a file was not written, I use NaN for the diameter and 1 for nParticle
        if (cloud_dir / "d").is_file():
            d = read_scalars(cloud_dir / "d")
        else:
            d = np.full(n, np.nan)
        if (cloud_dir / "nParticle").is_file():
            n_particle = read_scalars(cloud_dir / "nParticle")
        else:
            n_particle = np.ones(n)

        if len(d) != n or len(n_particle) != n:
            raise ValueError(f"Length mismatch at t={t}: positions {n}, d {len(d)}, nParticle {len(n_particle)}")
        frames.append({"t": t, "pos": pos, "d": d, "nParticle": n_particle})

    if not frames:
        raise FileNotFoundError(f"No parcel data found for cloud '{cloud}' in {case}")
    return frames


def read_injection(case):
    """Injector settings and the flow rate table from constant/sprayCloudProperties.

    I read these from the file and do not type them in, so that the injection velocity figure
    always uses the same values as the solver. Returns the table ('t', 'profile') and
    massTotal, duration, outerDiameter, Cd and thetaOuter.
    """
    file = next((case / "constant").glob("*CloudProperties"))
    text = re.sub(r"//[^\n]*", "", file.read_text())

    def read_value(name, prefix=""):
        match = re.search(rf"\b{name}\s+{prefix}({NUMBER})\s*;", text)
        if not match:
            raise ValueError(f"'{name}' not found in {file}")
        return float(match.group(1))

    table = re.search(r"flowRateProfile\s+table\s*\((.*?)\)\s*;", text, flags=re.S)
    if not table:
        raise ValueError(f"flowRateProfile table not found in {file}")
    # The table is a list of (time value) pairs
    pairs = re.findall(rf"\(\s*({NUMBER})\s+({NUMBER})\s*\)", table.group(1))
    pairs = np.array(pairs, dtype=float)

    return {
        "t": pairs[:, 0],
        "profile": pairs[:, 1],
        "massTotal": read_value("massTotal"),
        "duration": read_value("duration"),
        "outerDiameter": read_value("outerDiameter"),
        "Cd": read_value("Cd", r"constant\s+"),
        "thetaOuter": read_value("thetaOuter", r"constant\s+"),
    }
