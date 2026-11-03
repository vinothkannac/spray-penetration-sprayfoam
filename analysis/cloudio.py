"""Readers for the OpenFOAM files this study needs: Lagrangian parcels and the injector table.

Design notes (why I wrote my own reader)
- I read ASCII output with a few regular expressions instead of using a package such as
  fluidfoam or PyFoam. The files I need (positions, d, nParticle) are simple lists, and a
  dependency-free reader is easy to audit and fails loudly on anything unexpected.
- The alternative is to let OpenFOAM do the work: `postProcess -func` or a `cloudInfo`
  function object can write the same statistics during the run, and VTK export
  (`foamToVTK -lagrangianFields`) can be read with pyvista. I did not do this because I want
  the metric definitions (see spray_metrics.py) to live in one Python place and to be
  re-computable without re-running the CFD.
- Binary output is not supported. make_cases.py sets writeFormat ascii. The alternative
  (binary) is faster and smaller; I accepted the cost because the parcel files are small.
- I reject barycentric positions (Foundation OpenFOAM >= 7) with a clear error instead of
  guessing the conversion, because a silently wrong position would corrupt every result.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_TRIPLE = re.compile(rf"\(\s*({_NUM})\s+({_NUM})\s+({_NUM})\s*\)")
_PAIR = re.compile(rf"\(\s*({_NUM})\s+({_NUM})\s*\)")
_HEADER = re.compile(r"FoamFile\s*\{.*?\}", re.S)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")


def _read_body(path: Path) -> str:
    """File text without the FoamFile header and without comments."""
    text = path.read_text()
    text = _BLOCK_COMMENT.sub("", text)
    text = _LINE_COMMENT.sub("", text)
    return _HEADER.sub("", text)


def read_positions(path: Path) -> np.ndarray:
    """(N, 3) parcel positions in m. Raises if only barycentric coordinates are found."""
    body = _read_body(path)
    if not body.strip() or re.match(r"^\s*0\s*\(\s*\)", body):
        return np.empty((0, 3))
    vals = _TRIPLE.findall(body)
    if not vals:
        raise ValueError(f"No (x y z) entries found in {path}. The file may hold barycentric "
                         "coordinates (Foundation OpenFOAM >= 7); use ESI OpenFOAM or a version "
                         "that writes 'positions'.")
    return np.array(vals, dtype=float)


def read_scalars(path: Path) -> np.ndarray:
    """1-D array from an OpenFOAM scalar list ('N ( v1 v2 ... )')."""
    body = _read_body(path)
    i, j = body.find("("), body.rfind(")")
    if i < 0 or j < i:
        return np.empty(0)
    return np.array(body[i + 1: j].split(), dtype=float)


def list_times(case: Path) -> list[tuple[float, Path]]:
    """Numeric time directories of a case, sorted."""
    out = []
    for p in case.iterdir():
        if not p.is_dir():
            continue
        try:
            out.append((float(p.name), p))
        except ValueError:
            continue
    return sorted(out)


def find_cloud(case: Path) -> str:
    names = []
    for _, tdir in list_times(case):
        lag = tdir / "lagrangian"
        if lag.is_dir():
            names += [d.name for d in lag.iterdir() if d.is_dir()]
    if not names:
        raise FileNotFoundError(f"No lagrangian/<cloud> directories under {case}. "
                                "Did sprayFoam run and write parcels?")
    return sorted(set(names))[0]


def read_cloud(case: Path, cloud: str | None = None) -> list[dict]:
    """One dict per written time: positions (m), d (m) and nParticle (ones if not written)."""
    cloud = cloud or find_cloud(case)
    frames = []
    for t, tdir in list_times(case):
        cdir = tdir / "lagrangian" / cloud
        pfile = cdir / "positions"
        if not pfile.is_file():
            continue
        pos = read_positions(pfile)
        n = len(pos)
        d = read_scalars(cdir / "d") if (cdir / "d").is_file() else np.full(n, np.nan)
        npart = read_scalars(cdir / "nParticle") if (cdir / "nParticle").is_file() else np.ones(n)
        if len(d) != n or len(npart) != n:
            raise ValueError(f"Length mismatch at t={t}: positions {n}, d {len(d)}, nParticle {len(npart)}")
        frames.append({"t": t, "pos": pos, "d": d, "nParticle": npart})
    if not frames:
        raise FileNotFoundError(f"No parcel data found for cloud '{cloud}' in {case}")
    return frames


def read_injection(case: Path) -> dict:
    """Injector settings and the flow-rate table from constant/sprayCloudProperties.

    I parse these instead of hard-coding them so that the injection-velocity figure in the README
    is always drawn from the file the solver actually used. Returns a dict with the table
    ('t', 'profile') and the scalars massTotal, duration, outerDiameter, Cd, thetaOuter.
    """
    f = next((case / "constant").glob("*CloudProperties"))
    text = _LINE_COMMENT.sub("", f.read_text())

    def scalar(name: str, prefix: str = "") -> float:
        m = re.search(rf"\b{name}\s+{prefix}({_NUM})\s*;", text)
        if not m:
            raise ValueError(f"'{name}' not found in {f}")
        return float(m.group(1))

    table = re.search(r"flowRateProfile\s+table\s*\((.*?)\)\s*;", text, re.S)
    if not table:
        raise ValueError(f"flowRateProfile table not found in {f}")
    pairs = np.array(_PAIR.findall(table.group(1)), dtype=float)
    return {
        "t": pairs[:, 0], "profile": pairs[:, 1],
        "massTotal": scalar("massTotal"), "duration": scalar("duration"),
        "outerDiameter": scalar("outerDiameter"), "Cd": scalar("Cd", "constant\\s+"),
        "thetaOuter": scalar("thetaOuter", "constant\\s+"),
    }
