"""Penetration length and spray angle computed directly from Lagrangian parcel data.

Design notes
- Penetration metric. I define the penetration as S99: the axial distance that contains 99 % of
  the liquid mass. Parcel mass is proportional to nParticle * d^3. I chose a mass percentile
  because the absolute furthest parcel (S_max) is set by a single fast droplet and is noisy
  and mesh dependent, and a percentile is also close to how experiments define the liquid
  boundary (a threshold on the integrated extinction signal). S95 and S_max are written as well
  so that you can see how much the choice matters.
  Alternatives: a threshold on the projected mass per cell (closer to a real optical
  criterion), the Siebers 95 %-of-liquid-mass criterion (use percentile 95 in params.json), or
  the position of a liquid volume fraction iso-surface of 0.001 % on the Eulerian grid.
- Spray angle. I fit a straight line to the 95th percentile of the radial parcel position as a
  function of axial distance between 20 % and 80 % of the penetration. This avoids the
  nozzle region (the injector cone, not the spray, dominates there) and the tip (the head
  vortex widens it). The full angle is 2*atan(slope).
  Alternatives: the Naber-Siebers definition (angle of the cone that encloses the spray over
  the first 60 % of the penetration), or a mass-weighted standard deviation of r(s).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def axial_radial(pos: np.ndarray, origin, direction) -> tuple[np.ndarray, np.ndarray]:
    """Axial distance s along the spray axis and radial distance r from it, in m.

    I use the injector axis from params.json (not the global y axis) so that the same code works
    for a tilted injector.
    """
    d = unit(direction)
    rel = pos - np.asarray(origin, dtype=float)
    s = rel @ d
    perp = rel - np.outer(s, d)
    return s, np.linalg.norm(perp, axis=1)


def liquid_mass_weight(d: np.ndarray, n_particle: np.ndarray) -> np.ndarray:
    """Parcel liquid mass up to a constant: nParticle * d^3.

    Falls back to nParticle if no diameter was written (then every droplet counts equally, which
    is a worse approximation of mass, so I would rather make sure d is written).
    """
    if np.all(np.isnan(d)):
        return n_particle.astype(float)
    return n_particle * d ** 3


def mass_percentile_axial(s: np.ndarray, w: np.ndarray, q: float) -> float:
    """Axial distance within which q % of the liquid mass is found."""
    if len(s) == 0:
        return float("nan")
    order = np.argsort(s)
    cum = np.cumsum(w[order])
    return float(s[order][np.searchsorted(cum, q / 100.0 * cum[-1])])


def spray_angle_from_parcels(s: np.ndarray, r: np.ndarray, s_tip: float, frac=(0.2, 0.8),
                             r_percentile: float = 95.0, nbins: int = 12, min_parcels: int = 30) -> float:
    """Full spray angle in degrees from a linear fit of the radial envelope r_q(s).

    I use 12 axial bins because it gives a stable fit with the ~10^4 parcels of the tutorial;
    with many more parcels, more bins would reduce the noise of the slope further.
    """
    lo, hi = frac[0] * s_tip, frac[1] * s_tip
    edges = np.linspace(lo, hi, nbins + 1)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (s >= a) & (s < b)
        if sel.sum() >= max(3, min_parcels // nbins):
            xs.append(0.5 * (a + b))
            ys.append(np.percentile(r[sel], r_percentile))
    if len(xs) < 4:
        return float("nan")
    slope = np.polyfit(xs, ys, 1)[0]
    return float(np.degrees(2.0 * np.arctan(slope)))


def parcel_timeseries(frames: list[dict], origin, direction, params_metrics: dict) -> pd.DataFrame:
    """S_max, S<q> and the spray angle for every written time."""
    rows = []
    qs = params_metrics["percentiles"]
    for fr in frames:
        pos = fr["pos"]
        row = {"t": fr["t"], "n_parcels": len(pos)}
        if len(pos) == 0:
            rows.append({**row, "S_max": np.nan, **{f"S{q}": np.nan for q in qs}, "theta_parcel_deg": np.nan})
            continue
        s, r = axial_radial(pos, origin, direction)
        w = liquid_mass_weight(fr["d"], fr["nParticle"])
        row["S_max"] = float(s.max())
        for q in qs:
            row[f"S{q}"] = mass_percentile_axial(s, w, q)
        ref = row[f"S{params_metrics['primary_percentile']}"]
        row["theta_parcel_deg"] = spray_angle_from_parcels(
            s, r, ref, params_metrics["angle_axial_fraction"], params_metrics["angle_radial_percentile"])
        rows.append(row)
    return pd.DataFrame(rows)
