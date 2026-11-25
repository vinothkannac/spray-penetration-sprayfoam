"""Penetration length and spray angle, calculated from the parcel data.

Why I measure them this way
- Penetration: I use S99, the axial distance from the nozzle that holds 99 % of the liquid
  mass. The mass of a parcel is proportional to nParticle * d^3. I did not use the furthest
  parcel (S_max) because it is set by one fast droplet, so it is noisy and depends on the
  mesh. A percentile is also closer to experiments, which use a threshold to find the edge of
  the liquid. I also save S95 and S_max to show how much this choice matters.
  Other options are a threshold on the projected mass, or a very small value of the liquid
  volume fraction on the mesh.
- Spray angle: in each axial bin I take the 95th percentile of the radial position, fit a
  straight line from 20 % to 80 % of the penetration, and use angle = 2*atan(slope). I leave
  out the part near the nozzle, where the injector cone dominates, and the tip, where the
  head vortex makes the spray wider.
  Another option is the Naber-Siebers cone over the first 60 % of the penetration.
"""
import numpy as np
import pandas as pd


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def axial_radial(pos, origin, direction):
    """Distance s along the spray axis and distance r away from the axis, in m.

    I use the injector axis from params.json and not simply the y axis, so that the code also
    works for a tilted injector.
    """
    axis = unit(direction)
    relative = pos - np.asarray(origin, dtype=float)
    s = relative @ axis
    sideways = relative - np.outer(s, axis)
    r = np.linalg.norm(sideways, axis=1)
    return s, r


def liquid_mass_weight(d, n_particle):
    """Liquid mass of each parcel, without the constant factors: nParticle * d^3.

    If no diameter was written, I use nParticle only. Then every droplet counts the same,
    which is a poor estimate of the mass, so it is better to have d in the output.
    """
    if np.all(np.isnan(d)):
        return n_particle.astype(float)
    return n_particle * d ** 3


def mass_percentile_axial(s, weight, percent):
    """Axial distance that contains the given percentage of the liquid mass."""
    if len(s) == 0:
        return float("nan")
    order = np.argsort(s)                      # parcels from the nozzle to the tip
    cumulative_mass = np.cumsum(weight[order])
    target_mass = percent / 100.0 * cumulative_mass[-1]
    index = np.searchsorted(cumulative_mass, target_mass)
    return float(s[order][index])


def spray_angle_from_parcels(s, r, s_tip, frac=(0.2, 0.8), r_percentile=95.0, nbins=12,
                             min_parcels=30):
    """Full spray angle in degrees.

    I use 12 axial bins because this gave a stable fit with the roughly 10^4 parcels of the
    tutorial. With many more parcels, more bins would be possible.
    """
    edges = np.linspace(frac[0] * s_tip, frac[1] * s_tip, nbins + 1)
    # I skip bins with only a few parcels, because their percentile is not reliable
    min_per_bin = max(3, min_parcels // nbins)

    bin_centres = []
    bin_radius = []
    for i in range(nbins):
        in_bin = (s >= edges[i]) & (s < edges[i + 1])
        if in_bin.sum() >= min_per_bin:
            bin_centres.append(0.5 * (edges[i] + edges[i + 1]))
            bin_radius.append(np.percentile(r[in_bin], r_percentile))

    if len(bin_centres) < 4:
        return float("nan")
    slope = np.polyfit(bin_centres, bin_radius, 1)[0]
    return float(np.degrees(2.0 * np.arctan(slope)))


def parcel_timeseries(frames, origin, direction, params_metrics):
    """Table with S_max, the percentiles and the spray angle for every written time."""
    percentiles = params_metrics["percentiles"]
    primary = f"S{params_metrics['primary_percentile']}"
    rows = []
    for frame in frames:
        pos = frame["pos"]
        row = {"t": frame["t"], "n_parcels": len(pos)}

        if len(pos) == 0:
            row["S_max"] = np.nan
            for q in percentiles:
                row[f"S{q}"] = np.nan
            row["theta_parcel_deg"] = np.nan
            rows.append(row)
            continue

        s, r = axial_radial(pos, origin, direction)
        weight = liquid_mass_weight(frame["d"], frame["nParticle"])
        row["S_max"] = float(s.max())
        for q in percentiles:
            row[f"S{q}"] = mass_percentile_axial(s, weight, q)
        row["theta_parcel_deg"] = spray_angle_from_parcels(
            s, r, row[primary], params_metrics["angle_axial_fraction"],
            params_metrics["angle_radial_percentile"])
        rows.append(row)
    return pd.DataFrame(rows)
