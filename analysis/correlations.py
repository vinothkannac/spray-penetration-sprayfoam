"""Empirical spray correlations used as the reference, and the grid convergence index.

All quantities in SI units.

Hiroyasu & Arai (1990), liquid tip penetration of a non-evaporating diesel spray:
    S = 0.39 * sqrt(2*dp/rho_l) * t                 for t <  t_break
    S = 2.95 * (dp/rho_a)**0.25 * sqrt(d0 * t)      for t >= t_break
    t_break = 29 * rho_l * d0 / sqrt(rho_a * dp)

Design notes
- Why this correlation. I needed a reference that needs only the nozzle diameter, the injection
  velocity and the two densities, because those are the quantities the tutorial gives me. Its
  two regimes (linear jet, then square-root-in-time break-up regime) are also what a spray
  simulation should reproduce qualitatively.
  Alternatives: Naber & Siebers (1996) is better calibrated for liquid-length and angle at
  engine-like conditions; Dent (1971) is similar in form; Lagrangian results should ultimately
  be compared with measured data (for instance an Engine Combustion Network "Spray A" case).
- The pressure drop. The tutorial specifies a velocity, not a pressure, so I use
  dp = 0.5 * rho_l * U^2, i.e. a velocity coefficient of 1. A real nozzle has Cv < 1, which would
  raise dp for the same U and increase the predicted penetration slightly. I kept Cv = 1 because
  I have no nozzle data; the argument `cv` lets you change it.
- Extrapolation. The correlation was fitted to diesel sprays at far higher injection pressure than
  this tutorial injector. The comparison is therefore "agreement with an empirical trend", not
  validation.
- Please check the constants 0.39, 2.95 and 29 against SAE 900475 before you quote them.
"""
from __future__ import annotations

import numpy as np


def pressure_drop(u0: float, rho_l: float, cv: float = 1.0) -> float:
    """dp from the injection velocity, u0 = cv * sqrt(2*dp/rho_l)."""
    return 0.5 * rho_l * (u0 / cv) ** 2


def hiroyasu_arai(t, d0: float, u0: float, rho_l: float, rho_a: float, cv: float = 1.0):
    t = np.asarray(t, dtype=float)
    dp = pressure_drop(u0, rho_l, cv)
    t_break = 29.0 * rho_l * d0 / np.sqrt(rho_a * dp)
    s_early = 0.39 * np.sqrt(2.0 * dp / rho_l) * t
    s_late = 2.95 * (dp / rho_a) ** 0.25 * np.sqrt(d0 * np.maximum(t, 0.0))
    return np.where(t < t_break, s_early, s_late)


def break_up_time(d0: float, u0: float, rho_l: float, rho_a: float, cv: float = 1.0) -> float:
    dp = pressure_drop(u0, rho_l, cv)
    return float(29.0 * rho_l * d0 / np.sqrt(rho_a * dp))


def naber_siebers_half_angle_deg(rho_a: float, rho_l: float, c: float = 0.26) -> float:
    """Full spray angle (despite the function name) from the Naber-Siebers relation
    tan(theta/2) = c * [(rho_a/rho_l)^0.19 - 0.0043*sqrt(rho_l/rho_a)].

    The constant c is nozzle dependent (0.26 is the value I took for a diesel nozzle). I only use
    this as a rough comparison for the spray angle, not as a validation target.
    """
    tan_half = c * ((rho_a / rho_l) ** 0.19 - 0.0043 * np.sqrt(rho_l / rho_a))
    return float(np.degrees(2.0 * np.arctan(tan_half)))


def gas_density(p_pa: float, t_k: float, r_specific: float = 287.0) -> float:
    """Ideal gas density. I ignore the real-gas correction of air at 5 MPa (a few percent)."""
    return p_pa / (r_specific * t_k)


def gci(f_fine: float, f_med: float, f_coarse: float, n_fine: int, n_med: int, n_coarse: int,
        fs: float = 1.25, dim: int = 3, p_formal: float = 2.0) -> dict:
    """Grid convergence index after Celik et al. (2008) with three meshes.

    Design notes
    - The refinement ratio comes from the cell counts, r = (N_fine/N_med)^(1/dim). My meshes are
      scaled uniformly, so this is exact for the structured blockMesh used here.
    - Three levels give an observed order p from the fixed-point iteration of Celik et al.
      (eq. 3.7). For a Lagrangian spray I do not trust a large p: the penetration depends on
      the injection, break-up and drag models as well as on the grid, and a few meshes can
      agree by chance. I therefore also return a conservative GCI that assumes the formal order
      `p_formal` (gci_fine_pct_formal) and a flag `order_plausible` (p <= 1.5 * p_formal, my own
      rule of thumb). Report the conservative value when the flag is False.
    - If the three values are not monotone, no order or GCI is computed and only the spread is
      returned.
    """
    r21 = (n_fine / n_med) ** (1 / dim)
    r32 = (n_med / n_coarse) ** (1 / dim)
    e21 = f_med - f_fine
    e32 = f_coarse - f_med
    out = {"r21": r21, "r32": r32, "eps21": e21, "eps32": e32,
           "spread_pct": 100 * (max(f_fine, f_med, f_coarse) - min(f_fine, f_med, f_coarse)) / abs(f_fine)}
    if e21 == 0 or e32 == 0 or e32 / e21 <= 0:
        out.update(monotone=False, order=None, gci_fine_pct=None, f_extrapolated=None,
                   gci_fine_pct_formal=None, order_plausible=None)
        return out
    s = np.sign(e32 / e21)
    p = 2.0
    for _ in range(100):
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-6:
            p = p_new
            break
        p = p_new
    f_ext = (r21 ** p * f_fine - f_med) / (r21 ** p - 1)
    gci_fine = fs * abs(e21 / f_fine) / (r21 ** p - 1) * 100
    gci_formal = fs * abs(e21 / f_fine) / (r21 ** p_formal - 1) * 100
    out.update(monotone=True, order=float(p), gci_fine_pct=float(gci_fine), f_extrapolated=float(f_ext),
               gci_fine_pct_formal=float(gci_formal), order_plausible=bool(p <= 1.5 * p_formal))
    return out
