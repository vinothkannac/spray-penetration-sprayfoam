"""The reference correlation and the grid convergence index (GCI). All values are in SI units.

Hiroyasu & Arai (1990), tip penetration of a diesel spray that does not evaporate:
    S = 0.39 * sqrt(2*dp/rho_l) * t                 for t <  t_break
    S = 2.95 * (dp/rho_a)**0.25 * sqrt(d0 * t)      for t >= t_break
    t_break = 29 * rho_l * d0 / sqrt(rho_a * dp)

Why I chose this correlation
- It only needs the nozzle diameter, the injection velocity and the two densities, which I
  can get from the tutorial. It also has two stages (first linear, then square root of time)
  that a spray simulation should show too.
  Naber & Siebers (1996) and Dent (1971) are other options. In the end the results should be
  compared with measured data, for example the ECN "Spray A" case.
- Pressure drop: the tutorial gives a velocity and not a pressure, so I use
  dp = 0.5 * rho_l * U^2, which means a velocity coefficient of 1. A real nozzle has a value
  below 1. That would give a higher dp for the same velocity and a slightly longer predicted
  penetration. I kept 1 because I have no nozzle data. The argument cv changes it.
- The correlation was fitted to diesel sprays at much higher injection pressure than this
  tutorial uses. So my comparison shows agreement with an empirical trend. It is not a
  validation.
- The constants 0.39, 2.95 and 29 still need to be checked against SAE 900475 before they
  are quoted.
"""
import numpy as np


def pressure_drop(u0, rho_l, cv=1.0):
    """Pressure drop from the injection velocity, using u0 = cv * sqrt(2*dp/rho_l)."""
    return 0.5 * rho_l * (u0 / cv) ** 2


def break_up_time(d0, u0, rho_l, rho_a, cv=1.0):
    dp = pressure_drop(u0, rho_l, cv)
    return float(29.0 * rho_l * d0 / np.sqrt(rho_a * dp))


def hiroyasu_arai(t, d0, u0, rho_l, rho_a, cv=1.0):
    t = np.asarray(t, dtype=float)
    dp = pressure_drop(u0, rho_l, cv)
    t_break = break_up_time(d0, u0, rho_l, rho_a, cv)
    s_before = 0.39 * np.sqrt(2.0 * dp / rho_l) * t
    s_after = 2.95 * (dp / rho_a) ** 0.25 * np.sqrt(d0 * np.maximum(t, 0.0))
    # np.where picks the right formula for every time value
    return np.where(t < t_break, s_before, s_after)


def naber_siebers_full_angle_deg(rho_a, rho_l, c=0.26):
    """Full spray angle from the Naber-Siebers relation
    tan(theta/2) = c * [(rho_a/rho_l)^0.19 - 0.0043*sqrt(rho_l/rho_a)].

    The constant c depends on the nozzle. I took 0.26 for a diesel nozzle. I only use this
    angle as a rough comparison and not as a target.
    """
    tan_half = c * ((rho_a / rho_l) ** 0.19 - 0.0043 * np.sqrt(rho_l / rho_a))
    return float(np.degrees(2.0 * np.arctan(tan_half)))


def gas_density(p_pa, t_k, r_specific=287.0):
    """Ideal gas density. I ignore the real gas correction for air at 5 MPa (a few percent)."""
    return p_pa / (r_specific * t_k)


def gci(f_fine, f_med, f_coarse, n_fine, n_med, n_coarse, fs=1.25, dim=3, p_formal=2.0):
    """Grid convergence index for three meshes, following Celik et al. (2008).

    My notes
    - I get the refinement ratio from the cell counts, r = (N_fine/N_med)^(1/dim). This works
      here because I scale my structured meshes uniformly.
    - Three meshes give an observed order p, which is solved by repeating eq. 3.7 of Celik et
      al. until it stops changing. For a Lagrangian spray I do not trust a large p, because
      the penetration also depends on the injection, break-up and drag models, and three
      meshes can agree by chance. So I also calculate a safer GCI with the formal order
      p_formal, and a flag order_plausible. The limit p <= 1.5 * p_formal is my own rule of
      thumb. When the flag is False, the safer value should be reported.
    - If the three values do not change in one direction, I only return the spread.
    """
    r21 = (n_fine / n_med) ** (1 / dim)
    r32 = (n_med / n_coarse) ** (1 / dim)
    e21 = f_med - f_fine
    e32 = f_coarse - f_med
    values = [f_fine, f_med, f_coarse]
    result = {"r21": r21, "r32": r32, "eps21": e21, "eps32": e32,
              "spread_pct": 100 * (max(values) - min(values)) / abs(f_fine)}

    monotone = bool(e21 != 0 and e32 != 0 and e32 / e21 > 0)
    result["monotone"] = monotone
    if not monotone:
        for key in ("order", "gci_fine_pct", "f_extrapolated", "gci_fine_pct_formal", "order_plausible"):
            result[key] = None
        return result

    # Find the observed order p by fixed point iteration, starting from p = 2
    s = np.sign(e32 / e21)
    p = 2.0
    for _ in range(100):
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_old = p
        p = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p - p_old) < 1e-6:
            break

    f_extrapolated = (r21 ** p * f_fine - f_med) / (r21 ** p - 1)
    gci_observed = fs * abs(e21 / f_fine) / (r21 ** p - 1) * 100
    gci_formal = fs * abs(e21 / f_fine) / (r21 ** p_formal - 1) * 100

    result["order"] = float(p)
    result["gci_fine_pct"] = float(gci_observed)
    result["f_extrapolated"] = float(f_extrapolated)
    result["gci_fine_pct_formal"] = float(gci_formal)
    result["order_plausible"] = bool(p <= 1.5 * p_formal)
    return result
