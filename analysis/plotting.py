"""All the figures of the study. I keep them in one file so that they have the same style.

Why I made the figures this way
- One style for every figure: the same font, a colour-blind-safe palette (Okabe-Ito), no top
  or right border, a light grid, a bold title with a grey subtitle, panel letters (a), (b),
  ... and units on every axis. I put the captions in the README and not in the images, so
  that I can reuse the PNG files, for example in slides.
- Each mesh has the same colour in every figure (coarse blue, medium green, fine orange-red),
  so the reader does not have to learn the legend again.
- I save PNG files at 200 dpi because they are easy to show in the README. For vector output
  in a report, SAVE_KW can be changed to {"format": "pdf"}.
- The plotted results come from the analysis. I do not type result values into the figures.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import image_pipeline as ip
import render_images as ri
import spray_metrics as sm

COLORS = {"coarse": "#0072B2", "medium": "#009E73", "fine": "#D55E00", "corr": "#222222",
          "image": "#CC79A7", "ref": "#7F7F7F", "window": "#E8E8E8"}
SAVE_KW = {"dpi": 200, "facecolor": "white"}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 100, "lines.linewidth": 1.8,
    })


def add_titles(fig, title, subtitle, y=0.995):
    fig.text(0.012, y, title, fontsize=13, fontweight="bold", va="top", ha="left")
    # I scale the gap with the figure height, so it looks the same on every figure
    subtitle_y = y - 0.05 * (5.0 / fig.get_figheight())
    fig.text(0.012, subtitle_y, subtitle, fontsize=9, color="#555555", va="top", ha="left")


def add_panel_letter(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right")


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)


def format_cells(mesh_info, level):
    """Number of cells with a comma as thousands separator."""
    return f"{mesh_info[level]['n_cells']:,}"


# --------------------------------------------------------------------------------------- setup
def plot_setup(params, mesh_info, inj_tbl, out):
    """Chamber, injector cone, the three meshes near the nozzle, and cell size against nozzle size."""
    inj = params["injector"]
    half = inj_tbl["thetaOuter"]
    th = np.radians(half)
    y0 = inj["origin_m"][1] * 1e3  # nozzle position along the chamber axis (mm)
    d_noz = inj["nozzle_diameter_m"] * 1e6
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.8, 1, 1, 1], height_ratios=[1.15, 1], left=0.05, right=0.985,
                          top=0.83, bottom=0.09, wspace=0.3, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0])
    ax.add_patch(Rectangle((-10, 0), 20, 100, fc="#F4F4F4", ec="#333333", lw=1.4))
    cone_length = 50.0  # I draw only the first 50 mm of the cone, so it stays inside the walls
    for sgn in (-1, 1):
        ax.plot([0, sgn * cone_length * np.tan(th)], [y0, y0 - cone_length], color=COLORS["fine"], lw=1.4)
    ax.fill([0, -cone_length * np.tan(th), cone_length * np.tan(th)], [y0, y0 - cone_length, y0 - cone_length], color=COLORS["fine"], alpha=0.12, lw=0)
    ax.plot([0, 0], [y0, 5], color="#333333", lw=0.8, ls=(0, (6, 3)))
    ax.plot([0], [y0], marker="v", color="#333333", ms=7)
    ax.annotate(f"injector (d = {d_noz:.0f} µm)", (0, y0), (-9, 107), fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": "#555555"}, annotation_clip=False)
    ax.annotate("", (14, 0), (14, 100), arrowprops={"arrowstyle": "<->", "color": "#555555"}, annotation_clip=False)
    ax.text(15.5, 50, "100 mm", rotation=90, va="center", fontsize=8.5, color="#555555")
    ax.annotate("", (-10, -5), (10, -5), arrowprops={"arrowstyle": "<->", "color": "#555555"}, annotation_clip=False)
    ax.text(0, -10, "20 mm", ha="center", fontsize=8.5, color="#555555")
    ax.text(0, 38, f"injection cone\nfull angle {2 * half:.0f}°", ha="center", va="center", fontsize=8.5, color=COLORS["fine"])
    ax.text(0, 20, f"quiescent air\n{params['ambient']['pressure_Pa'] / 1e6:g} MPa, {params['cold_temperature_K']:g} K",
            ha="center", va="center", fontsize=8.5, bbox={"fc": "#F4F4F4", "ec": "none", "pad": 2})
    ax.set_xlim(-18, 22)
    ax.set_ylim(-14, 112)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral position (mm)")
    ax.set_ylabel("axial position (mm)")
    ax.grid(False)
    add_panel_letter(ax, "a")
    ax.set_title("Chamber", loc="left")

    sizes = {}
    for k, lv in enumerate(params["mesh_levels"]):
        a = fig.add_subplot(gs[0, k + 1])
        nx, ny, nz = mesh_info[lv]["cells_per_direction"]
        dx, dz = 20.0 / nx, 100.0 / nz
        sizes[lv] = (dx, dz)
        for x in np.arange(-10, 10 + 1e-9, dx):
            a.plot([x, x], [90, 100], color="#888888", lw=0.5)
        for z in np.arange(100, 90 - 1e-9, -dz):
            a.plot([-10, 10], [z, z], color="#888888", lw=0.5)
        for sgn in (-1, 1):
            a.plot([0, sgn * (y0 - 90) * np.tan(th)], [y0, 90], color=COLORS["fine"], lw=1.4)
        a.plot([0, 0], [y0, 90], color="#333333", lw=0.8, ls=(0, (6, 3)))
        a.plot([0], [y0], marker="v", color="#333333", ms=6, zorder=5)
        a.set_xlim(-5, 5)
        a.set_ylim(90, 100)
        a.set_aspect("equal")
        a.set_xlabel("lateral position (mm)")
        a.grid(False)
        if k == 0:
            a.set_ylabel("axial position (mm)")
        a.set_title(f"{lv}: {nx}×{ny}×{nz} = {format_cells(mesh_info, lv)} cells", fontsize=9, color=COLORS[lv],
                    fontweight="bold", loc="left")
        add_panel_letter(a, "bcd"[k])
    b = fig.add_subplot(gs[1, 1:])
    lv_list = list(sizes)
    y = np.arange(len(lv_list))
    b.barh(y - 0.18, [sizes[lv][0] * 1e3 for lv in lv_list], 0.34, color=[COLORS[lv] for lv in lv_list])
    b.barh(y + 0.18, [sizes[lv][1] * 1e3 for lv in lv_list], 0.34, color=[COLORS[lv] for lv in lv_list], alpha=0.45)
    b.axvline(d_noz, color="#222222", ls="--", lw=1.3)
    b.text(d_noz + 15, -0.62, f"nozzle diameter {d_noz:.0f} µm", fontsize=8.5, va="center")
    for yi, lv in zip(y, lv_list):
        b.text(sizes[lv][0] * 1e3 + 15, yi - 0.18, f"{sizes[lv][0] * 1e3:.0f} µm lateral", va="center", fontsize=8)
        b.text(sizes[lv][1] * 1e3 + 15, yi + 0.18, f"{sizes[lv][1] * 1e3:.0f} µm axial", va="center", fontsize=8)
    b.set_yticks(y, lv_list)
    b.invert_yaxis()
    b.set_xlim(0, 1700)
    b.set_ylim(len(lv_list) - 0.4, -0.9)
    b.set_xlabel("cell edge length (µm)")
    b.set_title("The nozzle is smaller than a single cell on every mesh", loc="left")
    b.grid(axis="y", visible=False)
    add_panel_letter(b, "e")
    add_titles(fig, "Test case and mesh levels",
            "OpenFOAM aachenBomb tutorial: constant-volume chamber, single solid-cone injector on the axis; "
            "(b)–(d) zoom on the injector end of each mesh", 0.985)
    save_figure(fig, out)


# ----------------------------------------------------------------------------------- injection
def injection_velocity(inj_tbl, rho_l, n=400):
    """Time, mass flow rate (kg/s) and velocity (m/s). I use the same formula as the solver's
    ConeNozzleInjection, so that the figure shows the velocity of the simulation.

    U(t) = massTotal * profile(t) / integral(profile) / (rho * Cd * A)   [flowRateAndDischarge]
    """
    t_tab, pr = inj_tbl["t"], inj_tbl["profile"]
    dur = inj_tbl["duration"]
    t = np.linspace(0, dur, n)
    tt = np.linspace(0, dur, 20001)
    integral = np.trapezoid(np.interp(tt, t_tab, pr), tt)
    mdot = inj_tbl["massTotal"] * np.interp(t, t_tab, pr) / integral
    area = 0.25 * np.pi * inj_tbl["outerDiameter"] ** 2
    return t, mdot, mdot / (rho_l * inj_tbl["Cd"] * area)


def plot_injection(params, inj_tbl, out):
    inj = params["injector"]
    t, mdot, u = injection_velocity(inj_tbl, inj["liquid_density_kg_m3"])
    w0, w1 = params["metrics"]["eval_window_s"]
    sel = t <= params["end_time_s"]
    u_mean = float(np.mean(u[sel]))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True, gridspec_kw={"hspace": 0.12})
    fig.subplots_adjust(top=0.84, left=0.11, right=0.97, bottom=0.09)
    for a in (a1, a2):
        a.axvspan(w0 * 1e3, w1 * 1e3, color=COLORS["window"], zorder=0)
    a1.plot(t * 1e3, mdot * 1e3, color=COLORS["corr"])
    a1.set_ylabel("mass flow rate (g/s)")
    a1.text(w0 * 1e3 + 0.01, a1.get_ylim()[1] * 0.12, "analysis window", fontsize=8.5, color="#555555")
    a2.plot(t * 1e3, u, color=COLORS["fine"], label="injection velocity U(t)")
    a2.axhline(inj["injection_velocity_m_s"], color=COLORS["corr"], ls="--", lw=1.3,
               label=f"constant {inj['injection_velocity_m_s']:.0f} m/s used in the correlation")
    a2.set_ylabel("injection velocity (m/s)")
    a2.set_xlabel("time after start of injection (ms)")
    a2.set_ylim(0, u.max() * 1.18)
    a2.legend(loc="lower right")
    add_panel_letter(a1, "a")
    add_panel_letter(a2, "b")
    add_titles(fig, "Injection history of the tutorial",
            f"massTotal {inj_tbl['massTotal'] * 1e6:.1f} mg, Cd {inj_tbl['Cd']}, nozzle d = "
            f"{inj_tbl['outerDiameter'] * 1e6:.0f} µm, ρ_l = {inj['liquid_density_kg_m3']:.0f} kg/m³ (estimate)", 0.985)
    save_figure(fig, out)
    return {"u_peak_m_s": float(u.max()), "u_mean_0_to_end_m_s": u_mean}


# -------------------------------------------------------------------------------------- parcels
def plot_parcels(frames_by_level, series, params, mesh_info, prim, out):
    """Parcel cloud at the last written time for each mesh level."""
    inj, img = params["injector"], params["images"]
    d_ax = sm.unit(inj["direction"])
    lat_ax = ri.lateral_axis(inj["direction"], img["view_lateral_axis"])
    levels = list(frames_by_level)
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.6), sharey=True)
    fig.subplots_adjust(top=0.80, left=0.07, right=0.88, bottom=0.12, wspace=0.12)
    s_max = max(series[lv][prim].iloc[-1] for lv in levels) * 1e3
    sc = None
    for k, (lv, a) in enumerate(zip(levels, axes)):
        fr = frames_by_level[lv][-1]
        rel = fr["pos"] - np.asarray(inj["origin_m"])
        s, lat = rel @ d_ax * 1e3, rel @ lat_ax * 1e3
        dia = fr["d"] * 1e6
        idx = rng.permutation(len(s))[:25000]
        idx = idx[np.argsort(dia[idx])[::-1]]  # I draw small droplets last, so they are on top
        sc = a.scatter(lat[idx], s[idx], c=dia[idx], s=2.2, cmap="viridis", vmin=0, vmax=np.percentile(dia, 99.5),
                       rasterized=True, linewidths=0)
        s99 = series[lv][prim].iloc[-1] * 1e3
        a.axhline(s99, color=COLORS["corr"], ls="--", lw=1.2)
        a.text(9.6, s99 + 0.8, f"{prim} = {s99:.1f} mm", ha="right", va="top", fontsize=8.5)
        a.set_xlim(-10, 10)
        a.set_ylim(s_max * 1.2, -1)
        a.set_aspect("equal")
        a.set_xlabel("lateral position (mm)")
        a.set_title(f"{lv}: {format_cells(mesh_info, lv)} cells\n{len(s):,} parcels", color=COLORS[lv], fontweight="bold",
                    fontsize=9.5, loc="left")
        add_panel_letter(a, "abc"[k])
    axes[0].set_ylabel("axial distance from nozzle (mm)")
    cax = fig.add_axes([0.9, 0.12, 0.015, 0.68])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("parcel diameter (µm)")
    t_end = frames_by_level[levels[0]][-1]["t"] * 1e3
    add_titles(fig, f"Lagrangian parcels at t = {t_end:.2f} ms",
            "Side projection of all three meshes (up to 25,000 randomly drawn parcels each); dashed = penetration length",
            0.985)
    save_figure(fig, out)


# -------------------------------------------------------------------------------- penetration
def plot_penetration(series, params, idf, prim, out):
    levels = list(series)
    w0, w1 = params["metrics"]["eval_window_s"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.1})
    fig.subplots_adjust(top=0.865, left=0.1, right=0.97, bottom=0.08)
    for a in (a1, a2):
        a.axvspan(w0 * 1e3, w1 * 1e3, color=COLORS["window"], zorder=0)
    ref = series[levels[1]]
    a1.plot(ref["t"] * 1e3, ref["S_corr"] * 1e3, "--", color=COLORS["corr"], lw=2, label="Hiroyasu–Arai correlation")
    for lv in levels:
        d = series[lv]
        a1.plot(d["t"] * 1e3, d[prim] * 1e3, "-o", ms=3.2, color=COLORS[lv], label=f"sprayFoam, {lv} mesh")
        err = 100 * (d[prim] - d["S_corr"]) / d["S_corr"]
        a2.plot(d["t"] * 1e3, err, "-o", ms=3, color=COLORS[lv])
    if idf is not None:
        a1.plot(idf["t"] * 1e3, idf["S_image"] * 1e3, "x", ms=6, mew=1.6, color=COLORS["image"],
                label="image pipeline (medium mesh)")
    a1.set_ylabel(f"penetration length {prim} (mm)")
    a1.legend(loc="upper left")
    a1.text(0.985, 0.04, "evaluation window", transform=a1.transAxes, ha="right", fontsize=8.5, color="#555555")
    a2.axhline(0, color=COLORS["corr"], lw=1.0, ls="--")
    a2.set_ylim(-60, 40)
    a2.set_ylabel("deviation from\ncorrelation (%)")
    a2.set_xlabel("time after start of injection (ms)")
    add_panel_letter(a1, "a")
    add_panel_letter(a2, "b")
    add_titles(fig, "Liquid penetration: sprayFoam vs. Hiroyasu–Arai",
            f"Non-evaporating spray (293 K, 5 MPa); {prim} = axial distance containing {prim[1:]} % of the liquid mass", 0.985)
    save_figure(fig, out)


# ----------------------------------------------------------------------------------- convergence
def plot_convergence(mean_s, cells, series, table, gci_result, params, prim, theta_ns, half_angle, out):
    levels = list(mean_s)
    n = np.array([cells[lv] for lv in levels], dtype=float)
    h = (n[-1] / n) ** (1 / 3)  # cell size relative to the fine mesh
    y = np.array([mean_s[lv] * 1e3 for lv in levels])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1.35, 0.9, 1.1], "wspace": 0.38})
    fig.subplots_adjust(top=0.78, left=0.06, right=0.985, bottom=0.14)
    a = axes[0]
    offsets = {"coarse": ((-10, 0), "right", "center"), "medium": ((-8, -12), "right", "top"),
               "fine": ((10, 10), "left", "bottom")}
    for lv, hi, yi in zip(levels, h, y):
        a.plot(hi, yi, "o", ms=8, color=COLORS[lv], zorder=4)
        off, ha, va = offsets.get(lv, ((8, 8), "left", "bottom"))
        a.annotate(f"{lv}\n{cells[lv]:,} cells", (hi, yi), textcoords="offset points", xytext=off, ha=ha, va=va,
                   fontsize=8.5, color=COLORS[lv])
    hh = np.linspace(0, h.max() * 1.05, 100)
    if gci_result["monotone"]:
        r = gci_result["r21"]
        # I draw the curve through the two finest meshes for the observed and the formal order
        curves = [(gci_result["order"], ":", f"observed order p = {gci_result['order']:.1f}"),
                  (2.0, "--", "formal order p = 2")]
        for p, line_style, label in curves:
            f_ext = (r ** p * y[2] - y[1]) / (r ** p - 1)  # value for zero cell size
            slope = (y[2] - f_ext) / h[2] ** p
            a.plot(hh, f_ext + slope * hh ** p, ls=line_style, color="#555555", lw=1.3,
                   label=f"{label}: S(h→0) = {f_ext:.1f} mm")
            a.plot(0, f_ext, "*", ms=11, color="#555555", clip_on=False, zorder=5)
        a.legend(loc="lower right", title="Richardson extrapolation", title_fontsize=8.5)
    a.set_xlim(-0.03, h.max() * 1.1)
    a.set_xlabel("relative cell size  h / h_fine  =  (N_fine / N)^(1/3)")
    a.set_ylabel(f"mean {prim} over the window (mm)")
    add_panel_letter(a, "a")
    a.set_title("Mesh convergence of the penetration", loc="left")

    a = axes[1]
    x = np.arange(len(levels))
    bias = table["bias_vs_corr_pct"].values
    mape = table["MAPE_vs_corr_pct"].values
    a.bar(x - 0.2, bias, 0.38, color=[COLORS[lv] for lv in levels], label="mean deviation (bias)")
    a.bar(x + 0.2, mape, 0.38, color=[COLORS[lv] for lv in levels], alpha=0.45, label="mean absolute deviation")
    for xi, b, m in zip(x, bias, mape):
        a.text(xi - 0.2, b - 1.5 if b < 0 else b + 0.5, f"{b:+.1f}", ha="center", va="top" if b < 0 else "bottom",
               fontsize=8)
        a.text(xi + 0.2, m + 0.5, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
    a.axhline(0, color="#333333", lw=0.8)
    a.set_xticks(x, levels)
    a.set_ylabel("deviation from Hiroyasu–Arai (%)")
    a.set_ylim(min(bias.min() * 1.25, -5), max(mape.max() * 1.25, 5))
    a.legend(loc="lower right", fontsize=8, handles=[
        Rectangle((0, 0), 1, 1, fc="#777777", label="mean deviation (bias)"),
        Rectangle((0, 0), 1, 1, fc="#777777", alpha=0.45, label="mean absolute deviation")])
    add_panel_letter(a, "b")
    a.set_title("Agreement with the correlation", loc="left")

    a = axes[2]
    th = table["theta_parcel_deg_mean"].values
    a.plot(n, th, "-", color="#999999", lw=1.2, zorder=1)
    for lv, ni, ti in zip(levels, n, th):
        a.plot(ni, ti, "o", ms=8, color=COLORS[lv], zorder=3)
    a.axhline(2 * half_angle, color=COLORS["fine"], ls="--", lw=1.2)
    a.text(n.min(), 2 * half_angle + 0.3, f"injector cone, {2 * half_angle:.0f}°", fontsize=8.5, color=COLORS["fine"], va="bottom")
    a.axhline(theta_ns, color=COLORS["ref"], ls=":", lw=1.4)
    a.text(n.min() * 1.02, theta_ns + 0.15, f"Naber–Siebers, {theta_ns:.1f}°", fontsize=8.5, color="#555555", va="bottom")
    a.set_xscale("log")
    a.set_xticks([1e5, 2e5, 4e5], ["100k", "200k", "400k"])
    a.minorticks_off()
    a.set_xlabel("number of cells")
    a.set_ylabel("spray full angle from parcels (°)")
    add_panel_letter(a, "c")
    a.set_title("Spray angle", loc="left")
    if gci_result["monotone"]:
        sub = (f"Observed order p = {gci_result['order']:.1f} → GCI(fine) = {gci_result['gci_fine_pct']:.1f} %; "
               f"with the formal order p = 2 → {gci_result['gci_fine_pct_formal']:.1f} %")
        if not gci_result["order_plausible"]:
            sub += "  (p is implausibly high for a Lagrangian spray: quote the second value)"
    else:
        sub = f"Not monotone: spread between the meshes {gci_result['spread_pct']:.1f} % (no GCI computed)"
    add_titles(fig, "Mesh sensitivity", sub, 0.985)
    save_figure(fig, out)


# -------------------------------------------------------------------------------- image pipeline
def image_extent(img, px):
    """Image edges in mm for imshow. The spray axis is at lateral position 0."""
    ny, nx = img.shape
    return [0, nx * px * 1e3, -ny * px * 5e2, ny * px * 5e2]


def draw_mask_outline(ax, mask, ext, line_width):
    x = np.linspace(ext[0], ext[1], mask.shape[1])
    y = np.linspace(ext[2], ext[3], mask.shape[0])
    ax.contour(x, y, mask.astype(float), levels=[0.5], colors=COLORS["fine"], linewidths=line_width)


def plot_pipeline_steps(images, bg, times, idf, params, out, prim):
    """All processing steps for the last frame."""
    px = params["images"]["pixel_size_m"]
    i = len(images) - 1
    frame = images[i].astype(np.float64)
    sig = ip.flat_field(frame, bg)
    mask, info = ip.segment(frame, bg)
    s_tip, theta = ip.penetration_and_angle(mask, px)
    xs, w = ip.width_profile(mask, px)
    slope, icpt = np.polyfit(xs, w, 1)
    # Centre line of the spray: the mean row of every column that contains spray
    column_centres = []
    for column in range(mask.shape[1]):
        if mask[:, column].any():
            column_centres.append(np.where(mask[:, column])[0].mean())
    centre = np.mean(column_centres)
    yc = (centre + 0.5) * px * 1e3 - mask.shape[0] * px * 5e2
    ax_mm = max(15.0, np.ceil(s_tip * 1e3 * 1.35 / 5) * 5)
    ext = image_extent(frame, px)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))
    fig.subplots_adjust(top=0.84, left=0.06, right=0.98, bottom=0.14, hspace=0.38, wspace=0.28)
    lo, hi = np.percentile(bg, [1, 99.9])
    kw = {"origin": "lower", "extent": ext, "aspect": "equal"}
    im0 = axes[0, 0].imshow(frame, cmap="gray", vmin=0, vmax=hi, **kw)
    axes[0, 0].set_title("1  Raw frame (16 bit)", loc="left")
    axes[0, 1].imshow(bg, cmap="gray", vmin=0, vmax=hi, **kw)
    axes[0, 1].set_title("2  Background (mean of 8 spray-free frames)", loc="left")
    im2 = axes[1, 0].imshow(sig, cmap="magma", vmin=0, vmax=0.6, **kw)
    axes[1, 0].set_title("3  Flat-field signal  1 − I / I₀, smoothed", loc="left")
    axes[1, 1].imshow(frame, cmap="gray", vmin=0, vmax=hi, **kw)
    draw_mask_outline(axes[1, 1], mask, ext, 1.2)
    xx = np.array([xs.min(), xs.max()]) * 1e3
    for sg in (-1, 1):
        axes[1, 1].plot(xx, yc + sg * 0.5 * (slope * xx * 1e-3 + icpt) * 1e3, color=COLORS["image"], lw=1.2, ls="--")
    axes[1, 1].axvline(s_tip * 1e3, color=COLORS["image"], lw=1.4)
    axes[1, 1].set_title(f"4  Mask, penetration {s_tip * 1e3:.1f} mm, angle {theta:.1f}°", loc="left")
    for k, a in enumerate(axes.flat):
        a.set_xlim(0, ax_mm)
        a.set_ylim(-ax_mm * 0.22, ax_mm * 0.22)
        a.grid(False)
        a.set_xlabel("axial distance (mm)")
        a.set_ylabel("lateral (mm)")
        add_panel_letter(a, "abcd"[k])
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.03, pad=0.015, label="extinction  1 − I / I₀")
    handles = [Line2D([0], [0], color=COLORS["fine"], label=f"mask (threshold {info['threshold']:.3f}, noise σ {info['noise_sigma']:.4f})"),
               Line2D([0], [0], color=COLORS["image"], ls="--", label="linear width fit, 20–80 % of length"),
               Line2D([0], [0], color=COLORS["image"], label="penetration")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    add_titles(fig, "Image-processing pipeline, step by step",
            f"Frame at t = {times[i] * 1e3:.2f} ms rendered from the medium-mesh parcels, {px * 1e6:.0f} µm/pixel", 0.985)
    save_figure(fig, out)


def plot_frame_sequence(images, masks, times, idf, params, out):
    px = params["images"]["pixel_size_m"]
    w0 = params["metrics"]["eval_window_s"][0]
    # I show four frames, evenly spread over the frames from the start of the window
    in_window = []
    for k, t in enumerate(times):
        if t >= w0:
            in_window.append(k)
    idx = []
    for fraction in (0, 1 / 3, 2 / 3, 1):
        idx.append(in_window[int(round(fraction * (len(in_window) - 1)))])
    s_all = max(idf["S_image"].iloc[idx]) * 1e3
    ax_mm = max(15.0, np.ceil(s_all * 1.3 / 5) * 5)
    fig, axes = plt.subplots(len(idx), 1, figsize=(8.5, 1.85 * len(idx) + 1.9), sharex=True)
    fig.subplots_adjust(top=0.88, left=0.08, right=0.98, bottom=0.08, hspace=0.18)
    for a, k in zip(axes, idx):
        ext = image_extent(images[k], px)
        a.imshow(images[k], cmap="gray", origin="lower", extent=ext, aspect="equal", vmin=0)
        draw_mask_outline(a, masks[k], ext, 1.0)
        s_img = idf.loc[idf["t"] == times[k], "S_image"].iloc[0] * 1e3
        a.axvline(s_img, color=COLORS["image"], lw=1.4, ls="--")
        a.text(s_img + 0.4, ax_mm * 0.17, f"{s_img:.1f} mm", color=COLORS["image"], fontsize=8.5, fontweight="bold",
               bbox={"fc": "black", "ec": "none", "pad": 1.5, "alpha": 0.6})
        a.text(0.6, ax_mm * 0.19, f"t = {times[k] * 1e3:.2f} ms", color="#111111", fontsize=9, va="top", fontweight="bold")
        a.set_xlim(0, ax_mm)
        a.set_ylim(-ax_mm * 0.22 * 0.9, ax_mm * 0.22 * 0.9)
        a.grid(False)
        a.set_ylabel("lateral (mm)")
    axes[-1].set_xlabel("axial distance from nozzle (mm)")
    add_titles(fig, "Rendered high-speed-style frames and detected spray",
            "Dark = liquid on a bright uneven background; orange contour = detected mask, dashed line = detected tip", 0.985)
    save_figure(fig, out)


def plot_image_validation(idf, params, prim, theta_ns, out):
    w0, w1 = params["metrics"]["eval_window_s"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), gridspec_kw={"wspace": 0.28})
    fig.subplots_adjust(top=0.78, left=0.07, right=0.985, bottom=0.14)
    a = axes[0]
    lim = [0, max(idf["S_image"].max(), idf[prim].max()) * 1e3 * 1.08]
    a.plot(lim, lim, color=COLORS["corr"], lw=1.0)
    band = np.array(lim)
    a.fill_between(lim, band * 0.95, band * 1.05, color=COLORS["window"], label="±5 %")
    sc = a.scatter(idf[prim] * 1e3, idf["S_image"] * 1e3, c=idf["t"] * 1e3, cmap="viridis", s=34, zorder=3)
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_aspect("equal")
    a.set_xlabel(f"parcel-based {prim} (mm)")
    a.set_ylabel("image-based penetration (mm)")
    a.legend(loc="upper left")
    fig.colorbar(sc, ax=a, fraction=0.046, pad=0.03, label="time (ms)")
    a.set_title("Penetration: image vs. parcels", loc="left")
    add_panel_letter(a, "a")
    a = axes[1]
    a.axvspan(w0 * 1e3, w1 * 1e3, color=COLORS["window"], zorder=0)
    a.plot(idf["t"] * 1e3, idf["theta_parcel_deg"], "-o", ms=3.5, color=COLORS["medium"], label="from parcels")
    a.plot(idf["t"] * 1e3, idf["theta_image_deg"], "-s", ms=3.5, color=COLORS["image"], label="from images")
    a.axhline(theta_ns, color=COLORS["ref"], ls=":", lw=1.4, label=f"Naber–Siebers {theta_ns:.1f}°")
    a.set_xlabel("time after start of injection (ms)")
    a.set_ylabel("spray full angle (°)")
    a.legend(loc="upper right")
    a.set_title("Spray angle: image vs. parcels", loc="left")
    add_panel_letter(a, "b")
    add_titles(fig, "Image pipeline against its own ground truth",
            "Frames rendered from the medium-mesh parcels; shaded band in (b) = evaluation window", 0.985)
    save_figure(fig, out)
