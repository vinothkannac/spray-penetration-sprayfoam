#!/usr/bin/env python3
"""Post-process the three mesh-level spray cases and compare them with the Hiroyasu-Arai correlation.

Outputs in --out:
  summary.json, validation_table.md/.csv   all numbers quoted in the README
  metrics_<level>.csv                      per-time parcel metrics (S_max, S95, S99, spray angle)
  image_metrics.csv                        penetration and angle extracted from the rendered frames
  figures/*.png                            the figures used in the README
  frames/                                  rendered 16-bit frames (not versioned)

Design notes
- One script, one pass: it reads each case once, computes the metrics, renders the frames, and
  draws every figure. I kept the plotting in plotting.py so that this file stays about the numbers.
- The script refuses to run while the injector values in params.json are null. I added this
  after realising that a Hiroyasu-Arai curve built from guessed inputs looks plausible and is
  wrong, so a hard stop is safer than a warning.
- Evaluation window. I average the errors over 0.2-1.0 ms. Before 0.2 ms the Lagrangian cloud
  has only a few parcels and the injection ramps up (see figures/02_injection.png), so the
  percentile is dominated by start-up noise. Change `eval_window_s` in params.json to see how
  sensitive the table is to this choice.
- Mesh sensitivity uses the mean of S99 over the window, one number per mesh. The alternative is
  to compare the full curves (for example with an L2 norm); I preferred one scalar because the
  GCI procedure of Celik et al. (2008) is defined for scalars.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloudio
import correlations as corr
import image_pipeline as ip
import plotting as pl
import render_images as ri
import spray_metrics as sm


def require_inputs(params: dict) -> None:
    missing = [k for k in ("nozzle_diameter_m", "injection_velocity_m_s", "liquid_density_kg_m3")
               if params["injector"].get(k) is None]
    if missing:
        sys.exit("params.json: fill injector." + ", injector.".join(missing) +
                 " from constant/sprayCloudProperties of your case (see make_cases.py output). "
                 "The correlation comparison is meaningless with guessed values.")


def window_mask(t: np.ndarray, window) -> np.ndarray:
    return (t >= window[0] - 1e-12) & (t <= window[1] + 1e-12)


def pct_err(sim: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return 100.0 * (sim - ref) / ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.json")
    ap.add_argument("--cases", default="cases")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    params = json.loads(Path(a.params).read_text())
    require_inputs(params)
    out = Path(a.out)
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pl.apply_style()

    inj, amb, met = params["injector"], params["ambient"], params["metrics"]
    rho_a = corr.gas_density(amb["pressure_Pa"], amb["temperature_K"], amb["gas_constant_J_per_kgK"])
    prim = f"S{met['primary_percentile']}"
    levels = list(params["mesh_levels"].keys())  # coarse, medium, fine (the order matters for the GCI)

    # ---- parcel metrics per mesh level
    series, mesh_info, frames_by_level = {}, {}, {}
    for lv in levels:
        case = Path(a.cases) / lv
        frames_by_level[lv] = cloudio.read_cloud(case)
        df = sm.parcel_timeseries(frames_by_level[lv], inj["origin_m"], inj["direction"], met)
        df["S_corr"] = corr.hiroyasu_arai(df["t"].values, inj["nozzle_diameter_m"],
                                          inj["injection_velocity_m_s"], inj["liquid_density_kg_m3"], rho_a)
        df.to_csv(out / f"metrics_{lv}.csv", index=False)
        series[lv] = df
        mesh_info[lv] = json.loads((case / "mesh_info.json").read_text())
    cells = {lv: mesh_info[lv]["n_cells"] for lv in levels}

    # ---- error against the correlation
    rows, window = [], met["eval_window_s"]
    for lv in levels:
        df = series[lv]
        sel = window_mask(df["t"].values, window) & df[prim].notna().values
        e = pct_err(df.loc[sel, prim].values, df.loc[sel, "S_corr"].values)
        last = df[df[prim].notna()].iloc[-1]
        rows.append({
            "mesh": lv, "cells": cells[lv],
            f"mean_{prim}_mm": 1e3 * df.loc[sel, prim].mean(),
            "MAPE_vs_corr_pct": float(np.mean(np.abs(e))),
            "bias_vs_corr_pct": float(np.mean(e)),
            "err_at_last_time_pct": float(pct_err(last[prim], last["S_corr"])),
            "t_last_ms": 1e3 * float(last["t"]),
            "theta_parcel_deg_mean": float(df.loc[sel, "theta_parcel_deg"].mean()),
            "n_parcels_last": int(last["n_parcels"]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out / "validation_table.csv", index=False)

    # ---- mesh convergence of the window-mean primary metric (fine, medium, coarse)
    f = {lv: series[lv].loc[window_mask(series[lv]["t"].values, window), prim].mean() for lv in levels}
    g = corr.gci(f[levels[2]], f[levels[1]], f[levels[0]], cells[levels[2]], cells[levels[1]], cells[levels[0]])

    # ---- images from the rendered parcels (medium mesh by default)
    img_summary, idf, images, masks, times, bg = None, None, None, None, None, None
    if params["images"]["enabled"]:
        lv = params["images"]["mesh_level"]
        frames = frames_by_level[lv]
        images, bg = ri.render_sequence(frames, inj["origin_m"], inj["direction"], params["images"])
        times = [fr["t"] for fr in frames]
        ri.save_sequence(images, bg, times, out / "frames")
        idf, masks = ip.analyse_sequence(images, bg, times, params["images"]["pixel_size_m"])
        idf = idf.merge(series[lv][["t", prim, "S_max", "theta_parcel_deg", "S_corr"]], on="t")
        idf.to_csv(out / "image_metrics.csv", index=False)
        ok = window_mask(idf["t"].values, window) & idf["S_image"].notna().values
        d_img_vs_parcel = pct_err(idf.loc[ok, "S_image"].values, idf.loc[ok, prim].values)
        d_img_vs_corr = pct_err(idf.loc[ok, "S_image"].values, idf.loc[ok, "S_corr"].values)
        img_summary = {
            "mesh_level": lv, "n_frames": len(images), "pixel_size_um": 1e6 * params["images"]["pixel_size_m"],
            "image_vs_parcel_mean_abs_pct": float(np.mean(np.abs(d_img_vs_parcel))),
            "image_vs_parcel_bias_pct": float(np.mean(d_img_vs_parcel)),
            "image_vs_Smax_mean_pct": float(np.mean(pct_err(idf.loc[ok, "S_image"].values, idf.loc[ok, "S_max"].values))),
            "image_vs_corr_mean_abs_pct": float(np.mean(np.abs(d_img_vs_corr))),
            "theta_image_deg_mean": float(idf.loc[ok, "theta_image_deg"].mean()),
            "theta_parcel_deg_mean": float(idf.loc[ok, "theta_parcel_deg"].mean()),
        }

    theta_ns = corr.naber_siebers_half_angle_deg(rho_a, inj["liquid_density_kg_m3"], met["naber_siebers_c"])

    # ---- figures
    inj_tbl = cloudio.read_injection(Path(a.cases) / levels[1])
    pl.plot_setup(params, mesh_info, inj_tbl, figs / "01_setup.png")
    inj_stats = pl.plot_injection(params, inj_tbl, figs / "02_injection.png")
    pl.plot_parcels(frames_by_level, series, params, mesh_info, prim, figs / "03_parcels.png")
    pl.plot_penetration(series, params, idf, prim, figs / "04_penetration.png")
    pl.plot_convergence(f, cells, series, table, g, params, prim, theta_ns, inj_tbl["thetaOuter"],
                        figs / "05_mesh_convergence.png")
    if idf is not None:
        pl.plot_pipeline_steps(images, bg, times, idf, params, figs / "06_pipeline_steps.png", prim)
        pl.plot_frame_sequence(images, masks, times, idf, params, figs / "07_frame_sequence.png")
        pl.plot_image_validation(idf, params, prim, theta_ns, figs / "08_image_validation.png")

    summary = {
        "mode": params["mode"], "primary_metric": prim, "eval_window_ms": [1e3 * w for w in window],
        "rho_air_kg_m3": rho_a,
        "break_up_time_us": 1e6 * corr.break_up_time(inj["nozzle_diameter_m"], inj["injection_velocity_m_s"],
                                                      inj["liquid_density_kg_m3"], rho_a),
        "naber_siebers_full_angle_deg": theta_ns,
        "injection": inj_stats,
        "validation": table.round(3).to_dict(orient="records"),
        "gci": {k: (None if v is None else (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)))
                for k, v in g.items()},
        "images": img_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(table, g, img_summary, summary, out, prim)
    print((out / "validation_table.md").read_text())


def _write_markdown(table, g, img, summary, out: Path, prim: str) -> None:
    lines = ["| Mesh | Cells | mean " + prim + " (mm) | MAPE vs correlation (%) | bias (%) | error at t_end (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in table.to_dict(orient="records"):
        lines.append(f"| {r['mesh']} | {r['cells']:,} | {r[f'mean_{prim}_mm']:.2f} | {r['MAPE_vs_corr_pct']:.1f} | "
                     f"{r['bias_vs_corr_pct']:+.1f} | {r['err_at_last_time_pct']:+.1f} |")
    lines.append("")
    if g["monotone"]:
        lines.append(f"Grid convergence: monotone, observed order p = {g['order']:.2f}, GCI (fine) = "
                     f"{g['gci_fine_pct']:.1f} %; with the formal order p = 2: GCI (fine) = {g['gci_fine_pct_formal']:.1f} %."
                     + ("" if g["order_plausible"] else " The observed order is implausibly high; use the p = 2 value."))
    else:
        lines.append(f"Grid convergence: NOT monotone; spread between the three meshes = {g['spread_pct']:.1f} % "
                     "(no order or GCI computed; typical for Lagrangian spray models).")
    if img:
        lines.append(f"Image pipeline ({img['mesh_level']} mesh, {img['n_frames']} frames, "
                     f"{img['pixel_size_um']:.0f} um/px): penetration differs from the parcel-based {prim} by "
                     f"{img['image_vs_parcel_mean_abs_pct']:.1f} % on average (bias {img['image_vs_parcel_bias_pct']:+.1f} %).")
    lines.append(f"Hiroyasu-Arai break-up time: {summary['break_up_time_us']:.1f} us; "
                 f"air density {summary['rho_air_kg_m3']:.1f} kg/m3.")
    (out / "validation_table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
