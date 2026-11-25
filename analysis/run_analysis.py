#!/usr/bin/env python3
"""Analyse the three spray cases and compare them with the Hiroyasu-Arai correlation.

Files written to the --out folder:
  summary.json, validation_table.md/.csv   all numbers quoted in the README
  metrics_<level>.csv                      parcel results for every time
  image_metrics.csv                        penetration and angle from the rendered frames
  figures/*.png                            the figures in the README
  frames/                                  rendered 16-bit frames (not saved in git)

How I set it up
- One script does everything: it reads each case once, calculates the results, renders the
  frames and makes the figures. I put the plotting in plotting.py so that this file is mainly
  about the numbers.
- The script stops if the injector values in params.json are null. I chose to stop and not
  only warn, because a correlation curve made from guessed values looks believable but is
  wrong.
- I average the errors from 0.2 to 1.0 ms. Before 0.2 ms there are only a few parcels and the
  injection is still ramping up (see figures/02_injection.png), so I expect start-up noise to
  dominate there. eval_window_s in params.json changes this window.
- For the mesh study I use one number per mesh, the mean S99 in the window. I could compare
  the full curves instead, but the GCI method of Celik et al. (2008) works on single values.
"""
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


def require_inputs(params):
    """Stop if an injector value is still missing."""
    missing = []
    for key in ("nozzle_diameter_m", "injection_velocity_m_s", "liquid_density_kg_m3"):
        if params["injector"].get(key) is None:
            missing.append(key)
    if missing:
        sys.exit("params.json: fill injector." + ", injector.".join(missing) +
                 " from constant/sprayCloudProperties of your case (see make_cases.py output). "
                 "The correlation comparison is meaningless with guessed values.")


def window_mask(t, window):
    # The small 1e-12 keeps the end times inside the window despite rounding errors
    return (t >= window[0] - 1e-12) & (t <= window[1] + 1e-12)


def percent_error(value, reference):
    return 100.0 * (value - reference) / reference


def analyse_parcels(params, cases_dir, levels, rho_a, out_dir):
    """Parcel results for every mesh level."""
    injector = params["injector"]
    series, mesh_info, frames_by_level = {}, {}, {}
    for level in levels:
        case = cases_dir / level
        frames_by_level[level] = cloudio.read_cloud(case)
        table = sm.parcel_timeseries(frames_by_level[level], injector["origin_m"], injector["direction"],
                                     params["metrics"])
        table["S_corr"] = corr.hiroyasu_arai(table["t"].values, injector["nozzle_diameter_m"],
                                             injector["injection_velocity_m_s"],
                                             injector["liquid_density_kg_m3"], rho_a)
        table.to_csv(out_dir / f"metrics_{level}.csv", index=False)
        series[level] = table
        mesh_info[level] = json.loads((case / "mesh_info.json").read_text())
    return series, mesh_info, frames_by_level


def compare_with_correlation(series, cells, levels, window, primary):
    """One row of errors against the correlation for every mesh."""
    rows = []
    for level in levels:
        data = series[level]
        selected = window_mask(data["t"].values, window) & data[primary].notna().values
        errors = percent_error(data.loc[selected, primary].values, data.loc[selected, "S_corr"].values)
        last = data[data[primary].notna()].iloc[-1]
        rows.append({
            "mesh": level,
            "cells": cells[level],
            f"mean_{primary}_mm": 1e3 * data.loc[selected, primary].mean(),
            "MAPE_vs_corr_pct": float(np.mean(np.abs(errors))),
            "bias_vs_corr_pct": float(np.mean(errors)),
            "err_at_last_time_pct": float(percent_error(last[primary], last["S_corr"])),
            "t_last_ms": 1e3 * float(last["t"]),
            "theta_parcel_deg_mean": float(data.loc[selected, "theta_parcel_deg"].mean()),
            "n_parcels_last": int(last["n_parcels"]),
        })
    return pd.DataFrame(rows)


def summarise_images(image_table, level, n_frames, pixel_size, window, primary):
    """Compare the image results with the parcel results and the correlation."""
    ok = window_mask(image_table["t"].values, window) & image_table["S_image"].notna().values
    s_image = image_table.loc[ok, "S_image"].values
    vs_parcel = percent_error(s_image, image_table.loc[ok, primary].values)
    vs_corr = percent_error(s_image, image_table.loc[ok, "S_corr"].values)
    vs_s_max = percent_error(s_image, image_table.loc[ok, "S_max"].values)
    return {
        "mesh_level": level,
        "n_frames": n_frames,
        "n_frames_eval": int(ok.sum()),
        "pixel_size_um": 1e6 * pixel_size,
        "image_vs_parcel_mean_abs_pct": float(np.mean(np.abs(vs_parcel))),
        "image_vs_parcel_bias_pct": float(np.mean(vs_parcel)),
        "image_vs_Smax_mean_pct": float(np.mean(vs_s_max)),
        "image_vs_corr_mean_abs_pct": float(np.mean(np.abs(vs_corr))),
        "theta_image_deg_mean": float(image_table.loc[ok, "theta_image_deg"].mean()),
        "theta_parcel_deg_mean": float(image_table.loc[ok, "theta_parcel_deg"].mean()),
    }


def to_json_values(gci_result):
    """json cannot save NumPy types, so I convert them to normal Python types."""
    converted = {}
    for key, value in gci_result.items():
        if value is None:
            converted[key] = None
        elif isinstance(value, (bool, np.bool_)):
            converted[key] = bool(value)
        else:
            converted[key] = float(value)
    return converted


def write_markdown(table, gci_result, image_summary, summary, out_dir, primary):
    lines = [f"| Mesh | Cells | mean {primary} (mm) | MAPE vs correlation (%) | bias (%) | error at t_end (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in table.to_dict(orient="records"):
        lines.append(f"| {row['mesh']} | {row['cells']:,} | {row[f'mean_{primary}_mm']:.2f} | "
                     f"{row['MAPE_vs_corr_pct']:.1f} | {row['bias_vs_corr_pct']:+.1f} | "
                     f"{row['err_at_last_time_pct']:+.1f} |")
    lines.append("")

    if gci_result["monotone"]:
        text = (f"Grid convergence: monotone, observed order p = {gci_result['order']:.2f}, "
                f"GCI (fine) = {gci_result['gci_fine_pct']:.1f} %; with the formal order p = 2: "
                f"GCI (fine) = {gci_result['gci_fine_pct_formal']:.1f} %.")
        if not gci_result["order_plausible"]:
            text += " The observed order is implausibly high; use the p = 2 value."
    else:
        text = (f"Grid convergence: NOT monotone; spread between the three meshes = "
                f"{gci_result['spread_pct']:.1f} % (no order or GCI computed; typical for Lagrangian spray models).")
    lines.append(text)

    if image_summary:
        lines.append(f"Image pipeline ({image_summary['mesh_level']} mesh, {image_summary['n_frames_eval']} of "
                     f"{image_summary['n_frames']} frames in the window, "
                     f"{image_summary['pixel_size_um']:.0f} um/px): penetration differs from the parcel-based "
                     f"{primary} by {image_summary['image_vs_parcel_mean_abs_pct']:.1f} % on average "
                     f"(bias {image_summary['image_vs_parcel_bias_pct']:+.1f} %).")

    lines.append(f"Hiroyasu-Arai break-up time: {summary['break_up_time_us']:.1f} us; "
                 f"air density {summary['rho_air_kg_m3']:.1f} kg/m3.")
    (out_dir / "validation_table.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.json")
    parser.add_argument("--cases", default="cases")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    params = json.loads(Path(args.params).read_text())
    require_inputs(params)
    cases_dir = Path(args.cases)
    out_dir = Path(args.out)
    figure_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.apply_style()

    injector = params["injector"]
    ambient = params["ambient"]
    metrics = params["metrics"]
    window = metrics["eval_window_s"]
    primary = f"S{metrics['primary_percentile']}"
    rho_a = corr.gas_density(ambient["pressure_Pa"], ambient["temperature_K"], ambient["gas_constant_J_per_kgK"])
    levels = list(params["mesh_levels"])  # coarse, medium, fine (the GCI needs this order)

    # 1. Parcel results for every mesh
    series, mesh_info, frames_by_level = analyse_parcels(params, cases_dir, levels, rho_a, out_dir)
    cells = {}
    for level in levels:
        cells[level] = mesh_info[level]["n_cells"]

    # 2. Errors against the correlation
    table = compare_with_correlation(series, cells, levels, window, primary)
    table.to_csv(out_dir / "validation_table.csv", index=False)

    # 3. Mesh convergence of the mean value in the window
    mean_by_level = {}
    for level in levels:
        in_window = window_mask(series[level]["t"].values, window)
        mean_by_level[level] = series[level].loc[in_window, primary].mean()
    coarse, medium, fine = levels
    gci_result = corr.gci(mean_by_level[fine], mean_by_level[medium], mean_by_level[coarse],
                          cells[fine], cells[medium], cells[coarse])

    # 4. Render frames from the parcels and analyse them (medium mesh by default)
    image_summary = None
    image_table = None
    if params["images"]["enabled"]:
        level = params["images"]["mesh_level"]
        pixel_size = params["images"]["pixel_size_m"]
        frames = frames_by_level[level]
        images, bg = ri.render_sequence(frames, injector["origin_m"], injector["direction"], params["images"])
        times = [frame["t"] for frame in frames]
        ri.save_sequence(images, bg, times, out_dir / "frames")

        image_table, masks = ip.analyse_sequence(images, bg, times, pixel_size)
        parcel_columns = series[level][["t", primary, "S_max", "theta_parcel_deg", "S_corr"]]
        image_table = image_table.merge(parcel_columns, on="t")
        image_table.to_csv(out_dir / "image_metrics.csv", index=False)
        image_summary = summarise_images(image_table, level, len(images), pixel_size, window, primary)

    theta_ns = corr.naber_siebers_full_angle_deg(rho_a, injector["liquid_density_kg_m3"], metrics["naber_siebers_c"])

    # 5. Figures
    injection_table = cloudio.read_injection(cases_dir / medium)
    pl.plot_setup(params, mesh_info, injection_table, figure_dir / "01_setup.png")
    injection_stats = pl.plot_injection(params, injection_table, figure_dir / "02_injection.png")
    pl.plot_parcels(frames_by_level, series, params, mesh_info, primary, figure_dir / "03_parcels.png")
    pl.plot_penetration(series, params, image_table, primary, figure_dir / "04_penetration.png")
    pl.plot_convergence(mean_by_level, cells, series, table, gci_result, params, primary, theta_ns,
                        injection_table["thetaOuter"], figure_dir / "05_mesh_convergence.png")
    if image_table is not None:
        pl.plot_pipeline_steps(images, bg, times, image_table, params, figure_dir / "06_pipeline_steps.png", primary)
        pl.plot_frame_sequence(images, masks, times, image_table, params, figure_dir / "07_frame_sequence.png")
        pl.plot_image_validation(image_table, params, primary, theta_ns, figure_dir / "08_image_validation.png")

    # 6. Summary files
    break_up = corr.break_up_time(injector["nozzle_diameter_m"], injector["injection_velocity_m_s"],
                                  injector["liquid_density_kg_m3"], rho_a)
    summary = {
        "mode": params["mode"],
        "primary_metric": primary,
        "eval_window_ms": [1e3 * window[0], 1e3 * window[1]],
        "rho_air_kg_m3": rho_a,
        "break_up_time_us": 1e6 * break_up,
        "naber_siebers_full_angle_deg": theta_ns,
        "injection": injection_stats,
        "validation": table.round(3).to_dict(orient="records"),
        "gci": to_json_values(gci_result),
        "images": image_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(table, gci_result, image_summary, summary, out_dir, primary)
    print((out_dir / "validation_table.md").read_text())


if __name__ == "__main__":
    main()
