#!/usr/bin/env python3
"""Replot a saved Tiberius Fleck fit with an interpretable model decomposition."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def read_contact_indices(input_path: Path) -> tuple[int, int]:
    values: dict[str, int] = {}
    for raw_line in input_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in {"contact1", "contact4"} and value:
            values[key] = int(value)
    return values["contact1"], values["contact4"]


def current_value(parameter) -> float:
    return float(getattr(parameter, "currVal", parameter))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot an existing Fleck fit without rerunning the MCMC."
    )
    parser.add_argument("run_dir", type=Path, help="Saved white-light Fleck fit directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF (default: RUN_DIR/fitted_model_wb0001_components_corrected.pdf)",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure after saving")
    parser.add_argument(
        "--colour-blind",
        "--color-blind",
        dest="colour_blind",
        action="store_true",
        help="Use an Okabe-Ito colour-blind-safe model palette",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    default_name = (
        "fitted_model_wb0001_components_colour_blind.pdf"
        if args.colour_blind
        else "fitted_model_wb0001_components_corrected.pdf"
    )
    output = args.output or run_dir / default_name
    combined_colour = "#D55E00" if args.colour_blind else "r"
    systematics_colour = "#0072B2" if args.colour_blind else "g"

    model = load_pickle(run_dir / "prod_model_wb0001.pickle")
    time = np.asarray(load_pickle(run_dir / "sigma_clipped_time_wb0001.pickle"), dtype=float)
    flux = np.asarray(load_pickle(run_dir / "sigma_clipped_flux_wb0001.pickle"), dtype=float)
    error = np.asarray(load_pickle(run_dir / "rescaled_errors_wb0001.pickle"), dtype=float)
    model_inputs = np.asarray(
        load_pickle(run_dir / "sigma_clipped_model_inputs_wb0001.pickle"), dtype=float
    )

    combined_model = np.asarray(model.calc(time, model_inputs), dtype=float)
    systematics = np.asarray(model.red_noise_poly(time, model_inputs), dtype=float)
    spotted_transit = combined_model / systematics

    contact1, contact4 = read_contact_indices(run_dir / "fitting_input.txt")
    exposure_index = np.arange(time.size)
    out_of_transit = (exposure_index < contact1) | (exposure_index > contact4)
    spotted_baseline = float(np.median(spotted_transit[out_of_transit]))

    # Move the constant spotted-star baseline between the two plotted factors.
    # Their product remains exactly equal to the fitted combined model.
    displayed_systematics = systematics * spotted_baseline
    displayed_spotted_transit = spotted_transit / spotted_baseline

    t0 = current_value(model.pars["t0"])
    hours = (time - t0) * 24.0
    residuals_ppm = (flux - combined_model) * 1.0e6

    # Match the dimensions and visual styling of the standard Tiberius
    # fitted_model_wb0001.pdf output.
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(8.0, 4.5056),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.04},
    )

    ax.errorbar(
        hours,
        flux,
        error,
        fmt="o",
        ms=4,
        color="k",
        ecolor="k",
        capsize=0,
        alpha=0.5,
        zorder=15
    )
    ax.plot(
        hours,
        combined_model,
        color=combined_colour,
        lw=1.5,
        zorder=10,
    )
    ax.plot(
        hours,
        displayed_systematics,
        color=systematics_colour,
        lw=1.5,
    )
    ax.plot(
        hours,
        displayed_spotted_transit,
        color="0.75",
        ls="--",
        lw=1.5,
        zorder=9,
    )
    ax.set_ylabel("Normalised flux", fontsize=14)

    residual_ax.errorbar(
        hours,
        residuals_ppm,
        error * 1.0e6,
        fmt="o",
        ms=4,
        color="k",
        ecolor="k",
        capsize=0,
        alpha=0.5,
    )
    residual_ax.axhline(0.0, color="k", ls="--")
    residual_ax.set_ylabel("Residuals (ppm)", fontsize=14)
    residual_ax.set_xlabel("Time from mid-transit (hours)", fontsize=14)

    for axis in (ax, residual_ax):
        axis.tick_params(bottom=True, top=True, left=True, right=True, direction="inout")
        axis.tick_params(which="minor", bottom=True, top=True, left=True, right=True, direction="inout")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=500)
    print(f"Saved {output}")
    print(f"Applied spotted-star baseline factor: {spotted_baseline:.8f}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
