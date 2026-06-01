import argparse
import re
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


plt.rcParams["image.origin"] = "lower"

DIAG_NAME_RE = re.compile(r"frame_(\d+)_star_(\d+)\.npz$")


def gaussian_profile(x, amplitude, mean, sigma, offset):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset


def moffat_profile(x, amplitude, mean, alpha, beta, offset):
    return amplitude * (1.0 + ((x - mean) / alpha) ** 2) ** (-beta) + offset

def double_gaussian_profile(x, amplitude, mean, sigma_core, sigma_wing, wing_fraction, offset):
    wing_fraction = np.clip(wing_fraction, 0.0, 0.95)
    core_fraction = 1.0 - wing_fraction
    return amplitude * (
        core_fraction * np.exp(-0.5 * ((x - mean) / sigma_core) ** 2)
        + wing_fraction * np.exp(-0.5 * ((x - mean) / sigma_wing) ** 2)
    ) + offset

def evaluate_profile(record, row_index, x, include_offset=True):
    amplitude = float(record["profile_amplitude"][row_index])
    centre = float(record["profile_centre"][row_index])
    scale = float(record["profile_scale"][row_index])
    offset = float(record["profile_offset"][row_index]) if include_offset else 0.0

    if record["profile_model"] == "empirical":
        return None
    if record["profile_model"] == "gaussian":
        return gaussian_profile(x, amplitude, centre, scale, offset)
    if record["profile_model"] == "moffat":
        beta = float(record["profile_beta"][row_index])
        return moffat_profile(x, amplitude, centre, scale, beta, offset)
    if record["profile_model"] == "double_gaussian":
        sigma_wing = float(record.get("profile_scale_secondary", np.full_like(record["profile_scale"], np.nan))[row_index])
        wing_fraction = float(record.get("profile_mix_fraction", np.full_like(record["profile_scale"], np.nan))[row_index])
        return double_gaussian_profile(x, amplitude, centre, scale, sigma_wing, wing_fraction, offset)
    raise ValueError("Unsupported profile model '%s'" % record["profile_model"])


def get_profile_plot_grid(record, x_left, x_right):
    oversampling = 1
    if "profile_fit_oversampling" in record:
        try:
            oversampling = max(int(record["profile_fit_oversampling"]), 1)
        except Exception:
            oversampling = 1

    x_left = float(x_left)
    x_right = float(x_right)
    if x_right <= x_left:
        return np.array([x_left], dtype=float)

    if oversampling <= 1:
        return np.arange(int(np.floor(x_left)), int(np.ceil(x_right)) + 1, dtype=float)

    count = max(int(np.ceil((x_right - x_left) * oversampling)) + 1, 2)
    return np.linspace(x_left, x_right, count)


def compute_data_ylim(*series_groups, padding_frac=0.08, min_pad=1.0):
    values = []
    for x_values, y_values, x_left, x_right in series_groups:
        if x_values is None or y_values is None:
            continue
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values >= x_left) & (x_values <= x_right)
        if np.any(mask):
            values.append(y_values[mask])

    if len(values) == 0:
        return None

    y = np.concatenate(values)
    if len(y) == 0:
        return None

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    span = max(y_max - y_min, min_pad)
    pad = max(min_pad, padding_frac * span)
    return y_min - pad, y_max + pad


def parse_rows(row_text):
    if row_text is None:
        return None
    return [int(x.strip()) for x in row_text.split(",") if x.strip() != ""]


def resolve_row_index(requested_row, row_numbers):
    if requested_row is None:
        return len(row_numbers) // 2

    matches = np.where(row_numbers == requested_row)[0]
    if len(matches) > 0:
        return int(matches[0])

    if 0 <= requested_row < len(row_numbers):
        return int(requested_row)

    raise ValueError("Requested row %s is outside the diagnostic file." % requested_row)


def resolve_input_path(input_path):
    path = Path(input_path).expanduser()
    if path.is_dir():
        if list(path.glob("frame_*_star_*.npz")):
            return path
        nested = path / "pickled_objects" / "extraction_diagnostics"
        if nested.is_dir():
            return nested
    return path


def parse_diag_name(path):
    match = DIAG_NAME_RE.match(path.name)
    if match is None:
        raise ValueError("Diagnostic filename '%s' does not match frame/star naming." % path.name)
    return int(match.group(1)), int(match.group(2))


def load_record(path):
    with np.load(path, allow_pickle=True) as diag:
        record = {key: diag[key] for key in diag.files}

    record["path"] = str(path)
    record["frame_index"] = int(record["frame_index"].item())
    record["star_index"] = int(record["star_index"].item())
    record["science_frame"] = str(record["science_frame"].item())
    record["profile_enabled"] = bool(record["profile_enabled"].item())
    record["profile_model"] = str(record["profile_model"].item())
    record["region_aperture"] = int(record["region_aperture"])
    record["region_background_used"] = int(record["region_background_used"])
    record["region_background_rejected"] = int(record["region_background_rejected"])
    record["region_contaminant"] = int(record["region_contaminant"])
    record["region_profile_fit"] = int(record["region_profile_fit"])
    return record


def resolve_science_frame_path(record):
    frame_path = Path(record["science_frame"]).expanduser()
    if frame_path.is_absolute() and frame_path.exists():
        return frame_path

    candidates = [Path.cwd() / frame_path]
    diag_path = Path(record["path"]).expanduser()
    if len(diag_path.parents) >= 3:
        reduction_dir = diag_path.parents[2]
        candidates.append(reduction_dir / frame_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return frame_path


@lru_cache(maxsize=None)
def get_frame_time_value(frame_path_str):
    frame_path = Path(frame_path_str).expanduser()
    try:
        header = fits.getheader(frame_path, 0)
    except Exception:
        return np.nan

    for key in ("MJD-OBS", "MJD_OBS", "MJD", "BJD_TDB", "BJD", "JD"):
        if key in header:
            try:
                return float(header[key])
            except Exception:
                continue
    return np.nan


def build_frame_groups(input_path):
    path = resolve_input_path(input_path)

    if path.is_file():
        frame_index, _ = parse_diag_name(path)
        diag_dir = path.parent
        files = sorted(diag_dir.glob("frame_*_star_*.npz"))
        groups = {}
        for diag_file in files:
            diag_frame, diag_star = parse_diag_name(diag_file)
            groups.setdefault(diag_frame, {})[diag_star] = diag_file
        return [groups[frame_index]]

    if not path.is_dir():
        raise FileNotFoundError("%s is not a diagnostic file or directory." % path)

    files = sorted(path.glob("frame_*_star_*.npz"))
    if len(files) == 0:
        raise FileNotFoundError("No diagnostic .npz files found in %s" % path)

    groups = {}
    for diag_file in files:
        frame_index, star_index = parse_diag_name(diag_file)
        groups.setdefault(frame_index, {})[star_index] = diag_file

    return [groups[key] for key in sorted(groups.keys())]


def load_frame_records(frame_group):
    return [load_record(frame_group[key]) for key in sorted(frame_group.keys())]


def build_row_context(record, requested_row):
    row_index = resolve_row_index(requested_row, record["row_numbers"])
    x = np.arange(record["frame_data"].shape[1])
    row = record["frame_data"][row_index]
    row_background = record["background_model"][row_index]
    finite_background = np.isfinite(row_background)

    row_background_subtracted = row.copy()
    row_background_subtracted[finite_background] = row[finite_background] - row_background[finite_background]

    row_flags = record["region_flags"][row_index]
    aperture_mask = (row_flags & record["region_aperture"]) > 0
    background_used_mask = (row_flags & record["region_background_used"]) > 0
    background_rejected_mask = (row_flags & record["region_background_rejected"]) > 0
    contaminant_mask = (row_flags & record["region_contaminant"]) > 0
    profile_fit_mask = (row_flags & record["region_profile_fit"]) > 0

    ap_left = int(record["aperture_left"][row_index])
    ap_right = int(record["aperture_right"][row_index])
    bg_left_start = int(record["background_left_start"][row_index])
    bg_left_end = int(record["background_left_end"][row_index])
    bg_right_start = int(record["background_right_start"][row_index])
    bg_right_end = int(record["background_right_end"][row_index])
    fit_left = int(record["profile_fit_left"][row_index]) if "profile_fit_left" in record else -1
    fit_right = int(record["profile_fit_right"][row_index]) if "profile_fit_right" in record else -1

    valid_bounds = [ap_left, ap_right, bg_left_start, bg_left_end, bg_right_start, bg_right_end]
    valid_bounds = [value for value in valid_bounds if value >= 0]
    if len(valid_bounds) > 0:
        roi_left = min(valid_bounds)
        roi_right = max(valid_bounds)
    else:
        roi_left = 0
        roi_right = len(x) - 1

    roi_pad = max(2, int(np.ceil(0.05 * max(roi_right - roi_left, 1))))
    x_left = max(0, roi_left - roi_pad)
    x_right = min(len(x) - 1, roi_right + roi_pad)

    if fit_left >= 0 and fit_right > fit_left:
        zoom_left = max(0, fit_left - roi_pad)
        zoom_right = min(len(x) - 1, fit_right + roi_pad)
    else:
        zoom_left = max(0, ap_left - roi_pad)
        zoom_right = min(len(x) - 1, ap_right + roi_pad)

    ignored_spans = []
    if fit_left >= 0 and fit_right > fit_left:
        if fit_left < ap_left:
            ignored_spans.append((fit_left, min(ap_left, fit_right)))
        if fit_right > ap_right:
            ignored_spans.append((max(ap_right, fit_left), fit_right))

    profile_y = None
    source_y = None
    profile_x = None
    source_x = None
    if record["profile_enabled"] and bool(record["profile_success"][row_index]):
        profile_x = get_profile_plot_grid(record, x_left, x_right)
        source_x = get_profile_plot_grid(record, zoom_left, zoom_right)
        profile_y = evaluate_profile(record, row_index, profile_x, include_offset=True)
        source_y = evaluate_profile(record, row_index, source_x, include_offset=False)

    trace_centre = float(record["trace"][row_index])
    core_exclusion_half_width = float(record["profile_core_exclusion_half_width"]) if "profile_core_exclusion_half_width" in record else 0.0
    fit_peak_region_left = fit_left if fit_left >= 0 else ap_left
    fit_peak_region_right = fit_right if fit_right > fit_peak_region_left else ap_right
    peak_slice = row_background_subtracted[fit_peak_region_left:fit_peak_region_right]
    if len(peak_slice) == 0 or not np.any(np.isfinite(peak_slice)):
        peak_slice = row_background_subtracted[max(ap_left, 0):max(ap_right, 0)]
    norm = np.nanmax(peak_slice) if len(peak_slice) > 0 else np.nan
    if not np.isfinite(norm) or norm <= 0:
        norm = 1.0

    return {
        "record": record,
        "row_index": row_index,
        "detector_row": int(record["row_numbers"][row_index]),
        "x": x,
        "row": row,
        "row_background": row_background,
        "row_background_subtracted": row_background_subtracted,
        "finite_background": finite_background,
        "aperture_mask": aperture_mask,
        "background_used_mask": background_used_mask,
        "background_rejected_mask": background_rejected_mask,
        "contaminant_mask": contaminant_mask,
        "profile_fit_mask": profile_fit_mask,
        "ap_left": ap_left,
        "ap_right": ap_right,
        "bg_left_start": bg_left_start,
        "bg_left_end": bg_left_end,
        "bg_right_start": bg_right_start,
        "bg_right_end": bg_right_end,
        "fit_left": fit_left,
        "fit_right": fit_right,
        "ignored_spans": ignored_spans,
        "profile_x": profile_x,
        "profile_y": profile_y,
        "source_x": source_x,
        "source_y": source_y,
        "trace_centre": trace_centre,
        "core_exclusion_half_width": core_exclusion_half_width,
        "x_left": x_left,
        "x_right": x_right,
        "zoom_left": zoom_left,
        "zoom_right": zoom_right,
        "norm": norm,
    }


def build_profile_summary_lines(context):
    record = context["record"]
    row_index = context["row_index"]
    aperture_width = context["ap_right"] - context["ap_left"]
    enclosed_width = float(2.0 * record["profile_radius"][row_index]) if "profile_radius" in record else np.nan
    fwhm = float(record["profile_fwhm"][row_index]) if "profile_fwhm" in record else np.nan
    trace_centre = float(context["trace_centre"])

    lines = [
        "centre = %.2f px" % trace_centre,
        "aperture = [%d, %d) width = %d px" % (context["ap_left"], context["ap_right"], aperture_width),
    ]

    if record["profile_enabled"] and bool(record["profile_success"][row_index]):
        if record["profile_model"] == "empirical":
            lines.append("half-max width = %.2f px" % fwhm)
        else:
            lines.append("FWHM = %.2f px" % fwhm)
        if np.isfinite(enclosed_width):
            percentile = float(record.get("profile_percentile", np.array(np.nan)).item()) if np.ndim(record.get("profile_percentile", np.array(np.nan))) == 0 else float(record["profile_percentile"])
            lines.append("enclosed width = %.2f px (%.1f%%)" % (enclosed_width, percentile * 100.0))

        model = record["profile_model"]
        if model == "gaussian":
            lines.append("sigma = %.2f px" % float(record["profile_scale"][row_index]))
        elif model == "moffat":
            lines.append(
                "alpha = %.2f px, beta = %.2f"
                % (float(record["profile_scale"][row_index]), float(record["profile_beta"][row_index]))
            )
        elif model == "double_gaussian":
            lines.append(
                "sigma_core = %.2f px, sigma_wing = %.2f px"
                % (
                    float(record["profile_scale"][row_index]),
                    float(record["profile_scale_secondary"][row_index]),
                )
            )
            lines.append("wing fraction = %.3f" % float(record["profile_mix_fraction"][row_index]))
        elif model == "empirical":
            lines.append("empirical cumulative-light aperture")

        if "profile_core_exclusion_half_width" in record:
            core_exclusion = float(record["profile_core_exclusion_half_width"])
            if np.isfinite(core_exclusion) and core_exclusion > 0:
                lines.append("core excluded = %.2f px" % core_exclusion)
        if "profile_max_width" in record:
            max_width = float(record["profile_max_width"])
            if np.isfinite(max_width) and max_width > 0:
                lines.append("max width = %.2f px" % max_width)
        if "profile_fit_oversampling" in record:
            fit_oversampling = int(record["profile_fit_oversampling"])
            if fit_oversampling > 1:
                lines.append("fit oversampling = %dx" % fit_oversampling)
    else:
        lines.append("profile fit unavailable")

    return lines


def add_profile_summary_box(ax, context):
    text = "\n".join(build_profile_summary_lines(context))
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "0.8", "boxstyle": "round,pad=0.35"},
    )


def shade_region(ax, start, end, color, alpha, label=None):
    if start >= 0 and end > start:
        ax.axvspan(start, end, color=color, alpha=alpha, label=label)


def plot_frame_panel(ax, context):
    record = context["record"]
    frame_data = record["frame_data"]
    vmin, vmax = np.nanpercentile(frame_data, [5, 95])
    ax.imshow(frame_data, vmin=vmin, vmax=vmax, aspect="auto")
    ax.plot(record["trace"], np.arange(len(record["trace"])), color="white", lw=0.9, label="trace")
    ax.plot(record["aperture_left"], np.arange(len(record["aperture_left"])), color="tab:red", lw=0.9, label="aperture")
    ax.plot(record["aperture_right"], np.arange(len(record["aperture_right"])), color="tab:red", lw=0.9)
    ax.plot(record["background_left_end"], np.arange(len(record["background_left_end"])), color="tab:cyan", lw=0.8, ls="--", label="background")
    ax.plot(record["background_right_start"], np.arange(len(record["background_right_start"])), color="tab:cyan", lw=0.8, ls="--")
    if np.any(record["background_left_start"] >= 0):
        ax.plot(record["background_left_start"], np.arange(len(record["background_left_start"])), color="tab:cyan", lw=0.8, ls="--")
    if np.any(record["background_right_end"] >= 0):
        ax.plot(record["background_right_end"], np.arange(len(record["background_right_end"])), color="tab:cyan", lw=0.8, ls="--")
    ax.axhline(context["row_index"], color="yellow", lw=0.9, ls=":")
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Local Y row")
    ax.set_title("Star %d frame view" % record["star_index"])
    ax.legend(loc="upper right", fontsize=8, framealpha=1)


def plot_row_panel(ax, context):
    record = context["record"]
    x = context["x"]
    row = context["row"]
    background_fit = context["row_background"]

    shade_region(ax, context["bg_left_start"], context["bg_left_end"], "tab:cyan", 0.10, "background")
    shade_region(ax, context["bg_right_start"], context["bg_right_end"], "tab:cyan", 0.10)
    for i, (start, end) in enumerate(context["ignored_spans"]):
        shade_region(ax, start, end, "0.6", 0.10, "ignored" if i == 0 else None)
    shade_region(ax, context["ap_left"], context["ap_right"], "tab:red", 0.10, "trace kept")
    if context["core_exclusion_half_width"] > 0:
        shade_region(
            ax,
            context["trace_centre"] - context["core_exclusion_half_width"],
            context["trace_centre"] + context["core_exclusion_half_width"],
            "0.4",
            0.08,
            "core excluded",
        )

    ax.plot(x, row, linestyle="None", marker=".", ms=3, color="black", label="raw data")
    if np.any(context["finite_background"]):
        ax.plot(x[context["finite_background"]], background_fit[context["finite_background"]], color="tab:green", lw=1.1, label="background fit")
    if np.any(context["background_rejected_mask"]):
        ax.plot(x[context["background_rejected_mask"]], row[context["background_rejected_mask"]], "x", color="tab:orange", ms=5, label="rejected")
    if np.any(context["contaminant_mask"]):
        ax.plot(x[context["contaminant_mask"]], row[context["contaminant_mask"]], "x", color="tab:red", ms=5, label="masked")
    if context["profile_y"] is not None and context["profile_x"] is not None:
        ax.plot(context["profile_x"], context["profile_y"], color="tab:purple", lw=1.2, label="%s fit" % record["profile_model"])
    ax.axvline(context["trace_centre"], color="0.35", ls="--", lw=1.0, label="trace centre")
    ax.set_xlim(context["x_left"], context["x_right"])
    ylim = compute_data_ylim(
        (x, row, context["x_left"], context["x_right"]),
        (x[context["finite_background"]], background_fit[context["finite_background"]], context["x_left"], context["x_right"]),
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel("Counts")
    ax.set_title("Star %d, row %d" % (record["star_index"], context["detector_row"]))
    ax.legend(loc="upper right", fontsize=8, framealpha=1)


def plot_zoom_panel(ax, context):
    record = context["record"]
    x = context["x"]
    row_sub = context["row_background_subtracted"]

    for i, (start, end) in enumerate(context["ignored_spans"]):
        shade_region(ax, start, end, "0.6", 0.10, "ignored" if i == 0 else None)
    shade_region(ax, context["ap_left"], context["ap_right"], "tab:red", 0.10, "trace kept")

    ax.plot(x, row_sub, linestyle="None", marker=".", ms=3, color="black", label="raw - background")
    if context["source_y"] is not None and context["source_x"] is not None:
        ax.plot(context["source_x"], context["source_y"], color="tab:purple", lw=1.2, label="%s source" % record["profile_model"])
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax.axvline(context["trace_centre"], color="0.35", ls="--", lw=1.0, label="trace centre")
    ax.set_xlim(context["zoom_left"], context["zoom_right"])
    ylim = compute_data_ylim(
        (x, row_sub, context["zoom_left"], context["zoom_right"]),
    )
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("X pixel")
    ax.set_ylabel("Background-subtracted counts")
    ax.set_title("Star %d fit-window zoom" % record["star_index"])
    ax.legend(loc="upper right", fontsize=8, framealpha=1)


def create_single_frame_figure(records, requested_row):
    nstars = len(records)
    fig = plt.figure(figsize=(20, 5.2 * nstars))
    gs = fig.add_gridspec(nstars, 3, width_ratios=[1.2, 1.0, 0.9])

    contexts = [build_row_context(record, requested_row) for record in records]

    for i, context in enumerate(contexts):
        plot_frame_panel(fig.add_subplot(gs[i, 0]), context)
        plot_row_panel(fig.add_subplot(gs[i, 1]), context)
        plot_zoom_panel(fig.add_subplot(gs[i, 2]), context)

    frame_label = Path(records[0]["science_frame"]).name
    fig.suptitle("Frame %d: %s" % (records[0]["frame_index"], frame_label), fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def create_row_export_figure(records, requested_row):
    contexts = [build_row_context(record, requested_row) for record in records]
    nstars = len(contexts)
    fig, axes = plt.subplots(nstars, 1, figsize=(12, 4.2 * nstars), squeeze=False)

    for i, context in enumerate(contexts):
        ax = axes[i, 0]
        plot_row_panel(ax, context)
        add_profile_summary_box(ax, context)

    detector_row = contexts[0]["detector_row"]
    frame_label = Path(records[0]["science_frame"]).name
    fig.suptitle(
        "Frame %d: %s, row %d"
        % (records[0]["frame_index"], frame_label, detector_row),
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig, detector_row


def plot_profile_stack(ax, contexts, joy=False):
    cmap = plt.get_cmap("tab10")
    min_x = None
    max_x = None
    y_data_min = None
    y_data_max = None

    for idx, context in enumerate(contexts):
        color = cmap(idx % 10)
        x_rel = context["x"] - context["trace_centre"]
        row_norm = context["row_background_subtracted"] / context["norm"]
        source_norm = None if context["source_y"] is None else context["source_y"] / context["norm"]

        this_left = context["x_left"] - context["trace_centre"]
        this_right = context["x_right"] - context["trace_centre"]
        min_x = this_left if min_x is None else min(min_x, this_left)
        max_x = this_right if max_x is None else max(max_x, this_right)
        data_mask = (x_rel >= this_left) & (x_rel <= this_right) & np.isfinite(row_norm)
        if np.any(data_mask):
            current_min = np.nanmin(row_norm[data_mask])
            current_max = np.nanmax(row_norm[data_mask])
            y_data_min = current_min if y_data_min is None else min(y_data_min, current_min)
            y_data_max = current_max if y_data_max is None else max(y_data_max, current_max)

        if joy:
            offset = 1.25 * idx
            ax.plot(x_rel, row_norm + offset, ".", ms=2.5, color=color)
            if source_norm is not None:
                ax.plot(x_rel, source_norm + offset, lw=1.0, color=color)
            ax.plot([context["ap_left"] - context["trace_centre"], context["ap_left"] - context["trace_centre"]], [offset - 0.05, offset + 1.05], ls="--", lw=0.8, color=color, alpha=0.7)
            ax.plot([context["ap_right"] - context["trace_centre"], context["ap_right"] - context["trace_centre"]], [offset - 0.05, offset + 1.05], ls="--", lw=0.8, color=color, alpha=0.7)
            ax.text(min_x if min_x is not None else -10, offset + 0.1, "row %d" % context["detector_row"], fontsize=8, ha="left", va="bottom")
        else:
            ax.plot(x_rel, row_norm, ".", ms=2.5, color=color, alpha=0.7, label="row %d" % context["detector_row"])
            if source_norm is not None:
                ax.plot(x_rel, source_norm, lw=1.1, color=color)
            ax.axvline(context["ap_left"] - context["trace_centre"], ls="--", lw=0.8, color=color, alpha=0.5)
            ax.axvline(context["ap_right"] - context["trace_centre"], ls="--", lw=0.8, color=color, alpha=0.5)

    ax.axvline(0.0, color="0.4", ls="--", lw=0.8)
    if min_x is not None and max_x is not None:
        pad = max(0.5, 0.05 * max(max_x - min_x, 1.0))
        ax.set_xlim(min_x - pad, max_x + pad)

    if joy:
        if y_data_min is not None and y_data_max is not None:
            top_offset = 1.25 * (len(contexts) - 1) if len(contexts) > 0 else 0.0
            y_pad = max(0.05, 0.08 * max(y_data_max - y_data_min, 1.0))
            ax.set_ylim(y_data_min - y_pad, top_offset + y_data_max + y_pad)
        ax.set_yticks([])
        ax.set_ylabel("Stacked rows")
    else:
        if y_data_min is not None and y_data_max is not None:
            y_pad = max(0.05, 0.08 * max(y_data_max - y_data_min, 1.0))
            ax.set_ylim(y_data_min - y_pad, y_data_max + y_pad)
        ax.set_ylabel("Normalized bg-subtracted counts")
        ax.legend(loc="upper right", fontsize=8, framealpha=1)
    ax.set_xlabel("Relative X pixel from trace centre")


def create_stack_figure(records, requested_rows, joy=False):
    if requested_rows is None or len(requested_rows) == 0:
        raise ValueError("Use --rows row1,row2,... with --overlay or --joy.")

    nstars = len(records)
    fig, axes = plt.subplots(nstars, 1, figsize=(12, 4.3 * nstars), squeeze=False)

    for i, record in enumerate(records):
        contexts = [build_row_context(record, requested_row) for requested_row in requested_rows]
        plot_profile_stack(axes[i, 0], contexts, joy=joy)
        axes[i, 0].set_title(
            "Star %d %s rows"
            % (record["star_index"], "joy" if joy else "overlay")
        )

    frame_label = Path(records[0]["science_frame"]).name
    fig.suptitle("Frame %d: %s" % (records[0]["frame_index"], frame_label), fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def get_metric_specs_for_record(record):
    if not record["profile_enabled"]:
        return [
            {"key": "aperture_width", "title": "extraction width", "colorbar": "Pixels"},
        ]

    common = [
        {"key": "aperture_width", "title": "extraction width", "colorbar": "Pixels"},
        {"key": "profile_fwhm", "title": "profile FWHM", "colorbar": "Pixels"},
        {"key": "profile_width", "title": "enclosed width", "colorbar": "Pixels"},
    ]

    if record["profile_model"] == "gaussian":
        return common + [
            {"key": "profile_scale", "title": "sigma", "colorbar": "Pixels"},
        ]

    if record["profile_model"] == "moffat":
        return common + [
            {"key": "profile_scale", "title": "alpha", "colorbar": "Pixels"},
            {"key": "profile_beta", "title": "beta", "colorbar": "Value"},
        ]

    if record["profile_model"] == "empirical":
        return [
            {"key": "aperture_width", "title": "extraction width", "colorbar": "Pixels"},
            {"key": "profile_fwhm", "title": "empirical half-max width", "colorbar": "Pixels"},
            {"key": "profile_width", "title": "enclosed width", "colorbar": "Pixels"},
        ]

    if record["profile_model"] == "double_gaussian":
        return common + [
            {"key": "profile_scale", "title": "core sigma", "colorbar": "Pixels"},
            {"key": "profile_scale_secondary", "title": "wing sigma", "colorbar": "Pixels"},
            {"key": "profile_mix_fraction", "title": "wing fraction", "colorbar": "Fraction"},
        ]

    return common


def build_metric_matrix(records_by_frame, star_index, key, row_numbers):
    columns = []
    for frame_records in records_by_frame:
        record = frame_records.get(star_index)
        if record is None:
            columns.append(np.full(len(row_numbers), np.nan, dtype=float))
            continue

        if not np.array_equal(record["row_numbers"], row_numbers):
            raise ValueError("Row numbering is inconsistent across diagnostic frames for star %d." % star_index)

        if key == "aperture_width":
            values = record["aperture_right"].astype(float) - record["aperture_left"].astype(float)
        elif key == "profile_fwhm":
            values = record["profile_fwhm"].astype(float)
            if "profile_success" in record:
                values = values.copy()
                values[~record["profile_success"].astype(bool)] = np.nan
        elif key == "profile_width":
            values = 2.0 * record["profile_radius"].astype(float)
            if "profile_success" in record:
                values = values.copy()
                values[~record["profile_success"].astype(bool)] = np.nan
        elif key in ["profile_scale", "profile_scale_secondary", "profile_beta", "profile_mix_fraction"]:
            if key not in record:
                values = np.full(len(row_numbers), np.nan, dtype=float)
            else:
                values = record[key].astype(float)
                if "profile_success" in record:
                    values = values.copy()
                    values[~record["profile_success"].astype(bool)] = np.nan
        else:
            raise ValueError("Unknown width key '%s'." % key)

        columns.append(values)

    return np.column_stack(columns)


def plot_width_heatmap(ax, metric_matrix, frame_indices, row_numbers, title, colorbar_label):
    finite = np.isfinite(metric_matrix)
    if np.any(finite):
        vmin = np.nanpercentile(metric_matrix[finite], 5)
        vmax = np.nanpercentile(metric_matrix[finite], 95)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = np.nanmin(metric_matrix[finite])
            vmax = np.nanmax(metric_matrix[finite])
    else:
        vmin, vmax = 0.0, 1.0

    image = ax.imshow(
        metric_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[frame_indices[0] - 0.5, frame_indices[-1] + 0.5, row_numbers[0] - 0.5, row_numbers[-1] + 0.5],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Detector row")
    return image, colorbar_label


def create_widths_figure(frame_groups):
    records_by_frame = []
    star_indices = set()
    frame_indices = []

    for frame_group in frame_groups:
        frame_records = {}
        for record in load_frame_records(frame_group):
            frame_records[record["star_index"]] = record
            star_indices.add(record["star_index"])
        if len(frame_records) == 0:
            continue
        frame_indices.append(next(iter(frame_records.values()))["frame_index"])
        records_by_frame.append(frame_records)

    if len(records_by_frame) == 0:
        raise ValueError("No diagnostic records were loaded.")

    star_indices = sorted(star_indices)
    nstars = len(star_indices)
    metric_specs_by_star = {}
    max_cols = 1

    for star_index in star_indices:
        reference_record = next((frame_records[star_index] for frame_records in records_by_frame if star_index in frame_records), None)
        if reference_record is None:
            continue
        metric_specs = get_metric_specs_for_record(reference_record)
        metric_specs_by_star[star_index] = metric_specs
        max_cols = max(max_cols, len(metric_specs))

    fig, axes = plt.subplots(nstars, max_cols, figsize=(5.4 * max_cols, 4.8 * nstars), squeeze=False)

    for row_idx, star_index in enumerate(star_indices):
        reference_record = next((frame_records[star_index] for frame_records in records_by_frame if star_index in frame_records), None)
        if reference_record is None:
            continue

        row_numbers = reference_record["row_numbers"]
        metric_specs = metric_specs_by_star[star_index]

        for col_idx, metric_spec in enumerate(metric_specs):
            metric_matrix = build_metric_matrix(records_by_frame, star_index, metric_spec["key"], row_numbers)
            image, colorbar_label = plot_width_heatmap(
                axes[row_idx, col_idx],
                metric_matrix,
                frame_indices,
                row_numbers,
                "Star %d %s" % (star_index, metric_spec["title"]),
                metric_spec["colorbar"],
            )
            axis = axes[row_idx, col_idx]
            cbar = fig.colorbar(image, ax=axis, pad=0.02)
            cbar.set_label(colorbar_label)

        for col_idx in range(len(metric_specs), max_cols):
            axes[row_idx, col_idx].set_visible(False)

    fig.suptitle("Width diagnostics across all frames", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def export_row_images(frame_groups, requested_row, output_dir):
    if requested_row is None:
        raise ValueError("Use -r/--row with --export-row-images.")

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    for frame_group in frame_groups:
        records = load_frame_records(frame_group)
        fig, detector_row = create_row_export_figure(records, requested_row)
        frame_index = records[0]["frame_index"]
        out_file = output_path / ("frame_%05d_row_%04d.png" % (frame_index, detector_row))
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)


def create_time_overlay_figure(frame_groups, requested_rows, subtitle_suffix=None):
    if requested_rows is None or len(requested_rows) == 0:
        raise ValueError("Use -r/--row or --rows row1,row2,... with --time-overlay.")

    records_by_frame = [load_frame_records(frame_group) for frame_group in frame_groups]
    star_indices = sorted(
        {
            record["star_index"]
            for frame_records in records_by_frame
            for record in frame_records
        }
    )
    n_panel_rows = len(star_indices) * len(requested_rows)
    fig, axes = plt.subplots(n_panel_rows, 2, figsize=(14, 4.1 * n_panel_rows), squeeze=False)

    row_counter = 0
    for star_index in star_indices:
        for requested_row in requested_rows:
            contexts = []
            for frame_records in records_by_frame:
                record = next((item for item in frame_records if item["star_index"] == star_index), None)
                if record is None:
                    continue
                contexts.append(build_row_context(record, requested_row))

            if len(contexts) == 0:
                continue

            overlay_ax = axes[row_counter, 0]
            mean_ax = axes[row_counter, 1]
            row_counter += 1

            min_x = min(context["x_left"] for context in contexts)
            max_x = max(context["x_right"] for context in contexts)
            frame_alpha = min(0.25, max(0.04, 8.0 / max(len(contexts), 1)))

            stack = []
            for context in contexts:
                x = context["x"]
                mask = (x >= min_x) & (x <= max_x)
                overlay_ax.plot(x[mask], context["row"][mask], color="tab:blue", alpha=frame_alpha, lw=0.8)
                stack.append(context["row"])

            stack = np.asarray(stack, dtype=float)
            mean_profile = np.nanmean(stack, axis=0)
            std_profile = np.nanstd(stack, axis=0)
            x = contexts[0]["x"]
            mask = (x >= min_x) & (x <= max_x)

            overlay_ax.set_xlim(min_x, max_x)
            overlay_ylim = compute_data_ylim(
                *[(context["x"], context["row"], min_x, max_x) for context in contexts]
            )
            if overlay_ylim is not None:
                overlay_ax.set_ylim(*overlay_ylim)
            overlay_ax.set_title("Star %d row %d raw profiles over time" % (star_index, contexts[0]["detector_row"]))
            overlay_ax.set_xlabel("X pixel")
            overlay_ax.set_ylabel("Counts")

            mean_ax.plot(x[mask], mean_profile[mask], color="tab:blue", lw=1.2, label="mean")
            mean_ax.fill_between(
                x[mask],
                mean_profile[mask] - std_profile[mask],
                mean_profile[mask] + std_profile[mask],
                color="tab:blue",
                alpha=0.20,
                label=r"$\pm1\sigma$",
            )
            mean_ax.set_xlim(min_x, max_x)
            mean_ylim = compute_data_ylim(
                (x, mean_profile - std_profile, min_x, max_x),
                (x, mean_profile + std_profile, min_x, max_x),
            )
            if mean_ylim is not None:
                mean_ax.set_ylim(*mean_ylim)
            mean_ax.set_title("Star %d row %d mean ± 1σ" % (star_index, contexts[0]["detector_row"]))
            mean_ax.set_xlabel("X pixel")
            mean_ax.set_ylabel("Counts")
            mean_ax.legend(loc="upper right", fontsize=8, framealpha=1)

    for hide_idx in range(row_counter, n_panel_rows):
        axes[hide_idx, 0].set_visible(False)
        axes[hide_idx, 1].set_visible(False)

    title = "Raw row-profile evolution across all frames"
    if subtitle_suffix is not None and str(subtitle_suffix).strip() != "":
        title = "%s\n%s" % (title, subtitle_suffix)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def prepare_time_overlay_panels(frame_groups, requested_rows):
    records_by_frame = [load_frame_records(frame_group) for frame_group in frame_groups]
    star_indices = sorted(
        {
            record["star_index"]
            for frame_records in records_by_frame
            for record in frame_records
        }
    )

    panel_data = []
    for star_index in star_indices:
        for requested_row in requested_rows:
            contexts = []
            for frame_idx, frame_records in enumerate(records_by_frame):
                record = next((item for item in frame_records if item["star_index"] == star_index), None)
                if record is None:
                    continue
                context = build_row_context(record, requested_row)
                context["frame_idx"] = frame_idx
                contexts.append(context)

            if len(contexts) == 0:
                continue

            min_x = min(context["x_left"] for context in contexts)
            max_x = max(context["x_right"] for context in contexts)
            x = contexts[0]["x"]
            mask = (x >= min_x) & (x <= max_x)
            stack = np.asarray([context["row"] for context in contexts], dtype=float)
            mean_profile = np.nanmean(stack, axis=0)
            std_profile = np.nanstd(stack, axis=0)
            overlay_ylim = compute_data_ylim(
                *[(context["x"], context["row"], min_x, max_x) for context in contexts]
            )
            mean_ylim = compute_data_ylim(
                (x, mean_profile - std_profile, min_x, max_x),
                (x, mean_profile + std_profile, min_x, max_x),
            )

            panel_data.append(
                {
                    "star_index": star_index,
                    "requested_row": requested_row,
                    "detector_row": contexts[0]["detector_row"],
                    "contexts": contexts,
                    "x": x,
                    "mask": mask,
                    "min_x": min_x,
                    "max_x": max_x,
                    "mean_profile": mean_profile,
                    "std_profile": std_profile,
                    "overlay_ylim": overlay_ylim,
                    "mean_ylim": mean_ylim,
                }
            )

    return panel_data


def get_frame_times_from_groups(frame_groups):
    times = []
    for frame_group in frame_groups:
        records = load_frame_records(frame_group)
        frame_time = np.nan
        for record in records:
            frame_path = resolve_science_frame_path(record)
            frame_time = get_frame_time_value(str(frame_path))
            if np.isfinite(frame_time):
                break
        times.append(frame_time)
    return np.asarray(times, dtype=float)


def play_time_overlay_build(frame_groups, requested_rows, delay):
    if requested_rows is None or len(requested_rows) == 0:
        raise ValueError("Use -r/--row or --rows row1,row2,... with --time-overlay-build.")

    frame_times = get_frame_times_from_groups(frame_groups)
    panel_data = prepare_time_overlay_panels(frame_groups, requested_rows)
    if len(panel_data) == 0:
        raise ValueError("No matching diagnostic rows were found for --time-overlay-build.")

    n_panel_rows = len(panel_data)
    fig, axes = plt.subplots(n_panel_rows, 2, figsize=(14, 4.1 * n_panel_rows), squeeze=False)
    highlight_artists = []

    for row_counter, panel in enumerate(panel_data):
        overlay_ax = axes[row_counter, 0]
        mean_ax = axes[row_counter, 1]
        contexts = panel["contexts"]
        x = panel["x"]
        mask = panel["mask"]
        min_x = panel["min_x"]
        max_x = panel["max_x"]
        frame_alpha = min(0.25, max(0.04, 8.0 / max(len(contexts), 1)))

        for context in contexts:
            overlay_ax.plot(x[mask], context["row"][mask], color="tab:blue", alpha=frame_alpha, lw=0.8)

        overlay_ax.set_xlim(min_x, max_x)
        if panel["overlay_ylim"] is not None:
            overlay_ax.set_ylim(*panel["overlay_ylim"])
        overlay_ax.set_title("Star %d row %d raw profiles over time" % (panel["star_index"], panel["detector_row"]))
        overlay_ax.set_xlabel("X pixel")
        overlay_ax.set_ylabel("Counts")

        mean_ax.plot(x[mask], panel["mean_profile"][mask], color="tab:blue", lw=1.2, label="mean")
        mean_ax.fill_between(
            x[mask],
            panel["mean_profile"][mask] - panel["std_profile"][mask],
            panel["mean_profile"][mask] + panel["std_profile"][mask],
            color="tab:blue",
            alpha=0.20,
            label=r"$\pm1\sigma$",
        )
        mean_ax.set_xlim(min_x, max_x)
        if panel["mean_ylim"] is not None:
            mean_ax.set_ylim(*panel["mean_ylim"])
        mean_ax.set_title("Star %d row %d mean ± 1σ" % (panel["star_index"], panel["detector_row"]))
        mean_ax.set_xlabel("X pixel")
        mean_ax.set_ylabel("Counts")

        overlay_current, = overlay_ax.plot([], [], color="tab:orange", lw=2.4, alpha=0.95, label="current frame", zorder=5)
        mean_current, = mean_ax.plot([], [], color="tab:orange", lw=2.2, alpha=0.95, label="current frame", zorder=5)
        overlay_ax.legend(loc="upper right", fontsize=8, framealpha=1)
        mean_ax.legend(loc="upper right", fontsize=8, framealpha=1)
        highlight_artists.append((overlay_current, mean_current))

    title_text = fig.suptitle("Raw row-profile evolution across all frames", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    plt.show(block=False)

    n_steps = max(len(panel["contexts"]) for panel in panel_data)
    for end_idx in range(1, n_steps + 1):
        subtitle = "highlighting frame %d" % end_idx
        if end_idx > 1 and (end_idx - 1) < len(frame_times) and np.isfinite(frame_times[end_idx - 1]) and np.isfinite(frame_times[end_idx - 2]):
            delta_minutes = (frame_times[end_idx - 1] - frame_times[end_idx - 2]) * 24.0 * 60.0
            subtitle += " | latest gap = %.2f min" % delta_minutes
        title_text.set_text("Raw row-profile evolution across all frames\n%s" % subtitle)

        for panel, artists in zip(panel_data, highlight_artists):
            overlay_current, mean_current = artists
            current_context = next((context for context in panel["contexts"] if context["frame_idx"] == end_idx - 1), None)
            if current_context is None:
                overlay_current.set_data([], [])
                mean_current.set_data([], [])
                continue
            x = panel["x"]
            mask = panel["mask"]
            overlay_current.set_data(x[mask], current_context["row"][mask])
            mean_current.set_data(x[mask], current_context["row"][mask])

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if end_idx < n_steps:
            plt.pause(delay)

    plt.show()
    plt.close(fig)


def export_time_overlay_all_rows(frame_groups, output_dir):
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    reference_records = load_frame_records(frame_groups[0])
    if len(reference_records) == 0:
        raise ValueError("No diagnostic records were loaded.")
    row_numbers = reference_records[0]["row_numbers"]

    for detector_row in row_numbers:
        fig = create_time_overlay_figure(frame_groups, [int(detector_row)])
        out_file = output_path / ("time_overlay_row_%04d.png" % int(detector_row))
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)


def create_time_stack_figure(frame_groups, requested_rows):
    if requested_rows is None or len(requested_rows) == 0:
        raise ValueError("Use -r/--row or --rows row1,row2,... with --time-stack.")

    records_by_frame = [load_frame_records(frame_group) for frame_group in frame_groups]
    star_indices = sorted(
        {
            record["star_index"]
            for frame_records in records_by_frame
            for record in frame_records
        }
    )
    n_panel_rows = len(star_indices) * len(requested_rows)
    fig, axes = plt.subplots(n_panel_rows, 1, figsize=(14, 4.8 * n_panel_rows), squeeze=False)

    row_counter = 0
    for star_index in star_indices:
        for requested_row in requested_rows:
            contexts = []
            for frame_records in records_by_frame:
                record = next((item for item in frame_records if item["star_index"] == star_index), None)
                if record is None:
                    continue
                context = build_row_context(record, requested_row)
                frame_path = resolve_science_frame_path(record)
                context["frame_time"] = get_frame_time_value(str(frame_path))
                contexts.append(context)

            if len(contexts) == 0:
                continue

            ax = axes[row_counter, 0]
            row_counter += 1

            min_x = min(context["x_left"] for context in contexts)
            max_x = max(context["x_right"] for context in contexts)
            x_span = max(max_x - min_x, 1)

            profile_spans = []
            for context in contexts:
                x = context["x"]
                mask = (x >= min_x) & (x <= max_x)
                row_values = context["row"][mask]
                finite_values = row_values[np.isfinite(row_values)]
                if len(finite_values) == 0:
                    continue
                low = np.nanpercentile(finite_values, 5)
                high = np.nanpercentile(finite_values, 99)
                profile_spans.append(max(high - low, 1.0))

            separation = 1.15 * max(profile_spans) if len(profile_spans) > 0 else 1.0

            time_values = np.asarray([context["frame_time"] for context in contexts], dtype=float)
            finite_time = np.isfinite(time_values)
            if np.count_nonzero(finite_time) >= 2:
                finite_times = time_values[finite_time]
                relative_minutes = (time_values - np.nanmin(finite_times)) * 24.0 * 60.0
                delta_minutes = np.diff(np.sort(finite_times)) * 24.0 * 60.0
                positive_deltas = delta_minutes[delta_minutes > 0]
                median_gap = np.nanmedian(positive_deltas) if len(positive_deltas) > 0 else 1.0
                if not np.isfinite(median_gap) or median_gap <= 0:
                    median_gap = 1.0
                offsets = (relative_minutes / median_gap) * separation
                spacing_label = "vertical spacing follows actual time gaps"
            else:
                offsets = np.arange(len(contexts), dtype=float) * separation
                relative_minutes = np.full(len(contexts), np.nan)
                spacing_label = "vertical spacing uses exposure index"

            colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(contexts)))
            label_x = max_x + 0.02 * x_span

            for idx, (context, offset, color) in enumerate(zip(contexts, offsets, colors)):
                x = context["x"]
                mask = (x >= min_x) & (x <= max_x)
                profile = context["row"][mask] + offset
                ax.plot(x[mask], profile, color=color, lw=1.0)

                gap_text = ""
                if idx > 0 and np.isfinite(relative_minutes[idx]) and np.isfinite(relative_minutes[idx - 1]):
                    gap_text = " (+%.2f min)" % (relative_minutes[idx] - relative_minutes[idx - 1])
                label_y = np.nanmedian(profile[-max(3, min(8, len(profile))):])
                ax.text(
                    label_x,
                    label_y,
                    "F%d%s" % (context["record"]["frame_index"], gap_text),
                    fontsize=7,
                    color=color,
                    va="center",
                )

            ax.set_xlim(min_x, label_x + 0.12 * x_span)
            ax.set_title(
                "Star %d row %d raw profiles stacked by time"
                % (star_index, contexts[0]["detector_row"])
            )
            ax.set_xlabel("X pixel")
            ax.set_ylabel("Counts + offset")
            ax.text(
                0.01,
                0.98,
                spacing_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
            )

    for hide_idx in range(row_counter, n_panel_rows):
        axes[hide_idx, 0].set_visible(False)

    fig.suptitle("Raw row-profile sequence across all frames", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def get_output_path(output_arg, frame_index, mode, multiple):
    if output_arg is None:
        return "diagnostic_%s_frame_%05d.png" % (mode, frame_index)

    output_path = Path(output_arg).expanduser()
    if not multiple:
        return str(output_path)

    if output_path.suffix != "":
        return str(output_path.with_name("%s_frame_%05d%s" % (output_path.stem, frame_index, output_path.suffix)))

    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path / ("diagnostic_%s_frame_%05d.png" % (mode, frame_index)))


def main():
    parser = argparse.ArgumentParser(description="Inspect extraction diagnostics for one frame or a whole directory.")
    parser.add_argument("diagnostic_input", help="A diagnostic .npz file or a directory containing frame_*_star_*.npz")
    parser.add_argument("-r", "--row", type=int, help="Detector row number or 0-based local row index for the default viewer.")
    parser.add_argument("--rows", help="Comma-separated rows for overlay or joy modes.")
    parser.add_argument("--overlay", action="store_true", help="Overlay the requested rows for each star, aligned on the trace centre.")
    parser.add_argument("--joy", action="store_true", help="Joy-style stacked plot of the requested rows for each star, aligned on the trace centre.")
    parser.add_argument("--widths", action="store_true", help="Show extraction/profile widths for every row in every frame.")
    parser.add_argument("--time-overlay", action="store_true", help="Overlay raw row profiles from all frames and show the mean ± 1σ band.")
    parser.add_argument("--time-overlay-build", action="store_true", help="Show the time-overlay plot building up cumulatively: 1, 1-2, 1-3, ...")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay in seconds between cumulative time-overlay updates.")
    parser.add_argument("--export-time-overlay-all-rows", action="store_true", help="Save one time-overlay figure per diagnostic row into the output directory.")
    parser.add_argument("--time-stack", action="store_true", help="Stack raw row profiles by exposure, using actual inter-frame time gaps when available.")
    parser.add_argument("--export-row-images", action="store_true", help="Export one saved row-profile figure per frame for the chosen row.")
    parser.add_argument("-s", "--save_figure", action="store_true", help="Save figure(s) as well as showing them.")
    parser.add_argument("-o", "--output", help="Optional output filename, prefix, or directory for saved figures.")
    args = parser.parse_args()

    selected_modes = [args.overlay, args.joy, args.widths, args.time_overlay, args.time_stack, args.export_row_images]
    if sum(bool(mode) for mode in selected_modes) > 1:
        raise ValueError("Choose only one of --overlay, --joy, --widths, --time-overlay, --time-stack, or --export-row-images.")
    if args.time_overlay_build and not args.time_overlay:
        raise ValueError("--time-overlay-build must be used together with --time-overlay.")
    if args.export_time_overlay_all_rows and not args.time_overlay:
        raise ValueError("--export-time-overlay-all-rows must be used together with --time-overlay.")

    requested_rows = parse_rows(args.rows)
    if requested_rows is None and args.row is not None:
        requested_rows = [args.row]
    frame_groups = build_frame_groups(args.diagnostic_input)
    multiple_frames = len(frame_groups) > 1

    if args.widths:
        fig = create_widths_figure(frame_groups)
        if args.save_figure:
            output_path = get_output_path(args.output, 0, "widths", False)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return

    if args.time_overlay:
        if args.export_time_overlay_all_rows:
            export_dir = args.output if args.output is not None else "time_overlay_all_rows"
            export_time_overlay_all_rows(frame_groups, export_dir)
            return

        if args.time_overlay_build:
            play_time_overlay_build(frame_groups, requested_rows, max(args.delay, 0.0))
            return

        fig = create_time_overlay_figure(frame_groups, requested_rows)
        if args.save_figure:
            output_path = get_output_path(args.output, 0, "time_overlay", False)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return

    if args.time_stack:
        fig = create_time_stack_figure(frame_groups, requested_rows)
        if args.save_figure:
            output_path = get_output_path(args.output, 0, "time_stack", False)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return

    if args.export_row_images:
        export_dir = args.output if args.output is not None else ("row_%04d_exports" % int(args.row))
        export_row_images(frame_groups, args.row, export_dir)
        return

    for frame_group in frame_groups:
        records = load_frame_records(frame_group)

        if args.overlay:
            fig = create_stack_figure(records, requested_rows, joy=False)
            mode = "overlay"
        elif args.joy:
            fig = create_stack_figure(records, requested_rows, joy=True)
            mode = "joy"
        else:
            fig = create_single_frame_figure(records, args.row)
            mode = "single"

        if args.save_figure:
            output_path = get_output_path(args.output, records[0]["frame_index"], mode, multiple_frames)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
