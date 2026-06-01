import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from astropy.visualization import ZScaleInterval

from plot_extraction_diagnostics import (
    build_frame_groups,
    build_row_context,
    get_excluded_column_display_label,
    load_frame_records,
    resolve_science_frame_path,
    shade_record_excluded_columns_on_frame,
)

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
    }
)

EFOSC2_BINNED_ARCSEC_PER_PIXEL = 0.24
SUMMARY_IMAGE_CMAP = "viridis"
SUMMARY_SPATIAL_PAD_PIXELS = 8
SUMMARY_LINE_WIDTH = 1.6
SUMMARY_DASHED_LINE_WIDTH = 1.3
SUMMARY_MARKER_SIZE = 5.2
SUMMARY_PLUS_EDGE_WIDTH = 1.3
SUMMARY_TRACE_OVERLAY_ALPHA = 0.95
SUMMARY_BACKGROUND_OVERLAY_ALPHA = 0.80
SUMMARY_MASKED_OVERLAY_ALPHA = 0.90
SUMMARY_IMAGE_INTERPOLATION = "nearest"
SUMMARY_ROW_DATA_LINESTYLE = "None"
SUMMARY_ROW_SHOW_SELECTED_WAVELENGTH = True
SUMMARY_ZSCALE_CONTRAST = 0.25
SUMMARY_ROW_EDGE_MASK_PIXELS = 2

ROW_DATA_STYLE = {
    "color": "#1b7837",
    "marker": "+",
    "ms": SUMMARY_MARKER_SIZE,
    "mew": SUMMARY_PLUS_EDGE_WIDTH,
    "lw": 0.0,
    "ls": SUMMARY_ROW_DATA_LINESTYLE,
}
ROW_MASKED_STYLE = {
    "color": "#d73027",
    "marker": "x",
    "ms": SUMMARY_MARKER_SIZE,
    "mew": 1.2,
    "linestyle": "None",
}
ROW_BACKGROUND_STYLE = {
    "color": "#2166ac",
    "marker": "o",
    "ms": 4.4,
    "mew": 0.0,
    "linestyle": "None",
}
TRACE_CENTRE_STYLE = {"color": "#1b7837", "lw": SUMMARY_LINE_WIDTH}
APERTURE_EDGE_STYLE = {"color": "#1b7837", "lw": SUMMARY_DASHED_LINE_WIDTH, "ls": "--"}
MASK_BOUNDARY_STYLE = {"color": "#d73027", "lw": SUMMARY_DASHED_LINE_WIDTH, "ls": "--"}
BACKGROUND_BOUNDARY_STYLE = {"color": "#2166ac", "lw": SUMMARY_DASHED_LINE_WIDTH, "ls": "--"}
SPECTRUM_STYLE = {"color": "#d73027", "lw": 1.6}
SELECTED_WAVELENGTH_STYLE = {"color": "deepskyblue", "lw": 1.0, "alpha": 0.95}


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def normalise_flux(flux):
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(flux) & (flux != 0)
    if not np.any(finite):
        return np.full_like(flux, np.nan, dtype=float)

    reference = float(np.nanmedian(flux[finite]))
    if not np.isfinite(reference) or reference == 0:
        reference = float(np.nanmax(np.abs(flux[finite])))
    if not np.isfinite(reference) or reference == 0:
        return np.full_like(flux, np.nan, dtype=float)

    result = np.full_like(flux, np.nan, dtype=float)
    result[finite] = flux[finite] / reference
    return result


def contiguous_true_spans(mask):
    mask = np.asarray(mask, dtype=bool)
    spans = []
    if mask.size == 0:
        return spans

    in_span = False
    start = 0
    for idx, flag in enumerate(mask):
        if flag and not in_span:
            start = idx
            in_span = True
        elif not flag and in_span:
            spans.append((start, idx))
            in_span = False
    if in_span:
        spans.append((start, len(mask)))
    return spans


def resolve_reduction_dir(records):
    diag_path = Path(records[0]["path"]).expanduser().resolve()
    return diag_path.parents[2]


def load_resampled_spectrum_bundle(reduction_dir):
    pickled_dir = Path(reduction_dir) / "pickled_objects"
    resampled_dir = pickled_dir / "improved_resampling"
    if not resampled_dir.is_dir():
        raise FileNotFoundError("Could not find improved_resampling in %s" % str(pickled_dir))

    return {
        "star1_flux": np.asarray(load_pickle(resampled_dir / "star1_flux_resampled.pickle"), dtype=float),
        "star2_flux": np.asarray(load_pickle(resampled_dir / "star2_flux_resampled.pickle"), dtype=float),
        "wavelength": np.asarray(load_pickle(resampled_dir / "wavelength_solution.pickle"), dtype=float),
    }


def build_row_to_wavelength_mappers(record, wavelength_solution):
    row_count = len(record["row_numbers"])
    wavelength = np.asarray(wavelength_solution, dtype=float)
    if wavelength.ndim != 1 or wavelength.size != row_count:
        return None, None

    finite = np.isfinite(wavelength)
    if not np.any(finite):
        return None, None
    if not np.all(finite):
        sample_rows = np.arange(row_count, dtype=float)
        wavelength = np.interp(sample_rows, sample_rows[finite], wavelength[finite])

    local_rows = np.arange(row_count, dtype=float)
    if wavelength[0] > wavelength[-1]:
        wavelength = wavelength[::-1]
        local_rows = local_rows[::-1]

    def row_to_wavelength(y):
        y = np.asarray(y, dtype=float)
        return np.interp(y, np.arange(row_count, dtype=float), wavelength, left=wavelength[0], right=wavelength[-1])

    def wavelength_to_row(w):
        w = np.asarray(w, dtype=float)
        return np.interp(w, wavelength, np.arange(row_count, dtype=float), left=0.0, right=float(row_count - 1))

    return row_to_wavelength, wavelength_to_row


def build_row_number_to_wavelength_mappers(record, wavelength_solution):
    row_numbers = np.asarray(record["row_numbers"], dtype=float)
    wavelength = np.asarray(wavelength_solution, dtype=float)
    if row_numbers.ndim != 1 or wavelength.ndim != 1 or row_numbers.size != wavelength.size:
        return None, None

    finite = np.isfinite(row_numbers) & np.isfinite(wavelength)
    if not np.any(finite):
        return None, None

    row_numbers = row_numbers[finite]
    wavelength = wavelength[finite]
    if wavelength[0] > wavelength[-1]:
        wavelength = wavelength[::-1]
        row_numbers = row_numbers[::-1]

    def wavelength_to_row_number(w):
        w = np.asarray(w, dtype=float)
        return np.interp(w, wavelength, row_numbers, left=row_numbers[0], right=row_numbers[-1])

    def row_number_to_wavelength(y):
        y = np.asarray(y, dtype=float)
        return np.interp(y, row_numbers, wavelength, left=wavelength[0], right=wavelength[-1])

    return wavelength_to_row_number, row_number_to_wavelength


def centers_to_edges(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        return np.array([], dtype=float)
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def compute_star_spatial_crop(record, pad=SUMMARY_SPATIAL_PAD_PIXELS):
    frame_data = np.asarray(record.get("frame_data", np.empty((0, 0))), dtype=float)
    if frame_data.ndim != 2 or frame_data.shape[1] == 0:
        return 0, 0

    candidates = []
    for key in (
        "trace",
        "aperture_left",
        "aperture_right",
        "background_left_start",
        "background_left_end",
        "background_right_start",
        "background_right_end",
    ):
        values = np.asarray(record.get(key, np.array([])), dtype=float)
        finite = values[np.isfinite(values)]
        finite = finite[finite >= 0]
        if finite.size > 0:
            candidates.append(finite)

    if len(candidates) == 0:
        return 0, int(frame_data.shape[1] - 1)

    lower = int(np.floor(min(np.nanmin(values) for values in candidates))) - int(max(pad, 0))
    upper = int(np.ceil(max(np.nanmax(values) for values in candidates))) + int(max(pad, 0))
    lower = max(0, lower)
    upper = min(int(frame_data.shape[1] - 1), upper)
    return lower, upper


def build_ignored_point_mask(row_context):
    x = np.asarray(row_context["x"], dtype=float)
    ignored_mask = np.asarray(row_context["excluded_column_mask"], dtype=bool).copy()

    ignored_spans = build_combined_ignored_spans(row_context)
    for left, right in ignored_spans:
        ignored_mask |= (x >= float(left)) & (x < float(right))
    return ignored_mask


def unique_boundary_pairs(spans):
    unique = []
    seen = set()
    for left, right in spans:
        key = (round(float(left), 6), round(float(right), 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append((float(left), float(right)))
    return unique


def build_background_gap_spans(row_context):
    spans = []
    bg_left_end = int(row_context["bg_left_end"])
    ap_left = int(row_context["ap_left"])
    ap_right = int(row_context["ap_right"])
    bg_right_start = int(row_context["bg_right_start"])

    if bg_left_end >= 0 and ap_left > bg_left_end:
        spans.append((bg_left_end, ap_left))
    if bg_right_start >= 0 and bg_right_start > ap_right:
        spans.append((ap_right, bg_right_start))
    return spans


def build_combined_ignored_spans(row_context):
    spans = list(row_context.get("ignored_spans", []))
    spans.extend(build_background_gap_spans(row_context))
    return unique_boundary_pairs(spans)


def collect_wavelength_ordered_trace_arrays(record, wavelength_solution):
    wavelength = np.asarray(wavelength_solution, dtype=float)
    row_numbers = np.asarray(record["row_numbers"], dtype=float)
    trace = np.asarray(record["trace"], dtype=float)
    ap_left = np.asarray(record["aperture_left"], dtype=float)
    ap_right = np.asarray(record["aperture_right"], dtype=float)
    bg_left_start = np.asarray(record["background_left_start"], dtype=float)
    bg_left_end = np.asarray(record["background_left_end"], dtype=float)
    bg_right_start = np.asarray(record["background_right_start"], dtype=float)
    bg_right_end = np.asarray(record["background_right_end"], dtype=float)

    finite = np.isfinite(wavelength)
    if not np.any(finite):
        raise ValueError("No finite wavelength values were found for the selected frame.")

    order = np.argsort(wavelength[finite])
    wavelength_sorted = wavelength[finite][order]
    row_numbers_sorted = row_numbers[finite][order]

    arrays = {
        "trace": trace[finite][order],
        "ap_left": ap_left[finite][order],
        "ap_right": ap_right[finite][order],
        "bg_left_start": bg_left_start[finite][order],
        "bg_left_end": bg_left_end[finite][order],
        "bg_right_start": bg_right_start[finite][order],
        "bg_right_end": bg_right_end[finite][order],
    }
    return wavelength_sorted, row_numbers_sorted, arrays


def choose_frame(records_by_frame, requested_frame_index=None):
    if len(records_by_frame) == 0:
        raise ValueError("No diagnostic frames were found.")

    if requested_frame_index is None:
        frame_pos = len(records_by_frame) // 2
        return frame_pos, records_by_frame[frame_pos]

    for frame_pos, records in enumerate(records_by_frame):
        if int(records[0]["frame_index"]) == int(requested_frame_index):
            return frame_pos, records

    available = [int(records[0]["frame_index"]) for records in records_by_frame]
    raise ValueError(
        "Frame index %s was not found. Available diagnostic frames run from %d to %d."
        % (str(requested_frame_index), min(available), max(available))
    )


def choose_star_record(records, star_index):
    for record in records:
        if int(record["star_index"]) == int(star_index):
            return record
    available = [int(record["star_index"]) for record in records]
    raise ValueError("Star index %d was not found in this frame. Available stars: %s" % (int(star_index), available))


def resolve_spectrum_index(frame_position, frame_index, spectra_length, n_frames):
    if spectra_length == n_frames and 0 <= frame_position < spectra_length:
        return int(frame_position)
    if 1 <= int(frame_index) <= spectra_length:
        return int(frame_index) - 1
    if 0 <= frame_position < spectra_length:
        return int(frame_position)
    raise ValueError(
        "Could not map diagnostic frame %d to the resampled spectra (n_spectra=%d, n_diagnostic_frames=%d)."
        % (int(frame_index), int(spectra_length), int(n_frames))
    )


def compute_panel_ylim(x, y_arrays, left, right):
    x = np.asarray(x, dtype=float)
    mask = (x >= float(left)) & (x <= float(right))
    values = []
    for y in y_arrays:
        arr = np.asarray(y, dtype=float)
        finite = np.isfinite(arr) & mask
        if np.any(finite):
            values.append(arr[finite])
    if len(values) == 0:
        return None
    stacked = np.concatenate(values)
    ymin = float(np.nanmin(stacked))
    ymax = float(np.nanmax(stacked))
    span = ymax - ymin
    if not np.isfinite(span) or span <= 0:
        span = max(abs(ymax), 1.0) * 0.05
    return ymin - 0.08 * span, ymax + 0.10 * span


def add_row_wavelength_label(row_context, wavelength_solution):
    wavelength = np.asarray(wavelength_solution, dtype=float)
    row_index = int(row_context["row_index"])
    if wavelength.ndim == 1 and 0 <= row_index < len(wavelength) and np.isfinite(wavelength[row_index]):
        return " (~%.0f $\\AA$)" % float(wavelength[row_index])
    return ""


def plot_raw_frame_panel(ax, record, row_context, wavelength_solution, show_bottom_wavelength_axis=True):
    frame_data = np.asarray(record["frame_data"], dtype=float)
    wavelength = np.asarray(wavelength_solution, dtype=float)
    if frame_data.ndim != 2 or wavelength.ndim != 1 or frame_data.shape[0] != wavelength.size:
        raise ValueError("Frame data and wavelength solution do not have compatible shapes.")

    spatial_min, spatial_max = compute_star_spatial_crop(record)
    spatial_pixels = np.arange(spatial_min, spatial_max + 1, dtype=float)
    wavelength_sorted, row_numbers_sorted, trace_arrays = collect_wavelength_ordered_trace_arrays(record, wavelength_solution)
    wavelength_order = np.argsort(wavelength)
    cropped = np.asarray(frame_data[:, spatial_min : spatial_max + 1], dtype=float)[wavelength_order, :].T

    arcsec_centres = (spatial_pixels - float(spatial_min)) * EFOSC2_BINNED_ARCSEC_PER_PIXEL

    finite_image = cropped[np.isfinite(cropped)]
    if finite_image.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = ZScaleInterval(contrast=SUMMARY_ZSCALE_CONTRAST).get_limits(finite_image)
        if not np.isfinite(vmin):
            vmin = float(np.nanmin(finite_image))
        if not np.isfinite(vmax):
            vmax = float(np.nanmax(finite_image))
        if vmax <= vmin:
            vmax = vmin + 1.0

    image = ax.imshow(
        cropped,
        origin="lower",
        aspect="auto",
        interpolation=SUMMARY_IMAGE_INTERPOLATION,
        cmap=SUMMARY_IMAGE_CMAP,
        vmin=vmin,
        vmax=vmax,
        extent=[
            float(np.nanmin(wavelength_sorted)),
            float(np.nanmax(wavelength_sorted)),
            float(np.nanmin(arcsec_centres) - 0.5 * EFOSC2_BINNED_ARCSEC_PER_PIXEL),
            float(np.nanmax(arcsec_centres) + 0.5 * EFOSC2_BINNED_ARCSEC_PER_PIXEL),
        ],
        rasterized=True,
    )

    selected_row_wavelength = np.nan
    row_index = int(row_context["row_index"])
    if 0 <= row_index < wavelength.size and np.isfinite(wavelength[row_index]):
        selected_row_wavelength = float(wavelength[row_index])
    if SUMMARY_ROW_SHOW_SELECTED_WAVELENGTH and np.isfinite(selected_row_wavelength):
        ax.axvline(selected_row_wavelength, **SELECTED_WAVELENGTH_STYLE)

    def xpixel_to_arcsec_pixels(values):
        values = np.asarray(values, dtype=float)
        return (values - float(spatial_min)) * EFOSC2_BINNED_ARCSEC_PER_PIXEL

    trace_arcsec = xpixel_to_arcsec_pixels(trace_arrays["trace"])
    ap_left_arcsec = xpixel_to_arcsec_pixels(trace_arrays["ap_left"])
    ap_right_arcsec = xpixel_to_arcsec_pixels(trace_arrays["ap_right"])
    bg_left_start_arcsec = xpixel_to_arcsec_pixels(trace_arrays["bg_left_start"])
    bg_left_end_arcsec = xpixel_to_arcsec_pixels(trace_arrays["bg_left_end"])
    bg_right_start_arcsec = xpixel_to_arcsec_pixels(trace_arrays["bg_right_start"])
    bg_right_end_arcsec = xpixel_to_arcsec_pixels(trace_arrays["bg_right_end"])

    ax.fill_between(
        wavelength_sorted,
        bg_left_start_arcsec,
        bg_left_end_arcsec,
        color=BACKGROUND_BOUNDARY_STYLE["color"],
        alpha=0.10,
        linewidth=0.0,
    )
    ax.fill_between(
        wavelength_sorted,
        bg_right_start_arcsec,
        bg_right_end_arcsec,
        color=BACKGROUND_BOUNDARY_STYLE["color"],
        alpha=0.10,
        linewidth=0.0,
    )
    ax.fill_between(
        wavelength_sorted,
        bg_left_end_arcsec,
        ap_left_arcsec,
        color=MASK_BOUNDARY_STYLE["color"],
        alpha=0.10,
        linewidth=0.0,
    )
    ax.fill_between(
        wavelength_sorted,
        ap_right_arcsec,
        bg_right_start_arcsec,
        color=MASK_BOUNDARY_STYLE["color"],
        alpha=0.10,
        linewidth=0.0,
    )

    ax.plot(wavelength_sorted, trace_arcsec, alpha=SUMMARY_TRACE_OVERLAY_ALPHA, **TRACE_CENTRE_STYLE)
    ax.plot(wavelength_sorted, ap_left_arcsec, alpha=SUMMARY_TRACE_OVERLAY_ALPHA, **APERTURE_EDGE_STYLE)
    ax.plot(wavelength_sorted, ap_right_arcsec, alpha=SUMMARY_TRACE_OVERLAY_ALPHA, **APERTURE_EDGE_STYLE)
    ax.plot(wavelength_sorted, bg_left_end_arcsec, alpha=SUMMARY_MASKED_OVERLAY_ALPHA, **MASK_BOUNDARY_STYLE)
    ax.plot(wavelength_sorted, bg_right_start_arcsec, alpha=SUMMARY_MASKED_OVERLAY_ALPHA, **MASK_BOUNDARY_STYLE)
    ax.plot(wavelength_sorted, bg_left_start_arcsec, alpha=SUMMARY_BACKGROUND_OVERLAY_ALPHA, **BACKGROUND_BOUNDARY_STYLE)
    ax.plot(wavelength_sorted, bg_left_end_arcsec, alpha=SUMMARY_BACKGROUND_OVERLAY_ALPHA, **BACKGROUND_BOUNDARY_STYLE)
    ax.plot(wavelength_sorted, bg_right_start_arcsec, alpha=SUMMARY_BACKGROUND_OVERLAY_ALPHA, **BACKGROUND_BOUNDARY_STYLE)
    ax.plot(wavelength_sorted, bg_right_end_arcsec, alpha=SUMMARY_BACKGROUND_OVERLAY_ALPHA, **BACKGROUND_BOUNDARY_STYLE)

    if show_bottom_wavelength_axis:
        ax.set_xlabel("Wavelength ($\\AA$)")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_ylabel("Along slit [arcsec]")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def arcsec_to_xpixel(arcsec_values):
        arcsec_values = np.asarray(arcsec_values, dtype=float)
        return arcsec_values / EFOSC2_BINNED_ARCSEC_PER_PIXEL + float(spatial_min)

    def xpixel_to_arcsec(xpixel_values):
        xpixel_values = np.asarray(xpixel_values, dtype=float)
        return (xpixel_values - float(spatial_min)) * EFOSC2_BINNED_ARCSEC_PER_PIXEL

    secondary_y = ax.secondary_yaxis("right", functions=(arcsec_to_xpixel, xpixel_to_arcsec))
    secondary_y.set_ylabel("X pixel")
    secondary_y.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, prune="upper"))
    secondary_y.yaxis.labelpad = 6

    wavelength_to_row_number, row_number_to_wavelength = build_row_number_to_wavelength_mappers(record, wavelength_solution)
    if wavelength_to_row_number is not None and row_number_to_wavelength is not None:
        secondary_x = ax.secondary_xaxis("top", functions=(wavelength_to_row_number, row_number_to_wavelength))
        secondary_x.set_xlabel("Y pixel")
        secondary_x.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, prune="upper"))
        secondary_x.xaxis.labelpad = 6

    ax.set_xlim(float(np.nanmin(wavelength_sorted)), float(np.nanmax(wavelength_sorted)))
    ax.set_ylim(
        float(np.nanmin(arcsec_centres) - 0.5 * EFOSC2_BINNED_ARCSEC_PER_PIXEL),
        float(np.nanmax(arcsec_centres) + 0.5 * EFOSC2_BINNED_ARCSEC_PER_PIXEL),
    )

    return image


def plot_row_profile_panel(ax, row_context, wavelength_solution):
    x = np.asarray(row_context["x"], dtype=float)
    row = np.asarray(row_context["row"], dtype=float)
    background_mask_full = np.asarray(row_context["background_used_mask"], dtype=bool)
    background_positions = x[background_mask_full]
    edge_pad = max(int(SUMMARY_ROW_EDGE_MASK_PIXELS), 0)
    if background_positions.size > 0:
        x_display_left = float(np.nanmin(background_positions)) - edge_pad - 0.5
        x_display_right = float(np.nanmax(background_positions)) + edge_pad + 0.5
    else:
        x_display_left = float(max(0, int(row_context["x_left"])))
        x_display_right = float(min(len(x) - 1, int(row_context["x_right"])))

    panel_mask = np.isfinite(x) & (x >= x_display_left) & (x <= x_display_right)
    ignored_mask = build_ignored_point_mask(row_context) & panel_mask
    background_mask = background_mask_full & panel_mask
    if background_positions.size > 0 and edge_pad > 0:
        outer_edge_mask = (
            ((x < float(np.nanmin(background_positions))) & (x >= float(np.nanmin(background_positions)) - edge_pad))
            | ((x > float(np.nanmax(background_positions))) & (x <= float(np.nanmax(background_positions)) + edge_pad))
        )
        ignored_mask |= outer_edge_mask & panel_mask
    data_mask = np.asarray(np.isfinite(row), dtype=bool) & panel_mask
    data_mask &= ~ignored_mask & ~background_mask

    ax.plot(
        x[data_mask],
        row[data_mask],
        color=ROW_DATA_STYLE["color"],
        lw=ROW_DATA_STYLE["lw"],
        ls=ROW_DATA_STYLE["ls"],
        marker=ROW_DATA_STYLE["marker"],
        ms=ROW_DATA_STYLE["ms"],
        mew=ROW_DATA_STYLE["mew"],
        label="Data",
        zorder=2,
    )

    if np.any(background_mask):
        ax.plot(
            x[background_mask],
            row[background_mask],
            color=ROW_BACKGROUND_STYLE["color"],
            linestyle="None",
            marker=ROW_BACKGROUND_STYLE["marker"],
            ms=ROW_BACKGROUND_STYLE["ms"],
            label="Background",
            zorder=4,
        )

    if np.any(ignored_mask):
        ax.plot(
            x[ignored_mask],
            row[ignored_mask],
            color=ROW_MASKED_STYLE["color"],
            linestyle=ROW_MASKED_STYLE["linestyle"],
            marker=ROW_MASKED_STYLE["marker"],
            ms=ROW_MASKED_STYLE["ms"],
            mew=ROW_MASKED_STYLE["mew"],
            label="Masked",
            zorder=5,
        )

    ax.axvline(float(row_context["trace_centre"]), label="Aperture centre", zorder=3, **TRACE_CENTRE_STYLE)
    ax.axvline(float(row_context["ap_left"]), label="Captured region", zorder=3, **APERTURE_EDGE_STYLE)
    ax.axvline(float(row_context["ap_right"]), zorder=3, **APERTURE_EDGE_STYLE)

    ignored_boundaries = build_combined_ignored_spans(row_context)
    used_mask_label = False
    for left, right in ignored_boundaries:
        ax.axvline(float(left), label="Masked region" if not used_mask_label else None, zorder=3, **MASK_BOUNDARY_STYLE)
        ax.axvline(float(right), zorder=3, **MASK_BOUNDARY_STYLE)
        used_mask_label = True

    blue_label_used = False
    for edge in (
        row_context["bg_left_start"],
        row_context["bg_left_end"],
        row_context["bg_right_start"],
        row_context["bg_right_end"],
    ):
        ax.axvline(
            float(edge),
            label="Sky background" if not blue_label_used else None,
            zorder=3,
            **BACKGROUND_BOUNDARY_STYLE,
        )
        blue_label_used = True

    ax.set_xlim(x_display_left, x_display_right)
    ylim = compute_panel_ylim(x, [row], x_display_left, x_display_right)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("X pixel")
    ax.set_ylabel("Counts at Y = %d" % int(row_context["detector_row"]))
    ax.tick_params(axis="y", left=False, labelleft=False)

    right_axis = ax.twinx()
    right_axis.set_ylim(ax.get_ylim())
    right_axis.set_ylabel("Counts")
    right_axis.tick_params(axis="y", labelsize=13)
    right_axis.patch.set_alpha(0.0)
    right_axis.grid(False)

    legend_handles = [
        Line2D(
            [],
            [],
            color=ROW_DATA_STYLE["color"],
            marker=ROW_DATA_STYLE["marker"],
            linestyle="None",
            markersize=ROW_DATA_STYLE["ms"],
            markeredgewidth=ROW_DATA_STYLE["mew"],
            label="Data",
        ),
        Line2D(
            [],
            [],
            color=ROW_BACKGROUND_STYLE["color"],
            marker=ROW_BACKGROUND_STYLE["marker"],
            linestyle="None",
            markersize=ROW_BACKGROUND_STYLE["ms"],
            markeredgewidth=ROW_BACKGROUND_STYLE["mew"],
            label="Background",
        ),
        Line2D(
            [],
            [],
            color=ROW_MASKED_STYLE["color"],
            marker=ROW_MASKED_STYLE["marker"],
            linestyle="None",
            markersize=ROW_MASKED_STYLE["ms"],
            markeredgewidth=ROW_MASKED_STYLE["mew"],
            label="Masked",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95)


def plot_spectrum_panel(ax, spectrum_bundle, spectrum_index, selected_star, selected_wavelength=None):
    wavelength = np.asarray(spectrum_bundle["wavelength"], dtype=float)
    flux_key = "star%d_flux" % int(selected_star)
    if flux_key not in spectrum_bundle:
        raise ValueError("Spectrum bundle does not contain %s." % flux_key)
    star_flux = normalise_flux(spectrum_bundle[flux_key][spectrum_index])

    valid = np.isfinite(wavelength) & np.isfinite(star_flux) & (star_flux > 0)
    if not np.any(valid):
        raise ValueError("No finite wavelength-calibrated spectrum points were found for the selected frame.")

    ax.plot(wavelength[valid], star_flux[valid], **SPECTRUM_STYLE)
    if SUMMARY_ROW_SHOW_SELECTED_WAVELENGTH and selected_wavelength is not None and np.isfinite(selected_wavelength):
        ax.axvline(float(selected_wavelength), **SELECTED_WAVELENGTH_STYLE)
    ax.set_xlabel("Wavelength ($\\AA$)")
    ax.set_ylabel("Normalised flux")

    ymin = float(np.nanmin(star_flux[valid]))
    ymax = float(np.nanmax(star_flux[valid]))
    span = ymax - ymin
    if not np.isfinite(span) or span <= 0:
        span = 0.05
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.08 * span)
    ax.set_xlim(float(np.nanmin(wavelength[valid])), float(np.nanmax(wavelength[valid])))


def create_extraction_summary_bundle(diagnostic_input, frame_index=None, star_index=1, requested_row=None):
    frame_groups = build_frame_groups(diagnostic_input)
    records_by_frame = [load_frame_records(group) for group in frame_groups]
    frame_position, records = choose_frame(records_by_frame, requested_frame_index=frame_index)
    record = choose_star_record(records, star_index)
    row_context = build_row_context(record, requested_row)
    reduction_dir = resolve_reduction_dir(records)
    spectrum_bundle = load_resampled_spectrum_bundle(reduction_dir)
    spectrum_index = resolve_spectrum_index(
        frame_position,
        int(records[0]["frame_index"]),
        len(spectrum_bundle["star1_flux"]),
        len(records_by_frame),
    )
    selected_wavelength = np.nan
    wavelength = np.asarray(spectrum_bundle["wavelength"], dtype=float)
    if 0 <= int(row_context["row_index"]) < len(wavelength) and np.isfinite(wavelength[int(row_context["row_index"])]):
        selected_wavelength = float(wavelength[int(row_context["row_index"])])

    frame_label = resolve_science_frame_path(record).name
    output_dir = reduction_dir / "diagnostic_plots" / "extraction_summary_figures"

    return {
        "records": records,
        "record": record,
        "row_context": row_context,
        "spectrum_bundle": spectrum_bundle,
        "spectrum_index": spectrum_index,
        "frame_label": frame_label,
        "frame_index": int(record["frame_index"]),
        "star_index": int(record["star_index"]),
        "row_label": int(row_context["detector_row"]),
        "selected_wavelength": selected_wavelength,
        "output_dir": output_dir,
    }


def create_combined_summary_figure(bundle):
    fig = plt.figure(figsize=(14.2, 7.6), constrained_layout=False)
    outer_gs = GridSpec(1, 2, figure=fig, width_ratios=[1.06, 1.0], wspace=0.20)
    left_gs = outer_gs[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.02)

    raw_ax = fig.add_subplot(left_gs[0, 0])
    spectrum_ax = fig.add_subplot(left_gs[1, 0], sharex=raw_ax)
    row_ax = fig.add_subplot(outer_gs[0, 1])

    plot_raw_frame_panel(
        raw_ax,
        bundle["record"],
        bundle["row_context"],
        bundle["spectrum_bundle"]["wavelength"],
        show_bottom_wavelength_axis=False,
    )
    plot_spectrum_panel(
        spectrum_ax,
        bundle["spectrum_bundle"],
        bundle["spectrum_index"],
        bundle["star_index"],
        selected_wavelength=bundle["selected_wavelength"],
    )
    plot_row_profile_panel(row_ax, bundle["row_context"], bundle["spectrum_bundle"]["wavelength"])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.96, wspace=0.20)
    return fig


def create_single_panel_figure(kind, bundle):
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    if kind == "raw_frame":
        plot_raw_frame_panel(
            ax,
            bundle["record"],
            bundle["row_context"],
            bundle["spectrum_bundle"]["wavelength"],
            show_bottom_wavelength_axis=True,
        )
    elif kind == "row_profile":
        plot_row_profile_panel(ax, bundle["row_context"], bundle["spectrum_bundle"]["wavelength"])
    elif kind == "spectrum":
        plot_spectrum_panel(
            ax,
            bundle["spectrum_bundle"],
            bundle["spectrum_index"],
            bundle["star_index"],
            selected_wavelength=bundle["selected_wavelength"],
        )
    else:
        raise ValueError("Unknown panel kind: %s" % str(kind))
    fig.tight_layout()
    return fig


def save_summary_figures(bundle, output_dir=None, dpi=300):
    output_path = Path(output_dir) if output_dir is not None else Path(bundle["output_dir"])
    output_path = output_path.expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = "frame_%05d_star_%d_row_%04d" % (
        int(bundle["frame_index"]),
        int(bundle["star_index"]),
        int(bundle["row_label"]),
    )

    combined_fig = create_combined_summary_figure(bundle)
    combined_path = output_path / ("%s_extraction_summary_combined.pdf" % base_name)
    combined_fig.savefig(combined_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(combined_fig)

    panel_paths = {}
    for kind in ("raw_frame", "row_profile", "spectrum"):
        fig = create_single_panel_figure(kind, bundle)
        out_path = output_path / ("%s_%s.pdf" % (base_name, kind))
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        panel_paths[kind] = out_path

    return {
        "output_dir": output_path,
        "combined": combined_path,
        "panels": panel_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Create a publication-style extraction summary figure from saved reduction diagnostics.")
    parser.add_argument("diagnostic_input", help="Reduction directory, extraction_diagnostics directory, or a single diagnostic .npz file.")
    parser.add_argument("--frame-index", type=int, help="Diagnostic frame index to plot. Defaults to the middle saved frame.")
    parser.add_argument("--star-index", type=int, default=1, help="Star index to annotate in the raw-frame and row-profile panels.")
    parser.add_argument("-r", "--row", type=int, help="Detector row number or 0-based local row index for the extraction-row panel.")
    parser.add_argument("--output-dir", help="Directory for the saved PDF figures. Defaults to diagnostic_plots/extraction_summary_figures inside the reduction directory.")
    parser.add_argument("--dpi", type=int, default=300, help="Save DPI for the PDF figures.")
    parser.add_argument("--show", action="store_true", help="Also show the combined figure interactively after saving.")
    args = parser.parse_args()

    bundle = create_extraction_summary_bundle(
        args.diagnostic_input,
        frame_index=args.frame_index,
        star_index=args.star_index,
        requested_row=args.row,
    )
    saved = save_summary_figures(bundle, output_dir=args.output_dir, dpi=max(int(args.dpi), 72))

    print("Saved combined summary to %s" % str(saved["combined"]))
    for kind, path in saved["panels"].items():
        print("Saved %s panel to %s" % (kind, str(path)))

    if args.show:
        fig = create_combined_summary_figure(bundle)
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
