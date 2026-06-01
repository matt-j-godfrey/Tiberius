import argparse
import pickle
import re
import shutil
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.patches import Rectangle


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


def get_record_scalar(record, key, default=np.nan):
    value = record.get(key, default)
    try:
        return value.item()
    except Exception:
        return value


def get_profile_aggregation_mode(record):
    return str(record.get("profile_aggregation_mode", "row_by_row"))


def describe_profile_aggregation_mode(mode):
    mode = str(mode)
    if mode == "summed_per_exposure":
        return "summed per exposure"
    if mode == "normalized_summed_per_exposure":
        return "summed per exposure (normalized)"
    return "row by row"


def normalize_excluded_column_mode(mode):
    text = "asymmetric" if mode is None else str(mode).strip().lower()
    aliases = {
        "mask": "mask_only",
        "masked": "mask_only",
        "interpolated": "interpolate",
    }
    return aliases.get(text, text)


def mask_values_in_column_strips(values, excluded_column_ranges):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or excluded_column_ranges is None or len(excluded_column_ranges) == 0:
        return np.zeros(values.shape, dtype=bool)

    mask = np.zeros(values.shape, dtype=bool)
    for left, right in excluded_column_ranges:
        mask |= (values >= float(left)) & (values < float(right))
    return mask


def interpolate_masked_columns_1d(row, column_mask):
    row = np.asarray(row, dtype=float)
    column_mask = np.asarray(column_mask, dtype=bool)
    if row.size == 0 or not np.any(column_mask):
        return np.array(row, copy=True)

    result = np.array(row, copy=True)
    x = np.arange(len(result), dtype=float)
    good_mask = np.isfinite(result) & (~column_mask)
    if np.count_nonzero(good_mask) == 0:
        result[column_mask] = np.nan
        return result
    if np.count_nonzero(good_mask) == 1:
        result[column_mask] = result[good_mask][0]
        return result

    result[column_mask] = np.interp(x[column_mask], x[good_mask], result[good_mask])
    return result


def _coerce_strip_array(raw):
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0:
        return np.empty((0, 2), dtype=float)
    raw = np.atleast_2d(raw)
    if raw.shape[-1] != 2:
        raw = raw.reshape(-1, 2)
    return raw


def get_record_excluded_column_ranges(record, row_index=None):
    if row_index is not None and "effective_excluded_column_strips" in record:
        raw = np.asarray(record.get("effective_excluded_column_strips", np.empty((0, 0, 2))), dtype=float)
        if raw.ndim == 3 and 0 <= int(row_index) < raw.shape[0]:
            row_raw = _coerce_strip_array(raw[int(row_index)])
            ranges = [
                (float(left), float(right))
                for left, right in row_raw
                if np.isfinite(left) and np.isfinite(right) and right > left
            ]
            if len(ranges) > 0:
                return ranges

    raw = _coerce_strip_array(record.get("excluded_column_strips", np.array([], dtype=float)))
    return [(float(left), float(right)) for left, right in raw if np.isfinite(left) and np.isfinite(right) and right > left]


def prepare_row_for_excluded_columns(row, excluded_column_ranges, mode):
    row = np.asarray(row, dtype=float)
    if row.size == 0:
        return np.array(row, copy=True), np.zeros(row.shape, dtype=bool)

    excluded_mask = mask_values_in_column_strips(np.arange(len(row), dtype=float), excluded_column_ranges)
    if not np.any(excluded_mask):
        return np.array(row, copy=True), excluded_mask

    mode = normalize_excluded_column_mode(mode)
    if mode == "interpolate":
        return interpolate_masked_columns_1d(row, excluded_mask), excluded_mask

    result = np.array(row, copy=True)
    if mode == "mask_only":
        result[excluded_mask] = np.nan
    return result, excluded_mask


def get_excluded_column_display_label(mode):
    mode = normalize_excluded_column_mode(mode)
    if mode == "mask_only":
        return "masked bad columns"
    if mode == "interpolate":
        return "interpolated bad columns"
    return "excluded bad columns"


def compute_relative_excluded_spans(trace, excluded_column_ranges):
    trace = np.asarray(trace, dtype=float)
    finite_trace = trace[np.isfinite(trace)]
    if finite_trace.size == 0 or excluded_column_ranges is None or len(excluded_column_ranges) == 0:
        return []

    spans = []
    for left, right in excluded_column_ranges:
        rel_left = float(np.nanmedian(float(left) - finite_trace))
        rel_right = float(np.nanmedian(float(right) - finite_trace))
        if np.isfinite(rel_left) and np.isfinite(rel_right) and rel_right > rel_left:
            spans.append((rel_left, rel_right))
    return spans


def get_record_relative_excluded_spans(record):
    trace = np.asarray(record.get("trace", np.array([])), dtype=float)
    effective = np.asarray(record.get("effective_excluded_column_strips", np.empty((0, 0, 2))), dtype=float)
    if effective.ndim == 3 and effective.shape[0] > 0 and effective.shape[2] == 2 and trace.size >= effective.shape[0]:
        spans = []
        for strip_index in range(effective.shape[1]):
            left_values = []
            right_values = []
            for row_index in range(effective.shape[0]):
                left = effective[row_index, strip_index, 0]
                right = effective[row_index, strip_index, 1]
                if not np.isfinite(left) or not np.isfinite(right) or right <= left:
                    continue
                if not np.isfinite(trace[row_index]):
                    continue
                left_values.append(float(left - trace[row_index]))
                right_values.append(float(right - trace[row_index]))
            if len(left_values) > 0:
                spans.append((float(np.nanmedian(left_values)), float(np.nanmedian(right_values))))
        if len(spans) > 0:
            return spans

    return compute_relative_excluded_spans(trace, get_record_excluded_column_ranges(record))


def shade_relative_excluded_spans(ax, relative_spans, mode, alpha=0.15):
    if relative_spans is None or len(relative_spans) == 0:
        return

    used_label = False
    for left, right in relative_spans:
        ax.axvspan(
            float(left),
            float(right),
            color="gold",
            alpha=alpha,
            lw=0,
            label=get_excluded_column_display_label(mode) if not used_label else None,
        )
        used_label = True


def interpolate_profile_to_relative_grid(relative_x, x_rel, profile_values):
    relative_x = np.asarray(relative_x, dtype=float)
    x_rel = np.asarray(x_rel, dtype=float)
    profile_values = np.asarray(profile_values, dtype=float)

    finite = np.isfinite(x_rel) & np.isfinite(profile_values)
    if np.count_nonzero(finite) < 2:
        return np.full(relative_x.shape, np.nan, dtype=np.float32)

    x_rel_finite = x_rel[finite]
    profile_finite = profile_values[finite]
    interpolated = np.interp(relative_x, x_rel_finite, profile_finite, left=np.nan, right=np.nan)
    x_min = float(np.nanmin(x_rel_finite))
    x_max = float(np.nanmax(x_rel_finite))
    interpolated[(relative_x < x_min) | (relative_x > x_max)] = np.nan
    return interpolated.astype(np.float32)


def uses_summed_profile(record):
    return bool(record.get("profile_enabled", False)) and get_profile_aggregation_mode(record) in (
        "summed_per_exposure",
        "normalized_summed_per_exposure",
    )


def uses_normalized_summed_profile(record):
    return bool(record.get("profile_enabled", False)) and get_profile_aggregation_mode(record) == "normalized_summed_per_exposure"


def evaluate_aggregate_profile(record, x, include_offset=True):
    model_name = str(record.get("profile_model", "fixed"))
    if model_name == "empirical":
        return None

    amplitude = float(get_record_scalar(record, "aggregate_profile_amplitude", np.nan))
    centre = float(get_record_scalar(record, "aggregate_profile_centre", 0.0))
    scale = float(get_record_scalar(record, "aggregate_profile_scale", np.nan))
    offset = float(get_record_scalar(record, "aggregate_profile_offset", 0.0)) if include_offset else 0.0

    if model_name == "gaussian":
        return gaussian_profile(x, amplitude, centre, scale, offset)
    if model_name == "moffat":
        beta = float(get_record_scalar(record, "aggregate_profile_beta", np.nan))
        return moffat_profile(x, amplitude, centre, scale, beta, offset)
    if model_name == "double_gaussian":
        sigma_wing = float(get_record_scalar(record, "aggregate_profile_scale_secondary", np.nan))
        wing_fraction = float(get_record_scalar(record, "aggregate_profile_mix_fraction", np.nan))
        return double_gaussian_profile(x, amplitude, centre, scale, sigma_wing, wing_fraction, offset)
    raise ValueError("Unsupported profile model '%s'" % model_name)


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


def compute_summed_left_ylim(source_sum, interp_y):
    data_series = []
    for values in (source_sum, interp_y):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            data_series.append(finite)

    if len(data_series) == 0:
        return None

    y = np.concatenate(data_series)
    y_max = float(np.nanmax(y))
    lower_anchor = min(0.0, float(np.nanpercentile(y, 1)))
    span = max(y_max - lower_anchor, 1.0)
    lower = lower_anchor - max(1.0, 0.05 * span)
    if y_max <= lower:
        y_max = lower + max(1.0, 0.05 * span)
    return lower, y_max


def compute_summed_overlay_ylim(row_stack):
    row_stack = np.asarray(row_stack, dtype=float)
    finite = row_stack[np.isfinite(row_stack)]
    if finite.size == 0:
        return None

    y_max = float(np.nanmax(finite))
    lower_anchor = min(0.0, float(np.nanpercentile(finite, 1)))
    span = max(y_max - lower_anchor, 1.0)
    lower = min(
        lower_anchor - max(0.02, 0.03 * max(y_max, 1.0), 0.04 * span),
        -0.05 * max(y_max, 1.0),
    )
    if y_max <= lower:
        y_max = lower + max(1.0, 0.05 * span)
    return lower, y_max


def compute_zoom_wing_ylim(context):
    x = np.asarray(context["x"], dtype=float)
    row_sub = np.asarray(context["row_background_subtracted"], dtype=float)
    zoom_left = float(context["zoom_left"])
    zoom_right = float(context["zoom_right"])
    trace_centre = float(context["trace_centre"])
    aperture_width = max(float(context["ap_right"] - context["ap_left"]), 1.0)
    zoom_width = max(zoom_right - zoom_left, 1.0)

    exclusion_half_width = max(
        float(context.get("core_exclusion_half_width", 0.0)),
        0.18 * aperture_width,
        0.06 * zoom_width,
        3.0,
    )
    edge_band_width = max(2.0, 0.12 * aperture_width)

    raw_zoom_mask = np.isfinite(x) & np.isfinite(row_sub) & (x >= zoom_left) & (x <= zoom_right)
    raw_wing_mask = raw_zoom_mask & (np.abs(x - trace_centre) >= exclusion_half_width)
    raw_edge_mask = raw_wing_mask & (np.abs(np.abs(x - trace_centre) - exclusion_half_width) <= edge_band_width)

    wing_values = []
    edge_values = []

    if np.any(raw_wing_mask):
        wing_values.append(row_sub[raw_wing_mask])
    if np.any(raw_edge_mask):
        edge_values.append(row_sub[raw_edge_mask])

    if context["source_x"] is not None and context["source_y"] is not None:
        source_x = np.asarray(context["source_x"], dtype=float)
        source_y = np.asarray(context["source_y"], dtype=float)
        source_zoom_mask = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & (source_x >= zoom_left)
            & (source_x <= zoom_right)
        )
        source_wing_mask = source_zoom_mask & (np.abs(source_x - trace_centre) >= exclusion_half_width)
        source_edge_mask = source_wing_mask & (
            np.abs(np.abs(source_x - trace_centre) - exclusion_half_width) <= edge_band_width
        )
        if np.any(source_wing_mask):
            wing_values.append(source_y[source_wing_mask])
        if np.any(source_edge_mask):
            edge_values.append(source_y[source_edge_mask])

    if len(wing_values) == 0:
        return compute_data_ylim((x, row_sub, zoom_left, zoom_right), padding_frac=0.03, min_pad=1.0)

    wing_values = np.concatenate([values[np.isfinite(values)] for values in wing_values if np.any(np.isfinite(values))])
    if len(wing_values) == 0:
        return compute_data_ylim((x, row_sub, zoom_left, zoom_right), padding_frac=0.03, min_pad=1.0)

    if len(edge_values) > 0:
        edge_values = np.concatenate([values[np.isfinite(values)] for values in edge_values if np.any(np.isfinite(values))])
    else:
        edge_values = np.array([], dtype=float)

    if len(edge_values) > 0:
        upper_ref = float(np.nanmax(edge_values))
    else:
        upper_ref = float(np.nanpercentile(wing_values, 92))

    lower_ref = float(np.nanpercentile(wing_values, 5))
    wing_span = max(upper_ref - lower_ref, 1.0)

    upper = upper_ref + max(0.10 * wing_span, 2.0)
    lower = min(0.0, lower_ref) - max(0.08 * wing_span, 2.0)
    lower = max(lower, -0.35 * max(upper, 1.0))

    peak_value = np.nan
    if context["source_y"] is not None:
        source_finite = np.asarray(context["source_y"], dtype=float)
        if np.any(np.isfinite(source_finite)):
            peak_value = float(np.nanmax(source_finite))
    if not np.isfinite(peak_value):
        row_finite = row_sub[np.isfinite(row_sub)]
        if len(row_finite) > 0:
            peak_value = float(np.nanmax(row_finite))

    if np.isfinite(peak_value) and peak_value > 0:
        upper = max(upper, 0.10 * peak_value)

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return compute_data_ylim((x, row_sub, zoom_left, zoom_right), padding_frac=0.03, min_pad=1.0)

    return lower, upper


def parse_rows(row_text):
    if row_text is None:
        return None
    return [int(x.strip()) for x in row_text.split(",") if x.strip() != ""]


def parse_index_spec(index_text):
    if index_text is None or str(index_text).strip() == "":
        return []

    indices = []
    for chunk in str(index_text).split(","):
        chunk = chunk.strip()
        if chunk == "":
            continue
        if ":" in chunk:
            start_text, end_text = chunk.split(":", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                start, end = end, start
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(chunk))
    return sorted(set(indices))


def parse_pixel_range_spec(range_text):
    if range_text is None or str(range_text).strip() == "":
        return None

    ranges = []
    for chunk in str(range_text).split(","):
        chunk = chunk.strip()
        if chunk == "":
            continue
        if ":" not in chunk:
            value = int(chunk)
            ranges.append((value, value))
            continue
        start_text, end_text = chunk.split(":", 1)
        start = int(start_text.strip())
        end = int(end_text.strip())
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges


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
    record["profile_aggregation_mode"] = str(record.get("profile_aggregation_mode", np.array("row_by_row")).item())
    record["excluded_column_mode"] = str(record.get("excluded_column_mode", np.array("asymmetric")).item())
    record["region_aperture"] = int(record["region_aperture"])
    record["region_background_used"] = int(record["region_background_used"])
    record["region_background_rejected"] = int(record["region_background_rejected"])
    record["region_contaminant"] = int(record["region_contaminant"])
    record["region_profile_fit"] = int(record["region_profile_fit"])
    if "aggregate_profile_success" in record:
        record["aggregate_profile_success"] = bool(record["aggregate_profile_success"].item())
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
    excluded_column_ranges = get_record_excluded_column_ranges(record, row_index=row_index)
    excluded_column_mode = normalize_excluded_column_mode(record.get("excluded_column_mode", "asymmetric"))
    row_used, excluded_column_mask = prepare_row_for_excluded_columns(
        row,
        excluded_column_ranges,
        excluded_column_mode,
    )
    row_background = record["background_model"][row_index]
    finite_background = np.isfinite(row_background)

    row_background_subtracted = row.copy()
    row_background_subtracted[finite_background] = row[finite_background] - row_background[finite_background]
    row_background_subtracted_used = row_used.copy()
    row_background_subtracted_used[finite_background] = row_used[finite_background] - row_background[finite_background]

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

    zoom_pad = 15
    zoom_left = max(0, ap_left - zoom_pad)
    zoom_right = min(len(x) - 1, ap_right + zoom_pad)

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
    if record["profile_enabled"] and bool(record["profile_success"][row_index]) and not uses_summed_profile(record):
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
        "row_used": row_used,
        "row_background": row_background,
        "row_background_subtracted": row_background_subtracted,
        "row_background_subtracted_used": row_background_subtracted_used,
        "finite_background": finite_background,
        "aperture_mask": aperture_mask,
        "background_used_mask": background_used_mask,
        "background_rejected_mask": background_rejected_mask,
        "contaminant_mask": contaminant_mask,
        "profile_fit_mask": profile_fit_mask,
        "excluded_column_ranges": excluded_column_ranges,
        "excluded_column_mode": excluded_column_mode,
        "excluded_column_mask": excluded_column_mask,
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
        "profile_aggregation_mode": get_profile_aggregation_mode(record),
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
    if len(context["excluded_column_ranges"]) > 0:
        lines.append("excluded columns = %s" % context["excluded_column_mode"])

    if uses_summed_profile(record):
        lines.append("aggregation = %s" % describe_profile_aggregation_mode(get_profile_aggregation_mode(record)))
        rows_used = int(get_record_scalar(record, "aggregate_profile_rows_used", 0))
        if rows_used > 0:
            lines.append("rows summed = %d" % rows_used)
        aggregate_ref_width = float(get_record_scalar(record, "aggregate_profile_reference_width", np.nan))
        aggregate_min_width = float(get_record_scalar(record, "aggregate_profile_min_width", np.nan))
        aggregate_max_width = float(get_record_scalar(record, "aggregate_profile_max_width", np.nan))
        if np.isfinite(aggregate_ref_width) and aggregate_ref_width > 0:
            lines.append("final width (median) = %.2f px" % aggregate_ref_width)
        if np.isfinite(aggregate_min_width) and np.isfinite(aggregate_max_width) and aggregate_max_width > aggregate_min_width:
            lines.append("final width range = %.2f -> %.2f px" % (aggregate_min_width, aggregate_max_width))

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


def add_aperture_width_box(ax, context):
    aperture_width = int(context["ap_right"] - context["ap_left"])
    left_half_width = float(context["trace_centre"] - context["ap_left"])
    right_half_width = float(context["ap_right"] - context["trace_centre"])
    text = (
        "width = %d px\n"
        "left = %.2f px\n"
        "right = %.2f px"
    ) % (aperture_width, left_half_width, right_half_width)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "0.8", "boxstyle": "round,pad=0.35"},
    )


def add_relative_aperture_edge_lines(ax, left_edge, right_edge, color="tab:red", label="final aperture"):
    if not (np.isfinite(left_edge) and np.isfinite(right_edge) and right_edge > left_edge):
        return

    ax.axvline(float(left_edge), color=color, lw=1.0, label=label)
    ax.axvline(float(right_edge), color=color, lw=1.0)


def shade_region(ax, start, end, color, alpha, label=None):
    if start >= 0 and end > start:
        ax.axvspan(start, end, color=color, alpha=alpha, label=label)


def shade_excluded_column_ranges(ax, excluded_column_ranges, x_left, x_right, label="excluded columns"):
    if excluded_column_ranges is None or len(excluded_column_ranges) == 0:
        return

    used_label = False
    for left, right in excluded_column_ranges:
        start = max(float(left), float(x_left))
        end = min(float(right), float(x_right))
        if end <= start:
            continue
        ax.axvspan(
            start,
            end,
            color="gold",
            alpha=0.15,
            lw=0,
            label=label if not used_label else None,
        )
        used_label = True


def shade_record_excluded_columns_on_frame(ax, record, fallback_ranges=None, label="excluded columns"):
    effective = np.asarray(record.get("effective_excluded_column_strips", np.empty((0, 0, 2))), dtype=float)
    if effective.ndim == 3 and effective.shape[0] > 0 and effective.shape[2] == 2:
        y = np.arange(effective.shape[0], dtype=float)
        used_label = False
        for strip_index in range(effective.shape[1]):
            left_values = effective[:, strip_index, 0]
            right_values = effective[:, strip_index, 1]
            finite = np.isfinite(left_values) & np.isfinite(right_values) & (right_values > left_values)
            if not np.any(finite):
                continue
            ax.fill_betweenx(
                y[finite],
                left_values[finite],
                right_values[finite],
                color="gold",
                alpha=0.15,
                lw=0,
                label=label if not used_label else None,
            )
            used_label = True
        if used_label:
            return

    shade_excluded_column_ranges(
        ax,
        fallback_ranges if fallback_ranges is not None else get_record_excluded_column_ranges(record),
        0,
        record["frame_data"].shape[1] - 1,
        label=label,
    )


def plot_frame_panel(ax, context):
    record = context["record"]
    frame_data = record["frame_data"]
    vmin, vmax = np.nanpercentile(frame_data, [5, 95])
    ax.imshow(frame_data, vmin=vmin, vmax=vmax, aspect="auto")
    shade_record_excluded_columns_on_frame(ax, record, fallback_ranges=context["excluded_column_ranges"])
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
    shade_excluded_column_ranges(
        ax,
        context["excluded_column_ranges"],
        context["x_left"],
        context["x_right"],
    )

    ax.plot(x, row, linestyle="None", marker=".", ms=3, color="black", label="raw data")
    if len(context["excluded_column_ranges"]) > 0 and context["excluded_column_mode"] in ("mask_only", "interpolate"):
        used_label = "used data"
        if context["excluded_column_mode"] == "interpolate":
            used_label = "used data (interpolated)"
        elif context["excluded_column_mode"] == "mask_only":
            used_label = "used data (masked)"
        ax.plot(x, context["row_used"], color="tab:orange", lw=1.0, alpha=0.9, label=used_label)
        if context["excluded_column_mode"] == "interpolate" and np.any(context["excluded_column_mask"]):
            interp_x = x[context["excluded_column_mask"]]
            interp_y = context["row_used"][context["excluded_column_mask"]]
            finite = np.isfinite(interp_y)
            if np.any(finite):
                ax.plot(
                    interp_x[finite],
                    interp_y[finite],
                    linestyle="None",
                    marker="o",
                    ms=3.5,
                    mfc="none",
                    mec="tab:orange",
                    label="interpolated columns",
                )
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
    row_sub_used = context["row_background_subtracted_used"]

    for i, (start, end) in enumerate(context["ignored_spans"]):
        shade_region(ax, start, end, "0.6", 0.10, "ignored" if i == 0 else None)
    shade_region(ax, context["ap_left"], context["ap_right"], "tab:red", 0.10, "trace kept")
    shade_excluded_column_ranges(
        ax,
        context["excluded_column_ranges"],
        context["zoom_left"],
        context["zoom_right"],
    )

    ax.plot(x, row_sub, linestyle="None", marker=".", ms=3, color="black", label="raw - background")
    if len(context["excluded_column_ranges"]) > 0 and context["excluded_column_mode"] in ("mask_only", "interpolate"):
        used_label = "used - background"
        if context["excluded_column_mode"] == "interpolate":
            used_label = "used - background (interpolated)"
        elif context["excluded_column_mode"] == "mask_only":
            used_label = "used - background (masked)"
        ax.plot(x, row_sub_used, color="tab:orange", lw=1.0, alpha=0.9, label=used_label)
        if context["excluded_column_mode"] == "interpolate" and np.any(context["excluded_column_mask"]):
            interp_x = x[context["excluded_column_mask"]]
            interp_y = row_sub_used[context["excluded_column_mask"]]
            finite = np.isfinite(interp_y)
            if np.any(finite):
                ax.plot(
                    interp_x[finite],
                    interp_y[finite],
                    linestyle="None",
                    marker="o",
                    ms=3.5,
                    mfc="none",
                    mec="tab:orange",
                    label="interpolated columns",
                )
    if context["source_y"] is not None and context["source_x"] is not None:
        ax.plot(context["source_x"], context["source_y"], color="tab:purple", lw=1.2, label="%s source" % record["profile_model"])
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax.axvline(context["trace_centre"], color="0.35", ls="--", lw=1.0, label="trace centre")
    ax.set_xlim(context["zoom_left"], context["zoom_right"])
    ylim = compute_zoom_wing_ylim(context)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("X pixel")
    ax.set_ylabel("Background-subtracted counts")
    ax.set_title("Star %d fit-window zoom" % record["star_index"])
    add_aperture_width_box(ax, context)
    ax.legend(loc="upper right", fontsize=8, framealpha=1)


def prepare_metric_series(frame_groups):
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
    metric_data = {}
    for star_index in star_indices:
        sample_record = None
        for frame_records in records_by_frame:
            if star_index in frame_records:
                sample_record = frame_records[star_index]
                break
        if sample_record is None:
            continue
        row_numbers = sample_record["row_numbers"]
        metric_data[star_index] = {
            "frame_indices": frame_indices,
            "row_numbers": row_numbers,
            "profile_width": build_metric_matrix(records_by_frame, star_index, "profile_width", row_numbers),
        }

    return metric_data


def plot_enclosed_width_panel(ax, metric_context, context):
    frame_indices = metric_context["frame_indices"]
    row_numbers = metric_context["row_numbers"]
    metric_matrix = metric_context["profile_width"]

    image, colorbar_label = plot_width_heatmap(
        ax,
        metric_matrix,
        frame_indices,
        row_numbers,
        "Star %d enclosed width" % context["record"]["star_index"],
        "Pixels",
    )

    frame_index = int(context["record"]["frame_index"])
    detector_row = int(context["detector_row"])
    highlight = Rectangle(
        (frame_index - 0.5, detector_row - 0.5),
        1.0,
        1.0,
        fill=False,
        edgecolor="red",
        linewidth=1.6,
    )
    ax.add_patch(highlight)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)


def create_single_frame_figure(records, requested_row, metric_data=None):
    nstars = len(records)
    show_metric_history = metric_data is not None
    if show_metric_history:
        fig = plt.figure(figsize=(24, 5.2 * nstars))
        gs = fig.add_gridspec(nstars, 4, width_ratios=[1.0, 1.2, 1.0, 0.9])
    else:
        fig = plt.figure(figsize=(18.5, 5.2 * nstars))
        gs = fig.add_gridspec(nstars, 3, width_ratios=[1.25, 1.0, 0.9])

    contexts = [build_row_context(record, requested_row) for record in records]

    for i, context in enumerate(contexts):
        if show_metric_history:
            if context["record"]["star_index"] in metric_data:
                plot_enclosed_width_panel(fig.add_subplot(gs[i, 0]), metric_data[context["record"]["star_index"]], context)
            else:
                placeholder_ax = fig.add_subplot(gs[i, 0])
                placeholder_ax.text(0.5, 0.5, "No enclosed-width history", ha="center", va="center")
                placeholder_ax.set_axis_off()
            plot_frame_panel(fig.add_subplot(gs[i, 1]), context)
            plot_row_panel(fig.add_subplot(gs[i, 2]), context)
            plot_zoom_panel(fig.add_subplot(gs[i, 3]), context)
        else:
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


def build_summed_profile_context(record):
    if not uses_summed_profile(record):
        raise ValueError("Record does not contain summed-profile diagnostics.")

    x = np.asarray(record.get("aggregate_profile_x", np.array([])), dtype=float)
    source_sum = np.asarray(record.get("aggregate_profile_source_sum", np.array([])), dtype=float)
    contributor_counts = np.asarray(record.get("aggregate_profile_contributor_counts", np.array([])), dtype=float)
    row_stack = np.asarray(record.get("aggregate_profile_row_stack", np.empty((0, 0))), dtype=float)
    interp_x = np.asarray(record.get("aggregate_profile_interp_x", np.array([])), dtype=float)
    interp_y = np.asarray(record.get("aggregate_profile_interp_y", np.array([])), dtype=float)

    if x.size == 0 or source_sum.size == 0:
        raise ValueError("Summed-profile diagnostics were requested but the saved arrays are empty.")

    if row_stack.ndim != 2 or row_stack.shape[1] != len(x):
        row_stack = np.empty((0, len(x)), dtype=float)

    overlay_ylim = compute_summed_overlay_ylim(row_stack)
    left_ylim = compute_summed_left_ylim(source_sum, interp_y)

    reference_width = float(get_record_scalar(record, "aggregate_profile_reference_width", np.nan))
    reference_radius = 0.5 * reference_width if np.isfinite(reference_width) and reference_width > 0 else np.nan
    fit_left = float(get_record_scalar(record, "aggregate_profile_fit_left", np.nan))
    fit_right = float(get_record_scalar(record, "aggregate_profile_fit_right", np.nan))
    rows_used = int(get_record_scalar(record, "aggregate_profile_rows_used", 0))
    core_exclusion_half_width = float(record.get("profile_core_exclusion_half_width", 0.0))
    profile_success = bool(get_record_scalar(record, "aggregate_profile_success", False))
    aggregation_mode = get_profile_aggregation_mode(record)
    normalized_rows = uses_normalized_summed_profile(record)
    excluded_column_mode = normalize_excluded_column_mode(record.get("excluded_column_mode", "asymmetric"))

    model_x = np.array([], dtype=float)
    model_y = np.array([], dtype=float)
    if profile_success and record["profile_model"] != "empirical":
        model_x = interp_x if interp_x.size > 0 else x
        model_y = evaluate_aggregate_profile(record, model_x, include_offset=False)

    mean_profile = np.nanmean(row_stack, axis=0) if row_stack.shape[0] > 0 else np.array([], dtype=float)
    median_profile = np.nanmedian(row_stack, axis=0) if row_stack.shape[0] > 0 else np.array([], dtype=float)
    trace = np.asarray(record.get("trace", np.array([])), dtype=float)
    aperture_left = np.asarray(record.get("aperture_left", np.array([])), dtype=float)
    aperture_right = np.asarray(record.get("aperture_right", np.array([])), dtype=float)
    relative_excluded_spans = get_record_relative_excluded_spans(record)
    display_ap_left = -reference_radius if np.isfinite(reference_radius) else np.nan
    display_ap_right = reference_radius if np.isfinite(reference_radius) else np.nan
    if trace.size == aperture_left.size == aperture_right.size and trace.size > 0:
        finite_edge_mask = np.isfinite(trace) & np.isfinite(aperture_left) & np.isfinite(aperture_right) & (aperture_right > aperture_left)
        if np.any(finite_edge_mask):
            display_ap_left = float(np.nanmedian(aperture_left[finite_edge_mask] - trace[finite_edge_mask]))
            display_ap_right = float(np.nanmedian(aperture_right[finite_edge_mask] - trace[finite_edge_mask]))

    return {
        "record": record,
        "x": x,
        "source_sum": source_sum,
        "contributor_counts": contributor_counts,
        "row_stack": row_stack,
        "interp_x": interp_x,
        "interp_y": interp_y,
        "model_x": model_x,
        "model_y": model_y,
        "reference_radius": reference_radius,
        "fit_left": fit_left,
        "fit_right": fit_right,
        "rows_used": rows_used,
        "core_exclusion_half_width": core_exclusion_half_width,
        "left_ylim": left_ylim,
        "overlay_ylim": overlay_ylim,
        "mean_profile": mean_profile,
        "median_profile": median_profile,
        "aggregation_mode": aggregation_mode,
        "aggregation_label": describe_profile_aggregation_mode(aggregation_mode),
        "normalized_rows": normalized_rows,
        "left_title": (
            "Star %d peak-normalized summed profile" % record["star_index"]
            if normalized_rows
            else "Star %d summed profile" % record["star_index"]
        ),
        "overlay_title": (
            "Star %d peak-normalized rows overlaid" % record["star_index"]
            if normalized_rows
            else "Star %d individual rows overlaid" % record["star_index"]
        ),
        "left_ylabel": (
            "Summed peak-normalized source"
            if normalized_rows
            else "Summed bg-subtracted counts"
        ),
        "overlay_ylabel": (
            "Peak-normalized per-row source"
            if normalized_rows
            else "Per-row bg-subtracted counts"
        ),
        "left_data_label": (
            "normalized-row sum"
            if normalized_rows
            else "summed data"
        ),
        "excluded_column_mode": excluded_column_mode,
        "relative_excluded_spans": relative_excluded_spans,
        "display_ap_left": display_ap_left,
        "display_ap_right": display_ap_right,
    }


def build_aligned_profile_context(record, display_pad=6):
    frame_data = np.asarray(record.get("frame_data", np.empty((0, 0))), dtype=float)
    background_model = np.asarray(record.get("background_model", np.empty((0, 0))), dtype=float)
    trace = np.asarray(record.get("trace", np.array([])), dtype=float)
    aperture_left = np.asarray(record.get("aperture_left", np.array([])), dtype=float)
    aperture_right = np.asarray(record.get("aperture_right", np.array([])), dtype=float)
    excluded_column_ranges = get_record_excluded_column_ranges(record)
    excluded_column_mode = normalize_excluded_column_mode(record.get("excluded_column_mode", "asymmetric"))

    if frame_data.ndim != 2 or frame_data.shape[0] == 0:
        raise ValueError("Record does not contain frame rows for aligned exposure diagnostics.")

    nrows, ncols = frame_data.shape
    if background_model.shape != frame_data.shape:
        background_model = np.zeros_like(frame_data, dtype=float)

    left_half_width = 0.0
    right_half_width = 0.0
    finite_edges = (
        np.isfinite(trace)
        & np.isfinite(aperture_left)
        & np.isfinite(aperture_right)
        & (aperture_right > aperture_left)
    )
    if np.any(finite_edges):
        left_half_width = float(np.nanmax(trace[finite_edges] - aperture_left[finite_edges]))
        right_half_width = float(np.nanmax(aperture_right[finite_edges] - trace[finite_edges]))

    effective_half_width = max(left_half_width, right_half_width, 8.0) + max(float(display_pad), 0.0)
    relative_x = np.arange(
        -int(np.ceil(effective_half_width)),
        int(np.ceil(effective_half_width)) + 1,
        dtype=float,
    )

    raw_stack = np.full((nrows, len(relative_x)), np.nan, dtype=np.float32)
    used_stack = np.full((nrows, len(relative_x)), np.nan, dtype=np.float32)
    masked_point_x = []
    masked_point_y = []
    interpolated_point_x = []
    interpolated_point_y = []

    for row_index in range(nrows):
        trace_centre = float(trace[row_index]) if row_index < len(trace) else np.nan
        if not np.isfinite(trace_centre):
            continue

        raw_row = np.asarray(frame_data[row_index], dtype=float)
        row_excluded_column_ranges = get_record_excluded_column_ranges(record, row_index=row_index)
        row_used, excluded_mask = prepare_row_for_excluded_columns(
            raw_row,
            row_excluded_column_ranges,
            excluded_column_mode,
        )
        background_row = np.asarray(background_model[row_index], dtype=float)
        finite_background = np.isfinite(background_row)

        raw_source = raw_row.copy()
        raw_source[finite_background] = raw_row[finite_background] - background_row[finite_background]
        used_source = row_used.copy()
        used_source[finite_background] = row_used[finite_background] - background_row[finite_background]

        columns = np.arange(ncols, dtype=float)
        x_rel = columns - trace_centre
        raw_stack[row_index] = interpolate_profile_to_relative_grid(relative_x, x_rel, raw_source)
        used_stack[row_index] = interpolate_profile_to_relative_grid(relative_x, x_rel, used_source)

        if np.any(excluded_mask):
            excluded_rel = x_rel[excluded_mask]
            if excluded_column_mode == "mask_only":
                excluded_raw = raw_source[excluded_mask]
                finite = np.isfinite(excluded_rel) & np.isfinite(excluded_raw)
                if np.any(finite):
                    masked_point_x.append(excluded_rel[finite])
                    masked_point_y.append(excluded_raw[finite])
            elif excluded_column_mode == "interpolate":
                excluded_used = used_source[excluded_mask]
                finite = np.isfinite(excluded_rel) & np.isfinite(excluded_used)
                if np.any(finite):
                    interpolated_point_x.append(excluded_rel[finite])
                    interpolated_point_y.append(excluded_used[finite])

    raw_counts = np.sum(np.isfinite(raw_stack), axis=0).astype(int)
    used_counts = np.sum(np.isfinite(used_stack), axis=0).astype(int)
    raw_sum = np.nansum(np.where(np.isfinite(raw_stack), raw_stack, 0.0), axis=0).astype(np.float32)
    used_sum = np.nansum(np.where(np.isfinite(used_stack), used_stack, 0.0), axis=0).astype(np.float32)
    raw_sum[raw_counts == 0] = np.nan
    used_sum[used_counts == 0] = np.nan

    raw_mean = np.full(len(relative_x), np.nan, dtype=np.float32)
    used_mean = np.full(len(relative_x), np.nan, dtype=np.float32)
    if len(relative_x) > 0:
        raw_mean = np.divide(
            raw_sum,
            raw_counts,
            out=np.full(len(relative_x), np.nan, dtype=np.float32),
            where=raw_counts > 0,
        )
        used_mean = np.divide(
            used_sum,
            used_counts,
            out=np.full(len(relative_x), np.nan, dtype=np.float32),
            where=used_counts > 0,
        )

    with np.errstate(all="ignore"):
        raw_median = np.nanmedian(raw_stack, axis=0) if raw_stack.shape[0] > 0 else np.array([], dtype=float)
        used_median = np.nanmedian(used_stack, axis=0) if used_stack.shape[0] > 0 else np.array([], dtype=float)

    left_ylim = compute_summed_left_ylim(used_sum, raw_sum)
    overlay_ylim = compute_summed_overlay_ylim(used_stack)
    relative_excluded_spans = get_record_relative_excluded_spans(record)

    display_ap_left = np.nan
    display_ap_right = np.nan
    if np.any(finite_edges):
        display_ap_left = float(np.nanmedian(aperture_left[finite_edges] - trace[finite_edges]))
        display_ap_right = float(np.nanmedian(aperture_right[finite_edges] - trace[finite_edges]))

    aperture_widths = aperture_right - aperture_left
    valid_widths = aperture_widths[np.isfinite(aperture_widths) & (aperture_widths > 0)]
    median_width = float(np.nanmedian(valid_widths)) if valid_widths.size > 0 else np.nan
    min_width = float(np.nanmin(valid_widths)) if valid_widths.size > 0 else np.nan
    max_width = float(np.nanmax(valid_widths)) if valid_widths.size > 0 else np.nan

    fwhm_values = np.asarray(record.get("profile_fwhm", np.array([])), dtype=float)
    valid_fwhm = fwhm_values[np.isfinite(fwhm_values)]
    median_fwhm = float(np.nanmedian(valid_fwhm)) if valid_fwhm.size > 0 else np.nan

    masked_point_x = np.concatenate(masked_point_x) if len(masked_point_x) > 0 else np.array([], dtype=float)
    masked_point_y = np.concatenate(masked_point_y) if len(masked_point_y) > 0 else np.array([], dtype=float)
    interpolated_point_x = np.concatenate(interpolated_point_x) if len(interpolated_point_x) > 0 else np.array([], dtype=float)
    interpolated_point_y = np.concatenate(interpolated_point_y) if len(interpolated_point_y) > 0 else np.array([], dtype=float)

    mode_name = "fixed aperture"
    if bool(record.get("profile_enabled", False)):
        mode_name = "%s percentile aperture (%s)" % (
            str(record.get("profile_model", "profile")),
            describe_profile_aggregation_mode(get_profile_aggregation_mode(record)),
        )

    return {
        "record": record,
        "x": relative_x,
        "raw_sum": raw_sum,
        "used_sum": used_sum,
        "raw_stack": raw_stack,
        "used_stack": used_stack,
        "raw_mean": raw_mean,
        "used_mean": used_mean,
        "raw_median": raw_median,
        "used_median": used_median,
        "left_ylim": left_ylim,
        "overlay_ylim": overlay_ylim,
        "display_ap_left": display_ap_left,
        "display_ap_right": display_ap_right,
        "median_width": median_width,
        "min_width": min_width,
        "max_width": max_width,
        "median_fwhm": median_fwhm,
        "rows_used": int(frame_data.shape[0]),
        "excluded_column_mode": excluded_column_mode,
        "relative_excluded_spans": relative_excluded_spans,
        "masked_point_x": masked_point_x,
        "masked_point_y": masked_point_y,
        "interpolated_point_x": interpolated_point_x,
        "interpolated_point_y": interpolated_point_y,
        "mode_name": mode_name,
    }


def build_summed_profile_summary_lines(context):
    record = context["record"]
    lines = [
        "aggregation = %s" % context["aggregation_label"],
        "rows summed = %d" % int(context["rows_used"]),
    ]

    reference_width = float(get_record_scalar(record, "aggregate_profile_reference_width", np.nan))
    min_width = float(get_record_scalar(record, "aggregate_profile_min_width", np.nan))
    max_width = float(get_record_scalar(record, "aggregate_profile_max_width", np.nan))
    if np.isfinite(reference_width) and reference_width > 0:
        lines.append("final width (median) = %.2f px" % reference_width)
    if np.isfinite(context["display_ap_left"]) and np.isfinite(context["display_ap_right"]) and context["display_ap_right"] > context["display_ap_left"]:
        lines.append("median edges = [%.2f, %.2f) px" % (context["display_ap_left"], context["display_ap_right"]))
    if np.isfinite(min_width) and np.isfinite(max_width):
        lines.append("final width range = %.2f -> %.2f px" % (min_width, max_width))
    if len(context.get("relative_excluded_spans", [])) > 0:
        lines.append("excluded columns = %s" % context["excluded_column_mode"])

    enclosed_width = float(get_record_scalar(record, "aggregate_profile_radius", np.nan)) * 2.0
    fwhm = float(get_record_scalar(record, "aggregate_profile_fwhm", np.nan))
    if np.isfinite(fwhm):
        if record["profile_model"] == "empirical":
            lines.append("half-max width = %.2f px" % fwhm)
        else:
            lines.append("FWHM = %.2f px" % fwhm)
    if np.isfinite(enclosed_width):
        percentile = float(record.get("profile_percentile", np.array(np.nan)).item()) if np.ndim(record.get("profile_percentile", np.array(np.nan))) == 0 else float(record["profile_percentile"])
        lines.append("enclosed width = %.2f px (%.1f%%)" % (enclosed_width, percentile * 100.0))

    model = record["profile_model"]
    if model == "moffat":
        alpha = float(get_record_scalar(record, "aggregate_profile_scale", np.nan))
        beta = float(get_record_scalar(record, "aggregate_profile_beta", np.nan))
        if np.isfinite(alpha):
            lines.append("alpha = %.2f px, beta = %.2f" % (alpha, beta))
    elif model == "empirical":
        lines.append("empirical cumulative-light aperture")

    core_exclusion = float(record.get("profile_core_exclusion_half_width", 0.0))
    if np.isfinite(core_exclusion) and core_exclusion > 0:
        lines.append("core excluded = %.2f px" % core_exclusion)

    fit_oversampling = int(record.get("profile_fit_oversampling", 1))
    if fit_oversampling > 1:
        lines.append("fit oversampling = %dx" % fit_oversampling)

    return lines


def build_aligned_profile_summary_lines(context):
    record = context["record"]
    lines = [
        "mode = %s" % context["mode_name"],
        "rows stacked = %d" % int(context["rows_used"]),
    ]

    if np.isfinite(context["median_width"]) and context["median_width"] > 0:
        lines.append("final width (median) = %.2f px" % context["median_width"])
    if np.isfinite(context["display_ap_left"]) and np.isfinite(context["display_ap_right"]) and context["display_ap_right"] > context["display_ap_left"]:
        lines.append("median edges = [%.2f, %.2f) px" % (context["display_ap_left"], context["display_ap_right"]))
    if np.isfinite(context["min_width"]) and np.isfinite(context["max_width"]):
        lines.append("final width range = %.2f -> %.2f px" % (context["min_width"], context["max_width"]))
    if len(context.get("relative_excluded_spans", [])) > 0:
        lines.append("excluded columns = %s" % context["excluded_column_mode"])

    if bool(record.get("profile_enabled", False)) and np.isfinite(context["median_fwhm"]):
        if str(record.get("profile_model", "moffat")) == "empirical":
            lines.append("median row half-max width = %.2f px" % context["median_fwhm"])
        else:
            lines.append("median row FWHM = %.2f px" % context["median_fwhm"])
    elif not bool(record.get("profile_enabled", False)):
        lines.append("fixed extraction aperture")

    return lines


def plot_summed_profile_left_panel(ax, context):
    record = context["record"]
    x = context["x"]
    if np.isfinite(context["fit_left"]) and np.isfinite(context["fit_right"]) and context["fit_right"] > context["fit_left"]:
        shade_region(ax, context["fit_left"], context["fit_right"], "tab:orange", 0.08, "fit window")
    if context["core_exclusion_half_width"] > 0:
        shade_region(
            ax,
            -context["core_exclusion_half_width"],
            context["core_exclusion_half_width"],
            "0.4",
            0.08,
            "core excluded",
        )
    shade_relative_excluded_spans(
        ax,
        context.get("relative_excluded_spans", []),
        context.get("excluded_column_mode", "asymmetric"),
    )

    ax.plot(x, context["source_sum"], linestyle="None", marker=".", ms=4, color="black", label=context["left_data_label"])
    if context["interp_x"].size > 0 and context["interp_y"].size > 0:
        fit_oversampling = int(record.get("profile_fit_oversampling", 1))
        ax.plot(
            context["interp_x"],
            context["interp_y"],
            linestyle="None",
            marker=".",
            ms=2.0,
            color="tab:orange",
            alpha=0.65,
            label="interpolated data (%dx)" % fit_oversampling,
        )
    if context["model_x"].size > 0 and context["model_y"].size > 0:
        ax.plot(context["model_x"], context["model_y"], color="tab:purple", lw=1.4, label="%s fit" % record["profile_model"])

    ax.axhline(0.0, color="0.45", ls="--", lw=0.9, zorder=0)
    ax.axvline(0.0, color="0.35", ls="--", lw=1.0, label="trace centre")
    add_relative_aperture_edge_lines(
        ax,
        context["display_ap_left"],
        context["display_ap_right"],
        color="tab:red",
        label="final aperture",
    )
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    if context["left_ylim"] is not None:
        ax.set_ylim(*context["left_ylim"])
    ax.set_xlabel("Relative X pixel from trace centre")
    ax.set_ylabel(context["left_ylabel"])
    ax.set_title(context["left_title"])
    text = "\n".join(build_summed_profile_summary_lines(context))
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
    ax.legend(loc="upper left", fontsize=8, framealpha=1)


def plot_summed_profile_overlay_panel(ax, context):
    x = context["x"]
    row_stack = context["row_stack"]
    nrows = row_stack.shape[0]
    line_alpha = min(0.22, max(0.015, 20.0 / max(nrows, 1)))

    if context["core_exclusion_half_width"] > 0:
        shade_region(
            ax,
            -context["core_exclusion_half_width"],
            context["core_exclusion_half_width"],
            "0.4",
            0.08,
            "core excluded",
        )
    shade_relative_excluded_spans(
        ax,
        context.get("relative_excluded_spans", []),
        context.get("excluded_column_mode", "asymmetric"),
    )

    for row_profile in row_stack:
        finite = np.isfinite(row_profile)
        if np.any(finite):
            ax.plot(x[finite], row_profile[finite], color="tab:blue", alpha=line_alpha, lw=0.8)

    if context["median_profile"].size > 0:
        finite = np.isfinite(context["median_profile"])
        if np.any(finite):
            ax.plot(x[finite], context["median_profile"][finite], color="tab:orange", lw=1.4, label="median row")
    if context["mean_profile"].size > 0:
        finite = np.isfinite(context["mean_profile"])
        if np.any(finite):
            ax.plot(x[finite], context["mean_profile"][finite], color="black", lw=1.2, alpha=0.9, label="mean row")

    ax.axhline(0.0, color="0.45", ls="--", lw=0.9, zorder=0)
    ax.axvline(0.0, color="0.35", ls="--", lw=1.0, label="trace centre")
    add_relative_aperture_edge_lines(
        ax,
        context["display_ap_left"],
        context["display_ap_right"],
        color="tab:red",
        label="final aperture",
    )
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    if context["overlay_ylim"] is not None:
        ax.set_ylim(*context["overlay_ylim"])
    ax.set_xlabel("Relative X pixel from trace centre")
    ax.set_ylabel(context["overlay_ylabel"])
    ax.set_title(context["overlay_title"])
    ax.legend(loc="upper left", fontsize=8, framealpha=1)


def plot_aligned_profile_left_panel(ax, context):
    x = context["x"]
    shade_relative_excluded_spans(
        ax,
        context.get("relative_excluded_spans", []),
        context.get("excluded_column_mode", "asymmetric"),
    )
    ax.plot(x, context["raw_sum"], linestyle="None", marker=".", ms=3.5, color="black", label="raw aligned sum")
    ax.plot(x, context["used_sum"], color="tab:orange", lw=1.3, label="used aligned sum")
    ax.axhline(0.0, color="0.45", ls="--", lw=0.9, zorder=0)
    ax.axvline(0.0, color="0.35", ls="--", lw=1.0, label="trace centre")
    add_relative_aperture_edge_lines(
        ax,
        context["display_ap_left"],
        context["display_ap_right"],
        color="tab:red",
        label="final aperture",
    )
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    if context["left_ylim"] is not None:
        ax.set_ylim(*context["left_ylim"])
    ax.set_xlabel("Relative X pixel from trace centre")
    ax.set_ylabel("Summed bg-subtracted counts")
    ax.set_title("Star %d aligned exposure profile" % context["record"]["star_index"])
    text = "\n".join(build_aligned_profile_summary_lines(context))
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
    ax.legend(loc="upper left", fontsize=8, framealpha=1)


def plot_aligned_profile_overlay_panel(ax, context):
    x = context["x"]
    used_stack = context["used_stack"]
    nrows = used_stack.shape[0]
    line_alpha = min(0.22, max(0.015, 20.0 / max(nrows, 1)))

    shade_relative_excluded_spans(
        ax,
        context.get("relative_excluded_spans", []),
        context.get("excluded_column_mode", "asymmetric"),
    )

    for row_profile in used_stack:
        finite = np.isfinite(row_profile)
        if np.any(finite):
            ax.plot(x[finite], row_profile[finite], color="tab:blue", alpha=line_alpha, lw=0.8)

    if len(context.get("relative_excluded_spans", [])) > 0 and context["raw_median"].size > 0:
        finite = np.isfinite(context["raw_median"])
        if np.any(finite):
            ax.plot(x[finite], context["raw_median"][finite], color="0.45", lw=1.0, ls="--", label="raw median row")

    if context["used_median"].size > 0:
        finite = np.isfinite(context["used_median"])
        if np.any(finite):
            ax.plot(x[finite], context["used_median"][finite], color="tab:orange", lw=1.4, label="used median row")
    if context["used_mean"].size > 0:
        finite = np.isfinite(context["used_mean"])
        if np.any(finite):
            ax.plot(x[finite], context["used_mean"][finite], color="black", lw=1.2, alpha=0.9, label="used mean row")

    if context["excluded_column_mode"] == "mask_only" and context["masked_point_x"].size > 0:
        ax.plot(
            context["masked_point_x"],
            context["masked_point_y"],
            linestyle="None",
            marker="x",
            ms=3.8,
            color="tab:red",
            alpha=0.75,
            label="ignored bad-column points",
        )
    if context["excluded_column_mode"] == "interpolate" and context["interpolated_point_x"].size > 0:
        ax.plot(
            context["interpolated_point_x"],
            context["interpolated_point_y"],
            linestyle="None",
            marker="o",
            ms=3.2,
            mfc="none",
            mec="tab:orange",
            alpha=0.8,
            label="interpolated bad-column points",
        )

    ax.axhline(0.0, color="0.45", ls="--", lw=0.9, zorder=0)
    ax.axvline(0.0, color="0.35", ls="--", lw=1.0, label="trace centre")
    add_relative_aperture_edge_lines(
        ax,
        context["display_ap_left"],
        context["display_ap_right"],
        color="tab:red",
        label="final aperture",
    )
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    if context["overlay_ylim"] is not None:
        ax.set_ylim(*context["overlay_ylim"])
    ax.set_xlabel("Relative X pixel from trace centre")
    ax.set_ylabel("Per-row bg-subtracted counts")
    ax.set_title("Star %d aligned per-row profiles" % context["record"]["star_index"])
    ax.legend(loc="upper left", fontsize=8, framealpha=1)


def create_summed_profile_figure(records):
    if not any(uses_summed_profile(record) for record in records):
        raise ValueError("No summed-profile diagnostics were found in this frame.")

    nstars = len(records)
    fig, axes = plt.subplots(nstars, 2, figsize=(14.5, 4.6 * nstars), squeeze=False)

    for row_idx, record in enumerate(records):
        if uses_summed_profile(record):
            context = build_summed_profile_context(record)
            plot_summed_profile_left_panel(axes[row_idx, 0], context)
            plot_summed_profile_overlay_panel(axes[row_idx, 1], context)
        else:
            for col_idx in range(2):
                axes[row_idx, col_idx].text(
                    0.5,
                    0.5,
                    "Star %d uses %s mode"
                    % (record["star_index"], get_profile_aggregation_mode(record)),
                    ha="center",
                    va="center",
                )
                axes[row_idx, col_idx].set_axis_off()

    frame_label = Path(records[0]["science_frame"]).name
    fig.suptitle("Frame %d: %s\nSummed-profile diagnostics" % (records[0]["frame_index"], frame_label), fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def create_exposure_profile_figure(records):
    nstars = len(records)
    fig, axes = plt.subplots(nstars, 2, figsize=(14.5, 4.6 * nstars), squeeze=False)

    for row_idx, record in enumerate(records):
        if uses_summed_profile(record):
            context = build_summed_profile_context(record)
            plot_summed_profile_left_panel(axes[row_idx, 0], context)
            plot_summed_profile_overlay_panel(axes[row_idx, 1], context)
        else:
            context = build_aligned_profile_context(record)
            plot_aligned_profile_left_panel(axes[row_idx, 0], context)
            plot_aligned_profile_overlay_panel(axes[row_idx, 1], context)

    frame_label = Path(records[0]["science_frame"]).name
    fig.suptitle("Frame %d: %s\nExposure-profile diagnostics" % (records[0]["frame_index"], frame_label), fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def export_summed_profile_all_frames(frame_groups, output_dir):
    output_path = Path(output_dir).expanduser()

    n_saved = 0
    for frame_group in frame_groups:
        records = load_frame_records(frame_group)
        if not any(uses_summed_profile(record) for record in records):
            continue
        if n_saved == 0:
            output_path.mkdir(parents=True, exist_ok=True)
        fig = create_summed_profile_figure(records)
        frame_index = records[0]["frame_index"]
        out_file = output_path / ("summed_profile_frame_%05d.png" % frame_index)
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1

    return n_saved


def export_profile_diagnostic_all_frames(frame_groups, output_dir, requested_row=None):
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for frame_group in frame_groups:
        records = load_frame_records(frame_group)
        if len(records) == 0:
            continue

        frame_index = records[0]["frame_index"]
        fig = create_exposure_profile_figure(records)
        out_file = output_path / ("profile_diagnostic_frame_%05d.png" % frame_index)

        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1

    return n_saved


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


def plot_metric_summary(ax, metric_matrix, frame_indices, title, y_label):
    mean_values = np.nanmean(metric_matrix, axis=0)
    std_values = np.nanstd(metric_matrix, axis=0)
    median_values = np.nanmedian(metric_matrix, axis=0)

    ax.plot(frame_indices, mean_values, color="tab:blue", lw=1.4, label="mean")
    ax.fill_between(
        frame_indices,
        mean_values - std_values,
        mean_values + std_values,
        color="tab:blue",
        alpha=0.18,
        label=r"$\pm1\sigma$",
    )
    ax.plot(frame_indices, median_values, color="tab:orange", lw=1.0, ls="--", label="median")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(y_label)
    ax.legend(loc="best")


def get_width_diagnostic_metric_specs(record):
    metrics = [
        {
            "key": "aperture_width",
            "heatmap_title": "final extraction width",
            "summary_title": "per-exposure final extraction width",
            "colorbar": "Width (pixels)",
            "ylabel": "Width (pixels)",
        }
    ]

    if bool(record.get("profile_enabled", False)):
        if str(record.get("profile_model", "")) == "empirical":
            heatmap_title = "empirical half-max width"
            summary_title = "per-exposure empirical half-max width"
        else:
            heatmap_title = "profile FWHM"
            summary_title = "per-exposure profile FWHM"
        metrics.append(
            {
                "key": "profile_fwhm",
                "heatmap_title": heatmap_title,
                "summary_title": summary_title,
                "colorbar": "Pixels",
                "ylabel": "Pixels",
            }
        )

    return metrics


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
    max_cols = 2

    for star_index in star_indices:
        reference_record = next((frame_records[star_index] for frame_records in records_by_frame if star_index in frame_records), None)
        if reference_record is None:
            continue
        metric_specs = get_width_diagnostic_metric_specs(reference_record)
        metric_specs_by_star[star_index] = metric_specs
        max_cols = max(max_cols, 2 * len(metric_specs))

    fig, axes = plt.subplots(nstars, max_cols, figsize=(5.6 * max_cols, 4.8 * nstars), squeeze=False)

    for row_idx, star_index in enumerate(star_indices):
        reference_record = next((frame_records[star_index] for frame_records in records_by_frame if star_index in frame_records), None)
        if reference_record is None:
            continue

        row_numbers = reference_record["row_numbers"]
        metric_specs = metric_specs_by_star[star_index]

        for metric_idx, metric_spec in enumerate(metric_specs):
            metric_matrix = build_metric_matrix(records_by_frame, star_index, metric_spec["key"], row_numbers)
            heatmap_col = 2 * metric_idx
            summary_col = heatmap_col + 1
            image, colorbar_label = plot_width_heatmap(
                axes[row_idx, heatmap_col],
                metric_matrix,
                frame_indices,
                row_numbers,
                "Star %d %s" % (star_index, metric_spec["heatmap_title"]),
                metric_spec["colorbar"],
            )
            axis = axes[row_idx, heatmap_col]
            cbar = fig.colorbar(image, ax=axis, pad=0.02)
            cbar.set_label(colorbar_label)
            plot_metric_summary(
                axes[row_idx, summary_col],
                metric_matrix,
                frame_indices,
                "Star %d %s" % (star_index, metric_spec["summary_title"]),
                metric_spec["ylabel"],
            )

        for col_idx in range(2 * len(metric_specs), max_cols):
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


def get_reduction_dir_from_frame_groups(frame_groups):
    if len(frame_groups) == 0:
        raise ValueError("No diagnostic frames were found.")

    first_records = load_frame_records(frame_groups[0])
    if len(first_records) == 0:
        raise ValueError("No diagnostic records were found in the first frame group.")

    diag_path = Path(first_records[0]["path"]).expanduser().resolve()
    if len(diag_path.parents) < 3:
        raise ValueError("Could not determine the reduction directory from %s" % diag_path)
    return diag_path.parents[2]


def running_median(values, window):
    values = np.asarray(values, dtype=float)
    n_values = len(values)
    if n_values == 0:
        return values.copy()

    window = max(int(window), 1)
    if window % 2 == 0:
        window += 1

    half = window // 2
    output = np.empty(n_values, dtype=float)
    for idx in range(n_values):
        left = max(0, idx - half)
        right = min(n_values, idx + half + 1)
        output[idx] = np.nanmedian(values[left:right])
    return output


def create_white_light_check_figure(frame_groups, highlight_indices=None, drop_indices=None, zoom_to_highlight=False, pixel_ranges=None):
    reduction_dir = get_reduction_dir_from_frame_groups(frame_groups)
    pickled_dir = reduction_dir / "pickled_objects"

    star1_path = pickled_dir / "star1_flux.pickle"
    star2_path = pickled_dir / "star2_flux.pickle"
    if not star1_path.exists() or not star2_path.exists():
        raise FileNotFoundError("Could not find star1_flux.pickle and star2_flux.pickle in %s" % pickled_dir)

    star1_flux = np.asarray(pickle.load(open(star1_path, "rb")), dtype=float)
    star2_flux = np.asarray(pickle.load(open(star2_path, "rb")), dtype=float)

    if star1_flux.ndim == 1:
        star1_flux = star1_flux[:, np.newaxis]
    if star2_flux.ndim == 1:
        star2_flux = star2_flux[:, np.newaxis]

    n_frames = min(len(frame_groups), star1_flux.shape[0], star2_flux.shape[0])
    n_pixels = min(star1_flux.shape[1], star2_flux.shape[1])
    star1_flux = star1_flux[:n_frames, :n_pixels]
    star2_flux = star2_flux[:n_frames, :n_pixels]

    if pixel_ranges is None or len(pixel_ranges) == 0:
        pixel_ranges = [(0, n_pixels - 1)]

    clipped_ranges = []
    for start, end in pixel_ranges:
        start_clip = max(0, int(start))
        end_clip = min(n_pixels - 1, int(end))
        if end_clip < start_clip:
            continue
        clipped_ranges.append((start_clip, end_clip))

    if len(clipped_ranges) == 0:
        raise ValueError("No valid pixel ranges remain after clipping to the extracted spectral pixel span.")

    x_label = "Exposure index"
    exposure_indices = np.arange(n_frames, dtype=int)
    drop_set = set(drop_indices or [])
    keep_mask = np.asarray([index not in drop_set for index in exposure_indices], dtype=bool)
    exposure_indices = exposure_indices[keep_mask]
    x_values = exposure_indices.astype(float)

    if len(exposure_indices) == 0:
        raise ValueError("No exposures remain after applying --drop-indices.")

    n_ranges = len(clipped_ranges)
    fig, axes = plt.subplots(2 * n_ranges, 1, figsize=(14, 7 * n_ranges), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for range_idx, (pix_start, pix_end) in enumerate(clipped_ranges):
        target_sum = np.sum(star1_flux[:, pix_start:pix_end + 1], axis=1)[keep_mask]
        comparison_sum = np.sum(star2_flux[:, pix_start:pix_end + 1], axis=1)[keep_mask]
        ratio_sum = target_sum / comparison_sum

        target_norm = target_sum / np.nanmedian(target_sum)
        comparison_norm = comparison_sum / np.nanmedian(comparison_sum)
        ratio_norm = ratio_sum / np.nanmedian(ratio_sum)

        smoothing_window = max(5, 2 * (len(x_values) // 20) + 1)
        target_trend = running_median(target_norm, smoothing_window)
        comparison_trend = running_median(comparison_norm, smoothing_window)
        ratio_trend = running_median(ratio_norm, smoothing_window)

        target_resid = target_norm / target_trend
        comparison_resid = comparison_norm / comparison_trend
        ratio_resid = ratio_norm / ratio_trend

        top_ax = axes[2 * range_idx]
        bottom_ax = axes[2 * range_idx + 1]

        top_ax.plot(x_values, target_norm, "ko-", markersize=4, linewidth=1.0, label="target raw")
        top_ax.plot(
            x_values,
            comparison_norm,
            color="tab:blue",
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            alpha=0.9,
            label="comparison raw",
        )
        top_ax.set_ylabel("Normalised flux")
        top_ax.legend(loc="best")
        top_ax.set_title("Target vs comparison check, pixels %d:%d" % (pix_start, pix_end))

        bottom_ax.plot(x_values, ratio_resid, "ko-", markersize=4, linewidth=1.0, label="target/comparison residual")
        bottom_ax.plot(
            x_values,
            target_resid,
            color="tab:red",
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            alpha=0.9,
            label="target residual",
        )
        bottom_ax.plot(
            x_values,
            comparison_resid,
            color="tab:blue",
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            alpha=0.9,
            label="comparison residual",
        )
        bottom_ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
        bottom_ax.set_xlabel(x_label)
        bottom_ax.set_ylabel("Residual / running median")
        bottom_ax.legend(loc="best")
        bottom_ax.set_title("Residual structure after running-median removal, pixels %d:%d" % (pix_start, pix_end))

        if highlight_indices is not None and len(highlight_indices) > 0:
            selected = np.isin(exposure_indices, np.asarray(highlight_indices, dtype=int))
            if np.any(selected):
                selected_x = x_values[selected]
                x0 = np.nanmin(selected_x)
                x1 = np.nanmax(selected_x)
                for ax in (top_ax, bottom_ax):
                    ax.axvspan(x0, x1, color="gold", alpha=0.15, lw=0)

                if zoom_to_highlight:
                    span = max(x1 - x0, 1e-6)
                    pad = max(0.15 * span, 2.0)
                    for ax in (top_ax, bottom_ax):
                        ax.set_xlim(x0 - pad, x1 + pad)

        for ax in (top_ax, bottom_ax):
            ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    return fig


def create_star_range_check_figure(frame_groups, pixel_ranges, highlight_indices=None, drop_indices=None, zoom_to_highlight=False):
    reduction_dir = get_reduction_dir_from_frame_groups(frame_groups)
    pickled_dir = reduction_dir / "pickled_objects"

    star1_path = pickled_dir / "star1_flux.pickle"
    star2_path = pickled_dir / "star2_flux.pickle"
    if not star1_path.exists() or not star2_path.exists():
        raise FileNotFoundError("Could not find star1_flux.pickle and star2_flux.pickle in %s" % pickled_dir)

    star1_flux = np.asarray(pickle.load(open(star1_path, "rb")), dtype=float)
    star2_flux = np.asarray(pickle.load(open(star2_path, "rb")), dtype=float)

    if star1_flux.ndim == 1:
        star1_flux = star1_flux[:, np.newaxis]
    if star2_flux.ndim == 1:
        star2_flux = star2_flux[:, np.newaxis]

    n_frames = min(len(frame_groups), star1_flux.shape[0], star2_flux.shape[0])
    n_pixels = min(star1_flux.shape[1], star2_flux.shape[1])
    star1_flux = star1_flux[:n_frames, :n_pixels]
    star2_flux = star2_flux[:n_frames, :n_pixels]

    clipped_ranges = []
    for start, end in (pixel_ranges or []):
        start_clip = max(0, int(start))
        end_clip = min(n_pixels - 1, int(end))
        if end_clip < start_clip:
            continue
        clipped_ranges.append((start_clip, end_clip))

    if len(clipped_ranges) == 0:
        raise ValueError("Use --pixel-ranges with at least one valid inclusive range, e.g. 200:400,401:600,601:800.")

    exposure_indices = np.arange(n_frames, dtype=int)
    drop_set = set(drop_indices or [])
    keep_mask = np.asarray([index not in drop_set for index in exposure_indices], dtype=bool)
    exposure_indices = exposure_indices[keep_mask]
    x_values = exposure_indices.astype(float)

    if len(exposure_indices) == 0:
        raise ValueError("No exposures remain after applying --drop-indices.")

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(clipped_ranges), 2)))
    smoothing_window = max(5, 2 * (len(x_values) // 20) + 1)

    for color, (pix_start, pix_end) in zip(colors, clipped_ranges):
        label = "%d:%d" % (pix_start, pix_end)
        star1_sum = np.sum(star1_flux[:, pix_start:pix_end + 1], axis=1)[keep_mask]
        star2_sum = np.sum(star2_flux[:, pix_start:pix_end + 1], axis=1)[keep_mask]

        star1_norm = star1_sum / np.nanmedian(star1_sum)
        star2_norm = star2_sum / np.nanmedian(star2_sum)
        star1_resid = star1_norm / running_median(star1_norm, smoothing_window)
        star2_resid = star2_norm / running_median(star2_norm, smoothing_window)

        axes[0].plot(x_values, star1_norm, marker="o", ms=3.5, lw=1.0, color=color, label=label)
        axes[1].plot(x_values, star1_resid, marker="o", ms=3.5, lw=1.0, color=color, label=label)
        axes[2].plot(x_values, star2_norm, marker="o", ms=3.5, lw=1.0, color=color, label=label)
        axes[3].plot(x_values, star2_resid, marker="o", ms=3.5, lw=1.0, color=color, label=label)

    axes[0].set_title("Star 1 flux by spectral-pixel range")
    axes[1].set_title("Star 1 residuals by spectral-pixel range")
    axes[2].set_title("Star 2 flux by spectral-pixel range")
    axes[3].set_title("Star 2 residuals by spectral-pixel range")

    axes[0].set_ylabel("Normalised flux")
    axes[1].set_ylabel("Residual / running median")
    axes[2].set_ylabel("Normalised flux")
    axes[3].set_ylabel("Residual / running median")
    axes[3].set_xlabel("Exposure index")
    axes[1].axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    axes[3].axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)

    if highlight_indices is not None and len(highlight_indices) > 0:
        selected = np.isin(exposure_indices, np.asarray(highlight_indices, dtype=int))
        if np.any(selected):
            selected_x = x_values[selected]
            x0 = np.nanmin(selected_x)
            x1 = np.nanmax(selected_x)
            for ax in axes:
                ax.axvspan(x0, x1, color="gold", alpha=0.15, lw=0)
            if zoom_to_highlight:
                span = max(x1 - x0, 1e-6)
                pad = max(0.15 * span, 2.0)
                for ax in axes:
                    ax.set_xlim(x0 - pad, x1 + pad)

    for ax in axes:
        ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)
        ax.legend(loc="best", fontsize=8, framealpha=1, title="pixels")

    fig.tight_layout()
    return fig


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


def sanitize_run_label(label):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label).strip())
    safe = safe.strip("._-")
    return safe if safe else "run"


def parse_run_collection_file(run_list_path):
    run_list_path = Path(run_list_path).expanduser().resolve()
    if not run_list_path.is_file():
        raise ValueError("Run list file '%s' was not found." % run_list_path)

    entries = []
    used_labels = {}
    for line_number, raw_line in enumerate(run_list_path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue

        if "|" in stripped:
            label_text, path_text = stripped.split("|", 1)
            display_label = label_text.strip()
            path_text = path_text.strip()
        else:
            display_label = None
            path_text = stripped

        run_path = Path(path_text).expanduser()
        if not run_path.is_absolute():
            run_path = (run_list_path.parent / run_path).resolve()
        else:
            run_path = run_path.resolve()

        if not run_path.exists():
            raise ValueError("Run path on line %d does not exist: %s" % (line_number, run_path))
        if not run_path.is_dir():
            raise ValueError("Run path on line %d is not a directory: %s" % (line_number, run_path))

        if display_label in (None, ""):
            display_label = run_path.name

        safe_label = sanitize_run_label(display_label)
        used_labels[safe_label] = used_labels.get(safe_label, 0) + 1
        if used_labels[safe_label] > 1:
            safe_label = "%s_%02d" % (safe_label, used_labels[safe_label])

        entries.append(
            {
                "label": display_label,
                "safe_label": safe_label,
                "path": run_path,
                "line_number": line_number,
            }
        )

    if len(entries) == 0:
        raise ValueError("Run list file '%s' did not contain any usable run directories." % run_list_path)

    return entries


def find_first_existing_path(run_path, relative_candidates):
    run_path = Path(run_path)
    for relative_path in relative_candidates:
        candidate = run_path / relative_path
        if candidate.exists():
            return candidate
    return None


def load_white_light_table(run_path):
    table_path = find_first_existing_path(run_path, ["white_light.dat", "white_light.txt"])
    if table_path is None:
        return None

    data = np.loadtxt(table_path)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError("White-light table '%s' does not have at least two columns." % table_path)

    time_values = np.asarray(data[:, 0], dtype=float)
    flux_values = np.asarray(data[:, 1], dtype=float)
    if data.shape[1] >= 3:
        err_values = np.asarray(data[:, 2], dtype=float)
    else:
        err_values = np.full_like(flux_values, np.nan, dtype=float)

    finite = np.isfinite(time_values) & np.isfinite(flux_values)
    time_values = time_values[finite]
    flux_values = flux_values[finite]
    err_values = err_values[finite]
    if len(flux_values) == 0:
        return None

    return {
        "path": table_path,
        "time": time_values,
        "flux": flux_values,
        "err": err_values,
    }


def compute_white_light_summary_metrics(flux_values):
    flux_values = np.asarray(flux_values, dtype=float)
    finite = flux_values[np.isfinite(flux_values)]
    if finite.size == 0:
        return {
            "median_flux": np.nan,
            "std_ppm": np.nan,
            "mad_ppm": np.nan,
            "point_to_point_ppm": np.nan,
        }

    median_flux = float(np.nanmedian(finite))
    if np.isfinite(median_flux) and median_flux != 0:
        normalized = finite / median_flux
    else:
        normalized = finite.copy()

    centered = normalized - np.nanmedian(normalized)
    std_ppm = float(np.nanstd(centered) * 1.0e6)
    mad_ppm = float(1.4826 * np.nanmedian(np.abs(centered - np.nanmedian(centered))) * 1.0e6)
    if len(normalized) >= 2:
        point_to_point_ppm = float(np.nanstd(np.diff(normalized)) / np.sqrt(2.0) * 1.0e6)
    else:
        point_to_point_ppm = np.nan

    return {
        "median_flux": median_flux,
        "std_ppm": std_ppm,
        "mad_ppm": mad_ppm,
        "point_to_point_ppm": point_to_point_ppm,
    }


def rank_summary_rows(summary_rows, metric_keys):
    for metric_key in metric_keys:
        ordered = sorted(
            summary_rows,
            key=lambda row: (
                np.inf if not np.isfinite(row[metric_key]) else row[metric_key],
                row["label"],
            ),
        )
        for rank_index, row in enumerate(ordered, start=1):
            row["rank_%s" % metric_key] = rank_index

    for row in summary_rows:
        row["rank_score"] = (
            float(row["rank_std_ppm"])
            + float(row["rank_mad_ppm"])
            + 2.0 * float(row["rank_point_to_point_ppm"])
        )

    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (
            row["rank_score"],
            row["rank_point_to_point_ppm"],
            row["rank_mad_ppm"],
            row["rank_std_ppm"],
            row["label"],
        ),
    )

    for overall_rank, row in enumerate(ranked_rows, start=1):
        row["overall_rank"] = overall_rank

    return ranked_rows


def create_collated_white_light_figure(run_series):
    if len(run_series) == 0:
        raise ValueError("No white-light curves were available to compare.")

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(run_series), 2)))

    for color, run_entry in zip(colors, run_series):
        flux = np.asarray(run_entry["flux"], dtype=float)
        median_flux = np.nanmedian(flux)
        if np.isfinite(median_flux) and median_flux != 0:
            y_values = flux / median_flux
        else:
            y_values = flux
        x_values = np.arange(len(y_values), dtype=int)

        ax.plot(
            x_values,
            y_values,
            marker="o",
            ms=3.5,
            lw=1.1,
            alpha=0.9,
            color=color,
            label=run_entry["label"],
        )

    ax.set_xlabel("Exposure index")
    ax.set_ylabel("White-light flux / median(flux)")
    ax.set_title("White-light comparison across runs")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2, lw=0.6)
    fig.tight_layout()
    return fig


def collate_run_diagnostics(run_list_path, output_dir=None):
    run_entries = parse_run_collection_file(run_list_path)
    run_list_path = Path(run_list_path).expanduser().resolve()

    if output_dir is None:
        output_root = run_list_path.with_suffix("")
        output_root = output_root.parent / ("%s_collated_diagnostics" % output_root.name)
    else:
        output_root = Path(output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = (Path.cwd() / output_root).resolve()

    width_output_dir = output_root / "width_diagnostics"
    capture_output_dir = output_root / "aperture_capture"
    white_light_output_dir = output_root / "white_light"
    for directory in (output_root, width_output_dir, capture_output_dir, white_light_output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "# Collated extraction diagnostics",
        "# label | run_path | width_diagnostics | aperture_capture_diagnostics | white_light_curve_numbered_wide | white_light_table",
    ]
    summary_rows = []
    white_light_series = []

    artifact_specs = [
        {
            "key": "width",
            "relative_candidates": [
                "diagnostic_plots/width_diagnostics/width_diagnostics.png",
                "width_diagnostics.png",
            ],
            "output_dir": width_output_dir,
            "suffix": "_width_diagnostics.png",
        },
        {
            "key": "capture",
            "relative_candidates": [
                "diagnostic_plots/aperture_capture/aperture_capture_diagnostics.png",
                "aperture_capture_diagnostics.png",
            ],
            "output_dir": capture_output_dir,
            "suffix": "_aperture_capture_diagnostics.png",
        },
        {
            "key": "white_light_pdf",
            "relative_candidates": ["white_light_curve_numbered_wide.pdf"],
            "output_dir": white_light_output_dir,
            "suffix": "_white_light_curve_numbered_wide.pdf",
        },
    ]

    for index, run_entry in enumerate(run_entries, start=1):
        file_prefix = "%02d_%s" % (index, run_entry["safe_label"])
        artifact_paths = {}

        for artifact_spec in artifact_specs:
            source_path = find_first_existing_path(run_entry["path"], artifact_spec["relative_candidates"])
            if source_path is None:
                artifact_paths[artifact_spec["key"]] = None
                continue
            destination_path = artifact_spec["output_dir"] / (file_prefix + artifact_spec["suffix"])
            shutil.copy2(source_path, destination_path)
            artifact_paths[artifact_spec["key"]] = destination_path

        white_light_table = load_white_light_table(run_entry["path"])
        if white_light_table is not None:
            metrics = compute_white_light_summary_metrics(white_light_table["flux"])
            white_light_series.append(
                {
                    "label": run_entry["label"],
                    "safe_label": run_entry["safe_label"],
                    "time": white_light_table["time"],
                    "flux": white_light_table["flux"],
                    "err": white_light_table["err"],
                }
            )
            summary_rows.append(
                {
                    "label": run_entry["label"],
                    "n_points": len(white_light_table["flux"]),
                    "median_flux": metrics["median_flux"],
                    "std_ppm": metrics["std_ppm"],
                    "mad_ppm": metrics["mad_ppm"],
                    "point_to_point_ppm": metrics["point_to_point_ppm"],
                }
            )
            white_light_table_path = white_light_table["path"]
        else:
            white_light_table_path = None

        manifest_lines.append(
            "%s | %s | %s | %s | %s | %s"
            % (
                run_entry["label"],
                run_entry["path"],
                artifact_paths["width"] if artifact_paths["width"] is not None else "missing",
                artifact_paths["capture"] if artifact_paths["capture"] is not None else "missing",
                artifact_paths["white_light_pdf"] if artifact_paths["white_light_pdf"] is not None else "missing",
                white_light_table_path if white_light_table_path is not None else "missing",
            )
        )

    manifest_path = output_root / "run_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")

    figure_path_png = None
    figure_path_pdf = None
    ranking_table_path = None
    fig = None
    if len(white_light_series) > 0:
        fig = create_collated_white_light_figure(white_light_series)
        figure_path_png = white_light_output_dir / "white_light_runs_overlay.png"
        figure_path_pdf = white_light_output_dir / "white_light_runs_overlay.pdf"
        fig.savefig(figure_path_png, dpi=200, bbox_inches="tight")
        fig.savefig(figure_path_pdf, bbox_inches="tight")

        summary_table_path = white_light_output_dir / "white_light_run_summary.txt"
        with open(summary_table_path, "w") as handle:
            handle.write("# run_label n_points median_flux std_ppm mad_ppm point_to_point_ppm\n")
            for row in summary_rows:
                handle.write(
                    "%s %d %.10f %.3f %.3f %.3f\n"
                    % (
                        sanitize_run_label(row["label"]),
                        int(row["n_points"]),
                        float(row["median_flux"]),
                        float(row["std_ppm"]),
                        float(row["mad_ppm"]),
                        float(row["point_to_point_ppm"]),
                    )
                )

        ranked_rows = rank_summary_rows(
            summary_rows,
            ["std_ppm", "mad_ppm", "point_to_point_ppm"],
        )
        ranking_table_path = white_light_output_dir / "white_light_run_ranking.txt"
        with open(ranking_table_path, "w") as handle:
            handle.write("# lower values are better for every metric\n")
            handle.write("# overall score = rank_std_ppm + rank_mad_ppm + 2 * rank_point_to_point_ppm\n")
            handle.write(
                "# overall_rank run_label rank_score rank_std_ppm rank_mad_ppm rank_point_to_point_ppm std_ppm mad_ppm point_to_point_ppm\n"
            )
            for row in ranked_rows:
                handle.write(
                    "%d %s %.1f %d %d %d %.3f %.3f %.3f\n"
                    % (
                        int(row["overall_rank"]),
                        sanitize_run_label(row["label"]),
                        float(row["rank_score"]),
                        int(row["rank_std_ppm"]),
                        int(row["rank_mad_ppm"]),
                        int(row["rank_point_to_point_ppm"]),
                        float(row["std_ppm"]),
                        float(row["mad_ppm"]),
                        float(row["point_to_point_ppm"]),
                    )
                )

    return {
        "output_root": output_root,
        "manifest_path": manifest_path,
        "white_light_figure": fig,
        "white_light_png": figure_path_png,
        "white_light_pdf": figure_path_pdf,
        "white_light_ranking": ranking_table_path,
        "n_runs": len(run_entries),
        "n_white_light": len(white_light_series),
    }


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
    parser.add_argument("diagnostic_input", help="A diagnostic .npz file, a directory containing frame_*_star_*.npz, or with --collate-runs a text file listing saved run directories.")
    parser.add_argument("-r", "--row", type=int, help="Detector row number or 0-based local row index for the default viewer.")
    parser.add_argument("--rows", help="Comma-separated rows for overlay or joy modes.")
    parser.add_argument("--overlay", action="store_true", help="Overlay the requested rows for each star, aligned on the trace centre.")
    parser.add_argument("--joy", action="store_true", help="Joy-style stacked plot of the requested rows for each star, aligned on the trace centre.")
    parser.add_argument("--widths", action="store_true", help="Show extraction/profile widths for every row in every frame.")
    parser.add_argument("--show-enclosed-width-history", action="store_true", help="For the default single-frame viewer, add the enclosed-width heatmap panel with the current frame/row highlighted.")
    parser.add_argument("--time-overlay", action="store_true", help="Overlay raw row profiles from all frames and show the mean ± 1σ band.")
    parser.add_argument("--time-overlay-build", action="store_true", help="Show the time-overlay plot building up cumulatively: 1, 1-2, 1-3, ...")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay in seconds between cumulative time-overlay updates.")
    parser.add_argument("--export-time-overlay-all-rows", action="store_true", help="Save one time-overlay figure per diagnostic row into the output directory.")
    parser.add_argument("--summed-profile", action="store_true", help="Show the per-exposure profile diagnostic used for aperture_profile_diagnostics. For summed modes this shows the summed fit on the left; otherwise it shows the aligned exposure profile.")
    parser.add_argument("--export-summed-profile-all-frames", action="store_true", help="With --summed-profile, save one per-exposure profile figure per frame into the output directory.")
    parser.add_argument("--white-light-check", action="store_true", help="Recreate the target-vs-comparison white-light diagnostic from saved flux pickles.")
    parser.add_argument("--star-range-check", action="store_true", help="Compare multiple spectral-pixel ranges within star 1 and within star 2 separately.")
    parser.add_argument("--pixel-ranges", help="Comma-separated inclusive spectral-pixel ranges for --white-light-check, e.g. 200:400,401:600,601:800.")
    parser.add_argument("--highlight-indices", help="Comma-separated exposure indices or inclusive ranges (e.g. 56:62) to highlight.")
    parser.add_argument("--drop-indices", help="Comma-separated exposure indices or inclusive ranges (e.g. 0:3,95) to omit from the white-light check.")
    parser.add_argument("--zoom-highlight", action="store_true", help="With --white-light-check, zoom to the highlighted exposure-index range instead of showing the full series.")
    parser.add_argument("--time-stack", action="store_true", help="Stack raw row profiles by exposure, using actual inter-frame time gaps when available.")
    parser.add_argument("--export-row-images", action="store_true", help="Export one saved row-profile figure per frame for the chosen row.")
    parser.add_argument("--collate-runs", action="store_true", help="Treat diagnostic_input as a text file listing saved run directories, copy key products into one comparison folder, and overlay the white-light curves.")
    parser.add_argument("-s", "--save_figure", action="store_true", help="Save figure(s) as well as showing them.")
    parser.add_argument("-o", "--output", help="Optional output filename, prefix, or directory for saved figures.")
    args = parser.parse_args()

    selected_modes = [args.overlay, args.joy, args.widths, args.time_overlay, args.summed_profile, args.white_light_check, args.star_range_check, args.time_stack, args.export_row_images, args.collate_runs]
    if sum(bool(mode) for mode in selected_modes) > 1:
        raise ValueError("Choose only one of --overlay, --joy, --widths, --time-overlay, --summed-profile, --white-light-check, --star-range-check, --time-stack, --export-row-images, or --collate-runs.")
    if args.time_overlay_build and not args.time_overlay:
        raise ValueError("--time-overlay-build must be used together with --time-overlay.")
    if args.export_time_overlay_all_rows and not args.time_overlay:
        raise ValueError("--export-time-overlay-all-rows must be used together with --time-overlay.")
    if args.export_summed_profile_all_frames and not args.summed_profile:
        raise ValueError("--export-summed-profile-all-frames must be used together with --summed-profile.")

    requested_rows = parse_rows(args.rows)
    if requested_rows is None and args.row is not None:
        requested_rows = [args.row]

    if args.collate_runs:
        result = collate_run_diagnostics(args.diagnostic_input, output_dir=args.output)
        print("Collated %d run(s) into %s" % (int(result["n_runs"]), result["output_root"]))
        print("Wrote manifest to %s" % result["manifest_path"])
        if result["n_white_light"] > 0 and result["white_light_png"] is not None:
            print(
                "Saved combined white-light comparison for %d run(s) to %s"
                % (int(result["n_white_light"]), result["white_light_png"])
            )
            if result["white_light_ranking"] is not None:
                print("Wrote white-light ranking to %s" % result["white_light_ranking"])
            backend_name = str(plt.get_backend()).lower()
            if result["white_light_figure"] is not None:
                if "agg" not in backend_name:
                    plt.show()
                plt.close(result["white_light_figure"])
        else:
            print("No white-light tables were found in the listed runs.")
        return

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

    if args.summed_profile:
        if args.export_summed_profile_all_frames:
            export_dir = args.output if args.output is not None else "aperture_profile_diagnostics"
            n_saved = export_profile_diagnostic_all_frames(frame_groups, export_dir, requested_row=args.row)
            if n_saved == 0:
                raise ValueError("No exposure-profile diagnostics were found in the supplied diagnostic bundle(s).")
            return

        shown = 0
        for frame_group in frame_groups:
            records = load_frame_records(frame_group)
            if len(records) == 0:
                continue
            fig = create_exposure_profile_figure(records)
            if args.save_figure:
                output_path = get_output_path(args.output, records[0]["frame_index"], "exposure_profile", multiple_frames)
                fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            shown += 1
        if shown == 0:
            raise ValueError("No exposure-profile diagnostics were found in the supplied diagnostic bundle(s).")
        return

    if args.white_light_check:
        highlight_indices = parse_index_spec(args.highlight_indices)
        drop_indices = parse_index_spec(args.drop_indices)
        pixel_ranges = parse_pixel_range_spec(args.pixel_ranges)
        fig = create_white_light_check_figure(
            frame_groups,
            highlight_indices=highlight_indices,
            drop_indices=drop_indices,
            zoom_to_highlight=args.zoom_highlight,
            pixel_ranges=pixel_ranges,
        )
        if args.save_figure:
            output_path = get_output_path(args.output, 0, "white_light_check", False)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return

    if args.star_range_check:
        highlight_indices = parse_index_spec(args.highlight_indices)
        drop_indices = parse_index_spec(args.drop_indices)
        pixel_ranges = parse_pixel_range_spec(args.pixel_ranges)
        fig = create_star_range_check_figure(
            frame_groups,
            pixel_ranges=pixel_ranges,
            highlight_indices=highlight_indices,
            drop_indices=drop_indices,
            zoom_to_highlight=args.zoom_highlight,
        )
        if args.save_figure:
            output_path = get_output_path(args.output, 0, "star_range_check", False)
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

    metric_data = None
    if args.show_enclosed_width_history:
        try:
            metric_data = prepare_metric_series(frame_groups)
        except Exception:
            metric_data = None

    for frame_group in frame_groups:
        records = load_frame_records(frame_group)

        if args.overlay:
            fig = create_stack_figure(records, requested_rows, joy=False)
            mode = "overlay"
        elif args.joy:
            fig = create_stack_figure(records, requested_rows, joy=True)
            mode = "joy"
        else:
            fig = create_single_frame_figure(records, args.row, metric_data=metric_data)
            mode = "single"

        if args.save_figure:
            output_path = get_output_path(args.output, records[0]["frame_index"], mode, multiple_frames)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
