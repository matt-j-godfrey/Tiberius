#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from scipy.stats import median_abs_deviation as mad
from scipy.ndimage import median_filter as MF
from scipy.ndimage import interpolation
from astropy.io import fits
import matplotlib.pyplot as plt
import time
import pickle
import os
import shutil
from collections import Counter
from pathlib import Path
from global_utils import parseInput
try:
    import astroscrappy
except:
    print("astroscrappy not imported, automatic cosmic ray detection can't be performed with lacosmic")
import copy
from cosmic_removal import interp_bad_pixels
from wavelength_calibration import rebin_spec
from astropy import units as u
from Keck_utils import Keck_order_masking as KO
from astropy.time import Time,TimeDelta

# Prevent matplotlib plotting frames upside down
plt.rcParams['image.origin'] = 'lower'

try:
    os.mkdir("./spectral_extraction_images/")
except:
      pass

# Always create diagnostic arrays for offline plotting
SAVE_DIAG = True  # can be toggled if ever needed

DIAG_DIR = Path("pickled_objects") / "extraction_diagnostics"
REGION_APERTURE = np.uint8(1)
REGION_BACKGROUND_USED = np.uint8(2)
REGION_BACKGROUND_REJECTED = np.uint8(4)
REGION_CONTAMINANT = np.uint8(8)
REGION_PROFILE_FIT = np.uint8(16)

DOUBLE_GAUSSIAN_MAX_WING_SIGMA_RATIO = 4.0
DOUBLE_GAUSSIAN_MAX_WING_FRACTION = 0.30
SUPPORTED_PROFILE_MODELS = ("moffat", "empirical")
SUPPORTED_PROFILE_AGGREGATION_MODES = (
    "row_by_row",
    "summed_per_exposure",
    "normalized_summed_per_exposure",
)
SUPPORTED_EXCLUDED_COLUMN_MODES = (
    "asymmetric",
    "symmetric_min",
    "mask_only",
    "interpolate",
)

def save_diag_npz(name, **arrays):
    """Save diagnostic arrays to compressed .npz in ./spectral_extraction_images."""
    out_dir = "./spectral_extraction_images"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **arrays)


def save_post_extraction_ancillary_products(output_dir, sky_avgs, nstars, airmass, obs_time_array):
    """Write the legacy sky_pickler.py products directly from extraction outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nstars = int(nstars)
    for star_index in range(nstars):
        sky_spec = np.asarray(sky_avgs[star_index::nstars])
        with open(output_path / ("sky%d.pickle" % (star_index + 1)), "wb") as handle:
            pickle.dump(sky_spec, handle)

    airmass_values = np.asarray(airmass, dtype=float)
    if airmass_values.size > 0:
        np.savetxt(output_path / "airmass.txt", airmass_values)

    mjd_times = np.asarray(obs_time_array, dtype=float)
    with open(output_path / "mjd_time.pickle", "wb") as handle:
        pickle.dump(mjd_times, handle)
    np.savetxt(output_path / "mjd_time.txt", mjd_times)


def gauss(x,amplitude,mean,std,offset):
    """A Gaussian with a fitted flux offset"""
    return amplitude*np.exp(-(x-mean)**2/(std**2))+offset

def gaussian_profile(x, amplitude, mean, sigma, offset):
    """Gaussian profile with the conventional sigma parameter."""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset

def moffat_profile(x, amplitude, mean, alpha, beta, offset):
    """Moffat profile with scale alpha and power-law index beta."""
    return amplitude * (1.0 + ((x - mean) / alpha) ** 2) ** (-beta) + offset

def double_gaussian_profile(x, amplitude, mean, sigma_core, sigma_wing, wing_fraction, offset):
    """Two-component Gaussian profile with a shared centre and background offset."""
    wing_fraction = np.clip(wing_fraction, 0.0, 0.95)
    core_fraction = 1.0 - wing_fraction
    return amplitude * (
        core_fraction * np.exp(-0.5 * ((x - mean) / sigma_core) ** 2)
        + wing_fraction * np.exp(-0.5 * ((x - mean) / sigma_wing) ** 2)
    ) + offset

def gaussian_profile_zero_bg(x, amplitude, mean, sigma):
    """Gaussian source profile after local background subtraction."""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def moffat_profile_zero_bg(x, amplitude, mean, alpha, beta):
    """Moffat source profile after local background subtraction."""
    return amplitude * (1.0 + ((x - mean) / alpha) ** 2) ** (-beta)

def double_gaussian_profile_zero_bg(x, amplitude, mean, sigma_core, sigma_wing, wing_fraction):
    """Two-component Gaussian source profile after local background subtraction."""
    wing_fraction = np.clip(wing_fraction, 0.0, 0.95)
    core_fraction = 1.0 - wing_fraction
    return amplitude * (
        core_fraction * np.exp(-0.5 * ((x - mean) / sigma_core) ** 2)
        + wing_fraction * np.exp(-0.5 * ((x - mean) / sigma_wing) ** 2)
    )

def evaluate_profile_source(model_name, x, params):
    """Evaluate a supported source-only profile on the provided grid."""
    if model_name == "gaussian":
        return gaussian_profile_zero_bg(x, params["amplitude"], params["centre"], params["scale"])
    if model_name == "moffat":
        return moffat_profile_zero_bg(x, params["amplitude"], params["centre"], params["scale"], params["beta"])
    if model_name == "double_gaussian":
        return double_gaussian_profile_zero_bg(
            x,
            params["amplitude"],
            params["centre"],
            params["scale"],
            params["scale_secondary"],
            params["mix_fraction"],
        )
    raise ValueError("Unsupported profile model '%s'" % model_name)

def evaluate_profile_with_background(model_name, x, params):
    """Evaluate a supported profile including its local background offset."""
    offset = params.get("offset", 0.0)
    if model_name == "gaussian":
        return gaussian_profile(x, params["amplitude"], params["centre"], params["scale"], offset)
    if model_name == "moffat":
        return moffat_profile(x, params["amplitude"], params["centre"], params["scale"], params["beta"], offset)
    if model_name == "double_gaussian":
        return double_gaussian_profile(
            x,
            params["amplitude"],
            params["centre"],
            params["scale"],
            params["scale_secondary"],
            params["mix_fraction"],
            offset,
        )
    raise ValueError("Unsupported profile model '%s'" % model_name)

def BIC(model,data,error,n):
    """Use to calculate the Bayesian Information Criterion."""
    residuals = (model-data)/error
    chi2 = np.sum(residuals*residuals)
    bic = chi2 + n
    return bic

def get_optional_input(input_dict, key, default=None):
    """Read optional extraction_input keys without breaking older input files."""
    try:
        value = input_dict[key]
    except Exception:
        return default

    if value is None:
        return default

    if isinstance(value, str) and value.strip() == "":
        return default

    return value

def parse_multi_value(raw_value, cast, nstars, default):
    """Parse comma-separated values and broadcast singletons across stars."""
    if raw_value is None:
        values = [default]
    else:
        if isinstance(raw_value, str):
            parts = [x.strip() for x in raw_value.split(",") if x.strip() != ""]
        else:
            parts = [raw_value]
        if len(parts) == 0:
            values = [default]
        else:
            values = [cast(x) for x in parts]

    if len(values) == 1:
        values = values * nstars

    if len(values) != nstars:
        raise ValueError("Expected either one value or one value per star in extraction_input.")

    return values


def normalize_profile_aggregation_mode(raw_value):
    """Normalise user-facing aliases for how percentile profiles are measured."""
    value = str(raw_value).strip().lower()
    aliases = {
        "": "row_by_row",
        "row": "row_by_row",
        "rows": "row_by_row",
        "per_row": "row_by_row",
        "rowwise": "row_by_row",
        "row_by_row": "row_by_row",
        "individual_rows": "row_by_row",
        "summed": "summed_per_exposure",
        "sum": "summed_per_exposure",
        "summed_rows": "summed_per_exposure",
        "sum_rows": "summed_per_exposure",
        "summed_per_exposure": "summed_per_exposure",
        "per_exposure_sum": "summed_per_exposure",
        "exposure_sum": "summed_per_exposure",
        "summed_profile": "summed_per_exposure",
        "normalised_sum": "normalized_summed_per_exposure",
        "normalised_summed": "normalized_summed_per_exposure",
        "normalised_summed_per_exposure": "normalized_summed_per_exposure",
        "normalized_sum": "normalized_summed_per_exposure",
        "normalized_summed": "normalized_summed_per_exposure",
        "normalized_sum_rows": "normalized_summed_per_exposure",
        "normalized_summed_rows": "normalized_summed_per_exposure",
        "normalized_summed_per_exposure": "normalized_summed_per_exposure",
        "normalised_profile_sum": "normalized_summed_per_exposure",
        "normalized_profile_sum": "normalized_summed_per_exposure",
        "normalised_summed_profile": "normalized_summed_per_exposure",
        "normalized_summed_profile": "normalized_summed_per_exposure",
    }
    if value not in aliases:
        raise ValueError(
            "Unsupported profile_aggregation_mode '%s'. Use one of: %s"
            % (raw_value, ", ".join(SUPPORTED_PROFILE_AGGREGATION_MODES))
        )
    return aliases[value]


def is_summed_profile_aggregation_mode(raw_value):
    """Return True when one shared profile is fit per exposure rather than per row."""
    return normalize_profile_aggregation_mode(raw_value) in (
        "summed_per_exposure",
        "normalized_summed_per_exposure",
    )


def uses_normalized_summed_profile(raw_value):
    """Return True when rows are peak-normalized before they are summed."""
    return normalize_profile_aggregation_mode(raw_value) == "normalized_summed_per_exposure"


def describe_profile_aggregation_mode(raw_value):
    """Human-readable label for diagnostic text."""
    canonical_mode = normalize_profile_aggregation_mode(raw_value)
    if canonical_mode == "row_by_row":
        return "row by row"
    if canonical_mode == "summed_per_exposure":
        return "summed per exposure"
    if canonical_mode == "normalized_summed_per_exposure":
        return "peak-normalized rows, then summed per exposure"
    return canonical_mode


def parse_excluded_column_strips(raw_value, oversampling_factor=1):
    """Parse detector-column strips such as '183' or '183:184,205:207' into merged half-open ranges."""
    if raw_value is None:
        return []

    text = str(raw_value).strip()
    if text == "":
        return []

    ranges = []
    scale = max(int(oversampling_factor), 1)
    for part in text.split(","):
        part = part.strip()
        if part == "":
            continue
        if ":" in part:
            start_text, end_text = [x.strip() for x in part.split(":", 1)]
            start = int(float(start_text)) * scale
            end_inclusive = int(float(end_text)) * scale
            left = min(start, end_inclusive)
            right = max(start, end_inclusive) + scale
        else:
            left = int(float(part)) * scale
            right = left + scale
        ranges.append((left, right))

    if len(ranges) == 0:
        return []

    ranges = sorted(ranges, key=lambda item: (item[0], item[1]))
    merged = [list(ranges[0])]
    for left, right in ranges[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [(int(left), int(right)) for left, right in merged]


def merge_excluded_column_ranges(excluded_column_ranges):
    if excluded_column_ranges is None:
        return []

    cleaned = []
    for item in excluded_column_ranges:
        if item is None or len(item) != 2:
            continue
        left = float(item[0])
        right = float(item[1])
        if not np.isfinite(left) or not np.isfinite(right) or right <= left:
            continue
        cleaned.append((left, right))

    if len(cleaned) == 0:
        return []

    cleaned = sorted(cleaned, key=lambda item: (item[0], item[1]))
    merged = [list(cleaned[0])]
    for left, right in cleaned[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [(float(left), float(right)) for left, right in merged]


def combine_excluded_column_ranges(*range_groups):
    combined = []
    for group in range_groups:
        if group is None:
            continue
        combined.extend(group)
    return merge_excluded_column_ranges(combined)


def normalize_row_excluded_column_ranges(row_excluded_column_ranges, nrows):
    if row_excluded_column_ranges is None:
        return None

    normalized = [[] for _ in range(int(max(nrows, 0)))]
    for i in range(min(len(row_excluded_column_ranges), len(normalized))):
        normalized[i] = merge_excluded_column_ranges(row_excluded_column_ranges[i])
    return normalized


def get_row_excluded_column_ranges(excluded_column_ranges=None, row_excluded_column_ranges=None, row_index=None):
    static_ranges = merge_excluded_column_ranges(excluded_column_ranges)
    if row_excluded_column_ranges is None or row_index is None:
        return static_ranges
    if row_index < 0 or row_index >= len(row_excluded_column_ranges):
        return static_ranges
    return combine_excluded_column_ranges(static_ranges, row_excluded_column_ranges[row_index])


def build_star_relative_excluded_column_ranges(excluded_column_ranges, source_trace, target_trace):
    base_ranges = merge_excluded_column_ranges(excluded_column_ranges)
    if len(base_ranges) == 0:
        return None

    source_trace = np.asarray(source_trace, dtype=float)
    target_trace = np.asarray(target_trace, dtype=float)
    nrows = int(min(len(source_trace), len(target_trace)))
    if nrows == 0:
        return None

    row_ranges = [[] for _ in range(len(target_trace))]
    for i in range(nrows):
        if not np.isfinite(source_trace[i]) or not np.isfinite(target_trace[i]):
            continue
        shift = float(target_trace[i]) - float(source_trace[i])
        row_ranges[i] = merge_excluded_column_ranges(
            [(left + shift, right + shift) for left, right in base_ranges]
        )
    return row_ranges


def pack_row_excluded_column_ranges(row_excluded_column_ranges, nrows):
    normalized = normalize_row_excluded_column_ranges(row_excluded_column_ranges, nrows)
    if normalized is None:
        return np.empty((0, 0, 2), dtype=np.float32)

    max_strips = max((len(ranges) for ranges in normalized), default=0)
    if max_strips == 0:
        return np.empty((len(normalized), 0, 2), dtype=np.float32)

    packed = np.full((len(normalized), max_strips, 2), np.nan, dtype=np.float32)
    for row_index, ranges in enumerate(normalized):
        for strip_index, (left, right) in enumerate(ranges):
            packed[row_index, strip_index, 0] = float(left)
            packed[row_index, strip_index, 1] = float(right)
    return packed


def normalize_excluded_column_mode(mode):
    text = "asymmetric" if mode is None else str(mode).strip().lower()
    aliases = {
        "mask": "mask_only",
        "masked": "mask_only",
        "interpolated": "interpolate",
    }
    return aliases.get(text, text)


def mask_values_in_column_strips(values, excluded_column_ranges):
    """Return a boolean mask for values that fall inside any excluded detector-column strip."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or excluded_column_ranges is None or len(excluded_column_ranges) == 0:
        return np.zeros(values.shape, dtype=bool)

    mask = np.zeros(values.shape, dtype=bool)
    for left, right in excluded_column_ranges:
        mask |= (values >= float(left)) & (values < float(right))
    return mask


def excluded_column_mode_clips_aperture(mode):
    return normalize_excluded_column_mode(mode) in ("asymmetric", "symmetric_min")


def excluded_column_mode_interpolates(mode):
    return normalize_excluded_column_mode(mode) == "interpolate"


def get_excluded_column_data_mask(values, excluded_column_ranges, mode="asymmetric"):
    if excluded_column_mode_interpolates(mode):
        values = np.asarray(values)
        return np.zeros(values.shape, dtype=bool)
    return mask_values_in_column_strips(values, excluded_column_ranges)


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


def prepare_row_for_excluded_columns(row, excluded_column_ranges, mode="asymmetric"):
    row = np.asarray(row, dtype=float)
    if row.size == 0:
        return np.array(row, copy=True), np.zeros(row.shape, dtype=bool)

    column_mask = mask_values_in_column_strips(np.arange(len(row), dtype=float), excluded_column_ranges)
    if not np.any(column_mask):
        return np.array(row, copy=True), column_mask

    if excluded_column_mode_interpolates(mode):
        return interpolate_masked_columns_1d(row, column_mask), column_mask

    return np.array(row, copy=True), column_mask


def clip_aperture_edges_against_column_strips(left_edge, right_edge, trace_centre, excluded_column_ranges):
    """Clip aperture edges so the contiguous aperture does not span across excluded detector strips."""
    left_edge = int(left_edge)
    right_edge = int(right_edge)
    if excluded_column_ranges is None or len(excluded_column_ranges) == 0:
        return left_edge, right_edge

    centre = float(trace_centre)
    for strip_left, strip_right in excluded_column_ranges:
        if strip_right <= left_edge or strip_left >= right_edge:
            continue
        if strip_right <= centre:
            left_edge = max(left_edge, int(strip_right))
        elif strip_left >= centre:
            right_edge = min(right_edge, int(strip_left))
        else:
            left_span = max(centre - left_edge, 0.0)
            right_span = max(right_edge - centre, 0.0)
            if left_span >= right_span:
                left_edge = max(left_edge, int(strip_right))
            else:
                right_edge = min(right_edge, int(strip_left))
    return left_edge, right_edge


def symmetrize_aperture_edges_from_min_span(left_edge, right_edge, trace_centre, ncols, buffer_left=0, buffer_right=0):
    """Make the aperture symmetric about the trace centre using the smaller surviving half-width."""
    centre = float(trace_centre)
    left_span = max(centre - float(left_edge), 0.0)
    right_span = max(float(right_edge) - centre, 0.0)
    radius = min(left_span, right_span)
    return compute_symmetric_aperture_edges(
        centre,
        radius,
        ncols,
        buffer_left=buffer_left,
        buffer_right=buffer_right,
    )


def apply_excluded_column_aperture_mode(
    left_edge,
    right_edge,
    trace_centre,
    excluded_column_ranges,
    ncols,
    buffer_left=0,
    buffer_right=0,
    mode="asymmetric",
):
    """Apply excluded-column clipping and optionally re-symmetrize the aperture about the trace centre."""
    mode = normalize_excluded_column_mode(mode)
    if not excluded_column_mode_clips_aperture(mode):
        return int(left_edge), int(right_edge)

    left_edge, right_edge = clip_aperture_edges_against_column_strips(
        left_edge,
        right_edge,
        trace_centre,
        excluded_column_ranges,
    )
    if mode == "symmetric_min":
        left_edge, right_edge = symmetrize_aperture_edges_from_min_span(
            left_edge,
            right_edge,
            trace_centre,
            ncols,
            buffer_left=buffer_left,
            buffer_right=buffer_right,
        )
    return int(left_edge), int(right_edge)

def resolve_input_path(base_dir, path_value):
    """Resolve a path from extraction_input relative to the input file location."""
    if path_value is None:
        return None

    path_value = str(path_value).strip()
    if path_value == "":
        return None

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)

def resolve_science_files(science_input, instrument, base_dir):
    """Resolve science inputs from a list file, glob pattern, or reference FITS frame."""
    import glob

    if science_input is None:
        raise ValueError("science_list is not set in extraction_input.")

    science_input = str(science_input).strip()
    if science_input == "":
        raise ValueError("science_list is empty in extraction_input.")

    fits_suffixes = (".fits", ".fit", ".fits.gz", ".fit.gz", ".fz")

    resolved_input = Path(resolve_input_path(base_dir, science_input))

    if resolved_input.exists() and not resolved_input.name.lower().endswith(fits_suffixes):
        listed_files = np.atleast_1d(np.loadtxt(str(resolved_input), str))
        resolved_files = []
        for listed_file in listed_files:
            listed_file = str(listed_file).strip()
            if listed_file == "":
                continue
            resolved_files.append(resolve_input_path(resolved_input.parent, listed_file))
        return np.atleast_1d(resolved_files)

    if resolved_input.is_absolute():
        glob_pattern = str(resolved_input)
    else:
        glob_pattern = str((base_dir / science_input).resolve())
    glob_matches = sorted(glob.glob(glob_pattern))
    if len(glob_matches) > 0:
        return np.atleast_1d(glob_matches)

    instrument = str(instrument).strip()
    if instrument == "ACAM":
        candidate_patterns = ["r*.fit", "r*.fits", "*.fit", "*.fits"]
    elif instrument == "EFOSC":
        candidate_patterns = ["EFOSC_Spectrum*.fits", "*.fits", "*.fit"]
    elif instrument == "Keck/NIRSPEC":
        candidate_patterns = ["*.fits", "*.fit"]
    elif "JWST" in instrument:
        candidate_patterns = ["*gainscalestep.fits", "*rateints.fits", "*.fits"]
    else:
        candidate_patterns = ["*.fits", "*.fit"]

    candidate_files = []
    for pattern in candidate_patterns:
        candidate_files.extend(glob.glob(str(base_dir / pattern)))
    candidate_files = sorted(dict.fromkeys(candidate_files))

    if len(candidate_files) == 0:
        raise FileNotFoundError(
            "science_list='%s' is neither a readable list file nor a matching glob, and no candidate FITS files were found relative to %s."
            % (science_input, base_dir)
        )

    basenames = [os.path.basename(x) for x in candidate_files]
    ref_name = os.path.basename(science_input)

    resolved_input_str = str(resolved_input)
    if resolved_input_str in candidate_files:
        start_index = candidate_files.index(resolved_input_str)
        return np.atleast_1d(candidate_files[start_index:])

    if ref_name in basenames:
        start_index = basenames.index(ref_name)
        return np.atleast_1d(candidate_files[start_index:])

    if any(char in science_input for char in "*?[]"):
        return np.atleast_1d(candidate_files)

    raise FileNotFoundError(
        "science_list='%s' was not found. Set it to a real list file, a glob pattern such as 'EFOSC_Spectrum*.fits', or an existing reference FITS filename."
        % science_input
    )

def compute_profile_radius(model_name, params, percentile, ncols, oversample=10, x_min=None, x_max=None):
    """Return the half-width containing the requested percentile of the fitted profile."""
    percentile = float(percentile)
    if percentile > 1:
        percentile /= 100.0

    percentile = np.clip(percentile, 1e-3, 0.9999)

    centre = params["centre"]
    if x_min is None:
        x_min = 0.0
    if x_max is None:
        x_max = float(ncols - 1)
    x_min = float(np.clip(x_min, 0.0, max(ncols - 1, 0)))
    x_max = float(np.clip(x_max, x_min, max(ncols - 1, 0)))
    x_dense = np.linspace(x_min, x_max, max(int(max(x_max - x_min, 1.0) * oversample), int(np.ceil(x_max - x_min)) + 1))

    model = evaluate_profile_source(model_name, x_dense, params)

    model = np.clip(model, 0, None)
    total_flux = model.sum()

    if not np.isfinite(total_flux) or total_flux <= 0:
        return None

    distances = np.abs(x_dense - centre)
    order = np.argsort(distances)
    cumulative = np.cumsum(model[order])
    idx = np.searchsorted(cumulative, percentile * total_flux, side="left")
    idx = min(idx, len(order) - 1)
    return float(distances[order][idx])

def compute_profile_fwhm(model_name, params, ncols, oversample=20):
    """Return the full width at half maximum for a supported source profile."""
    x_dense = np.linspace(0, ncols - 1, max(int(ncols * oversample), ncols))
    model = np.clip(evaluate_profile_source(model_name, x_dense, params), 0, None)
    if len(model) == 0:
        return np.nan

    peak = np.nanmax(model)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan

    half_max = 0.5 * peak
    mask = model >= half_max
    if not np.any(mask):
        return np.nan

    indices = np.where(mask)[0]
    return float(x_dense[indices[-1]] - x_dense[indices[0]])


def compute_percentile_radius_from_samples(x_values, sample_values, centre, percentile):
    """Return the symmetric half-width containing the requested fraction of sampled flux."""
    percentile = float(percentile)
    if percentile > 1:
        percentile /= 100.0
    percentile = np.clip(percentile, 1e-3, 0.9999)

    x_values = np.asarray(x_values, dtype=float)
    sample_values = np.asarray(sample_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(sample_values)
    if not np.any(finite):
        return None

    x_values = x_values[finite]
    sample_values = np.clip(sample_values[finite], 0.0, None)
    total_flux = np.sum(sample_values)
    if not np.isfinite(total_flux) or total_flux <= 0:
        return None

    distances = np.abs(x_values - float(centre))
    order = np.argsort(distances)
    cumulative = np.cumsum(sample_values[order])
    idx = np.searchsorted(cumulative, percentile * total_flux, side="left")
    idx = min(idx, len(order) - 1)
    return float(distances[order][idx])


def compute_fwhm_from_samples(x_values, sample_values):
    """Estimate the FWHM directly from sampled source values."""
    x_values = np.asarray(x_values, dtype=float)
    sample_values = np.asarray(sample_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(sample_values)
    if not np.any(finite):
        return np.nan

    x_values = x_values[finite]
    sample_values = np.clip(sample_values[finite], 0.0, None)
    peak = np.nanmax(sample_values)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan

    mask = sample_values >= 0.5 * peak
    if not np.any(mask):
        return np.nan

    indices = np.where(mask)[0]
    return float(x_values[indices[-1]] - x_values[indices[0]])


def compute_empirical_profile_metrics(x_values, source_values, centre, percentile):
    """Return a symmetric empirical aperture radius and FWHM from sampled source values."""
    radius = compute_percentile_radius_from_samples(x_values, source_values, centre, percentile)
    fwhm = compute_fwhm_from_samples(x_values, source_values)
    return radius, fwhm


def compute_empirical_capture_fraction(x_values, source_values, aperture_left, aperture_right):
    """Return the fraction of empirical source flux falling inside the supplied aperture."""
    x_values = np.asarray(x_values, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(source_values)
    if not np.any(finite):
        return np.nan, np.nan, np.nan

    x_values = x_values[finite]
    source_values = np.clip(source_values[finite], 0.0, None)
    total_flux = np.sum(source_values)
    if not np.isfinite(total_flux) or total_flux <= 0:
        return np.nan, np.nan, np.nan

    aperture_mask = (x_values >= float(aperture_left)) & (x_values < float(aperture_right))
    captured_flux = np.sum(source_values[aperture_mask])
    if not np.isfinite(captured_flux):
        return np.nan, np.nan, np.nan

    return float(captured_flux / total_flux), float(captured_flux), float(total_flux)


def save_fixed_width_capture_products(capture_matrices_by_star, capture_configs, row_numbers, obs_time_array, output_dir):
    """Save empirical aperture-capture diagnostics as summary tables and a heatmap/summary figure."""
    enabled_stars = [
        (star_index, np.asarray(capture_matrices_by_star[star_index], dtype=float))
        for star_index in range(len(capture_matrices_by_star))
        if bool(capture_configs[star_index].get("enabled", False))
    ]
    enabled_stars = [
        (star_index, matrix)
        for star_index, matrix in enabled_stars
        if matrix.size > 0 and matrix.ndim == 2 and matrix.shape[0] > 0 and matrix.shape[1] > 0
    ]
    if len(enabled_stars) == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    row_numbers = np.asarray(row_numbers, dtype=int)
    obs_time_array = np.asarray(obs_time_array, dtype=float)

    fig, axes = plt.subplots(len(enabled_stars), 2, figsize=(14, 4.6 * len(enabled_stars)), squeeze=False)

    for row_idx, (star_index, capture_matrix) in enumerate(enabled_stars):
        frame_indices = np.arange(1, capture_matrix.shape[0] + 1, dtype=int)
        capture_percent = 100.0 * capture_matrix
        mean_fraction = np.nanmean(capture_matrix, axis=1)
        std_fraction = np.nanstd(capture_matrix, axis=1)
        median_fraction = np.nanmedian(capture_matrix, axis=1)
        min_fraction = np.nanmin(capture_matrix, axis=1)
        max_fraction = np.nanmax(capture_matrix, axis=1)

        summary_table = np.column_stack(
            [
                frame_indices,
                obs_time_array[: capture_matrix.shape[0]],
                mean_fraction,
                std_fraction,
                median_fraction,
                min_fraction,
                max_fraction,
                100.0 * mean_fraction,
                100.0 * std_fraction,
                100.0 * median_fraction,
                100.0 * min_fraction,
                100.0 * max_fraction,
            ]
        )
        summary_path = table_dir / ("aperture_capture_summary_star%d.txt" % (star_index + 1))
        np.savetxt(
            summary_path,
            summary_table,
            fmt=["%d", "%.10f", "%.8f", "%.8f", "%.8f", "%.8f", "%.8f", "%.4f", "%.4f", "%.4f", "%.4f", "%.4f"],
            header=(
                "frame_index obs_time mean_fraction std_fraction median_fraction min_fraction max_fraction "
                "mean_percent std_percent median_percent min_percent max_percent"
            ),
        )

        per_row_table = np.column_stack(
            [
                frame_indices,
                obs_time_array[: capture_matrix.shape[0]],
                capture_matrix,
                capture_percent,
            ]
        )
        fraction_headers = ["fraction_row_%d" % int(row_no) for row_no in row_numbers]
        percent_headers = ["percent_row_%d" % int(row_no) for row_no in row_numbers]
        per_row_formats = ["%d", "%.10f"] + ["%.8f"] * len(fraction_headers) + ["%.4f"] * len(percent_headers)
        per_row_path = table_dir / ("aperture_capture_per_row_star%d.txt" % (star_index + 1))
        np.savetxt(
            per_row_path,
            per_row_table,
            fmt=per_row_formats,
            header="frame_index obs_time " + " ".join(fraction_headers + percent_headers),
        )

        heatmap_ax = axes[row_idx, 0]
        finite = np.isfinite(capture_percent)
        if np.any(finite):
            vmin = np.nanpercentile(capture_percent[finite], 5)
            vmax = np.nanpercentile(capture_percent[finite], 95)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin = np.nanmin(capture_percent[finite])
                vmax = np.nanmax(capture_percent[finite])
        else:
            vmin, vmax = 0.0, 100.0

        image = heatmap_ax.imshow(
            capture_percent.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[frame_indices[0] - 0.5, frame_indices[-1] + 0.5, row_numbers[0] - 0.5, row_numbers[-1] + 0.5],
            vmin=vmin,
            vmax=vmax,
        )
        heatmap_ax.set_title("Star %d empirical fraction captured by final aperture" % (star_index + 1))
        heatmap_ax.set_xlabel("Frame")
        heatmap_ax.set_ylabel("Detector row")
        cbar = fig.colorbar(image, ax=heatmap_ax, pad=0.02)
        cbar.set_label("Captured flux (%)")

        summary_ax = axes[row_idx, 1]
        summary_ax.plot(frame_indices, 100.0 * mean_fraction, color="tab:blue", lw=1.4, label="mean")
        summary_ax.fill_between(
            frame_indices,
            100.0 * (mean_fraction - std_fraction),
            100.0 * (mean_fraction + std_fraction),
            color="tab:blue",
            alpha=0.18,
            label=r"$\pm1\sigma$",
        )
        summary_ax.plot(frame_indices, 100.0 * median_fraction, color="tab:orange", lw=1.0, ls="--", label="median")
        summary_ax.set_title("Star %d per-exposure empirical capture fraction" % (star_index + 1))
        summary_ax.set_xlabel("Frame")
        summary_ax.set_ylabel("Captured flux (%)")
        summary_ax.legend(loc="best")

    fig.suptitle("Empirical aperture-capture diagnostics", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(output_dir / "aperture_capture_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_aperture_capture_products_from_diagnostics(diagnostic_input=".", output_dir=None):
    """Rebuild empirical aperture-capture summary products from saved diagnostic bundles."""
    try:
        from plot_extraction_diagnostics import (
            build_frame_groups,
            get_frame_time_value,
            load_frame_records,
            resolve_science_frame_path,
        )
    except Exception as exc:
        raise RuntimeError("could not import diagnostic helpers (%s)" % exc)

    def record_bool(record, key, default=False):
        value = record.get(key, np.array(default))
        try:
            return bool(value.item())
        except Exception:
            return bool(value)

    frame_groups = build_frame_groups(diagnostic_input)
    records_by_frame = []
    star_indices = set()
    obs_time_values = []

    for frame_group in frame_groups:
        frame_records = {}
        for record in load_frame_records(frame_group):
            frame_records[record["star_index"]] = record
            star_indices.add(record["star_index"])
        if len(frame_records) == 0:
            continue
        representative_record = next(iter(frame_records.values()))
        obs_time_values.append(float(get_frame_time_value(str(resolve_science_frame_path(representative_record)))))
        records_by_frame.append(frame_records)

    if len(records_by_frame) == 0:
        raise RuntimeError("no diagnostic records were loaded")
    if len(star_indices) == 0:
        raise RuntimeError("no stars were found in the saved diagnostics")

    max_star_index = max(star_indices)
    capture_matrices_by_star = [np.empty((0, 0), dtype=float) for _ in range(max_star_index)]
    capture_configs = [{"enabled": False} for _ in range(max_star_index)]
    common_row_numbers = None

    for star_index in sorted(star_indices):
        reference_record = next((frame_records[star_index] for frame_records in records_by_frame if star_index in frame_records), None)
        if reference_record is None:
            continue

        row_numbers = np.asarray(reference_record["row_numbers"], dtype=int)
        if common_row_numbers is None:
            common_row_numbers = row_numbers
        elif not np.array_equal(common_row_numbers, row_numbers):
            raise RuntimeError("row numbering is inconsistent across stars in the saved diagnostics")

        capture_rows = []
        capture_enabled = False
        for frame_records in records_by_frame:
            record = frame_records.get(star_index)
            if record is None:
                capture_rows.append(np.full(len(row_numbers), np.nan, dtype=float))
                continue

            if not np.array_equal(np.asarray(record["row_numbers"], dtype=int), row_numbers):
                raise RuntimeError("row numbering is inconsistent across diagnostic frames for star %d" % star_index)

            if record_bool(record, "fixed_width_capture_enabled", False):
                capture_enabled = True
                capture_rows.append(np.asarray(record["fixed_width_capture_fraction"], dtype=float))
            else:
                capture_rows.append(np.full(len(row_numbers), np.nan, dtype=float))

        if capture_enabled:
            capture_matrices_by_star[star_index - 1] = np.vstack(capture_rows)
            capture_configs[star_index - 1]["enabled"] = True

    if common_row_numbers is None or not any(cfg.get("enabled", False) for cfg in capture_configs):
        raise RuntimeError("no aperture-capture diagnostics were found in the saved bundles")

    output_dir = Path.cwd() if output_dir is None else Path(output_dir)
    save_fixed_width_capture_products(
        capture_matrices_by_star,
        capture_configs,
        common_row_numbers,
        np.asarray(obs_time_values, dtype=float),
        output_dir,
    )


def save_aperture_width_products_from_diagnostics(diagnostic_input=".", output_dir=None):
    """Save width/FWHM diagnostic figures and summary tables from saved diagnostics."""
    try:
        from plot_extraction_diagnostics import build_frame_groups, build_metric_matrix, create_widths_figure, load_frame_records
    except Exception as exc:
        raise RuntimeError("could not import diagnostic helpers (%s)" % exc)

    frame_groups = build_frame_groups(diagnostic_input)
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
        raise RuntimeError("no diagnostic records were loaded")

    output_dir = Path.cwd() if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    star_indices = sorted(star_indices)

    for star_index in star_indices:
        reference_record = next((frame_records[star_index] for frame_records in records_by_frame if star_index in frame_records), None)
        if reference_record is None:
            continue

        row_numbers = np.asarray(reference_record["row_numbers"], dtype=int)
        width_matrix = build_metric_matrix(records_by_frame, star_index, "aperture_width", row_numbers)
        mean_width = np.nanmean(width_matrix, axis=0)
        std_width = np.nanstd(width_matrix, axis=0)
        median_width = np.nanmedian(width_matrix, axis=0)
        min_width = np.nanmin(width_matrix, axis=0)
        max_width = np.nanmax(width_matrix, axis=0)

        summary_table = np.column_stack(
            [
                np.asarray(frame_indices, dtype=int),
                mean_width,
                std_width,
                median_width,
                min_width,
                max_width,
            ]
        )
        np.savetxt(
            output_dir / ("aperture_width_summary_star%d.txt" % int(star_index)),
            summary_table,
            fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f", "%.8f"],
            header="frame_index mean_width std_width median_width min_width max_width",
        )

        per_row_table = np.column_stack([np.asarray(frame_indices, dtype=int), width_matrix.T])
        row_headers = ["width_row_%d" % int(row_no) for row_no in row_numbers]
        np.savetxt(
            output_dir / ("aperture_width_per_row_star%d.txt" % int(star_index)),
            per_row_table,
            fmt=["%d"] + ["%.8f"] * len(row_headers),
            header="frame_index " + " ".join(row_headers),
        )

        if bool(reference_record.get("profile_enabled", False)):
            fwhm_matrix = build_metric_matrix(records_by_frame, star_index, "profile_fwhm", row_numbers)
            mean_fwhm = np.nanmean(fwhm_matrix, axis=0)
            std_fwhm = np.nanstd(fwhm_matrix, axis=0)
            median_fwhm = np.nanmedian(fwhm_matrix, axis=0)
            min_fwhm = np.nanmin(fwhm_matrix, axis=0)
            max_fwhm = np.nanmax(fwhm_matrix, axis=0)
            fwhm_summary_table = np.column_stack(
                [
                    np.asarray(frame_indices, dtype=int),
                    mean_fwhm,
                    std_fwhm,
                    median_fwhm,
                    min_fwhm,
                    max_fwhm,
                ]
            )
            np.savetxt(
                output_dir / ("profile_fwhm_summary_star%d.txt" % int(star_index)),
                fwhm_summary_table,
                fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f", "%.8f"],
                header="frame_index mean_fwhm std_fwhm median_fwhm min_fwhm max_fwhm",
            )
            fwhm_per_row_table = np.column_stack([np.asarray(frame_indices, dtype=int), fwhm_matrix.T])
            np.savetxt(
                output_dir / ("profile_fwhm_per_row_star%d.txt" % int(star_index)),
                fwhm_per_row_table,
                fmt=["%d"] + ["%.8f"] * len(row_headers),
                header="frame_index " + " ".join(["fwhm_row_%d" % int(row_no) for row_no in row_numbers]),
            )

    fig = create_widths_figure(frame_groups)
    fig.savefig(output_dir / "width_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def oversample_profile_series(x_values, y_values, oversampling):
    """Interpolate a 1D profile onto an exact sub-pixel grid for fitting/display."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite):
        return x_values, y_values, False

    x_values = x_values[finite]
    y_values = y_values[finite]
    oversampling = max(int(np.round(oversampling)), 1)
    if oversampling <= 1 or len(x_values) < 2:
        return x_values, y_values, False

    dense_count = (len(x_values) - 1) * oversampling + 1
    x_dense = np.linspace(x_values[0], x_values[-1], dense_count)

    try:
        spline_order = min(3, len(x_values) - 1)
        spline = UnivariateSpline(x_values, y_values, k=spline_order, s=0)
        y_dense = spline(x_dense)
    except Exception:
        y_dense = np.interp(x_dense, x_values, y_values)

    return x_dense, y_dense, True


def compute_symmetric_aperture_edges(centre, radius, ncols, buffer_left=0, buffer_right=0):
    """Convert a continuous symmetric half-width into clipped integer extraction edges."""
    centre = float(centre)
    radius = float(radius)
    left_edge = int(np.floor(centre - radius))
    right_edge = int(np.ceil(centre + radius))
    left_edge = max(int(buffer_left), left_edge)
    right_edge = min(int(ncols - buffer_right), right_edge)

    if right_edge <= left_edge:
        left_edge = max(int(buffer_left), int(np.floor(centre)))
        right_edge = min(int(ncols - buffer_right), left_edge + 1)

    return left_edge, right_edge


def build_hybrid_wing_profile(x_values, empirical_source, model_source, centre, core_exclusion_half_width):
    """Use measured core pixels with fitted wings when the fit excludes the bright central core."""
    empirical_source = np.asarray(empirical_source, dtype=float)
    model_source = np.asarray(model_source, dtype=float)
    hybrid = np.array(model_source, copy=True)

    if core_exclusion_half_width is not None and core_exclusion_half_width > 0:
        core_mask = np.abs(np.asarray(x_values, dtype=float) - float(centre)) <= float(core_exclusion_half_width)
        hybrid[core_mask] = empirical_source[core_mask]

    hybrid[~np.isfinite(hybrid)] = 0.0
    return np.clip(hybrid, 0.0, None)

def get_local_profile_background_stats(
    row,
    fit_left,
    fit_right,
    background_offset,
    background_width,
    buffer_left,
    buffer_right,
    excluded_column_ranges=None,
    excluded_column_mode="asymmetric",
):
    """Return a clipped local sky estimate and RMS from the configured background sidebands."""
    ncols = len(row)

    if background_width == 1:
        left_cols = np.arange(buffer_left, max(buffer_left, fit_left - background_offset), dtype=int)
        right_cols = np.arange(min(ncols - buffer_right, fit_right + background_offset), ncols - buffer_right, dtype=int)
    else:
        left_start = max(buffer_left, fit_left - background_offset - background_width)
        left_end = max(left_start, fit_left - background_offset)
        right_start = min(max(fit_right + background_offset, 0), ncols - buffer_right)
        right_end = min(right_start + background_width, ncols - buffer_right)
        left_cols = np.arange(left_start, left_end, dtype=int)
        right_cols = np.arange(right_start, right_end, dtype=int)

    bg_cols = np.concatenate((left_cols, right_cols))
    if len(bg_cols) > 0:
        bg_cols = np.unique(bg_cols)
        bg_cols = bg_cols[
            ~get_excluded_column_data_mask(
                bg_cols,
                excluded_column_ranges,
                excluded_column_mode,
            )
        ]

    if len(bg_cols) == 0:
        edge_samples = np.concatenate((row[fit_left:min(fit_left + 2, fit_right)], row[max(fit_right - 2, fit_left):fit_right]))
        edge_samples = edge_samples[np.isfinite(edge_samples)]
        if len(edge_samples) == 0:
            return {"level": 0.0, "rms": 1.0}
        return {
            "level": float(np.nanmedian(edge_samples)),
            "rms": float(max(np.nanstd(edge_samples), 1.0)),
        }

    bg_values = row[bg_cols]
    finite = np.isfinite(bg_values)
    bg_cols = bg_cols[finite]
    bg_values = bg_values[finite]

    if len(bg_values) == 0:
        return {"level": 0.0, "rms": 1.0}

    bg_med = np.nanmedian(bg_values)
    bg_std = np.nanstd(bg_values)
    if np.isfinite(bg_std) and bg_std > 0:
        keep = (bg_values >= bg_med - 3.0 * bg_std) & (bg_values <= bg_med + 3.0 * bg_std)
        bg_values = bg_values[keep]

    if len(bg_values) == 0:
        level = float(bg_med if np.isfinite(bg_med) else 0.0)
        return {"level": level, "rms": float(max(bg_std, 1.0) if np.isfinite(bg_std) else 1.0)}

    level = float(np.nanmedian(bg_values))
    rms = float(np.nanstd(bg_values))
    if not np.isfinite(rms) or rms <= 0:
        rms = 1.0
    return {"level": level, "rms": rms}


def estimate_local_profile_background(
    row,
    fit_left,
    fit_right,
    background_offset,
    background_width,
    buffer_left,
    buffer_right,
    excluded_column_ranges=None,
    excluded_column_mode="asymmetric",
):
    """Estimate a local sky level for profile fitting from the configured background sidebands."""
    return get_local_profile_background_stats(
        row,
        fit_left,
        fit_right,
        background_offset,
        background_width,
        buffer_left,
        buffer_right,
        excluded_column_ranges=excluded_column_ranges,
        excluded_column_mode=excluded_column_mode,
    )["level"]

def fit_aperture_profile(row, trace_centre, fit_half_width, percentile, model_name, fallback_width, ncols, background_offset, background_width, buffer_left=0, buffer_right=0, previous_fit=None, core_exclusion_half_width=0, max_width=None, fit_oversampling=1, excluded_column_ranges=None, excluded_column_mode="asymmetric"):
    """Fit a wing-focused Moffat profile or use empirical cumulative light to define the aperture."""

    excluded_column_mode = normalize_excluded_column_mode(excluded_column_mode)
    fit_half_width = max(int(np.ceil(fit_half_width)), 2)
    fallback_width = max(int(np.ceil(fallback_width)), 2)
    core_exclusion_half_width = max(float(core_exclusion_half_width), 0.0)
    if max_width is not None:
        max_width = float(max_width)
        if max_width <= 0:
            max_width = None

    effective_half_width = max(fit_half_width, int(np.ceil(fallback_width / 2.0 + max(background_offset, 0))))
    left = max(buffer_left, int(np.floor(trace_centre - effective_half_width)))
    right = min(ncols - buffer_right, int(np.ceil(trace_centre + effective_half_width + 1)))
    x = np.arange(left, right, dtype=float)
    row_working, _ = prepare_row_for_excluded_columns(row, excluded_column_ranges, excluded_column_mode)
    y = row_working[left:right]
    finite = np.isfinite(y)

    result = {
        "success": False,
        "model_name": model_name,
        "fit_window_left": left,
        "fit_window_right": right,
        "left_edge": max(0, int(np.round(trace_centre - fallback_width / 2.0))),
        "right_edge": min(ncols, max(0, int(np.round(trace_centre + fallback_width / 2.0)))),
        "centre": float(trace_centre),
        "amplitude": np.nan,
        "offset": np.nan,
        "scale": np.nan,
        "scale_secondary": np.nan,
        "beta": np.nan,
        "mix_fraction": np.nan,
        "radius": np.nan,
        "fwhm": np.nan,
        "interp_x": None,
        "interp_y": None,
    }

    if result["right_edge"] <= result["left_edge"]:
        result["right_edge"] = min(ncols, result["left_edge"] + fallback_width)
        result["left_edge"] = max(0, result["right_edge"] - fallback_width)

    if finite.sum() < 6:
        return result

    background_stats = get_local_profile_background_stats(
        row_working,
        left,
        right,
        background_offset,
        background_width,
        buffer_left,
        buffer_right,
        excluded_column_ranges=excluded_column_ranges,
        excluded_column_mode=excluded_column_mode,
    )
    offset_guess = float(background_stats["level"])
    background_rms = max(float(background_stats["rms"]), 1.0)
    source_window = np.clip(np.where(np.isfinite(y), y - offset_guess, 0.0), 0.0, None)
    excluded_window_mask = get_excluded_column_data_mask(x, excluded_column_ranges, excluded_column_mode)

    if model_name == "empirical":
        source_window_for_metrics = source_window.copy()
        if np.any(excluded_window_mask):
            source_window_for_metrics[excluded_window_mask] = 0.0
        radius, fwhm = compute_empirical_profile_metrics(x, source_window_for_metrics, trace_centre, percentile)
        if radius is None or not np.isfinite(radius):
            return result

        if max_width is not None:
            radius = min(radius, max_width / 2.0)
        left_edge, right_edge = compute_symmetric_aperture_edges(
            trace_centre,
            radius,
            ncols,
            buffer_left=buffer_left,
            buffer_right=buffer_right,
        )
        left_edge, right_edge = apply_excluded_column_aperture_mode(
            left_edge,
            right_edge,
            trace_centre,
            excluded_column_ranges,
            ncols,
            buffer_left=buffer_left,
            buffer_right=buffer_right,
            mode=excluded_column_mode,
        )

        result.update(
            {
                "success": True,
                "left_edge": int(left_edge),
                "right_edge": int(right_edge),
                "centre": float(trace_centre),
                "amplitude": float(np.nanmax(source_window)) if np.any(np.isfinite(source_window)) else np.nan,
                "offset": offset_guess,
                "radius": float(radius),
                "fwhm": float(fwhm),
            }
        )
        return result

    if model_name != "moffat":
        raise ValueError("Unsupported profile model '%s'. Use moffat or empirical." % model_name)

    x_fit = x[finite]
    y_fit = y[finite]
    y_fit_sub = y_fit - offset_guess
    positive_signal = np.clip(y_fit_sub, 0.0, None)
    if not np.any(np.isfinite(positive_signal)):
        return result

    amplitude_guess = np.nanmax(positive_signal)
    if not np.isfinite(amplitude_guess) or amplitude_guess <= 0:
        amplitude_guess = max(np.nanstd(y_fit_sub), 1.0)

    excluded_fit_mask = get_excluded_column_data_mask(x_fit, excluded_column_ranges, excluded_column_mode)
    wing_mask = ~excluded_fit_mask
    if core_exclusion_half_width > 0:
        wing_mask = wing_mask & (np.abs(x_fit - float(trace_centre)) >= core_exclusion_half_width)
        if wing_mask.sum() < 6:
            wing_mask = ~excluded_fit_mask

    fit_oversampling = max(int(np.round(fit_oversampling)), 1)
    x_fit_dense, y_fit_sub_dense, used_dense_grid = oversample_profile_series(
        x_fit,
        y_fit_sub,
        fit_oversampling,
    )
    if used_dense_grid:
        result["interp_x"] = x_fit_dense
        result["interp_y"] = y_fit_sub_dense + offset_guess

    excluded_fit_mask_dense = get_excluded_column_data_mask(
        x_fit_dense,
        excluded_column_ranges,
        excluded_column_mode,
    )
    wing_mask_dense = ~excluded_fit_mask_dense
    if core_exclusion_half_width > 0:
        wing_mask_dense = wing_mask_dense & (np.abs(x_fit_dense - float(trace_centre)) >= core_exclusion_half_width)
        if wing_mask_dense.sum() < 6:
            wing_mask_dense = ~excluded_fit_mask_dense

    x_fit_wings = x_fit_dense[wing_mask_dense]
    y_fit_wings = y_fit_sub_dense[wing_mask_dense]
    if len(y_fit_wings) < 6:
        return result

    wing_positive_signal = np.clip(y_fit_wings, 0.0, None)
    amplitude_guess = max(float(np.nanmax(wing_positive_signal)), amplitude_guess, 1.0)

    width_guess = float(max(1.0, min(effective_half_width / 2.0, fallback_width / 2.355 if fallback_width > 0 else 3.0)))
    beta_guess = 2.5
    if previous_fit is not None and previous_fit.get("success", False):
        prev_amp = float(previous_fit.get("amplitude", amplitude_guess))
        prev_scale = float(previous_fit.get("scale", width_guess))
        prev_beta = float(previous_fit.get("beta", beta_guess))
        if np.isfinite(prev_amp) and prev_amp > 0:
            amplitude_guess = prev_amp
        if np.isfinite(prev_scale) and prev_scale > 0:
            width_guess = float(np.clip(prev_scale, 0.3, max(2.0, effective_half_width * 2.0)))
        if np.isfinite(prev_beta) and prev_beta > 1.01:
            beta_guess = prev_beta

    sigma_fit = np.full_like(y_fit_wings, background_rms, dtype=float)

    try:
        def weighted_residuals(params):
            model = moffat_profile_zero_bg(x_fit_wings, params[0], float(trace_centre), params[1], params[2])
            return (model - y_fit_wings) / sigma_fit

        lower = [0.0, 0.2, 1.01]
        upper = [np.inf, effective_half_width * 3.0, 20.0]
        p0 = np.clip([amplitude_guess, width_guess, beta_guess], lower, upper)
        fit_result = optimize.least_squares(
            weighted_residuals,
            x0=p0,
            bounds=(lower, upper),
            loss="soft_l1",
            max_nfev=30000,
        )
        popt = fit_result.x
        params = {
            "amplitude": float(popt[0]),
            "centre": float(trace_centre),
            "scale": abs(float(popt[1])),
            "scale_secondary": np.nan,
            "beta": float(popt[2]),
            "mix_fraction": np.nan,
            "offset": offset_guess,
        }

        model_source = np.clip(evaluate_profile_source("moffat", x, params), 0.0, None)
        radius_source = model_source
        fwhm_source = model_source
        if core_exclusion_half_width > 0:
            hybrid_source = build_hybrid_wing_profile(
                x,
                source_window,
                model_source,
                trace_centre,
                core_exclusion_half_width,
            )
            radius_source = hybrid_source
            fwhm_source = hybrid_source

        radius = compute_percentile_radius_from_samples(x, radius_source, trace_centre, percentile)
        fwhm = compute_fwhm_from_samples(x, fwhm_source)
        if radius is None or not np.isfinite(radius):
            return result

        if max_width is not None:
            radius = min(radius, max_width / 2.0)

        left_edge, right_edge = compute_symmetric_aperture_edges(
            trace_centre,
            radius,
            ncols,
            buffer_left=buffer_left,
            buffer_right=buffer_right,
        )
        left_edge, right_edge = apply_excluded_column_aperture_mode(
            left_edge,
            right_edge,
            trace_centre,
            excluded_column_ranges,
            ncols,
            buffer_left=buffer_left,
            buffer_right=buffer_right,
            mode=excluded_column_mode,
        )

        result.update(
            {
                "success": True,
                "left_edge": int(left_edge),
                "right_edge": int(right_edge),
                "centre": float(trace_centre),
                "amplitude": float(popt[0]),
                "offset": offset_guess,
                "scale": abs(float(popt[1])),
                "beta": float(popt[2]),
                "radius": float(radius),
                "fwhm": float(fwhm),
            }
        )
        return result

    except Exception:
        return result


def fit_summed_aperture_profile(
    frame,
    trace,
    fit_half_width,
    percentile,
    model_name,
    fallback_width,
    ncols,
    background_offset,
    background_width,
    buffer_left=0,
    buffer_right=0,
    previous_fit=None,
    core_exclusion_half_width=0,
    max_width=None,
    fit_oversampling=1,
    excluded_column_ranges=None,
    row_excluded_column_ranges=None,
    excluded_column_mode="asymmetric",
    aggregation_mode="summed_per_exposure",
):
    """Fit one percentile aperture from the sum of all background-subtracted rows in an exposure."""

    excluded_column_mode = normalize_excluded_column_mode(excluded_column_mode)
    fit_half_width = max(int(np.ceil(fit_half_width)), 2)
    fallback_width = max(int(np.ceil(fallback_width)), 2)
    aggregation_mode = normalize_profile_aggregation_mode(aggregation_mode)
    normalize_rows_before_sum = uses_normalized_summed_profile(aggregation_mode)
    core_exclusion_half_width = max(float(core_exclusion_half_width), 0.0)
    if max_width is not None:
        max_width = float(max_width)
        if max_width <= 0:
            max_width = None

    frame = np.asarray(frame, dtype=float)
    trace = np.asarray(trace, dtype=float)
    nrows = frame.shape[0]

    effective_half_width = max(
        fit_half_width,
        int(np.ceil(fallback_width / 2.0 + max(background_offset, 0))),
    )
    relative_x = np.arange(-effective_half_width, effective_half_width + 1, dtype=float)

    row_source_stack = np.full((nrows, len(relative_x)), np.nan, dtype=np.float32)
    row_fit_left = np.full(nrows, -1, dtype=int)
    row_fit_right = np.full(nrows, -1, dtype=int)
    row_background_level = np.full(nrows, np.nan, dtype=np.float32)
    row_background_rms = np.full(nrows, np.nan, dtype=np.float32)

    result = {
        "success": False,
        "model_name": model_name,
        "fit_window_left": float(relative_x[0]) if len(relative_x) > 0 else np.nan,
        "fit_window_right": float(relative_x[-1]) if len(relative_x) > 0 else np.nan,
        "left_edge": np.nan,
        "right_edge": np.nan,
        "centre": 0.0,
        "amplitude": np.nan,
        "offset": 0.0,
        "scale": np.nan,
        "scale_secondary": np.nan,
        "beta": np.nan,
        "mix_fraction": np.nan,
        "radius": np.nan,
        "fwhm": np.nan,
        "interp_x": np.array([], dtype=float),
        "interp_y": np.array([], dtype=float),
        "relative_x": relative_x.astype(np.float32),
        "source_sum": np.full(len(relative_x), np.nan, dtype=np.float32),
        "contributor_counts": np.zeros(len(relative_x), dtype=np.int16),
        "row_source_stack": row_source_stack,
        "row_fit_left": row_fit_left,
        "row_fit_right": row_fit_right,
        "row_background_level": row_background_level,
        "row_background_rms": row_background_rms,
        "rows_used": 0,
        "row_normalization": "peak" if normalize_rows_before_sum else "none",
    }

    if len(relative_x) == 0 or nrows == 0:
        return result

    for i, row in enumerate(frame):
        row_excluded_ranges = get_row_excluded_column_ranges(
            excluded_column_ranges,
            row_excluded_column_ranges,
            i,
        )
        row_working, _ = prepare_row_for_excluded_columns(row, row_excluded_ranges, excluded_column_mode)
        centre = float(trace[i])
        left = max(buffer_left, int(np.floor(centre - effective_half_width)))
        right = min(ncols - buffer_right, int(np.ceil(centre + effective_half_width + 1)))
        row_fit_left[i] = left
        row_fit_right[i] = right
        if right <= left:
            continue

        x = np.arange(left, right, dtype=float)
        y = row_working[left:right]
        finite = np.isfinite(y)
        if finite.sum() < 2:
            continue

        background_stats = get_local_profile_background_stats(
            row_working,
            left,
            right,
            background_offset,
            background_width,
            buffer_left,
            buffer_right,
            excluded_column_ranges=row_excluded_ranges,
            excluded_column_mode=excluded_column_mode,
        )
        row_background_level[i] = float(background_stats["level"])
        row_background_rms[i] = float(background_stats["rms"])

        x_rel = x[finite] - centre
        source_values = y[finite] - float(background_stats["level"])
        if len(x_rel) < 2:
            continue

        interpolated = np.interp(relative_x, x_rel, source_values, left=np.nan, right=np.nan)
        x_rel_min = np.nanmin(x_rel)
        x_rel_max = np.nanmax(x_rel)
        interpolated[(relative_x < x_rel_min) | (relative_x > x_rel_max)] = np.nan

        absolute_x = relative_x + centre
        detector_mask = (absolute_x < float(buffer_left)) | (absolute_x >= float(ncols - buffer_right))
        if np.any(detector_mask):
            interpolated[detector_mask] = np.nan

        if row_excluded_ranges is not None and len(row_excluded_ranges) > 0:
            excluded_mask = get_excluded_column_data_mask(
                absolute_x,
                row_excluded_ranges,
                excluded_column_mode,
            )
            interpolated[excluded_mask] = np.nan

        if normalize_rows_before_sum:
            positive_profile = np.clip(np.where(np.isfinite(interpolated), interpolated, np.nan), 0.0, None)
            row_peak = np.nanmax(positive_profile)
            if not np.isfinite(row_peak) or row_peak <= 0:
                continue
            interpolated = interpolated / row_peak

        row_source_stack[i] = interpolated.astype(np.float32)

    contributor_counts = np.sum(np.isfinite(row_source_stack), axis=0).astype(np.int16)
    source_sum = np.nansum(np.where(np.isfinite(row_source_stack), row_source_stack, 0.0), axis=0).astype(np.float32)
    source_sum[contributor_counts == 0] = np.nan
    result["source_sum"] = source_sum
    result["contributor_counts"] = contributor_counts
    result["rows_used"] = int(np.count_nonzero(np.any(np.isfinite(row_source_stack), axis=1)))

    finite = np.isfinite(source_sum)
    if finite.sum() < 6:
        return result

    if model_name == "empirical":
        source_window_for_metrics = np.clip(np.where(np.isfinite(source_sum), source_sum, 0.0), 0.0, None)
        radius, fwhm = compute_empirical_profile_metrics(relative_x, source_window_for_metrics, 0.0, percentile)
        if radius is None or not np.isfinite(radius):
            return result

        if max_width is not None:
            radius = min(radius, max_width / 2.0)

        result.update(
            {
                "success": True,
                "amplitude": float(np.nanmax(source_window_for_metrics)) if np.any(np.isfinite(source_window_for_metrics)) else np.nan,
                "radius": float(radius),
                "fwhm": float(fwhm),
            }
        )
        return result

    if model_name != "moffat":
        raise ValueError("Unsupported profile model '%s'. Use moffat or empirical." % model_name)

    x_fit = relative_x[finite]
    y_fit_sub = source_sum[finite].astype(float)
    positive_signal = np.clip(y_fit_sub, 0.0, None)
    if not np.any(np.isfinite(positive_signal)):
        return result

    amplitude_guess = np.nanmax(positive_signal)
    if not np.isfinite(amplitude_guess) or amplitude_guess <= 0:
        amplitude_guess = max(np.nanstd(y_fit_sub), 1.0)

    wing_mask = np.ones_like(x_fit, dtype=bool)
    if core_exclusion_half_width > 0:
        wing_mask = wing_mask & (np.abs(x_fit) >= core_exclusion_half_width)
        if wing_mask.sum() < 6:
            wing_mask = np.ones_like(x_fit, dtype=bool)

    x_fit_dense, y_fit_sub_dense, used_dense_grid = oversample_profile_series(
        x_fit,
        y_fit_sub,
        fit_oversampling,
    )
    if used_dense_grid:
        result["interp_x"] = x_fit_dense
        result["interp_y"] = y_fit_sub_dense

    wing_mask_dense = np.ones_like(x_fit_dense, dtype=bool)
    if core_exclusion_half_width > 0:
        wing_mask_dense = wing_mask_dense & (np.abs(x_fit_dense) >= core_exclusion_half_width)
        if wing_mask_dense.sum() < 6:
            wing_mask_dense = np.ones_like(x_fit_dense, dtype=bool)

    x_fit_wings = x_fit_dense[wing_mask_dense]
    y_fit_wings = y_fit_sub_dense[wing_mask_dense]
    if len(y_fit_wings) < 6:
        return result

    wing_positive_signal = np.clip(y_fit_wings, 0.0, None)
    amplitude_guess = max(float(np.nanmax(wing_positive_signal)), amplitude_guess, 1.0)

    background_rms = np.sqrt(np.nansum(np.square(row_background_rms[np.isfinite(row_background_rms)])))
    if not np.isfinite(background_rms) or background_rms <= 0:
        background_rms = 1.0
    sigma_fit = np.full_like(y_fit_wings, background_rms, dtype=float)

    width_guess = float(max(1.0, min(effective_half_width / 2.0, fallback_width / 2.355 if fallback_width > 0 else 3.0)))
    beta_guess = 2.5
    if previous_fit is not None and previous_fit.get("success", False):
        prev_amp = float(previous_fit.get("amplitude", amplitude_guess))
        prev_scale = float(previous_fit.get("scale", width_guess))
        prev_beta = float(previous_fit.get("beta", beta_guess))
        if np.isfinite(prev_amp) and prev_amp > 0:
            amplitude_guess = prev_amp
        if np.isfinite(prev_scale) and prev_scale > 0:
            width_guess = float(np.clip(prev_scale, 0.3, max(2.0, effective_half_width * 2.0)))
        if np.isfinite(prev_beta) and prev_beta > 1.01:
            beta_guess = prev_beta

    try:
        def weighted_residuals(params):
            model = moffat_profile_zero_bg(x_fit_wings, params[0], 0.0, params[1], params[2])
            return (model - y_fit_wings) / sigma_fit

        lower = [0.0, 0.2, 1.01]
        upper = [np.inf, effective_half_width * 3.0, 20.0]
        p0 = np.clip([amplitude_guess, width_guess, beta_guess], lower, upper)
        fit_result = optimize.least_squares(
            weighted_residuals,
            x0=p0,
            bounds=(lower, upper),
            # The summed profiles are very high S/N, so the default robust loss can
            # collapse to a near-delta-function solution at the parameter bounds.
            loss="linear",
            max_nfev=30000,
        )
        popt = fit_result.x
        params = {
            "amplitude": float(popt[0]),
            "centre": 0.0,
            "scale": abs(float(popt[1])),
            "scale_secondary": np.nan,
            "beta": float(popt[2]),
            "mix_fraction": np.nan,
            "offset": 0.0,
        }

        model_source = np.clip(evaluate_profile_source("moffat", relative_x, params), 0.0, None)
        radius_source = model_source
        fwhm_source = model_source
        if core_exclusion_half_width > 0:
            empirical_source = np.clip(np.where(np.isfinite(source_sum), source_sum, 0.0), 0.0, None)
            hybrid_source = build_hybrid_wing_profile(
                relative_x,
                empirical_source,
                model_source,
                0.0,
                core_exclusion_half_width,
            )
            radius_source = hybrid_source
            fwhm_source = hybrid_source

        radius = compute_percentile_radius_from_samples(relative_x, radius_source, 0.0, percentile)
        fwhm = compute_fwhm_from_samples(relative_x, fwhm_source)
        if radius is None or not np.isfinite(radius):
            return result

        if max_width is not None:
            radius = min(radius, max_width / 2.0)

        result.update(
            {
                "success": True,
                "amplitude": float(popt[0]),
                "scale": abs(float(popt[1])),
                "beta": float(popt[2]),
                "radius": float(radius),
                "fwhm": float(fwhm),
            }
        )
        return result
    except Exception:
        return result

def save_extraction_diagnostic(output_name, **arrays):
    """Persist a compressed per-frame diagnostic bundle for inspection later."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DIAG_DIR / f"{output_name}.npz", **arrays)

def find_spectral_trace(frame,guess_location,search_width,gaussian_width,trace_poly_order,trace_spline_sf,star=None,verbose=False,co_add_rows=0,instrument=None,frame_no=0):
    """The function used to extract the location of a spectral trace either with a Gaussian or the argmax and then
    fits a nth order polynomial to these locations"""

    if "JWST" in instrument: # we need to extract only the first array since the frame is an array of (flux_frame,error_frame)
        frame = frame[0]

    if instrument == "Keck/NIRSPEC":
        search_frame = frame.copy()
        search_frame[search_frame < 0] = 0
    else:
        search_frame = frame

    if "JWST" in instrument:
        buffer_pixels = 0
    else:
        buffer_pixels = 5 # ignore edges of detector which might have abnormally high counts

    if guess_location-search_width < buffer_pixels:
        search_left_edge = buffer_pixels
    else:
        search_left_edge = guess_location-search_width

    if guess_location+search_width > np.shape(search_frame)[1] - buffer_pixels:
        search_right_edge = np.shape(search_frame)[1] - buffer_pixels
    else:
        search_right_edge = guess_location+search_width

    columns_of_interest = search_frame[:,search_left_edge:search_right_edge]
    nrows,ncols = np.shape(columns_of_interest)

    trace_centre = []
    fwhm = []
    gauss_std = [] # the standard deviation in pixels measured by the Gaussian

    row_array = np.arange(nrows)

    plot_row = 2000#nrows//2

    total_errors = [] # number of errors per frame
    force_verbose = False
    delay = verbose

    if delay == -1: # we're overriding force verbose
        override_force_verbose = True
        delay = 0
        verbose = False
    elif delay == -2:
        override_force_verbose = True
    else:
        override_force_verbose = False

    log = open('reduction_output.log','a')

    for i,row in enumerate(columns_of_interest):

        if co_add_rows != 0:
            if i < co_add_rows/2:
                row = np.nanmedian(columns_of_interest[i:i+co_add_rows],axis=0)
            elif i > ncols-co_add_rows/2:
                row = np.nanmedian(columns_of_interest[i-co_add_rows:i],axis=0)
            else:
                row = np.nanmedian(columns_of_interest[i-co_add_rows//2:i+co_add_rows//2],axis=0)

        x = np.arange(ncols)[np.isfinite(row)]+search_left_edge
        row = row[np.isfinite(row)]

        if len(row) == 0:
            trace_centre.append(0)
            fwhm.append(np.nan)
            gauss_std.append(np.nan)
            continue

        if instrument == "Keck/NIRSPEC":
            # clip out negative frame from A-B
            row_residuals_1 = row - np.median(row)
            keep_index_1 = row_residuals_1 >= -5*mad(row_residuals_1)
            x = x[keep_index_1]
            row = row[keep_index_1]

            # Now use a median filter to clip out cosmic rays which are sharp positive features
            row_median_filter = MF(row,5)
            row_residuals_2 = row - row_median_filter
            keep_index_2 = ((row_residuals_2 >= -5*mad(row_residuals_2)) & (row_residuals_2 <= 5*mad(row_residuals_2)))
            x = x[keep_index_2]
            row = row[keep_index_2]

        nerrors = 0 # running count of errors
        centre_guess = peak_counts_location = x[np.argmax(row)]
        amplitude = np.nanmax(row)
        amplitude_offset = np.nanmin(row)

        try:
            popt1,pcov1 = optimize.curve_fit(gauss,x,row,p0=[amplitude,centre_guess,gaussian_width,amplitude_offset])

            # Make sure fitted amplitude (with offset) is not less than 25% of the guess amplitude. - note for ACAM this number was 0.3 (70%).
            # print(search_left_edge,search_right_edge,popt1[1])
            if np.fabs(popt1[0] + popt1[-1] - amplitude) < amplitude * 0.75:
                TC = popt1[1] # trace centre
                trace_centre.append(TC)
                fwhm.append(popt1[2]*2*np.sqrt(2*np.log(2))) ### save width of gaussian as FWHM after applying conversion
                gauss_std.append(abs(popt1[2]))


            else:
                print('--- Unsatisfactory fit, appending argmax at row %d for trace %d'%(i+1,star+1))
                log.write('--- Unsatisfactory fit, appending argmax at row %d for trace %d \n'%(i+1,star+1))
                TC = centre_guess
                trace_centre.append(TC)
                nerrors += 1
                gauss_std.append(0) # append 0, this will be replaced by the mean of surrounding rows in extract_trace_flux
                fwhm.append(np.nan)


        except:
            print('--- Gaussian fit failed at row %d, appending argmax for trace %d'%(i+1,star+1))
            log.write('--- Gaussian fit failed at row %d, appending argmax for trace %d \n'%(i+1,star+1))
            TC = centre_guess
            trace_centre.append(TC)
            nerrors += 1
            gauss_std.append(0) # append 0, this will be replaced by the mean of surrounding rows in extract_trace_flux
            fwhm.append(np.nan)

        total_errors.append(nerrors)

        if nerrors > 0 and i == plot_row and not override_force_verbose:
            force_verbose = True
            delay = 5 # delay in seconds

        if verbose and i == plot_row or force_verbose:
            plt.figure(figsize=(8,6))
            plt.plot(x,row,'bo',ms=5,label='data')
            plt.axvline(centre_guess,label='guessed centre',color='grey',ls='--')
            plt.axvline(TC,label='fitted centre',color='r',ls='--')
            try:
                plt.plot(x,gauss(x,*popt1),'g',label='fit')
            except:
                pass
            plt.title('Trace detection, star %d'%(star+1))
            plt.xlabel('X pixel')
            plt.ylabel('Counts at row %d'%(i+1))
            plt.legend(loc='upper left',numpoints=1)

            if delay == -2:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(delay)
                plt.close()

            force_verbose = False

    log.close()

    # use a running median to smooth the centres, with a running box of 5 data points, before fitting with a polynomial
    trace_median_filter = MF(trace_centre,5)

    if trace_poly_order > 0: # we're using the user-defined polynomial order
        poly = np.poly1d(np.polyfit(row_array,trace_median_filter,trace_poly_order))
    else: # we use a polynomial of fourth order to find the outliers before fitting the spline (which may otherwise fit the outliers)
        poly = np.poly1d(np.polyfit(row_array,trace_median_filter,4))

    fitted_positions = poly(np.arange(nrows))
    old_fitted_positions = fitted_positions.copy() # before sigma clipping
    trace_residuals = np.array(trace_centre)-poly(row_array)

    # Clip 5 sigma outliers and refit
    std_residuals = mad(trace_residuals)
    clipped_trace_idx = (np.fabs(trace_residuals) <= 5*std_residuals)

    if trace_poly_order > 0: # we're using a polynomial for our final trace positions
        if len(row_array[clipped_trace_idx]) > 2:
            fitted_function = np.poly1d(np.polyfit(row_array[clipped_trace_idx],np.array(trace_centre)[clipped_trace_idx],trace_poly_order))
        else:
            fitted_function = poly

    if trace_spline_sf > 0: # we're using a spline for our final trace positions
        spline = UnivariateSpline(row_array,np.array(trace_centre),k=3,s=trace_spline_sf)
        old_fitted_positions = spline(np.arange(nrows))
        fitted_function = UnivariateSpline(row_array[clipped_trace_idx],np.array(trace_centre)[clipped_trace_idx],k=3,s=trace_spline_sf)


    y = np.arange(nrows)

    fitted_positions = fitted_function(np.arange(nrows))

    if sum(total_errors) > 10 and not override_force_verbose:
        force_verbose = True
        delay = 5

    if verbose or force_verbose and not override_force_verbose:
        plt.figure(figsize=(8,6))
        if instrument == "Keck/NIRSPEC":
            vmin,vmax = 0,500
        else:
            vmin,vmax = np.nanpercentile(search_frame,[10,90])
        plt.imshow(search_frame,vmin=vmin,vmax=vmax,aspect="auto")
        plt.plot(trace_centre,row_array,'r',label="row-by-row centre")
        plt.plot(fitted_positions,y,'k',label="fitted centres")
        plt.legend(framealpha=1)
        plt.title('Trace detection, star %d'%(star+1))
        plt.xlim(0,np.shape(frame)[1])
        plt.ylim(0,np.shape(frame)[0])
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.tight_layout()

        if delay == -2:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(delay)
            plt.close()

        if delay >= 5:
            plt.figure(figsize=(8,6))
            plt.plot(y,fitted_positions-old_fitted_positions)
            plt.xlabel('Y pixel')
            plt.ylabel('New poly - old poly')
            plt.title('Difference between trace polynomials before and after sigma clipping, star %d'%(star+1))
            plt.show(block=False)
            plt.pause(delay)
            plt.close()

        plt.figure(figsize=(5,8))
        plt.plot(trace_centre,row_array,'bo',ms=4,label='outlier (ignored)')
        plt.plot(np.array(trace_centre)[clipped_trace_idx],row_array[clipped_trace_idx],'ro',ms=4)
        plt.plot(fitted_positions,y,'k')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.ylim(0,nrows)
        plt.title('Trace fitting, star %d'%(star+1))
        plt.legend(numpoints=1)
        # ~ plt.show()
        if delay == -2:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(delay)
            plt.close()

        plt.figure(figsize=(8,4))
        plt.plot(row_array,np.array(trace_centre)-fitted_function(row_array),'bo',ms=4,label='outlier (ignored)')
        plt.plot(row_array[clipped_trace_idx],np.array(trace_centre)[clipped_trace_idx]-fitted_function(row_array)[clipped_trace_idx],'ro',ms=4)#,markerfacecolor='None')
        plt.title('Residuals of trace fitting, star %d'%(star+1))
        plt.ylabel('Residuals')
        plt.xlabel('X pixel')
        plt.xlim(0,nrows)
        plt.legend(numpoints=1)
        if delay == -2:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(delay)
            plt.close()

        if len(fwhm) > 0:
            plt.figure(figsize=(8,4))
            plt.plot(row_array[clipped_trace_idx],np.array(gauss_std)[clipped_trace_idx],label='Standard deviation of Gaussian')
            plt.plot(row_array[clipped_trace_idx],np.array(fwhm)[clipped_trace_idx],label="FWHM of trace")
            plt.title('Trace width, star %d'%(star+1))
            plt.ylabel('Width in pixels')
            plt.xlabel('X pixel')
            plt.xlim(0,nrows)
            plt.legend(numpoints=1)
            if delay == -2:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(delay)
                plt.close()

    if override_force_verbose and not verbose:
        delay = 0

    fwhm = np.array(fwhm)

    return fitted_positions, delay, np.median(fwhm[np.isfinite(fwhm)]), np.array(gauss_std)


def extract_trace_flux(frame,trace,aperture_width,background_offset,background_width,pre_flat_frame,poly_bg_order,am,exposure_time,verbose,star,mask,instrument,row_min,gauss_std,readout_speed,co_add_rows,rectify_frame,oversampling_factor,gain_file,readnoise_file,frame_no=0,frame_label=None,profile_config=None,save_diagnostics=False,forced_aperture_widths=None,return_aperture_widths=False,max_aperture_per_exposure=False,excluded_column_ranges=None,row_excluded_column_ranges=None,excluded_column_mode="asymmetric",fixed_width_capture_config=None):
    """Extract flux for a single trace using fixed, Gaussian-width, or profile-percentile apertures."""

    if verbose and verbose == -1:
        verbose = False

    profile_config = profile_config or {}
    profile_enabled = bool(profile_config.get("enabled", False))
    profile_model = str(profile_config.get("model", "moffat")).lower()
    profile_percentile = float(profile_config.get("percentile", 0.95))
    if profile_percentile > 1:
        profile_percentile /= 100.0
    profile_percentile = np.clip(profile_percentile, 1e-3, 0.9999)
    profile_fit_half_width = int(np.ceil(profile_config.get("fit_half_width", max(aperture_width, 10))))
    profile_core_exclusion_half_width = float(profile_config.get("core_exclusion_half_width", 0.0))
    profile_fit_oversampling = max(int(np.round(profile_config.get("fit_oversampling", 1))), 1)
    profile_aggregation_mode = normalize_profile_aggregation_mode(
        profile_config.get("aggregation_mode", "row_by_row")
    )
    profile_max_width = profile_config.get("max_width", None)
    if profile_max_width is not None:
        profile_max_width = float(profile_max_width)
        if profile_max_width <= 0:
            profile_max_width = None
    legacy_gaussian_aperture = gauss_std is not None and not profile_enabled
    fixed_width_capture_config = fixed_width_capture_config or {}
    fixed_width_capture_requested = bool(fixed_width_capture_config.get("enabled", False))
    fixed_width_capture_enabled = bool(fixed_width_capture_requested or save_diagnostics)
    fixed_width_capture_half_width = int(
        np.ceil(
            fixed_width_capture_config.get(
                "fit_half_width",
                max(aperture_width, 12),
            )
        )
    )
    excluded_column_mode = normalize_excluded_column_mode(excluded_column_mode)

    if profile_enabled and profile_model not in SUPPORTED_PROFILE_MODELS:
        raise ValueError(
            "Unsupported profile model '%s'. Use moffat or empirical in spectral_extraction_wingfocus.py."
            % profile_model
        )
    summed_profile_mode = bool(profile_enabled and is_summed_profile_aggregation_mode(profile_aggregation_mode))

    if instrument == 'ACAM':
        D = 420.
        h = 2420.
        if readout_speed.lower() == 'fast':
            gain = 1.86
            readnoise = 6.5
        elif readout_speed.lower() == 'slow':
            gain = 0.92
            readnoise = 3.7
        else:
            raise NameError("readout_speed for ACAM must be defined as either 'fast' or 'slow' in extraction_input")
        buffer_pixels = 20 * oversampling_factor
        dark_current = 4.

    elif instrument == 'EFOSC':
        D = 358.
        h = 2377.
        gain = 1.38
        readnoise = 12.6
        buffer_pixels = 65 * oversampling_factor
        dark_current = 7.

    elif instrument == 'Keck/NIRSPEC':
        D = 1000.
        h = 4000
        gain = 3.01
        readnoise = 11.56
        buffer_pixels = 0
        dark_current = 2520

    elif "JWST" in instrument:
        if gain_file is not None:
            gain = gain_file
        else:
            gain = 1

        if readnoise_file is not None:
            readnoise = readnoise_file
        else:
            readnoise = 0

        buffer_pixels = 0
        dark_current = 0
        error_frame = frame[1]
        frame = frame[0]

    else:
        raise NameError('Currently only set up for ACAM, EFOSC, Keck/NIRSPEC and JWST')

    if rectify_frame:
        frame = rectify_spatial(frame,trace)
        trace = np.ones_like(trace) * np.nanmedian(trace)
        if not profile_enabled:
            trace = trace.astype(int)
        if verbose:
            plt.figure()
            if instrument == "Keck/NIRSPEC":
                vmin,vmax = 0,500
            else:
                vmin,vmax = np.nanpercentile(frame,[10,90])
            plt.imshow(frame,vmin=vmin,vmax=vmax,aspect="auto")
            plt.axvline(np.nanmedian(trace),color='k',label="New trace")
            plt.xlabel("X pixel")
            plt.ylabel("X pixel")
            plt.legend()
            plt.title("Post-spatial rectification")
            if verbose == -2:
                plt.show()
            if verbose > 0:
                plt.show(block=False)
                plt.pause(verbose)
                plt.close()
    else:
        if profile_enabled:
            trace = np.asarray(trace, dtype=float)
        else:
            trace = np.round(trace).astype(int)

    W3 = 2.0
    h0 = 8000.
    if instrument == "ACAM" or instrument == "EFOSC":
        scintillation = 0.09 * D ** (-2.0 / 3.0) * (am ** W3) * np.exp(-h / h0) * (2.0 * exposure_time) ** (-0.5)
    else:
        scintillation = 0

    if "JWST" in instrument:
        frame = frame * exposure_time * gain
        error_frame = error_frame * exposure_time * gain
        pre_flat_frame = pre_flat_frame * exposure_time * gain
    else:
        frame = frame * gain
        pre_flat_frame = pre_flat_frame * gain

    nrows, ncols = np.shape(frame)
    excluded_column_ranges = merge_excluded_column_ranges(excluded_column_ranges)
    row_excluded_column_ranges = normalize_row_excluded_column_ranges(row_excluded_column_ranges, nrows)
    effective_row_excluded_column_ranges = [
        get_row_excluded_column_ranges(excluded_column_ranges, row_excluded_column_ranges, i)
        for i in range(nrows)
    ]
    x = np.arange(ncols)
    plot_frames = sorted(set([max(0, min(nrows - 1, 50)), nrows // 2, max(0, min(nrows - 1, nrows - 50))]))

    if forced_aperture_widths is not None:
        forced_aperture_widths = np.asarray(forced_aperture_widths, dtype=float)
        if forced_aperture_widths.shape[0] != nrows:
            raise ValueError("forced_aperture_widths must have one value per extracted row.")

    flux = []
    error = []
    sky_left = []
    sky_right = []
    sky_avg = []
    sky_poly = []
    raw_star_flux = []
    clipped_frame = []
    bkg_poly_orders_used = []
    flux_base_level_left = []
    flux_base_level_right = []
    max_counts = []
    error_from_readnoise = []
    error_from_scintillation = []
    error_from_source = []
    background_coeffs = []

    if legacy_gaussian_aperture:
        aperture_multiplication_factor = aperture_width
        aperture_width_array = []
        for i in range(nrows):
            if i <= 10:
                aperture_width_array.append(np.nanmedian(gauss_std[0:i+10]) * aperture_multiplication_factor)
            elif i >= nrows - 10:
                aperture_width_array.append(np.nanmedian(gauss_std[i-10:nrows]) * aperture_multiplication_factor)
            else:
                aperture_width_array.append(np.nanmedian(gauss_std[i-5:i+5]) * aperture_multiplication_factor)
        aperture_width_array = np.round(np.array(aperture_width_array)).astype(int)

    precomputed_profile_fits = None
    summed_profile_fit = None
    if max_aperture_per_exposure:
        if forced_aperture_widths is not None:
            finite_widths = forced_aperture_widths[np.isfinite(forced_aperture_widths)]
            shared_width = float(np.nanmax(finite_widths)) if len(finite_widths) > 0 else float(aperture_width)
            forced_aperture_widths = np.full(nrows, max(shared_width, 2.0), dtype=float)
        elif legacy_gaussian_aperture:
            finite_widths = aperture_width_array[np.isfinite(aperture_width_array)]
            shared_width = int(np.round(np.nanmax(finite_widths))) if len(finite_widths) > 0 else max(int(aperture_width), 2)
            aperture_width_array = np.full(nrows, max(shared_width, 2), dtype=int)
        elif profile_enabled and not summed_profile_mode:
            precomputed_profile_fits = [None] * nrows
            candidate_widths = np.full(nrows, max(int(aperture_width), 2), dtype=float)
            previous_precomputed_fit = None
            for i, row in enumerate(frame):
                row_excluded_ranges = effective_row_excluded_column_ranges[i]
                cached_fit = fit_aperture_profile(
                    row,
                    float(trace[i]),
                    profile_fit_half_width,
                    profile_percentile,
                    profile_model,
                    max(int(aperture_width), 2),
                    ncols,
                    background_offset,
                    background_width,
                    buffer_pixels_left if 'buffer_pixels_left' in locals() else buffer_pixels,
                    buffer_pixels_right if 'buffer_pixels_right' in locals() else buffer_pixels,
                    previous_fit=previous_precomputed_fit,
                    core_exclusion_half_width=profile_core_exclusion_half_width,
                    max_width=profile_max_width,
                    fit_oversampling=profile_fit_oversampling,
                    excluded_column_ranges=row_excluded_ranges,
                    excluded_column_mode=excluded_column_mode,
                )
                precomputed_profile_fits[i] = cached_fit
                if cached_fit["success"]:
                    previous_precomputed_fit = cached_fit.copy()
                    candidate_widths[i] = max(cached_fit["right_edge"] - cached_fit["left_edge"], 2)

            finite_widths = candidate_widths[np.isfinite(candidate_widths)]
            shared_width = float(np.nanmax(finite_widths)) if len(finite_widths) > 0 else float(aperture_width)
            forced_aperture_widths = np.full(nrows, max(shared_width, 2.0), dtype=float)

    if summed_profile_mode:
        summed_profile_fit = fit_summed_aperture_profile(
            frame,
            trace,
            profile_fit_half_width,
            profile_percentile,
            profile_model,
            max(int(aperture_width), 2),
            ncols,
            background_offset,
            background_width,
            buffer_left=buffer_pixels,
            buffer_right=buffer_pixels,
            previous_fit=None,
            core_exclusion_half_width=profile_core_exclusion_half_width,
            max_width=profile_max_width,
            fit_oversampling=profile_fit_oversampling,
            excluded_column_ranges=excluded_column_ranges,
            row_excluded_column_ranges=row_excluded_column_ranges,
            excluded_column_mode=excluded_column_mode,
            aggregation_mode=profile_aggregation_mode,
        )

    ap_left_arr = np.full(nrows, -1, dtype=int)
    ap_right_arr = np.full(nrows, -1, dtype=int)
    bkg_left_start_arr = np.full(nrows, -1, dtype=int)
    bkg_left_end_arr = np.full(nrows, -1, dtype=int)
    bkg_right_start_arr = np.full(nrows, -1, dtype=int)
    bkg_right_end_arr = np.full(nrows, -1, dtype=int)
    profile_fit_left_arr = np.full(nrows, -1, dtype=int)
    profile_fit_right_arr = np.full(nrows, -1, dtype=int)

    diag_enabled = bool(save_diagnostics)
    if diag_enabled:
        region_flags = np.zeros((nrows, ncols), dtype=np.uint8)
        background_model_frame = np.full((nrows, ncols), np.nan, dtype=np.float32)
        profile_success = np.zeros(nrows, dtype=bool)
        profile_amplitude = np.full(nrows, np.nan, dtype=np.float32)
        profile_centre = np.full(nrows, np.nan, dtype=np.float32)
        profile_scale = np.full(nrows, np.nan, dtype=np.float32)
        profile_scale_secondary = np.full(nrows, np.nan, dtype=np.float32)
        profile_beta = np.full(nrows, np.nan, dtype=np.float32)
        profile_mix_fraction = np.full(nrows, np.nan, dtype=np.float32)
        profile_offset = np.full(nrows, np.nan, dtype=np.float32)
        profile_radius = np.full(nrows, np.nan, dtype=np.float32)
        profile_fwhm = np.full(nrows, np.nan, dtype=np.float32)
        fixed_width_capture_fraction = np.full(nrows, np.nan, dtype=np.float32)
        fixed_width_capture_flux = np.full(nrows, np.nan, dtype=np.float32)
        fixed_width_total_flux = np.full(nrows, np.nan, dtype=np.float32)
        fixed_width_capture_left = np.full(nrows, -1, dtype=int)
        fixed_width_capture_right = np.full(nrows, -1, dtype=int)
    elif fixed_width_capture_enabled:
        fixed_width_capture_fraction = np.full(nrows, np.nan, dtype=np.float32)
        fixed_width_capture_flux = np.full(nrows, np.nan, dtype=np.float32)
        fixed_width_total_flux = np.full(nrows, np.nan, dtype=np.float32)
        fixed_width_capture_left = np.full(nrows, -1, dtype=int)
        fixed_width_capture_right = np.full(nrows, -1, dtype=int)

    lh_overlap = []
    rh_overlap = []
    log = open('reduction_output.log','a')
    previous_profile_fit = None

    for i, row in enumerate(frame):
        raw_row = np.asarray(row, dtype=float)
        row_excluded_ranges = effective_row_excluded_column_ranges[i]
        row_working, excluded_row_mask = prepare_row_for_excluded_columns(
            raw_row,
            row_excluded_ranges,
            excluded_column_mode,
        )
        pre_flat_row = np.asarray(pre_flat_frame[i], dtype=float)
        pre_flat_row_working, _ = prepare_row_for_excluded_columns(
            pre_flat_row,
            row_excluded_ranges,
            excluded_column_mode,
        )

        masked_regions = np.array([], dtype=int)
        if mask is not None:
            masked_regions = np.array(mask + int(np.round(trace[i])), dtype=int)
            masked_regions = masked_regions[(masked_regions >= 0) & (masked_regions < ncols)]

        if forced_aperture_widths is not None:
            current_aperture_width = max(int(np.round(forced_aperture_widths[i])), 2)
        elif legacy_gaussian_aperture:
            current_aperture_width = max(int(aperture_width_array[i]), 2)
            aperture_log = open('aperture_log.log','a')
            aperture_log.write("Trace %d, row %d, Gauss std = %f, aperture width = %f \n"%(star+1,i,gauss_std[i],current_aperture_width))
            aperture_log.close()
        else:
            current_aperture_width = max(int(aperture_width), 2)
        profile_fit = None

        if instrument == "Keck/NIRSPEC":
            if row[0] > 1000:
                buffer_pixels_left = 8 * oversampling_factor
                print("using 8 buffer pixels to the left, row %d=%f"%(i,row[0]))
            else:
                buffer_pixels_left = 0

            if row[-1] > 1000:
                buffer_pixels_right = 8 * oversampling_factor
                print("using 8 buffer pixels to the right, row %d=%f"%(i,row[-1]))
            else:
                buffer_pixels_right = 0
        else:
            buffer_pixels_left = buffer_pixels_right = buffer_pixels

        if profile_enabled:
            if summed_profile_mode:
                profile_fit = summed_profile_fit
            elif precomputed_profile_fits is not None:
                profile_fit = precomputed_profile_fits[i]
                if profile_fit["success"]:
                    previous_profile_fit = profile_fit.copy()
            else:
                profile_fit = fit_aperture_profile(
                    row_working,
                    float(trace[i]),
                    profile_fit_half_width,
                    profile_percentile,
                    profile_model,
                    current_aperture_width,
                    ncols,
                    background_offset,
                    background_width,
                    buffer_pixels_left,
                    buffer_pixels_right,
                    previous_fit=previous_profile_fit,
                    core_exclusion_half_width=profile_core_exclusion_half_width,
                    max_width=profile_max_width,
                    fit_oversampling=profile_fit_oversampling,
                    excluded_column_ranges=row_excluded_ranges,
                    excluded_column_mode=excluded_column_mode,
                )
                if profile_fit["success"]:
                    previous_profile_fit = profile_fit.copy()

        if profile_enabled:
            if summed_profile_mode:
                fit_window_left = int(profile_fit["row_fit_left"][i]) if profile_fit is not None else -1
                fit_window_right = int(profile_fit["row_fit_right"][i]) if profile_fit is not None else -1
                if forced_aperture_widths is not None:
                    aperture_left_hand_edge = int(np.floor(float(trace[i]) - current_aperture_width / 2.0))
                    aperture_right_hand_edge = int(np.ceil(float(trace[i]) + current_aperture_width / 2.0))
                elif profile_fit is not None and profile_fit["success"] and np.isfinite(profile_fit["radius"]):
                    aperture_left_hand_edge, aperture_right_hand_edge = compute_symmetric_aperture_edges(
                        float(trace[i]),
                        float(profile_fit["radius"]),
                        ncols,
                        buffer_left=buffer_pixels_left,
                        buffer_right=buffer_pixels_right,
                    )
                else:
                    aperture_left_hand_edge = int(np.floor(float(trace[i]) - current_aperture_width / 2.0))
                    aperture_right_hand_edge = int(np.ceil(float(trace[i]) + current_aperture_width / 2.0))
            else:
                aperture_left_hand_edge = profile_fit["left_edge"]
                aperture_right_hand_edge = profile_fit["right_edge"]
                fit_window_left = profile_fit["fit_window_left"]
                fit_window_right = profile_fit["fit_window_right"]
                if forced_aperture_widths is not None:
                    aperture_left_hand_edge = int(np.floor(float(trace[i]) - current_aperture_width / 2.0))
                    aperture_right_hand_edge = int(np.ceil(float(trace[i]) + current_aperture_width / 2.0))
        else:
            aperture_left_hand_edge = int(np.floor(trace[i] - current_aperture_width / 2.0))
            aperture_right_hand_edge = int(np.ceil(trace[i] + current_aperture_width / 2.0))
            fit_window_left = fit_window_right = -1

        aperture_left_hand_edge = max(aperture_left_hand_edge, buffer_pixels_left)
        aperture_right_hand_edge = min(aperture_right_hand_edge, ncols - buffer_pixels_right)
        aperture_left_hand_edge, aperture_right_hand_edge = apply_excluded_column_aperture_mode(
            aperture_left_hand_edge,
            aperture_right_hand_edge,
            trace[i],
            row_excluded_ranges,
            ncols,
            buffer_left=buffer_pixels_left,
            buffer_right=buffer_pixels_right,
            mode=excluded_column_mode,
        )

        if aperture_right_hand_edge <= aperture_left_hand_edge:
            midpoint = int(np.clip(np.round(trace[i]), buffer_pixels_left, ncols - buffer_pixels_right - 1))
            aperture_left_hand_edge = max(buffer_pixels_left, midpoint - max(1, current_aperture_width // 2))
            aperture_right_hand_edge = min(ncols - buffer_pixels_right, aperture_left_hand_edge + max(2, current_aperture_width))
            aperture_left_hand_edge = max(buffer_pixels_left, aperture_right_hand_edge - max(2, current_aperture_width))
            aperture_left_hand_edge, aperture_right_hand_edge = apply_excluded_column_aperture_mode(
                aperture_left_hand_edge,
                aperture_right_hand_edge,
                trace[i],
                row_excluded_ranges,
                ncols,
                buffer_left=buffer_pixels_left,
                buffer_right=buffer_pixels_right,
                mode=excluded_column_mode,
            )

        aperture_cols = np.arange(aperture_left_hand_edge, aperture_right_hand_edge, dtype=int)
        excluded_aperture_mask = mask_values_in_column_strips(aperture_cols, row_excluded_ranges)
        usable_aperture_cols = aperture_cols[
            ~get_excluded_column_data_mask(
                aperture_cols,
                row_excluded_ranges,
                excluded_column_mode,
            )
        ]
        if len(usable_aperture_cols) == 0:
            usable_aperture_cols = aperture_cols
        aperture_npix = max(len(usable_aperture_cols), 1)

        if background_width == 1:
            left_bkg_left_hand_edge = buffer_pixels_left
            right_bkg_right_hand_edge = ncols - buffer_pixels_right
        else:
            left_bkg_left_hand_edge = aperture_left_hand_edge - background_offset - background_width
            right_bkg_right_hand_edge = aperture_right_hand_edge + background_offset + background_width

            if left_bkg_left_hand_edge <= buffer_pixels_left:
                lh_overlap.append(buffer_pixels_left - left_bkg_left_hand_edge)
                left_bkg_left_hand_edge = buffer_pixels_left

            if right_bkg_right_hand_edge >= ncols - buffer_pixels_right:
                rh_overlap.append(right_bkg_right_hand_edge - (ncols - buffer_pixels_right))
                right_bkg_right_hand_edge = ncols - buffer_pixels_right

        left_bkg_right_hand_edge = aperture_left_hand_edge - background_offset
        right_bkg_left_hand_edge = aperture_right_hand_edge + background_offset

        if background_width == 1:
            if (left_bkg_right_hand_edge - left_bkg_left_hand_edge) < 10:
                lh_overlap.append(left_bkg_right_hand_edge - left_bkg_left_hand_edge)
            if (right_bkg_right_hand_edge - right_bkg_left_hand_edge) < 10:
                rh_overlap.append(right_bkg_right_hand_edge - right_bkg_left_hand_edge)

        left_bkg_left_hand_edge = max(left_bkg_left_hand_edge, buffer_pixels_left)
        left_bkg_right_hand_edge = max(left_bkg_left_hand_edge, min(left_bkg_right_hand_edge, ncols))
        right_bkg_left_hand_edge = min(max(right_bkg_left_hand_edge, 0), ncols)
        right_bkg_right_hand_edge = min(max(right_bkg_right_hand_edge, right_bkg_left_hand_edge), ncols)

        region_left = left_bkg_left_hand_edge
        region_right = max(right_bkg_right_hand_edge, aperture_right_hand_edge)

        bkg_cols = []
        if left_bkg_right_hand_edge > left_bkg_left_hand_edge:
            bkg_cols.extend(range(left_bkg_left_hand_edge, left_bkg_right_hand_edge))
        if right_bkg_right_hand_edge > right_bkg_left_hand_edge:
            bkg_cols.extend(range(right_bkg_left_hand_edge, right_bkg_right_hand_edge))
        bkg_cols = np.array(bkg_cols, dtype=int)

        if len(bkg_cols) > 0:
            bkg_cols = bkg_cols[
                ~get_excluded_column_data_mask(
                    bkg_cols,
                    row_excluded_ranges,
                    excluded_column_mode,
                )
            ]

        if mask is not None and len(bkg_cols) > 0:
            bkg_cols = np.array(sorted(set(bkg_cols).difference(masked_regions.tolist())), dtype=int)
            masked_regions = masked_regions[(masked_regions < region_right) & (masked_regions >= region_left)]

        if co_add_rows > 0 and len(bkg_cols) > 0:
            def get_background_row_samples(row_index):
                neighbor_row, _ = prepare_row_for_excluded_columns(
                    frame[int(row_index)],
                    effective_row_excluded_column_ranges[int(row_index)],
                    excluded_column_mode,
                )
                return neighbor_row[bkg_cols]

            if i < co_add_rows / 2:
                y = np.nanmedian(np.array([get_background_row_samples(r) for r in range(i, i + co_add_rows)]), axis=0)
            elif i > nrows - co_add_rows / 2:
                y = np.nanmedian(np.array([get_background_row_samples(r) for r in range(i - co_add_rows, i)]), axis=0)
            else:
                y = np.nanmedian(
                    np.array(
                        [
                            get_background_row_samples(r)
                            for r in range(i - int(co_add_rows / 2), i + int(co_add_rows / 2))
                        ]
                    ),
                    axis=0,
                )
        elif len(bkg_cols) > 0:
            y = row_working[bkg_cols]
        else:
            y = np.array([], dtype=float)

        if len(y) > 0:
            y_med = np.nanmedian(y)
            y_std = np.nanstd(y)
            keep_idx = (y <= y_med + 3 * y_std) & (y >= y_med - 3 * y_std)
            if instrument == "Keck/NIRSPEC":
                keep_idx = keep_idx & np.isfinite(y) & (abs(y) >= 1e-2)
        else:
            keep_idx = np.array([], dtype=bool)

        y_keep = y[keep_idx]
        bkg_cols_keep = bkg_cols[keep_idx] if len(bkg_cols) > 0 else np.array([], dtype=int)
        reject_idx = ~keep_idx if len(keep_idx) > 0 else np.array([], dtype=bool)
        y_reject = y[reject_idx] if len(y) > 0 else np.array([], dtype=float)
        bkg_cols_reject = bkg_cols[reject_idx] if len(bkg_cols) > 0 else np.array([], dtype=int)

        bg_x = np.arange(region_left, region_right)
        chosen_order = 0
        poly = np.poly1d([0.0])
        background_fit = np.zeros_like(bg_x, dtype=float)

        if poly_bg_order == 0:
            poly = np.poly1d([0.0])
            chosen_order = 0

        elif poly_bg_order == -1:
            if len(y_keep) > 0:
                poly = np.poly1d([np.median(y_keep)])
            else:
                poly = np.poly1d([0.0])
            chosen_order = 0
            background_fit = poly(bg_x)

        elif poly_bg_order == -2:
            best_bic = None
            best_poly = None
            max_order = min(4, max(len(y_keep) - 1, 0))
            if max_order >= 1:
                for order in range(1, max_order + 1):
                    candidate_poly = np.poly1d(np.polyfit(bkg_cols_keep, y_keep, order))
                    model = candidate_poly(bkg_cols_keep)
                    err = np.sqrt(np.clip(np.abs(y_keep), 1.0, None))
                    candidate_bic = BIC(model, y_keep, err, order * max(len(y_keep), 1))
                    if best_bic is None or candidate_bic < best_bic:
                        best_bic = candidate_bic
                        best_poly = candidate_poly
                        chosen_order = order
            if best_poly is not None:
                poly = best_poly
            background_fit = poly(bg_x)

        else:
            if len(y_keep) > poly_bg_order:
                poly = np.poly1d(np.polyfit(bkg_cols_keep, y_keep, poly_bg_order))
                chosen_order = poly_bg_order
            else:
                poly = np.poly1d([0.0])
                chosen_order = 0
            background_fit = poly(bg_x)

        if poly_bg_order in [0, -1] and len(bg_x) > 0:
            background_fit = poly(bg_x)

        bkg_poly_orders_used.append(chosen_order)
        background_coeffs.append(np.asarray(poly.coefficients, dtype=float))

        if "JWST" not in instrument:
            if len(y_keep) > 0 and np.any(bkg_cols_keep < left_bkg_right_hand_edge):
                sky_left.append(np.mean(y_keep[bkg_cols_keep < left_bkg_right_hand_edge]) / oversampling_factor)
            else:
                sky_left.append(np.nan)

            if len(y_keep) > 0 and np.any(bkg_cols_keep >= right_bkg_left_hand_edge):
                sky_right.append(np.mean(y_keep[bkg_cols_keep >= right_bkg_left_hand_edge]) / oversampling_factor)
            else:
                sky_right.append(np.nan)

        sky_avg.append(np.mean(y_keep) / oversampling_factor if len(y_keep) > 0 else np.nan)
        sky_poly.append(background_fit)

        if verbose and ((i + 1) == 1 or (i + 1) % 100 == 0 or i in plot_frames):
            print(
                "Frame %d, star %d: processed row %d/%d"
                % (frame_no + 1, star + 1, i + 1, nrows)
            )

        if verbose and i in plot_frames:
            fig = plt.figure(figsize=(9,6))
            ax = fig.add_subplot(111)
            local_pad = max(2, int(np.ceil(0.05 * max(region_right - region_left, 1))))
            ax.plot(
                x,
                raw_row,
                linestyle='None',
                marker='o',
                ms=3,
                color='tab:blue',
                label='raw data',
                zorder=4,
            )

            if profile_enabled and not summed_profile_mode and profile_fit is not None and profile_fit.get("interp_x") is not None:
                ax.plot(
                    profile_fit["interp_x"],
                    profile_fit["interp_y"],
                    linestyle='None',
                    marker='.',
                    ms=2.0,
                    color='tab:orange',
                    alpha=0.70,
                    label='interpolated data (%dx)' % profile_fit_oversampling,
                    zorder=2,
                )

            if len(bg_x) > 0:
                ax.plot(bg_x, background_fit, 'g', label="background fit")

            if len(bkg_cols_reject) > 0:
                ax.plot(bkg_cols_reject, y_reject, 'kx', label='background reject')

            if len(masked_regions) > 0:
                ax.plot(masked_regions, raw_row[masked_regions], 'rx', label='contaminant mask')

            if row_excluded_ranges is not None and len(row_excluded_ranges) > 0:
                label_used = False
                x_min_plot = max(0, region_left - local_pad)
                x_max_plot = min(ncols - 1, region_right + local_pad)
                for strip_left, strip_right in row_excluded_ranges:
                    if strip_right <= x_min_plot or strip_left >= x_max_plot:
                        continue
                    ax.axvspan(
                        max(strip_left, x_min_plot),
                        min(strip_right, x_max_plot),
                        color='tab:red',
                        alpha=0.06,
                        label='excluded columns' if not label_used else None,
                    )
                    label_used = True

            if profile_enabled and not summed_profile_mode and profile_fit is not None and profile_fit["success"] and profile_model == "moffat":
                model_x = profile_fit.get("interp_x")
                if model_x is None:
                    model_x = x
                model_y = evaluate_profile_with_background(profile_model, model_x, profile_fit)
                ax.plot(model_x, model_y, color='tab:purple', lw=1.6, label='%s fit' % profile_model, zorder=3)
                ax.axvspan(fit_window_left, fit_window_right, color='tab:orange', alpha=0.08, label='profile fit window')
                if profile_core_exclusion_half_width > 0:
                    ax.axvspan(
                        trace[i] - profile_core_exclusion_half_width,
                        trace[i] + profile_core_exclusion_half_width,
                        color='0.6',
                        alpha=0.10,
                        label='core excluded',
                    )
            elif profile_enabled and fit_window_left >= 0 and fit_window_right > fit_window_left:
                ax.axvspan(fit_window_left, fit_window_right, color='tab:orange', alpha=0.08, label='profile fit window')

            ax.axvline(trace[i], color='k', label="trace centre")
            ax.axvline(aperture_left_hand_edge, color='r', label="extraction aperture")
            ax.axvline(aperture_right_hand_edge, color='r')
            ax.axvline(left_bkg_left_hand_edge, color='r', ls='--', label="background regions")
            ax.axvline(left_bkg_right_hand_edge, color='r', ls='--')
            ax.axvline(right_bkg_left_hand_edge, color='r', ls='--')
            ax.axvline(right_bkg_right_hand_edge, color='r', ls='--')
            ax.set_title('Aperture locations before background subtraction, star %d'%(star+1))
            ax.set_xlabel('X pixel')
            ax.set_ylabel('Counts at row %d'%(i+1))
            ax.set_xlim(max(0, region_left - local_pad), min(ncols - 1, region_right + local_pad))
            x_min_plot, x_max_plot = ax.get_xlim()
            plot_mask = (x >= x_min_plot) & (x <= x_max_plot) & np.isfinite(row)
            if np.any(plot_mask):
                y_visible = row[plot_mask]
                y_span = max(float(np.nanmax(y_visible) - np.nanmin(y_visible)), 1.0)
                y_pad = max(1.0, 0.08 * y_span)
                ax.set_ylim(float(np.nanmin(y_visible) - y_pad), float(np.nanmax(y_visible) + y_pad))
            ax.legend(loc='upper left', numpoints=1, framealpha=1)

            if 'SAVE_FIG' in globals() and SAVE_FIG:
                plt.savefig(f"spectral_extraction_images/aperture_loc_before_subtract_with_back_fit_row_{i+1}_star_{star+1}_frame_{frame_no}.pdf",dpi=50)

            if verbose == -2:
                plt.show()
            if verbose > 0:
                plt.show(block=False)
                plt.pause(verbose)
                plt.close()

        raw_flux = np.sum(pre_flat_row_working[usable_aperture_cols]) / oversampling_factor

        if "JWST" not in instrument:
            ap_slice = row_working[usable_aperture_cols]
            if len(ap_slice) > 0:
                if len(np.shape(gain)) > 1:
                    max_counts.append(np.nanmax(ap_slice / gain[i][usable_aperture_cols] / oversampling_factor))
                else:
                    max_counts.append(np.nanmax(ap_slice / gain / oversampling_factor))
            else:
                max_counts.append(np.nan)

            if raw_flux > 0:
                if len(np.shape(readnoise)) > 1:
                    error_from_readnoise.append(np.sum(readnoise[i][usable_aperture_cols] / oversampling_factor / raw_flux))
                else:
                    error_from_readnoise.append(aperture_npix * readnoise / raw_flux)
                error_from_source.append(np.sqrt(raw_flux) / raw_flux)
            else:
                error_from_readnoise.append(np.nan)
                error_from_source.append(np.nan)
            error_from_scintillation.append(scintillation)

        background_subtracted = row_working[region_left:region_right] - background_fit
        clipped_frame.append(raw_row[region_left:region_right])

        usable_aperture_cols_region = usable_aperture_cols[(usable_aperture_cols >= region_left) & (usable_aperture_cols < region_right)] - region_left
        flux.append(np.sum(background_subtracted[usable_aperture_cols_region]) / oversampling_factor)

        if fixed_width_capture_enabled:
            capture_left = max(buffer_pixels_left, int(np.floor(float(trace[i]) - fixed_width_capture_half_width)))
            capture_right = min(ncols - buffer_pixels_right, int(np.ceil(float(trace[i]) + fixed_width_capture_half_width)))
            if capture_right <= capture_left:
                capture_right = min(ncols - buffer_pixels_right, capture_left + 1)

            capture_cols = np.arange(capture_left, capture_right, dtype=int)
            if len(capture_cols) > 0:
                capture_cols = capture_cols[
                    ~get_excluded_column_data_mask(
                        capture_cols,
                        row_excluded_ranges,
                        excluded_column_mode,
                    )
                ]
            if len(capture_cols) > 0 and len(masked_regions) > 0:
                capture_cols = capture_cols[~np.isin(capture_cols, masked_regions)]

            if len(capture_cols) > 0:
                capture_source = row_working[capture_cols] - poly(capture_cols)
                capture_fraction, capture_flux, total_capture_flux = compute_empirical_capture_fraction(
                    capture_cols,
                    capture_source,
                    aperture_left_hand_edge,
                    aperture_right_hand_edge,
                )
                fixed_width_capture_fraction[i] = capture_fraction
                fixed_width_capture_flux[i] = capture_flux
                fixed_width_total_flux[i] = total_capture_flux
                fixed_width_capture_left[i] = capture_left
                fixed_width_capture_right[i] = capture_right

        if "JWST" not in instrument:
            left_slice = background_subtracted[:max(0, left_bkg_right_hand_edge - region_left)]
            right_slice = background_subtracted[max(0, right_bkg_left_hand_edge - region_left):]
            flux_base_level_left.append(np.median(left_slice) / oversampling_factor if len(left_slice) > 0 else np.nan)
            flux_base_level_right.append(np.median(right_slice) / oversampling_factor if len(right_slice) > 0 else np.nan)

        aperture_sum = np.sum(row[usable_aperture_cols])
        if aperture_sum <= 0:
            error.append(np.nan)
        else:
            if instrument == "Keck/NIRSPEC":
                error.append(np.sqrt(aperture_sum / oversampling_factor + (aperture_npix / oversampling_factor) * readnoise**2 + dark_current * (aperture_npix / oversampling_factor) * exposure_time / 3600.))
            elif "JWST" in instrument:
                error.append(np.sqrt(np.sum((error_frame[i][usable_aperture_cols] / oversampling_factor) ** 2)))
            else:
                error.append(np.sqrt(aperture_sum / oversampling_factor + (aperture_npix / oversampling_factor) * readnoise**2 + scintillation**2))

        if "JWST" not in instrument:
            raw_star_flux.append(aperture_sum / oversampling_factor)

        if verbose and i in plot_frames:
            plt.figure(figsize=(9,6))
            local_x = np.arange(region_left, region_right)
            local_pad = max(2, int(np.ceil(0.05 * max(region_right - region_left, 1))))
            plt.plot(local_x, background_subtracted, linestyle='None', marker='o', ms=3, color='tab:blue', label='background subtracted')
            if len(bkg_cols_reject) > 0:
                plt.plot(bkg_cols_reject, y_reject - poly(bkg_cols_reject), 'kx', label='background reject')
            if len(masked_regions) > 0:
                masked_in_region = masked_regions[(masked_regions >= region_left) & (masked_regions < region_right)]
                if len(masked_in_region) > 0:
                    plt.plot(masked_in_region, background_subtracted[masked_in_region - region_left], 'rx', label='contaminant mask')

            if row_excluded_ranges is not None and len(row_excluded_ranges) > 0:
                label_used = False
                x_min_plot = max(0, region_left - local_pad)
                x_max_plot = min(ncols - 1, region_right + local_pad)
                for strip_left, strip_right in row_excluded_ranges:
                    if strip_right <= x_min_plot or strip_left >= x_max_plot:
                        continue
                    plt.axvspan(
                        max(strip_left, x_min_plot),
                        min(strip_right, x_max_plot),
                        color='tab:red',
                        alpha=0.06,
                        label='excluded columns' if not label_used else None,
                    )
                    label_used = True

            plt.axvline(trace[i], color='k', label="trace centre")
            plt.axvline(aperture_left_hand_edge, color='r', label="extraction aperture")
            plt.axvline(aperture_right_hand_edge, color='r')
            plt.axvline(left_bkg_right_hand_edge, color='r', ls='--', label="background regions")
            plt.axvline(right_bkg_left_hand_edge, color='r', ls='--')
            plt.axhline(0, color='k')
            plt.title('Aperture locations after background subtraction, star %d'%(star+1))
            plt.xlabel('X pixel')
            plt.ylabel('Background subtracted counts at row %d'%(i+1))
            plt.xlim(max(0, region_left - local_pad), min(ncols - 1, region_right + local_pad))
            x_min_plot, x_max_plot = plt.xlim()
            plot_mask = (local_x >= x_min_plot) & (local_x <= x_max_plot) & np.isfinite(background_subtracted)
            if np.any(plot_mask):
                y_visible = background_subtracted[plot_mask]
                y_span = max(float(np.nanmax(y_visible) - np.nanmin(y_visible)), 1.0)
                y_pad = max(1.0, 0.08 * y_span)
                plt.ylim(float(np.nanmin(y_visible) - y_pad), float(np.nanmax(y_visible) + y_pad))
            plt.legend(loc='upper left', numpoints=1, framealpha=1)
            if 'SAVE_FIG' in globals() and SAVE_FIG:
                plt.savefig(f"spectral_extraction_images/aperture_loc_after_subtract_row_{i+1}_star_{star+1}_frame_{frame_no}.pdf",dpi=50)

            if verbose == -2:
                plt.show()
            if verbose > 0:
                plt.show(block=False)
                plt.pause(verbose)
                plt.close()

        ap_left_arr[i] = aperture_left_hand_edge
        ap_right_arr[i] = aperture_right_hand_edge
        bkg_left_start_arr[i] = left_bkg_left_hand_edge
        bkg_left_end_arr[i] = left_bkg_right_hand_edge
        bkg_right_start_arr[i] = right_bkg_left_hand_edge
        bkg_right_end_arr[i] = right_bkg_right_hand_edge
        profile_fit_left_arr[i] = fit_window_left
        profile_fit_right_arr[i] = fit_window_right

        if diag_enabled:
            background_model_frame[i, region_left:region_right] = background_fit.astype(np.float32)
            if len(usable_aperture_cols) > 0:
                region_flags[i, usable_aperture_cols] |= REGION_APERTURE
            if len(bkg_cols_keep) > 0:
                region_flags[i, bkg_cols_keep] |= REGION_BACKGROUND_USED
            if len(bkg_cols_reject) > 0:
                region_flags[i, bkg_cols_reject] |= REGION_BACKGROUND_REJECTED
            if len(masked_regions) > 0:
                region_flags[i, masked_regions] |= REGION_CONTAMINANT
            if len(aperture_cols) > 0 and np.any(excluded_aperture_mask):
                region_flags[i, aperture_cols[excluded_aperture_mask]] |= REGION_CONTAMINANT
            if profile_enabled:
                region_flags[i, fit_window_left:fit_window_right] |= REGION_PROFILE_FIT
                profile_success[i] = bool(profile_fit["success"])
                profile_amplitude[i] = profile_fit["amplitude"]
                profile_centre[i] = float(trace[i]) if summed_profile_mode else profile_fit["centre"]
                profile_scale[i] = profile_fit["scale"]
                profile_scale_secondary[i] = profile_fit["scale_secondary"]
                profile_beta[i] = profile_fit["beta"]
                profile_mix_fraction[i] = profile_fit["mix_fraction"]
                profile_offset[i] = profile_fit["offset"]
                profile_radius[i] = profile_fit["radius"]
                profile_fwhm[i] = profile_fit["fwhm"]

    if verbose:
        plt.figure(figsize=(8,6))
        if instrument == "Keck/NIRSPEC":
            vmin,vmax = 0,500
        else:
            vmin,vmax = np.nanpercentile(frame,[50,70])
        plt.imshow(frame,vmin=vmin,vmax=vmax,aspect="auto")
        plt.plot(trace,np.arange(nrows),'k',label="fitted centre")
        plt.plot(ap_right_arr, np.arange(nrows), 'r', label="extraction aperture")
        plt.plot(ap_left_arr, np.arange(nrows), 'r')
        plt.plot(bkg_left_end_arr,np.arange(nrows),'r--',label="background region")
        plt.plot(bkg_right_start_arr,np.arange(nrows),'r--')
        if background_width != 1:
            plt.plot(bkg_left_start_arr,np.arange(nrows),'r--')
            plt.plot(bkg_right_end_arr,np.arange(nrows),'r--')
        plt.xlim(0,ncols)
        plt.ylim(0,nrows)
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.title('Aperture locations for star %d'%(star+1))
        plt.legend(framealpha=1)
        plt.tight_layout()
        if verbose == -2:
            plt.show()
        if verbose > 0:
            plt.show(block=False)
            plt.pause(verbose)
            plt.close()

        try:
            clipped_frame = np.array(clipped_frame)
            plt.figure(figsize=(12,8))
            plt.subplot(121)
            if instrument == "Keck/NIRSPEC":
                vmin,vmax = 0,500
            else:
                vmin,vmax = np.nanpercentile(clipped_frame,[10,50])
            plt.imshow(clipped_frame,vmin=vmin,vmax=vmax,aspect="auto")
            plt.title("Before background subtraction")
            plt.xlabel("Pixel column")
            plt.ylabel("Pixel row")

            plt.subplot(122)
            if instrument == "Keck/NIRSPEC":
                vmin,vmax = -500,1000
            else:
                vmin,vmax = np.nanpercentile(clipped_frame-np.array(sky_poly,dtype=object),[10,50])
            plt.imshow(clipped_frame-np.array(sky_poly),vmin=vmin,vmax=vmax,aspect="auto")
            plt.title("After background subtraction")
            plt.xlabel("Pixel column")
            if verbose == -2:
                plt.show()
            if verbose > 0:
                plt.show(block=False)
                plt.pause(verbose)
                plt.close()
        except Exception:
            pass

        plt.figure(figsize=(8,6))
        plt.plot(flux)
        plt.ylabel('Integrated counts (DN/s)')
        plt.xlabel('Y pixel')
        if verbose == -2:
            plt.show()
        if verbose > 0:
            plt.show(block=False)
            plt.pause(verbose)
            plt.close()

    if len(lh_overlap) != 0:
        print("For trace %d..."%(star+1))
        lh_counted = Counter(lh_overlap)
        for k in sorted(lh_counted.keys()):
            print("Left hand edge overlaps buffer pixels by %d pixels for %d rows"%(k,lh_counted[k]))
            log.write("Left hand edge overlaps buffer pixels by %d pixels for %d rows \n"%(k,lh_counted[k]))

    if len(rh_overlap) != 0:
        print("For trace %d..."%(star+1))
        rh_counted = Counter(rh_overlap)
        for k in sorted(rh_counted.keys()):
            print("Right hand edge overlaps buffer pixels by %d pixels for %d rows"%(k,rh_counted[k]))
            log.write("Right hand edge overlaps buffer pixels by %d pixels for %d rows"%(k,rh_counted[k]))

    log.close()

    aperture_widths_used = ap_right_arr - ap_left_arr
    finite_aperture_widths = aperture_widths_used[np.isfinite(aperture_widths_used)]
    if summed_profile_mode and summed_profile_fit is not None:
        aggregate_profile_success = bool(summed_profile_fit["success"])
        aggregate_profile_x = np.asarray(summed_profile_fit["relative_x"], dtype=np.float32)
        aggregate_profile_source_sum = np.asarray(summed_profile_fit["source_sum"], dtype=np.float32)
        aggregate_profile_contributor_counts = np.asarray(summed_profile_fit["contributor_counts"], dtype=np.int16)
        aggregate_profile_row_stack = np.asarray(summed_profile_fit["row_source_stack"], dtype=np.float32)
        aggregate_profile_interp_x = np.asarray(summed_profile_fit.get("interp_x", np.array([], dtype=float)), dtype=np.float32)
        aggregate_profile_interp_y = np.asarray(summed_profile_fit.get("interp_y", np.array([], dtype=float)), dtype=np.float32)
        aggregate_profile_rows_used = int(summed_profile_fit.get("rows_used", 0))
    else:
        aggregate_profile_success = False
        aggregate_profile_x = np.array([], dtype=np.float32)
        aggregate_profile_source_sum = np.array([], dtype=np.float32)
        aggregate_profile_contributor_counts = np.array([], dtype=np.int16)
        aggregate_profile_row_stack = np.empty((0, 0), dtype=np.float32)
        aggregate_profile_interp_x = np.array([], dtype=np.float32)
        aggregate_profile_interp_y = np.array([], dtype=np.float32)
        aggregate_profile_rows_used = 0

    if len(finite_aperture_widths) > 0:
        aggregate_profile_reference_width = float(np.nanmedian(finite_aperture_widths))
        aggregate_profile_min_width = float(np.nanmin(finite_aperture_widths))
        aggregate_profile_max_width = float(np.nanmax(finite_aperture_widths))
    else:
        aggregate_profile_reference_width = np.nan
        aggregate_profile_min_width = np.nan
        aggregate_profile_max_width = np.nan

    if diag_enabled:
        max_coeff_len = max(len(c) for c in background_coeffs) if len(background_coeffs) > 0 else 1
        background_coeff_array = np.full((nrows, max_coeff_len), np.nan, dtype=np.float32)
        for i, coeffs in enumerate(background_coeffs):
            background_coeff_array[i, -len(coeffs):] = coeffs

        save_extraction_diagnostic(
            "frame_%05d_star_%d" % (frame_no + 1, star + 1),
            frame_data=np.asarray(frame, dtype=np.float32),
            background_model=np.asarray(background_model_frame, dtype=np.float32),
            region_flags=region_flags,
            row_numbers=np.arange(row_min, row_min + nrows, dtype=int),
            trace=np.asarray(trace, dtype=np.float32),
            aperture_left=ap_left_arr,
            aperture_right=ap_right_arr,
            background_left_start=bkg_left_start_arr,
            background_left_end=bkg_left_end_arr,
            background_right_start=bkg_right_start_arr,
            background_right_end=bkg_right_end_arr,
            background_poly_order=np.asarray(bkg_poly_orders_used, dtype=np.int16),
            background_poly_coefficients=background_coeff_array,
            flux=np.asarray(flux, dtype=np.float32),
            error=np.asarray(error, dtype=np.float32),
            sky_avg=np.asarray(sky_avg, dtype=np.float32),
            profile_enabled=np.array(profile_enabled),
            profile_model=np.array(profile_model if profile_enabled else "fixed"),
            profile_aggregation_mode=np.array(profile_aggregation_mode if profile_enabled else "row_by_row"),
            profile_percentile=np.array(profile_percentile, dtype=np.float32),
            profile_fit_oversampling=np.array(profile_fit_oversampling, dtype=np.int16),
            profile_core_exclusion_half_width=np.array(profile_core_exclusion_half_width, dtype=np.float32),
            profile_max_width=np.array(-1.0 if profile_max_width is None else profile_max_width, dtype=np.float32),
            excluded_column_strips=np.asarray(excluded_column_ranges if excluded_column_ranges is not None else [], dtype=np.float32),
            effective_excluded_column_strips=pack_row_excluded_column_ranges(effective_row_excluded_column_ranges, nrows),
            excluded_column_mode=np.array(excluded_column_mode),
            profile_fit_left=profile_fit_left_arr if profile_enabled else np.full(nrows, -1, dtype=int),
            profile_fit_right=profile_fit_right_arr if profile_enabled else np.full(nrows, -1, dtype=int),
            profile_success=profile_success if profile_enabled else np.zeros(nrows, dtype=bool),
            profile_amplitude=profile_amplitude if profile_enabled else np.full(nrows, np.nan, dtype=np.float32),
            profile_centre=profile_centre if profile_enabled else np.full(nrows, np.nan, dtype=np.float32),
            profile_scale=profile_scale if profile_enabled else np.full(nrows, np.nan, dtype=np.float32),
            profile_scale_secondary=profile_scale_secondary if profile_enabled else np.full(nrows, np.nan, dtype=np.float32),
            profile_beta=profile_beta if profile_enabled else np.full(nrows, np.nan, dtype=np.float32),
            profile_mix_fraction=profile_mix_fraction if profile_enabled else np.full(nrows, np.nan, dtype=np.float32),
            profile_offset=profile_offset if profile_enabled else np.full(nrows, np.nan, dtype=np.float32),
            profile_radius=profile_radius if profile_enabled else np.full(nrows, np.nan, dtype=np.float32),
            profile_fwhm=profile_fwhm if profile_enabled else np.full(nrows, np.nan, dtype=np.float32),
            aggregate_profile_success=np.array(aggregate_profile_success),
            aggregate_profile_x=aggregate_profile_x,
            aggregate_profile_source_sum=aggregate_profile_source_sum,
            aggregate_profile_contributor_counts=aggregate_profile_contributor_counts,
            aggregate_profile_row_stack=aggregate_profile_row_stack,
            aggregate_profile_interp_x=aggregate_profile_interp_x,
            aggregate_profile_interp_y=aggregate_profile_interp_y,
            aggregate_profile_rows_used=np.array(aggregate_profile_rows_used, dtype=np.int32),
            aggregate_profile_fit_left=np.array(
                np.nan if summed_profile_fit is None else summed_profile_fit.get("fit_window_left", np.nan),
                dtype=np.float32,
            ),
            aggregate_profile_fit_right=np.array(
                np.nan if summed_profile_fit is None else summed_profile_fit.get("fit_window_right", np.nan),
                dtype=np.float32,
            ),
            aggregate_profile_amplitude=np.array(
                np.nan if summed_profile_fit is None else summed_profile_fit.get("amplitude", np.nan),
                dtype=np.float32,
            ),
            aggregate_profile_centre=np.array(
                np.nan if summed_profile_fit is None else summed_profile_fit.get("centre", np.nan),
                dtype=np.float32,
            ),
            aggregate_profile_scale=np.array(
                np.nan if summed_profile_fit is None else summed_profile_fit.get("scale", np.nan),
                dtype=np.float32,
            ),
            aggregate_profile_beta=np.array(
                np.nan if summed_profile_fit is None else summed_profile_fit.get("beta", np.nan),
                dtype=np.float32,
            ),
            aggregate_profile_offset=np.array(
                np.nan if summed_profile_fit is None else summed_profile_fit.get("offset", np.nan),
                dtype=np.float32,
            ),
            aggregate_profile_radius=np.array(
                np.nan if summed_profile_fit is None else summed_profile_fit.get("radius", np.nan),
                dtype=np.float32,
            ),
            aggregate_profile_fwhm=np.array(
                np.nan if summed_profile_fit is None else summed_profile_fit.get("fwhm", np.nan),
                dtype=np.float32,
            ),
            aggregate_profile_reference_width=np.array(aggregate_profile_reference_width, dtype=np.float32),
            aggregate_profile_min_width=np.array(aggregate_profile_min_width, dtype=np.float32),
            aggregate_profile_max_width=np.array(aggregate_profile_max_width, dtype=np.float32),
            aggregate_profile_row_normalization=np.array(
                "none" if summed_profile_fit is None else str(summed_profile_fit.get("row_normalization", "none"))
            ),
            fixed_width_capture_enabled=np.array(fixed_width_capture_enabled),
            fixed_width_capture_fraction=fixed_width_capture_fraction if fixed_width_capture_enabled else np.full(nrows, np.nan, dtype=np.float32),
            fixed_width_capture_flux=fixed_width_capture_flux if fixed_width_capture_enabled else np.full(nrows, np.nan, dtype=np.float32),
            fixed_width_total_flux=fixed_width_total_flux if fixed_width_capture_enabled else np.full(nrows, np.nan, dtype=np.float32),
            fixed_width_capture_left=fixed_width_capture_left if fixed_width_capture_enabled else np.full(nrows, -1, dtype=int),
            fixed_width_capture_right=fixed_width_capture_right if fixed_width_capture_enabled else np.full(nrows, -1, dtype=int),
            frame_index=np.array(frame_no + 1, dtype=np.int32),
            star_index=np.array(star + 1, dtype=np.int16),
            instrument=np.array(instrument),
            science_frame=np.array("" if frame_label is None else str(frame_label)),
            oversampling_factor=np.array(oversampling_factor, dtype=np.int16),
            region_aperture=np.array(REGION_APERTURE, dtype=np.uint8),
            region_background_used=np.array(REGION_BACKGROUND_USED, dtype=np.uint8),
            region_background_rejected=np.array(REGION_BACKGROUND_REJECTED, dtype=np.uint8),
            region_contaminant=np.array(REGION_CONTAMINANT, dtype=np.uint8),
            region_profile_fit=np.array(REGION_PROFILE_FIT, dtype=np.uint8),
        )
    if len(finite_aperture_widths) > 0:
        print(
            "  Trace %d aperture widths: %d -> %d px (median %.2f px)"
            % (
                star + 1,
                int(np.nanmin(finite_aperture_widths)),
                int(np.nanmax(finite_aperture_widths)),
                float(np.nanmedian(finite_aperture_widths)),
            )
        )
    if "JWST" in instrument:
        outputs = (np.array(flux),np.array(error),np.array(sky_avg))
    else:
        outputs = (np.array(flux),np.array(error),np.array(sky_avg),np.array(sky_left),np.array(sky_right),np.array(flux_base_level_left),np.array(flux_base_level_right),np.array(max_counts),np.array(error_from_readnoise),\
               np.array(error_from_scintillation),np.array(error_from_source),np.array(bkg_poly_orders_used),np.array(raw_star_flux))

    if return_aperture_widths:
        outputs = outputs + (np.array(aperture_widths_used),)
    if fixed_width_capture_requested:
        outputs = outputs + (np.array(fixed_width_capture_fraction),)
    return outputs

def extract_all_frame_fluxes(science_list,master_bias,master_flat,trace_dict,window_dict,extraction_dict,verbose=False,bad_pixel_mask=None,cosmic_pixel_mask=None,oversampling_factor=1,gain_file=None,readnoise_file=None):

    """The funtion that loops through all science frames,finding the trace locations, extracting the flux, and saving the final
    output."""

    # if verbose:
    #     if verbose == -1:
    #         verbose = False

    start_time = time.time()

    if master_bias is not None:
        master_bias = fits.open(master_bias)[0].data

    if master_flat is not None:
        master_flat = fits.open(master_flat)[0].data

    if bad_pixel_mask is not None:
        try:
            bad_pixel_mask = np.atleast_2d(fits.open(bad_pixel_mask)[0].data.astype(bool))
        except:
            bad_pixel_mask = np.atleast_2d(pickle.load(open(bad_pixel_mask,"rb")).astype(bool))
        use_mask = True

    if cosmic_pixel_mask is not None:
        cosmic_pixel_mask = pickle.load(open(cosmic_pixel_mask,"rb"))
        use_mask = True

    if bad_pixel_mask is None and cosmic_pixel_mask is None:
        use_mask = False

    rotate_frame = window_dict['rotate_frame']
    row_min = window_dict['row_min']
    row_max = window_dict['row_max']

    if gain_file is not None:
        gain_file = fits.getdata(gain_file)
        if rotate_frame:
            gain_file = np.flip(gain_file.T,axis=1)
        if oversampling_factor > 1:
            gain_file = resample_frame(gain_file,oversampling_factor,verbose=False)
        gain_file = gain_file[row_min:row_max]
        print("Mean gain = %.3f"%(gain_file.mean()))


    if readnoise_file is not None:
        readnoise_file = fits.getdata(readnoise_file)
        if rotate_frame:
            readnoise_file = np.flip(readnoise_file.T,axis=1)
        if oversampling_factor > 1:
            readnoise_file = resample_frame(readnoise_file,oversampling_factor,verbose=False)
        readnoise_file = readnoise_file[row_min:row_max]
        print("Mean readnoise = %.3f"%(readnoise_file.mean()))

    obs_time_array = []
    airmass = []
    exposure_time_array = []

    guess_location = trace_dict['guess_locations']
    search_width = trace_dict['search_width']
    gaussian_width = trace_dict['gaussian_width']
    trace_poly_order = trace_dict['trace_poly_order']
    trace_spline_sf = trace_dict['trace_spline_sf']
    if trace_spline_sf > 0 and trace_poly_order > 0:
        raise ValueError('Cannot use both a spline and polynomial fit to the trace, one of these must be set to zero in extraction input.')

    co_add_rows = trace_dict['co_add_rows']

    instrument = window_dict['instrument']
    readout_speed = window_dict['readout_speed']
    nwindows = window_dict['nwindows']


    aperture_width = extraction_dict['aperture_width']
    background_offset = extraction_dict['background_offset']
    background_width = extraction_dict['background_width']
    poly_bg_order = extraction_dict['poly_bg_order']
    rectify_frame = extraction_dict['rectify_frame']

    nstars = extraction_dict['nstars']
    masks = extraction_dict['masks']
    try:
        NIRSPEC_order = extraction_dict["NIRSPEC_order"]
    except:
        NIRSPEC_order = None
    use_lacosmic = extraction_dict['use_lacosmic']
    ACAM_linearity_correction = extraction_dict['ACAM_linearity_correction']
    gaussian_defined_aperture = extraction_dict['gaussian_defined_aperture']
    profile_apertures = extraction_dict['profile_apertures']
    fixed_width_capture_configs = extraction_dict.get('fixed_width_capture_configs', [{'enabled': False} for _ in range(nstars)])
    save_diagnostics = extraction_dict['save_diagnostics']
    share_star1_aperture_width = extraction_dict.get('share_star1_aperture_width', False)
    share_star1_relative_excluded_columns = extraction_dict.get('share_star1_relative_excluded_columns', False)
    max_aperture_per_exposure = extraction_dict.get('max_aperture_per_exposure', False)
    excluded_column_ranges = extraction_dict.get('excluded_column_ranges', [])
    excluded_column_mode = extraction_dict.get('excluded_column_mode', 'asymmetric')
    if gaussian_defined_aperture:
        aperture_log = open('aperture_log.log','w')
        aperture_log.close()

    stellar_fluxes = []
    stellar_errors = []
    sky_lefts = []
    sky_rights = []
    sky_avgs = []
    sky_polys = []
    base_lefts = []
    base_rights = []
    traces = []
    FWHM = []
    MAX_COUNTS = []
    raw_stellar_fluxes = []
    fixed_width_capture_fractions = [[] for _ in range(nstars)]
    cosmic_masked_pixels = []

    scintillation_error = []
    readnoise_error = []
    poisson_noise = []

    background_poly_order_used = []

    log = open('reduction_output.log','w')
    log.close()

    run_summary_lines = [
        "Extraction setup:",
        "  instrument = %s, nstars = %d, rows = %d:%d, frame oversampling = %d"
        % (instrument, nstars, row_min, row_max, oversampling_factor),
        "  rectify = %s, diagnostics = %s, share star1 widths = %s, max width per exposure = %s"
        % (bool(rectify_frame), bool(save_diagnostics), bool(share_star1_aperture_width), bool(max_aperture_per_exposure)),
    ]
    share_star1_relative_excluded_columns_active = bool(share_star1_relative_excluded_columns)
    if share_star1_relative_excluded_columns_active and excluded_column_mode != "mask_only":
        run_summary_lines.append(
            "  note: share_star1_relative_excluded_columns only applies with excluded_column_strip_mode = mask_only; ignoring it for mode = %s"
            % str(excluded_column_mode)
        )
        share_star1_relative_excluded_columns_active = False
    if share_star1_relative_excluded_columns_active and not excluded_column_ranges:
        run_summary_lines.append(
            "  note: share_star1_relative_excluded_columns requested but excluded_column_strips is blank; ignoring it"
        )
        share_star1_relative_excluded_columns_active = False
    if share_star1_relative_excluded_columns_active and nstars < 2:
        run_summary_lines.append(
            "  note: share_star1_relative_excluded_columns requested but only one star is being extracted; ignoring it"
        )
        share_star1_relative_excluded_columns_active = False
    if share_star1_relative_excluded_columns_active and nwindows != 1:
        run_summary_lines.append(
            "  note: share_star1_relative_excluded_columns currently supports only nwindows = 1; ignoring it"
        )
        share_star1_relative_excluded_columns_active = False
    if excluded_column_ranges:
        strip_summary = ", ".join(
            (
                "%d" % int(left / oversampling_factor)
                if (right - left) == oversampling_factor
                else "%d:%d" % (int(left / oversampling_factor), int(right / oversampling_factor) - 1)
            )
            for left, right in excluded_column_ranges
        )
        run_summary_lines.append("  excluded detector column strips = %s" % strip_summary)
        run_summary_lines.append("  excluded-column aperture mode = %s" % str(excluded_column_mode))
        if share_star1_relative_excluded_columns_active:
            run_summary_lines.append("  mirror star1 relative excluded columns onto later stars = True")
    elif excluded_column_mode in ("mask_only", "interpolate"):
        run_summary_lines.append(
            "  WARNING: excluded_column_strip_mode = %s but excluded_column_strips is blank, so no bad columns will be %s."
            % (
                str(excluded_column_mode),
                "masked" if excluded_column_mode == "mask_only" else "interpolated",
            )
        )
    for star_index in range(nstars):
        profile_cfg = profile_apertures[star_index]
        if profile_cfg.get("enabled", False):
            run_summary_lines.append(
                "  star %d: %s percentile aperture, aggregation = %s, keep %.1f%%, fit half-width = %d px, fit oversampling = %d, core exclusion = %.2f px, max width = %s"
                % (
                    star_index + 1,
                    str(profile_cfg.get("model", "moffat")),
                    str(profile_cfg.get("aggregation_mode", "row_by_row")),
                    float(profile_cfg.get("percentile", 0.95)) * (100.0 if float(profile_cfg.get("percentile", 0.95)) <= 1 else 1.0),
                    int(profile_cfg.get("fit_half_width", aperture_width[star_index])),
                    int(profile_cfg.get("fit_oversampling", 1)),
                    float(profile_cfg.get("core_exclusion_half_width", 0.0)),
                    "none" if profile_cfg.get("max_width", None) is None else "%.2f px" % float(profile_cfg.get("max_width")),
                )
            )
            if bool(max_aperture_per_exposure) and is_summed_profile_aggregation_mode(profile_cfg.get("aggregation_mode", "row_by_row")):
                run_summary_lines.append(
                    "    note: max_aperture_per_exposure is redundant in per-exposure summed-profile modes and is ignored here"
                )
        elif gaussian_defined_aperture:
            run_summary_lines.append(
                "  star %d: Gaussian-defined aperture, multiplier = %s"
                % (star_index + 1, aperture_width[star_index])
            )
        else:
            run_summary_lines.append(
                "  star %d: fixed aperture width = %d px"
                % (star_index + 1, aperture_width[star_index])
            )
        if fixed_width_capture_configs[star_index].get("enabled", False):
            run_summary_lines.append(
                "    empirical aperture-capture diagnostics = on, comparison half-width = %d px"
                % int(fixed_width_capture_configs[star_index].get("fit_half_width", aperture_width[star_index]))
            )

    print("\n".join(run_summary_lines))
    log = open('reduction_output.log','a')
    log.write("\n".join(run_summary_lines) + "\n")
    log.close()

    if "JWST" in instrument:
        fits_files = [fits.open(s,memmap=False) for s in science_list]
        nints = np.cumsum([f["SCI"].data.shape[0] for f in fits_files])
        total_nints = nints[-1]
        science_list = ["Integration %s"%i for i in range(total_nints)]

    for i,f in enumerate(science_list):
        shared_widths_from_star1 = None

        #print(f, '[%.1f%% complete, %d mins since start]'%((i+1)*100./len(science_list),(time.time()-start_time)/60))
        #log = open('reduction_output.log','a')
        #log.write('%s [%.1f%% complete, %d mins since start] \n'%(f,(i+1)*100./len(science_list),(time.time()-start_time)/60))
        #log.close()
        
        frame_display_name = os.path.basename(str(f))
        print(frame_display_name, '[%.1f%% complete, %d mins since start, %d/%d]' % ((i+1)*100./len(science_list), (time.time()-start_time)/60, i+1, len(science_list)))
        log = open('reduction_output.log', 'a')
        log.write('%s [%.1f%% complete, %d mins since start, %d/%d] \n' % (frame_display_name, (i+1)*100./len(science_list), (time.time()-start_time)/60, i+1, len(science_list)))
        log.close()

        if gaussian_defined_aperture:
            aperture_log = open('aperture_log.log','a')
            aperture_log.write('%s \n'%(f))
            aperture_log.close()

        if "JWST" not in instrument:
            fits_file = fits.open(f,memmap=False)
        else:
            jwst_fits_counter = np.digitize(i,nints)
            if jwst_fits_counter > 0:
                jwst_index_counter = i-nints[jwst_fits_counter]
            else:
                jwst_index_counter = i
            fits_file = fits_files[jwst_fits_counter]

        for window in range(1,nwindows+1):

            if master_bias is None and "JWST" not in instrument:
                if instrument == 'ACAM':
                    master_bias = np.zeros_like(fits_file[window].data)
                else:
                    master_bias = np.zeros_like(fits_file[window-1].data)

            if nwindows > 1:
                bias = master_bias[window-1]
            else:
                bias = master_bias

            if nwindows > 1 and master_flat is not None:
                flat = master_flat[window-1]
            else:
                flat = master_flat

            if window == 1:

                if instrument == "ACAM":
                    obs_time_array.append(fits_file[0].header['MJD-OBS'])
                    exposure_time = fits_file[0].header['EXPTIME']
                    exposure_time_array.append(exposure_time)
                    am = fits_file[0].header['AIRMASS']
                    airmass.append(am)

                elif instrument == "EFOSC":
                    obs_time_array.append(fits_file[0].header['MJD-OBS'])
                    exposure_time = fits_file[0].header['EXPTIME']
                    exposure_time_array.append(exposure_time)
                    am = fits_file[0].header['HIERARCH ESO TEL AIRM START']
                    airmass.append(am)

                elif "JWST" in instrument:
                    obs_time_array.append(fits_file["INT_TIMES"].data["int_mid_BJD_TDB"][jwst_index_counter])
                    exposure_time = fits_file[0].header["EFFINTTM"]
                    exposure_time_array.append(exposure_time)
                    am = 0
                    airmass.append(0)

                elif instrument == "Keck/NIRSPEC":
                    exposure_time = fits_file[0].header["ITIME"] / 1e3
                    exposure_time_array.append(exposure_time)
                    obs_date = fits_file[0].header["DATE-OBS"]
                    obs_start = Time(obs_date + "T" + fits_file[0].header["UTSTART"])
                    obs_mid = obs_start + TimeDelta(exposure_time/2,format='sec')
                    obs_time_array.append(obs_mid.mjd)
                    am = fits_file[0].header["AIRMASS"]
                    airmass.append(am)
                    m1temp = fits_file[0].header["SPEC1TMP"]
                    try: # saving m1temp to text file to save propagating through as a numpy array
                        new_tab = open("m1temp.txt","a")
                    except:
                        new_tab = open("m1temp.txt","w")
                    new_tab.write("%f \n"%(m1temp))
                    new_tab.close()

                else:
                    obs_time_array.append(0)
                    exposure_time = 0
                    exposure_time_array.append(0)
                    am = 0
                    airmass.append(0)

            if instrument == 'ACAM':
                frame = fits_file[window].data - bias
            elif "JWST" in instrument: # we're not performing a bias correction as this is done in jwst stage0
                frame = np.array([fits_file["SCI"].data[jwst_index_counter],fits_file["ERR"].data[jwst_index_counter]])
            else:
                frame = fits_file[window-1].data - bias

            uncorrected_frame = frame.astype(float)

            if master_flat is not None and "JWST" not in instrument: # this doesn't apply for jwst data as this is done in jwst stage0
                if instrument == 'ACAM':
                    frame = (fits_file[window].data - bias) / flat
                else:
                    frame = (fits_file[window-1].data - bias) / flat

            # replace inf with nan
            if "JWST" in instrument:
                if np.any(~np.isfinite(frame[0])):
                    frame[0][~np.isfinite(frame[0])] = np.nan
            else:
                if np.any(~np.isfinite(frame)):
                    frame[~np.isfinite(frame)] = np.nan


            if use_mask:
                if bad_pixel_mask is not None:
                    if len(bad_pixel_mask.shape) > 2:
                        bad_pixel_mask = bad_pixel_mask[i]
                if bad_pixel_mask is not None and cosmic_pixel_mask is None:
                    pixel_mask = bad_pixel_mask
                if bad_pixel_mask is None and cosmic_pixel_mask is not None:
                    pixel_mask = cosmic_pixel_mask[i]
                if bad_pixel_mask is not None and cosmic_pixel_mask is not None:
                    pixel_mask = bad_pixel_mask + cosmic_pixel_mask[i]

                if verbose != -1 and verbose != 0 and i == 0 or verbose != -1 and verbose != 0 and cosmic_pixel_mask is not None:
                    plt.figure()
                    plt.imshow(pixel_mask, interpolation='none',aspect="auto")
                    plt.title("Pixel mask, frame %d"%i)
                    plt.ylabel("Pixel column")
                    plt.xlabel("Pixel row")

                    # plt.savefig(f"spectral_extraction_images/pixel_mask_frame_{i}.pdf",dpi=50)

                    # if SAVE_FIG:
                    #     plt.savefig(f"spectral_extraction_images/pixel_mask_frame_{i}.pdf",dpi=50)

                    if verbose == -2:
                        plt.show()
                    if verbose > 0:
                        plt.show(block=False)
                        plt.pause(verbose)
                        plt.close()

                original_frame = frame.copy()
                if "JWST" in instrument:
                    frame = np.array([interp_bad_pixels(frame[0],pixel_mask,replace_with_medians=True),interp_bad_pixels(frame[1],pixel_mask,replace_with_medians=True)])
                else:
                    frame = interp_bad_pixels(frame,pixel_mask,replace_with_medians=True)

                if verbose != -1 and verbose != 0 and i == 0 or verbose != -1 and verbose != 0 and cosmic_pixel_mask is not None:
                    plt.figure()

                    plt.subplot(211)
                    if "JWST" in instrument:
                        vmin,vmax = np.nanpercentile(original_frame[0],[10,70])
                        plt.imshow(original_frame[0],vmin=vmin,vmax=vmax,aspect="auto")
                    # elif instrument == "Keck/NIRSPEC":
                    #     vmin,vmax = 0,500
                    else:
                        vmin,vmax = np.nanpercentile(original_frame,[10,70])
                        plt.imshow(original_frame,vmin=vmin,vmax=vmax,aspect="auto")
                    plt.title("Pre-pixel-masked frame")
                    plt.xticks(visible=False)
                    # ~ plt.xlabel("Pixel column")
                    plt.ylabel("Pixel row")


                    plt.subplot(212)
                    if "JWST" in instrument:
                        vmin,vmax = np.nanpercentile(frame[0],[10,70])
                        plt.imshow(frame[0],vmin=vmin,vmax=vmax,aspect="auto")
                    # elif instrument == "Keck/NIRSPEC":
                    #     vmin,vmax = 0,500
                    else:
                        vmin,vmax = np.nanpercentile(frame,[10,70])
                        plt.imshow(frame,vmin=vmin,vmax=vmax,aspect="auto")

                    plt.title("Post-pixel-masked frame")
                    plt.xlabel("Pixel column")
                    plt.ylabel("Pixel row")
                    # plt.savefig(f"spectral_extraction_images/pre_post_pixel_masked_frame_{i}.pdf",dpi=50)

                    if verbose == -2:
                        plt.show()
                    if verbose > 0:
                        plt.show(block=False)
                        plt.pause(verbose)
                        plt.close()

            else:
                pixel_mask=None


            if NIRSPEC_order is not None:
                if i == 0 and verbose:
                    v = verbose
                else:
                    v = False

                frame = KO.mask_NIRSPEC_data(frame,NIRSPEC_order,v)


            #if use_lacosmic and instrument == 'ACAM':
                #frame,_ = lacosmic.lacosmic(frame,0.5,15,15,effective_gain=1.9,readnoise=7)
            if use_lacosmic and instrument == "Keck/NIRSPEC" and cosmic_pixel_mask is None:
                cosmic_search_frame = copy.deepcopy(frame)
                cosmic_search_frame[~np.isfinite(cosmic_search_frame)] = 0
                cosmic_search_frame[cosmic_search_frame < 0] = 0

                # frame[~np.isfinite(frame)] = 0
                # frame[frame < 0] = 0
                cosmic_pixels,_ = astroscrappy.detect_cosmics(cosmic_search_frame[row_min:row_max], gain=3.01,readnoise=11.56, \
                                                          satlevel=np.inf, inmask=pixel_mask[row_min:row_max], sepmed=False, \
                                                          cleantype='medmask', fsmode='median',verbose=True,sigclip=5,objlim=10,niter=8)
                # frame[frame == 0] = np.nan
                frame[row_min:row_max] = interp_bad_pixels(frame[row_min:row_max],cosmic_pixels)

                if verbose != -1 and verbose != 0:
                    plt.figure(figsize=(4,12))
                    plt.imshow(cosmic_pixels,aspect="auto")
                    plt.title("Lacosmic-flagged cosmic pixels")
                    if verbose == -2:
                        plt.show()
                    if verbose > 0:
                        plt.show(block=False)
                        plt.pause(verbose)
                        plt.close()

                cosmic_masked_pixels.append(cosmic_pixels)

            if rotate_frame:
                if "JWST" in instrument:
                    frame = np.array([np.flip(frame[0].T,axis=1),np.flip(frame[1].T,axis=1)])
                    uncorrected_frame = np.array([np.flip(uncorrected_frame[0].T,axis=1),np.flip(uncorrected_frame[1].T,axis=1)])
                else:
                    frame = np.flip(frame.T,axis=1)
                    uncorrected_frame = np.flip(uncorrected_frame.T,axis=1)

            if "JWST" in instrument:
                frame = np.array([frame[0][row_min:row_max].astype(float),frame[1][row_min:row_max].astype(float)])
                uncorrected_frame = np.array([uncorrected_frame[0][row_min:row_max],uncorrected_frame[1][row_min:row_max]])
            else:
                frame = frame[row_min:row_max].astype(float)
                uncorrected_frame = uncorrected_frame[row_min:row_max]

            if oversampling_factor > 1:
                # nrows,ncols = frame.shape
                if "JWST" in instrument:
                    frame = np.array([resample_frame(frame[0],oversampling_factor,verbose=verbose),resample_frame(frame[1],oversampling_factor)])
                    uncorrected_frame = np.array([resample_frame(uncorrected_frame[0],oversampling_factor),resample_frame(uncorrected_frame[1],oversampling_factor)])
                else:
                    frame = resample_frame(frame,oversampling_factor,verbose=verbose)
                    uncorrected_frame = resample_frame(uncorrected_frame,oversampling_factor)
                # oversampling_factor = ((ncols-1)*oversampling+1)/ncols


            if ACAM_linearity_correction and instrument == 'ACAM':
                frame = ((-0.007/65000)*frame + 1)*frame # from ACAM webpages

            if nwindows == 1:
                loop_range = range(nstars)
            else:
                loop_range = range(1)

            star1_trace_for_relative_excluded_columns = None
            for star_number in loop_range:

                if nwindows > 1:
                    star_number += window - 1

                if search_width[star_number] > 0:
                    trace, force_verbose, fwhm, gauss_std = find_spectral_trace(frame,guess_location[star_number],search_width[star_number],gaussian_width,trace_poly_order,trace_spline_sf,star_number,verbose,co_add_rows,instrument,frame_no=i)
                else:
                    trace = np.ones(row_max-row_min)*guess_location[star_number]
                    fwhm = gauss_std = np.ones(row_max-row_min)
                    force_verbose = verbose

                if gaussian_defined_aperture:

                    # Smooth the FWHMs with a quadratic polynomial
                    gauss_std_poly = np.poly1d(np.polyfit(np.arange(0,row_max-row_min),gauss_std,trace_poly_order))
                    gauss_std_smooth = gauss_std_poly(np.arange(0,row_max-row_min))

                    # refit with outliers clipped
                    gauss_std_residuals = gauss_std - gauss_std_poly(np.arange(0,row_max-row_min))
                    gauss_std_keep_idx = abs(gauss_std_residuals) <= 4*np.std(gauss_std_residuals)

                    gauss_std_poly = np.poly1d(np.polyfit(np.arange(0,row_max-row_min)[gauss_std_keep_idx],gauss_std[gauss_std_keep_idx],trace_poly_order))
                    gauss_std_smooth = gauss_std_poly(np.arange(0,row_max-row_min))

                    if verbose != -1 and verbose != 0:
                        plt.figure()
                        plt.plot(np.arange(row_min,row_max),gauss_std,label="Std dev of trace")
                        plt.plot(np.arange(row_min,row_max)[~gauss_std_keep_idx],gauss_std[~gauss_std_keep_idx],"rx",label="Clipped outlier")
                        plt.plot(np.arange(row_min,row_max),gauss_std_smooth,label="Smoothed with polynomial (order = %d)"%trace_poly_order)
                        plt.xlabel("Pixel number")
                        plt.ylabel("Standard deviation (pixels)")
                        plt.title("Gaussian-defined aperture widths")
                        if verbose == -2:
                            plt.show()
                        if verbose > 0:
                            plt.show(block=False)
                            plt.pause(verbose)
                            plt.close()

                    trace_std = gauss_std_smooth*2*np.sqrt(2*np.log(2))

                else:
                    trace_std = None

                star_relative_excluded_column_ranges = None
                if share_star1_relative_excluded_columns_active:
                    if star_number == 0:
                        star1_trace_for_relative_excluded_columns = np.asarray(trace, dtype=float).copy()
                    elif star1_trace_for_relative_excluded_columns is not None:
                        star_relative_excluded_column_ranges = build_star_relative_excluded_column_ranges(
                            excluded_column_ranges,
                            star1_trace_for_relative_excluded_columns,
                            trace,
                        )

                capture_enabled_for_star = bool(fixed_width_capture_configs[star_number].get('enabled', False))

                if "JWST" in instrument: # only return the key arrays since the data files are so large and consume too much memory
                    jwst_outputs = extract_trace_flux(frame,trace,aperture_width[star_number],background_offset[star_number],\
                                                                                                    background_width[star_number],uncorrected_frame[0],poly_bg_order[star_number],am,\
                                                                                                    exposure_time,force_verbose,star_number,masks['mask%d'%(star_number+1)],instrument,row_min,trace_std,readout_speed,co_add_rows,rectify_frame,oversampling_factor,\
                                                                                                    gain_file,readnoise_file,frame_no=i,frame_label=f,profile_config=profile_apertures[star_number],fixed_width_capture_config=fixed_width_capture_configs[star_number],save_diagnostics=save_diagnostics,\
                                                                                                    forced_aperture_widths=shared_widths_from_star1 if (share_star1_aperture_width and star_number > 0) else None,\
                                                                                                    return_aperture_widths=(share_star1_aperture_width and star_number == 0),\
                                                                                                    max_aperture_per_exposure=max_aperture_per_exposure,\
                                                                                                    excluded_column_ranges=excluded_column_ranges,\
                                                                                                    row_excluded_column_ranges=star_relative_excluded_column_ranges,\
                                                                                                    excluded_column_mode=excluded_column_mode)
                    if share_star1_aperture_width and star_number == 0 and capture_enabled_for_star:
                        flux,error,sky_avg,shared_widths_from_star1,fixed_capture_fraction = jwst_outputs
                    elif share_star1_aperture_width and star_number == 0:
                        flux,error,sky_avg,shared_widths_from_star1 = jwst_outputs
                    elif capture_enabled_for_star:
                        flux,error,sky_avg,fixed_capture_fraction = jwst_outputs
                    else:
                        flux,error,sky_avg = jwst_outputs

                else:
                    non_jwst_outputs = extract_trace_flux(frame,trace,aperture_width[star_number],background_offset[star_number],\
                                                                                background_width[star_number],uncorrected_frame,poly_bg_order[star_number],am,\
                                                                                exposure_time,force_verbose,star_number,masks['mask%d'%(star_number+1)],instrument,row_min,trace_std,readout_speed,co_add_rows,rectify_frame,oversampling_factor,\
                                                                                gain_file,readnoise_file,frame_no=i,frame_label=f,profile_config=profile_apertures[star_number],fixed_width_capture_config=fixed_width_capture_configs[star_number],save_diagnostics=save_diagnostics,\
                                                                                forced_aperture_widths=shared_widths_from_star1 if (share_star1_aperture_width and star_number > 0) else None,\
                                                                                return_aperture_widths=(share_star1_aperture_width and star_number == 0),\
                                                                                max_aperture_per_exposure=max_aperture_per_exposure,\
                                                                                excluded_column_ranges=excluded_column_ranges,\
                                                                                row_excluded_column_ranges=star_relative_excluded_column_ranges,\
                                                                                excluded_column_mode=excluded_column_mode)
                    if share_star1_aperture_width and star_number == 0 and capture_enabled_for_star:
                        flux,error,sky_avg,sky_left,sky_right,base_left,base_right,max_counts,rn_error,scin_error,pois_error,bkg_poly_order,raw_star_flux,shared_widths_from_star1,fixed_capture_fraction = non_jwst_outputs
                    elif share_star1_aperture_width and star_number == 0:
                        flux,error,sky_avg,sky_left,sky_right,base_left,base_right,max_counts,rn_error,scin_error,pois_error,bkg_poly_order,raw_star_flux,shared_widths_from_star1 = non_jwst_outputs
                    elif capture_enabled_for_star:
                        flux,error,sky_avg,sky_left,sky_right,base_left,base_right,max_counts,rn_error,scin_error,pois_error,bkg_poly_order,raw_star_flux,fixed_capture_fraction = non_jwst_outputs
                    else:
                        flux,error,sky_avg,sky_left,sky_right,base_left,base_right,max_counts,rn_error,scin_error,pois_error,bkg_poly_order,raw_star_flux = non_jwst_outputs

                    plt.close("all")

                    sky_lefts.append(sky_left)
                    sky_rights.append(sky_right)
                    # sky_polys.append(sky_poly)
                    base_lefts.append(base_left)
                    base_rights.append(base_right)
                    MAX_COUNTS.append(max_counts)
                    scintillation_error.append(scin_error)
                    readnoise_error.append(rn_error)
                    poisson_noise.append(pois_error)
                    background_poly_order_used.append(bkg_poly_order)
                    raw_stellar_fluxes.append(raw_star_flux)

                stellar_fluxes.append(flux)
                stellar_errors.append(error)
                sky_avgs.append(sky_avg)
                traces.append(trace)
                FWHM.append(fwhm)
                if capture_enabled_for_star:
                    fixed_width_capture_fractions[star_number].append(np.asarray(fixed_capture_fraction, dtype=float))

        if "JWST" not in instrument:
            fits_file.close()

    try:
        os.mkdir("pickled_objects")
    except:
        pass

    for i in range(nstars):
        pickle.dump(np.array(stellar_fluxes[i::nstars]),open('pickled_objects/star%d_flux.pickle'%(i+1),'wb'))
        pickle.dump(np.array(stellar_errors[i::nstars]),open('pickled_objects/star%d_error.pickle'%(i+1),'wb'))
        pickle.dump(np.array(traces[i::nstars]),open('pickled_objects/x_positions_%d.pickle'%(i+1),'wb'))
        pickle.dump(np.array(FWHM[i::nstars]),open('pickled_objects/fwhm_%d.pickle'%(i+1),'wb'))
        pickle.dump(np.array(sky_avgs[i::nstars]),open('pickled_objects/background_avg_star%d.pickle'%(i+1),'wb'))
        if len(fixed_width_capture_fractions[i]) > 0 and fixed_width_capture_configs[i].get('enabled', False):
            pickle.dump(np.array(fixed_width_capture_fractions[i]),open('pickled_objects/fixed_width_capture_fraction_star%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(fixed_width_capture_fractions[i]),open('pickled_objects/aperture_capture_fraction_star%d.pickle'%(i+1),'wb'))

        if "JWST" not in instrument:
            pickle.dump(np.array(sky_lefts[i::nstars]),open('pickled_objects/sky_left_star%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(sky_rights[i::nstars]),open('pickled_objects/sky_right_star%d.pickle'%(i+1),'wb'))
            # pickle.dump(np.array(sky_polys[i::nstars]),open('pickled_objects/sky_poly_star%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(base_lefts[i::nstars]),open('pickled_objects/flux_base_level_left_star%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(base_rights[i::nstars]),open('pickled_objects/flux_base_level_right_star%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(MAX_COUNTS[i::nstars]),open('pickled_objects/max_counts_%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(scintillation_error[i::nstars]),open('pickled_objects/scintillation_error_%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(readnoise_error[i::nstars]),open('pickled_objects/readnoise_error_%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(poisson_noise[i::nstars]),open('pickled_objects/poisson_noise_%d.pickle'%(i+1),'wb'))
            pickle.dump(np.array(raw_stellar_fluxes[i::nstars]),open('pickled_objects/star%d_raw_flux.pickle'%(i+1),'wb'))

            if 0 in poly_bg_order:
                pickle.dump(np.array(background_poly_order_used[i::nstars]),open('background_poly_orders_used_%d.pickle'%(i+1),'wb'))

    if "JWST" not in instrument:
        pickle.dump(np.array(airmass),open('pickled_objects/airmass.pickle','wb'))
        pickle.dump(np.array(exposure_time_array),open('pickled_objects/exposure_times.pickle','wb'))

    pickle.dump(np.array(obs_time_array),open('pickled_objects/obs_time_array.pickle','wb'))
    pickle.dump(np.array(int(oversampling_factor)),open('pickled_objects/oversampling_factor.pickle','wb'))
    save_post_extraction_ancillary_products(
        "pickled_objects",
        sky_avgs,
        nstars,
        airmass,
        obs_time_array,
    )
    row_numbers = np.arange(row_min, row_min + (len(traces[0]) if len(traces) > 0 else 0), dtype=int)
    if len(row_numbers) > 0:
        pickle.dump(row_numbers, open('pickled_objects/fixed_width_capture_rows.pickle', 'wb'))
        pickle.dump(row_numbers, open('pickled_objects/aperture_capture_rows.pickle', 'wb'))

    if use_lacosmic and instrument == "Keck/NIRSPEC" and cosmic_pixel_mask is None:
        pickle.dump(np.array(cosmic_masked_pixels),open("pickled_objects/cosmic_masked_pixels.pickle","wb"))

    if instrument == "Keck/NIRSPEC":
        os.rename("m1temp.txt", "pickled_objects/m1temp.txt")

    return np.array(stellar_fluxes),np.array(stellar_errors),np.array(obs_time_array)


def generate_wl_curve(stellar_fluxes,stellar_errors,time,nstars,overwrite=True):

    """Generate the white light curve and output to table and figure"""

    star1 = stellar_fluxes[::nstars]
    error1 = stellar_errors[::nstars]

    if nstars > 1:
        star2 = stellar_fluxes[1::nstars]
        error2 = stellar_errors[1::nstars]

        ratio = np.sum(star1,axis=1)/np.sum(star2,axis=1)
        err_ratio = np.sqrt((np.sqrt(np.sum(error1**2,axis=1))/np.sum(star1,axis=1))**2 + (np.sqrt(np.sum(error2**2,axis=1))/np.sum(star2,axis=1))**2)*ratio

    else:
        ratio = np.mean(star1,axis=1)
        err_ratio = np.mean(error1,axis=1)

    if overwrite or not os.path.isfile('white_light.txt'):
        tab = open('white_light.txt','w')
        old_time = None
    else:
        tab = open('white_light.txt','a')
        old_time,old_ratio,old_err_ratio = np.loadtxt('white_light.txt',unpack=True)

    for i in range(len(ratio)):
        tab.write("%f %f %f \n"%(time[i],ratio[i],err_ratio[i]))

    tab.close()
    shutil.copyfile('white_light.txt', 'white_light.dat')

    if old_time is None:
        x_vals = time - int(time[0])
        y_vals = ratio
        y_err_vals = err_ratio
        x_label_zero = int(time[0])
    else:
        x_vals = np.hstack((old_time, time)) - int(old_time[0])
        y_vals = np.hstack((old_ratio, ratio))
        y_err_vals = np.hstack((old_err_ratio, err_ratio))
        x_label_zero = int(old_time[0])

    y_vals = np.asarray(y_vals, dtype=float)

    finite_flux = y_vals[np.isfinite(y_vals)]
    if len(finite_flux) > 0:
        median_flux = float(np.nanmedian(finite_flux))
        if np.isfinite(median_flux) and median_flux != 0:
            normalized_flux = finite_flux / median_flux
        else:
            normalized_flux = finite_flux.copy()
        centered_flux = normalized_flux - np.nanmedian(normalized_flux)
        std_ppm = float(np.nanstd(centered_flux) * 1.0e6)
        mad_ppm = float(1.4826 * np.nanmedian(np.abs(centered_flux - np.nanmedian(centered_flux))) * 1.0e6)
        if len(normalized_flux) >= 2:
            point_to_point_ppm = float(np.nanstd(np.diff(normalized_flux)) / np.sqrt(2.0) * 1.0e6)
        else:
            point_to_point_ppm = np.nan
    else:
        median_flux = np.nan
        std_ppm = np.nan
        mad_ppm = np.nan
        point_to_point_ppm = np.nan

    with open('white_light_metrics.txt', 'w') as metrics_file:
        metrics_file.write("# n_points median_flux std_ppm mad_ppm point_to_point_ppm\n")
        metrics_file.write(
            "%d %.10f %.3f %.3f %.3f\n"
            % (
                int(len(y_vals)),
                float(median_flux),
                float(std_ppm),
                float(mad_ppm),
                float(point_to_point_ppm),
            )
        )
    shutil.copyfile('white_light_metrics.txt', 'white_light_metrics.dat')

    y_span = np.nanmax(y_vals) - np.nanmin(y_vals)
    if not np.isfinite(y_span) or y_span <= 0:
        y_span = max(np.nanmax(np.abs(y_vals))*0.02, 1e-6)
    label_offset = 0.05 * y_span
    y_pad = 0.08 * y_span

    def style_numbered_white_light_plot(fig_width, fig_height, filename, add_faint_line=False, add_errorbars=False):
        plt.figure(figsize=(fig_width, fig_height))
        if add_errorbars:
            plt.errorbar(
                x_vals,
                y_vals,
                yerr=y_err_vals,
                fmt='o',
                color='k',
                ecolor='0.5',
                elinewidth=0.8,
                capsize=0,
                markersize=5,
                zorder=3,
            )
        else:
            plt.plot(x_vals, y_vals, 'ko', markersize=5, zorder=3)

        if add_faint_line:
            plt.plot(x_vals, y_vals, color='0.4', linewidth=1.0, alpha=0.5, zorder=2)

        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            plt.axvline(x, color='gray', linestyle='--', linewidth=0.5, alpha=0.25, zorder=1)
            plt.text(x, y + label_offset, str(i), fontsize=6, ha='center', va='bottom', rotation=90)

        plt.xlabel('Time (MJD/BJD - %d)' % x_label_zero)
        plt.ylabel('Flux')
        plt.ylim(np.nanmin(y_vals) - y_pad, np.nanmax(y_vals) + 1.6 * label_offset)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plt.figure(figsize=(8,6))
    if old_time is None:
        plt.plot(time-int(time[0]),ratio,'k.')
        plt.xlabel('Time (MJD/BJD - %d)'%int(time[0]))
    else:
        plt.plot(np.hstack((old_time,time))-int(old_time[0]),np.hstack((old_ratio,ratio)),'k.')
        plt.xlabel('Time (MJD/BJD - %d)'%int(old_time[0]))
    plt.ylabel('Flux')
    plt.savefig('white_light_curve.pdf')
    plt.close()

    style_numbered_white_light_plot(8, 6, 'white_light_curve_numbered.pdf')
    style_numbered_white_light_plot(14, 4.5, 'white_light_curve_numbered_wide.pdf', add_faint_line=True, add_errorbars=True)

    try:
        os.mkdir("./initial_WL_fit")
    except:
        pass

    pickle.dump(time-time[0],open("./initial_WL_fit/initial_WL_time.pickle","wb"))
    pickle.dump(ratio,open("./initial_WL_fit/initial_WL_flux.pickle","wb"))
    pickle.dump(err_ratio,open("./initial_WL_fit/initial_WL_err.pickle","wb"))

    return


def save_postrun_diagnostic_plots(
    diagnostic_input=".",
    time_overlay_rows=(200, 400, 600, 800, 1000),
    include_width_diagnostics=True,
    include_capture_diagnostics=True,
):
    """Save post-run diagnostic summary figures into diagnostic_plots/."""
    try:
        from plot_extraction_diagnostics import (
            build_frame_groups,
            create_time_overlay_figure,
            create_white_light_check_figure,
            export_profile_diagnostic_all_frames,
        )
    except Exception as exc:
        print("Skipping post-run diagnostic plot export: could not import plot_extraction_diagnostics.py (%s)" % exc)
        return

    try:
        frame_groups = build_frame_groups(diagnostic_input)
    except Exception as exc:
        print("Skipping post-run diagnostic plot export: no diagnostic bundles were found (%s)" % exc)
        return

    output_root = Path.cwd() / "diagnostic_plots"
    output_root.mkdir(parents=True, exist_ok=True)
    time_overlay_dir = output_root / "time_overlay"
    width_dir = output_root / "width_diagnostics"
    capture_dir = output_root / "aperture_capture"
    profile_dir = output_root / "aperture_profile_diagnostics"
    white_light_dir = output_root / "white_light"
    for path in (time_overlay_dir, width_dir, capture_dir, profile_dir, white_light_dir):
        path.mkdir(parents=True, exist_ok=True)

    try:
        if include_width_diagnostics:
            save_aperture_width_products_from_diagnostics(diagnostic_input=diagnostic_input, output_dir=width_dir)
            print("Saved width diagnostics into diagnostic_plots/width_diagnostics/")
    except Exception as exc:
        if include_width_diagnostics:
            print("Could not save width diagnostics: %s" % exc)

    for detector_row in time_overlay_rows:
        try:
            fig = create_time_overlay_figure(frame_groups, [int(detector_row)])
            output_name = time_overlay_dir / ("time_overlay_row_%04d.png" % int(detector_row))
            fig.savefig(output_name, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print("Saved diagnostic_plots/time_overlay/%s" % output_name.name)
        except Exception as exc:
            print("Could not save time-overlay for row %d: %s" % (int(detector_row), exc))

    try:
        saved_profiles = export_profile_diagnostic_all_frames(frame_groups, profile_dir)
        if saved_profiles > 0:
            print("Saved %d per-frame profile diagnostic figure(s) into diagnostic_plots/aperture_profile_diagnostics/" % int(saved_profiles))
    except Exception as exc:
        print("Could not save per-frame profile diagnostics: %s" % exc)

    try:
        if include_capture_diagnostics:
            save_aperture_capture_products_from_diagnostics(diagnostic_input=diagnostic_input, output_dir=capture_dir)
            print("Saved aperture-capture diagnostics into diagnostic_plots/aperture_capture/")
    except Exception as exc:
        if include_capture_diagnostics:
            print("Could not save aperture-capture diagnostics: %s" % exc)

    try:
        fig = create_white_light_check_figure(frame_groups)
        out_file = white_light_dir / "white_light_target_comparison_check.png"
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved diagnostic_plots/white_light/%s" % out_file.name)
    except Exception as exc:
        print("Could not save white-light target/comparison check: %s" % exc)


def main(input_file='extraction_input.txt'):
    input_file_path = Path(input_file).expanduser()
    if not input_file_path.is_absolute():
        input_file_path = (Path.cwd() / input_file_path).resolve()
    input_base_dir = input_file_path.parent

    input_dict = parseInput(str(input_file_path))

    oversampling_factor = input_dict["oversampling_factor"]
    if oversampling_factor is None:
        oversampling_factor = 1
    else:
        oversampling_factor = int(oversampling_factor)


    # order mask for Keck/NIRSPEC data
    if input_dict['instrument'] == 'Keck/NIRSPEC':

        NIRSPEC_order = input_dict["NIRSPEC_order"]
        trace_guess_locations,trace_search_widths = KO.get_guess_locations(NIRSPEC_order)
        nstars = 1
        trace_guess_locations *= oversampling_factor
        trace_search_widths *= oversampling_factor

    else:

        NIRSPEC_order = None

        trace_guess_locations = [int(x)*oversampling_factor for x in input_dict['trace_guess_locations'].split(",")]
        nstars = len(trace_guess_locations)

        # Update, added ability to have different extraction parameters for each star
        trace_search_widths = [int(x)*oversampling_factor for x in input_dict['trace_search_width'].split(",")]

        if len(trace_search_widths) == 1:
            trace_search_widths = trace_search_widths*nstars

    polybg_orders = [int(x) for x in input_dict['poly_bg_order'].split(",")]
    if len(polybg_orders) == 1:
        polybg_orders = polybg_orders*nstars


    gaussian_defined_aperture = bool(int(input_dict['gaussian_defined_aperture']))
    aperture_method = str(get_optional_input(input_dict, 'aperture_method', 'fixed')).strip().lower()

    if aperture_method in ['fixed', '']:
        profile_aperture_enabled = False
        default_profile_model = 'moffat'
    elif aperture_method == 'moffat_percentile':
        profile_aperture_enabled = True
        default_profile_model = 'moffat'
    elif aperture_method == 'empirical_percentile':
        profile_aperture_enabled = True
        default_profile_model = 'empirical'
    elif aperture_method == 'profile_percentile':
        profile_aperture_enabled = True
        default_profile_model = str(get_optional_input(input_dict, 'profile_model', 'moffat')).strip().lower()
    else:
        raise ValueError("Unsupported aperture_method '%s'. Use fixed, moffat_percentile, empirical_percentile or profile_percentile." % aperture_method)

    if gaussian_defined_aperture and profile_aperture_enabled:
        raise ValueError("gaussian_defined_aperture and aperture_method percentile modes are mutually exclusive.")

    if gaussian_defined_aperture:
        aperture_widths = [int(x) for x in input_dict['aperture_width'].split(",")]
    else:
        aperture_widths = [int(x)*oversampling_factor for x in input_dict['aperture_width'].split(",")]
    if len(aperture_widths) == 1:
        aperture_widths = aperture_widths*nstars


    background_offsets = [int(x)*oversampling_factor for x in input_dict['background_offset'].split(",")]
    if len(background_offsets) == 1:
        background_offsets = background_offsets*nstars

    background_widths = []
    for x in input_dict['background_width'].split(","):
        if int(x) > 1:
            background_widths.append(int(x)*oversampling_factor)
        else:
            background_widths.append(1)
    if len(background_widths) == 1:
        background_widths = background_widths*nstars

    profile_models = parse_multi_value(
        get_optional_input(input_dict, 'profile_model', default_profile_model),
        str,
        nstars,
        default_profile_model,
    )
    profile_models = [str(model).strip().lower() for model in profile_models]
    invalid_profile_models = [model for model in profile_models if model not in SUPPORTED_PROFILE_MODELS]
    if len(invalid_profile_models) > 0:
        raise ValueError(
            "Unsupported profile_model values %s. spectral_extraction_wingfocus.py supports only: %s"
            % (invalid_profile_models, ", ".join(SUPPORTED_PROFILE_MODELS))
        )
    profile_aggregation_modes_raw = parse_multi_value(
        get_optional_input(input_dict, 'profile_aggregation_mode', 'row_by_row'),
        str,
        nstars,
        'row_by_row',
    )
    profile_aggregation_modes = [
        normalize_profile_aggregation_mode(mode)
        for mode in profile_aggregation_modes_raw
    ]
    profile_fit_half_width_input = get_optional_input(input_dict, 'profile_fit_half_width', None)
    if profile_fit_half_width_input is None:
        profile_fit_half_widths = [max(12 * oversampling_factor, int(np.ceil(aperture_widths[i]))) for i in range(nstars)]
    else:
        profile_fit_half_widths = parse_multi_value(
            profile_fit_half_width_input,
            int,
            nstars,
            12,
        )
        profile_fit_half_widths = [int(x) * oversampling_factor for x in profile_fit_half_widths]
    profile_fit_oversampling_values = parse_multi_value(
        get_optional_input(input_dict, 'profile_fit_oversampling', 1),
        int,
        nstars,
        1,
    )
    profile_fit_oversampling_values = [max(int(x), 1) for x in profile_fit_oversampling_values]
    profile_percentiles = parse_multi_value(
        get_optional_input(input_dict, 'profile_percentile', 95.0),
        float,
        nstars,
        95.0,
    )
    profile_core_exclusion_half_widths = parse_multi_value(
        get_optional_input(input_dict, 'profile_core_exclusion_half_width', 0.0),
        float,
        nstars,
        0.0,
    )
    profile_core_exclusion_half_widths = [float(x) * oversampling_factor for x in profile_core_exclusion_half_widths]
    profile_max_widths_raw = parse_multi_value(
        get_optional_input(input_dict, 'profile_max_width', 0.0),
        float,
        nstars,
        0.0,
    )
    profile_max_widths = []
    for width in profile_max_widths_raw:
        if float(width) <= 0:
            profile_max_widths.append(None)
        else:
            profile_max_widths.append(float(width) * oversampling_factor)
    excluded_column_ranges = parse_excluded_column_strips(
        get_optional_input(input_dict, 'excluded_column_strips', None),
        oversampling_factor=oversampling_factor,
    )
    excluded_column_mode = normalize_excluded_column_mode(
        get_optional_input(input_dict, 'excluded_column_strip_mode', 'asymmetric')
    )
    if excluded_column_mode not in SUPPORTED_EXCLUDED_COLUMN_MODES:
        raise ValueError(
            "excluded_column_strip_mode must be one of: %s."
            % ", ".join(SUPPORTED_EXCLUDED_COLUMN_MODES)
        )

    profile_apertures = []
    for i in range(nstars):
        profile_apertures.append(
            {
                'enabled': profile_aperture_enabled,
                'model': str(profile_models[i]).strip().lower(),
                'aggregation_mode': str(profile_aggregation_modes[i]).strip().lower(),
                'percentile': float(profile_percentiles[i]),
                'fit_half_width': int(profile_fit_half_widths[i]),
                'fit_oversampling': int(profile_fit_oversampling_values[i]),
                'core_exclusion_half_width': float(profile_core_exclusion_half_widths[i]),
                'max_width': profile_max_widths[i],
            }
        )

    capture_enabled_input = get_optional_input(
        input_dict,
        'aperture_capture_diagnostics',
        get_optional_input(input_dict, 'fixed_width_capture_diagnostics', 1),
    )
    fixed_width_capture_enabled_values = parse_multi_value(
        capture_enabled_input,
        int,
        nstars,
        0,
    )
    fixed_width_capture_half_width_input = get_optional_input(
        input_dict,
        'aperture_capture_fit_half_width',
        get_optional_input(input_dict, 'fixed_width_capture_fit_half_width', None),
    )
    if fixed_width_capture_half_width_input is None:
        fixed_width_capture_half_widths = [
            max(int(np.ceil(aperture_widths[i])), 12 * oversampling_factor)
            for i in range(nstars)
        ]
    else:
        fixed_width_capture_half_widths = parse_multi_value(
            fixed_width_capture_half_width_input,
            int,
            nstars,
            12,
        )
        fixed_width_capture_half_widths = [int(x) * oversampling_factor for x in fixed_width_capture_half_widths]

    fixed_width_capture_configs = []
    for i in range(nstars):
        fixed_width_capture_configs.append(
            {
                'enabled': bool(int(fixed_width_capture_enabled_values[i])),
                'fit_half_width': int(fixed_width_capture_half_widths[i]),
            }
        )

    default_save_diagnostics = 1
    save_diagnostics = bool(int(get_optional_input(input_dict, 'save_extraction_diagnostics', default_save_diagnostics)))
    share_star1_aperture_width = bool(int(get_optional_input(input_dict, 'share_star1_aperture_width', 0)))
    share_star1_relative_excluded_columns = bool(int(get_optional_input(input_dict, 'share_star1_relative_excluded_columns', 0)))
    max_aperture_per_exposure = bool(int(get_optional_input(input_dict, 'max_aperture_per_exposure', 0)))


    mask_input = input_dict['masks']
    mask_width = input_dict['mask_width']
    if mask_input is not None:
        all_masks = mask_input.split(";")
        masked_region_list = []
        for i in range(nstars):
            masked_region_list.append([int(x)*oversampling_factor for x in all_masks[i].split(",") if x != ''])

        masks = create_masks(masked_region_list,nstars,int(mask_width)*oversampling_factor)

    else:
        masks = {}
        for i in range(1,nstars+1):
            masks['mask%d'%i] = None


    if input_dict["instrument"] == "ACAM":
        ACAM_linearity_correction = bool(int(input_dict['ACAM_linearity_correction']))
    else:
        ACAM_linearity_correction = False

    overwrite = bool(int(input_dict['overwrite']))


    trace_location_dict = {'guess_locations':trace_guess_locations,'search_width':trace_search_widths,\
                            'gaussian_width':int(input_dict['trace_gaussian_width'])*oversampling_factor,'trace_poly_order':int(input_dict['trace_poly_order']),\
                            'trace_spline_sf':float(input_dict['trace_spline_sf']),'co_add_rows':int(input_dict['co_add_rows'])}

    window_info_dict = {'instrument':input_dict['instrument'],'nwindows':int(input_dict['nwindows']),'row_min':int(input_dict['row_min']),'row_max':int(input_dict['row_max']),\
                        'readout_speed':input_dict['readout_speed'],"rotate_frame":bool(int(input_dict["rotate_frame"]))}

    extraction_params_dict = {'aperture_width':aperture_widths,'background_offset':background_offsets,\
                               'background_width':background_widths,'poly_bg_order':polybg_orders,\
                               'nstars':nstars,'masks':masks,'ACAM_linearity_correction':ACAM_linearity_correction,'gaussian_defined_aperture':gaussian_defined_aperture,\
                               "NIRSPEC_order":NIRSPEC_order,'use_lacosmic':bool(int(input_dict['use_lacosmic'])),"rectify_frame":bool(int(input_dict["rectify_data"])),\
                               'profile_apertures':profile_apertures,'save_diagnostics':save_diagnostics,\
                               'share_star1_aperture_width':share_star1_aperture_width,'max_aperture_per_exposure':max_aperture_per_exposure,\
                               'share_star1_relative_excluded_columns':share_star1_relative_excluded_columns,\
                               'excluded_column_ranges':excluded_column_ranges,'excluded_column_mode':excluded_column_mode,\
                               'fixed_width_capture_configs':fixed_width_capture_configs}

    v = int(input_dict['verbose'])

        # Enable figure generation for saving even if verbose==0
    try:
        if SAVE_FIG and v == 0:
            v = 1
    except NameError:
        print('Issue here with SAVE_FIG variable')
        pass

    science_files = resolve_science_files(input_dict['science_list'], input_dict['instrument'], input_base_dir)

    if not overwrite and os.path.isfile('white_light.txt'):
        test = np.loadtxt('white_light.txt')
        n = len(test[:,0])
        print("...loading from frame %d"%n)
        science_files = science_files[n:]


    bias = resolve_input_path(input_base_dir, input_dict['master_bias'])
    flat = resolve_input_path(input_base_dir, input_dict['master_flat'])
    bad_pixel_mask = resolve_input_path(input_base_dir, input_dict['bad_pixel_mask'])
    cosmic_pixel_mask = resolve_input_path(input_base_dir, input_dict['cosmic_pixel_mask'])

    if "JWST" in input_dict["instrument"]:
        gain_file = resolve_input_path(input_base_dir, input_dict["gain_file"])
        readnoise_file = resolve_input_path(input_base_dir, input_dict["readnoise_file"])
    else:
        gain_file = readnoise_file = None

    sf,se,time = extract_all_frame_fluxes(science_files,bias,flat,trace_location_dict,window_info_dict,extraction_params_dict,verbose=v,bad_pixel_mask=bad_pixel_mask,cosmic_pixel_mask=cosmic_pixel_mask,oversampling_factor=oversampling_factor,gain_file=gain_file,readnoise_file=readnoise_file)

    if input_dict["instrument"] == "Keck/NIRSPEC":
        f_norm = np.array([f/np.nanmean(f) for f in sf])
        master_spectrum = np.nanmedian(f_norm,axis=0)
        residual_spectra = f_norm-master_spectrum
        print("\n\nStandard deviation of residual spectra = %f\n"%(np.nanstd(residual_spectra)))
        log = open('reduction_output.log','a')
        log.write("\n\nStandard deviation of residual spectra = %f\n"%(np.nanstd(residual_spectra)))

    # ~ if nstars > 1:
    generate_wl_curve(sf,se,time,nstars,overwrite)
    save_postrun_diagnostic_plots(
        ".",
        time_overlay_rows=(200, 400, 600, 800, 1000),
        include_width_diagnostics=True,
        include_capture_diagnostics=True,
    )
    return


def create_masks(masked_region_list,nstars,mask_width):

    masks = {}

    for i in range(nstars):
        if mask_width is None:
            mask = [range(masked_region_list[i][j],masked_region_list[i][j+1]) for j in range(0,len(masked_region_list[i]),2)]
        else:
            mask_width = int(mask_width)
            mask = [range(masked_region_list[i][j]-mask_width//2,masked_region_list[i][j]+mask_width//2) for j in range(len(masked_region_list[i]))]

        flattened_mask = np.array([y for x in mask for y in x])
        if flattened_mask.size == 0: # array is empty
            flattened_mask = None
        masks['mask%d'%(i+1)] = flattened_mask

    return masks


def rectify_spatial(data, curve):
    """
    Shift data, column by column, along y-axis according to curve.
    Returns shifted image.
    Throws IndexError exception if length of curve
    is not equal to number of columns in data.
    """

    # shift curve to be centered at middle of order
    # and change sign so shift is corrective
    curve_p = -1.0 * curve
    curve_p = curve_p - np.median(curve_p)

    rectified = []
    for i in range(0, len(curve_p)):
        keep_index = np.isfinite(data[i])
        row = data[i]
        row[~keep_index] = 0

        rectified_row = interpolation.shift(row, curve_p[i], order=3, mode='nearest', prefilter=True)

        rectified_row[np.abs(rectified_row) <= 1e-30] = np.nan
        rectified.append(rectified_row)

    return((np.array(rectified)))


def resample_frame(data,oversampling=10,xmin=0,verbose=False):
    """A function that resamples all rows within an image to a greater sampling via linear interpolation. This is being tested as a method to deal with partial pixel extraction

    Inputs:
    data - the 2D spectral image
    oversampling - the number of sub-pixels in which to split each larger pixel into. Default=10
    xmin - if the data frame is a cut out of the larger frame, can define xmin as the left hand column for consistent x arrays. Default=0.
    verbose - True/False - do we want to plot the output of the resampling?

    Returns:
    data_resampled - the resampled image data"""

    nrows,ncols = data.shape
    old_x = np.arange(xmin,ncols)
    # new_x = np.arange(xmin,ncols-1+1/oversampling,1/oversampling)
    new_x = np.linspace(xmin,ncols,ncols*oversampling)

    data_resampled = np.array([np.interp(new_x,old_x,y) for y in data])

    if verbose:
        if verbose == -1:
            verbose = False

    if verbose:
        plt.figure()
        plt.plot(old_x,data[nrows//2],'ko',ms=6,mfc="k",label="Pre-oversampling",zorder=10)
        # plt.plot(old_x,data[nrows//2],'ko',ms=10,mfc="none",label="Pre-oversampling")
        plt.plot(new_x,data_resampled[nrows//2],'r.',label='Post-oversampling')
        plt.xlabel("X pixel")
        plt.ylabel("Counts (DN/s)")
        plt.title("Pixel resampling to deal with partial pixels")
        plt.legend()
        if verbose == -2:
            plt.show()
        if verbose > 0:
            plt.show(block=False)
            plt.pause(verbose)
            plt.close()

    return data_resampled

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--save_fig",
        action="store_true",
        help="Enable saving of figures instead of (or in addition to) showing them."
    )
    args = parser.parse_args()

    # pass this to your main or set a global flag
    SAVE_FIG = int(args.save_fig)
    print("SAVE_FIG =", SAVE_FIG)

        # If saving figures, suppress GUI windows
    if 'SAVE_FIG' in globals() and SAVE_FIG:
        try:
            import matplotlib
            matplotlib.use('Agg')  # non-interactive backend
        except Exception:
            pass
        # Prevent any window pops from plotting code
        def _noshow(*args, **kwargs):
            try:
                plt.close()
            except Exception:
                pass
        plt.show = _noshow


main()
