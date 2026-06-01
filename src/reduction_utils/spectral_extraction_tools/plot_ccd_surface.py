import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from astropy.io import fits
from astropy.visualization import ZScaleInterval


DEFAULT_COLOR_SCALE = "Viridis"
DEFAULT_DOWNSAMPLE = 8
DEFAULT_ZSCALE_CONTRAST = 0.25
DEFAULT_OUTPUT_SUFFIX = "_ccd_surface.html"


def find_default_hdu(hdul):
    for idx, hdu in enumerate(hdul):
        if getattr(hdu.data, "ndim", 0) == 2:
            return idx
    raise ValueError("No 2D image HDU was found in the supplied FITS file.")


def load_frame_data(path, hdu_index=None):
    with fits.open(path) as hdul:
        if hdu_index is None:
            hdu_index = find_default_hdu(hdul)
        data = np.asarray(hdul[hdu_index].data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Selected HDU %s is not 2D." % str(hdu_index))
    return data, int(hdu_index)


def block_reduce_mean(data, factor):
    factor = max(int(factor), 1)
    if factor == 1:
        return np.asarray(data, dtype=float)

    ny, nx = data.shape
    trimmed_ny = (ny // factor) * factor
    trimmed_nx = (nx // factor) * factor
    if trimmed_ny <= 0 or trimmed_nx <= 0:
        raise ValueError("Downsample factor %d is larger than the image dimensions %s." % (factor, data.shape))

    trimmed = np.asarray(data[:trimmed_ny, :trimmed_nx], dtype=float)
    reshaped = trimmed.reshape(trimmed_ny // factor, factor, trimmed_nx // factor, factor)
    return np.nanmean(reshaped, axis=(1, 3))


def compute_zscale_limits(data, contrast):
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin, vmax = ZScaleInterval(contrast=float(contrast)).get_limits(finite)
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(finite))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def build_surface_arrays(data, downsample, zscale_contrast, height_mode):
    reduced = block_reduce_mean(data, downsample)
    vmin, vmax = compute_zscale_limits(reduced, zscale_contrast)
    clipped = np.clip(reduced, vmin, vmax)

    if str(height_mode).lower() == "raw":
        height = reduced
        zaxis_title = "Counts"
    elif str(height_mode).lower() == "clipped":
        height = clipped
        zaxis_title = "Clipped counts"
    elif str(height_mode).lower() == "clipped_offset":
        height = clipped - vmin
        zaxis_title = "Clipped counts - zmin"
    elif str(height_mode).lower() == "log":
        # Shift to a strictly positive range before taking log10 so low-count
        # structure remains visible even when the z-scale minimum is near zero.
        height = np.log10(np.maximum(clipped - vmin + 1.0, 1.0e-6))
        zaxis_title = r"log10(clipped counts - zmin + 1)"
    else:
        raise ValueError("Unknown height_mode %s. Use raw, clipped, clipped_offset, or log." % str(height_mode))

    y = np.arange(reduced.shape[0], dtype=float) * int(max(downsample, 1))
    x = np.arange(reduced.shape[1], dtype=float) * int(max(downsample, 1))
    x_grid, y_grid = np.meshgrid(x, y)
    return {
        "x": x_grid,
        "y": y_grid,
        "height": height,
        "surfacecolor": clipped,
        "vmin": vmin,
        "vmax": vmax,
        "shape": reduced.shape,
        "zaxis_title": zaxis_title,
    }


def create_surface_figure(surface_bundle, title, colorscale, show_contours):
    contour_config = (
        {
            "z": {
                "show": True,
                "usecolormap": True,
                "highlightcolor": "#ffffff",
                "project": {"z": True},
            }
        }
        if show_contours
        else None
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=surface_bundle["x"],
                y=surface_bundle["y"],
                z=surface_bundle["height"],
                surfacecolor=surface_bundle["surfacecolor"],
                colorscale=colorscale,
                cmin=surface_bundle["vmin"],
                cmax=surface_bundle["vmax"],
                colorbar={"title": "Counts"},
                contours=contour_config,
                hovertemplate=(
                    "X pixel=%{x:.0f}<br>"
                    "Y pixel=%{y:.0f}<br>"
                    "Height=%{z:.1f}<br>"
                    "Counts=%{surfacecolor:.1f}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        scene={
            "xaxis_title": "X pixel",
            "yaxis_title": "Y pixel",
            "zaxis_title": surface_bundle["zaxis_title"],
            "aspectmode": "manual",
            "aspectratio": {
                "x": max(surface_bundle["x"].shape[1] / max(surface_bundle["x"].shape[0], 1), 0.5),
                "y": 1.0,
                "z": 0.35,
            },
            "camera": {
                "eye": {"x": 1.45, "y": -1.65, "z": 0.95},
            },
        },
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    return fig


def default_output_path(fits_path):
    fits_path = Path(fits_path).expanduser().resolve()
    return fits_path.with_name(fits_path.stem + DEFAULT_OUTPUT_SUFFIX)


def main():
    parser = argparse.ArgumentParser(
        description="Create an interactive 3D CCD surface plot from a FITS image."
    )
    parser.add_argument("fits_path", help="Path to the FITS frame to visualise.")
    parser.add_argument("--hdu", type=int, help="HDU index to use. Defaults to the first 2D image HDU.")
    parser.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_DOWNSAMPLE,
        help="Block-average downsample factor for interactive speed. Default: %(default)s",
    )
    parser.add_argument(
        "--colorscale",
        default=DEFAULT_COLOR_SCALE,
        help="Plotly colorscale name, for example Viridis, Cividis, Plasma, Inferno.",
    )
    parser.add_argument(
        "--zscale-contrast",
        type=float,
        default=DEFAULT_ZSCALE_CONTRAST,
        help="Astropy z-scale contrast used for the displayed counts range. Default: %(default)s",
    )
    parser.add_argument(
        "--height-mode",
        default="clipped_offset",
        choices=("raw", "clipped", "clipped_offset", "log"),
        help="How to map counts into surface height. Default: %(default)s",
    )
    parser.add_argument(
        "--no-contours",
        action="store_true",
        help="Disable projected contour lines on the 3D surface.",
    )
    parser.add_argument(
        "--output-html",
        help="Path for the interactive HTML output. Defaults to <fits_stem>_ccd_surface.html next to the FITS file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the interactive Plotly viewer after writing the HTML.",
    )
    args = parser.parse_args()

    fits_path = Path(args.fits_path).expanduser().resolve()
    data, used_hdu = load_frame_data(fits_path, hdu_index=args.hdu)
    surface_bundle = build_surface_arrays(
        data,
        downsample=max(int(args.downsample), 1),
        zscale_contrast=float(args.zscale_contrast),
        height_mode=args.height_mode,
    )

    title = (
        f"{fits_path.name} | HDU {used_hdu} | "
        f"shape={data.shape[1]}x{data.shape[0]} | downsample={max(int(args.downsample), 1)}"
    )
    fig = create_surface_figure(
        surface_bundle,
        title=title,
        colorscale=args.colorscale,
        show_contours=not args.no_contours,
    )

    output_html = Path(args.output_html).expanduser().resolve() if args.output_html else default_output_path(fits_path)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs=True)
    print(f"Saved interactive surface plot to {output_html}")

    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
