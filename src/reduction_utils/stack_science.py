#!/usr/bin/env python3
# Stack science frames into a single image to check for blends/contamination.
# Usage example:
#   python stack_science.py SCI_Gr11_list -inst EFOSC -m sum -v
#
# Author: adapted for EFOSC/ACAM from your master_arc pattern

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

plt.rcParams['image.origin'] = 'lower'

def read_list(path):
    return np.loadtxt(path, dtype=str)

def robust_combine(arr, method="median"):
    if method == "median":
        return np.median(arr, axis=0)
    if method == "mean":
        return np.mean(arr, axis=0)
    if method == "sum":
        return np.sum(arr, axis=0)
    raise ValueError("method must be one of: median, mean, sum")

def combine_efosc(filelist, verbose=False, eyeball=False, method="median"):
    files = read_list(filelist)
    stack = []
    if verbose or eyeball:
        plt.figure()
    keep = []

    for i, f in enumerate(files):
        with fits.open(f) as hdu:
            data = hdu[0].data.astype(float)

        if verbose or eyeball:
            plt.title(f'{i+1}/{len(files)}: {Path(f).name}')
            vmin, vmax = np.percentile(data[np.isfinite(data)], [5, 95])
            plt.imshow(data, vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.show(block=False)

        if eyeball:
            ans = input("Keep this frame? [g/b, default=g]: ").strip().lower()
            if ans == "b":
                if verbose:
                    print("  -> rejected")
                plt.clf()
                continue
        keep.append(f)
        stack.append(data)

        if verbose and not eyeball:
            plt.pause(0.4)
        if verbose or eyeball:
            plt.clf()

    if eyeball:
        with open(f"{filelist}_GOOD", "w") as gf:
            for f in keep:
                gf.write(f + "\n")

    stack = np.array(stack)
    return robust_combine(stack, method=method)

def combine_acam(filelist, verbose=False, eyeball=False, method="median"):
    files = read_list(filelist)
    # Determine number of windows from first file
    with fits.open(files[0]) as test:
        nwin = len(test) - 1  # ACAM windows start at ext=1

    if nwin not in (1, 2):
        raise RuntimeError(f"Unexpected ACAM window count: {nwin}")

    stacks = [[] for _ in range(nwin)]
    keep = []

    if verbose or eyeball:
        plt.figure()

    for i, f in enumerate(files):
        with fits.open(f) as hdul:
            for w in range(nwin):
                data = hdul[w+1].data.astype(float)
                stacks[w].append(data)

                if verbose:
                    plt.subplot(1, nwin, w+1)
                    vmin, vmax = np.percentile(data[np.isfinite(data)], [5, 95])
                    plt.imshow(data, vmin=vmin, vmax=vmax)
                    plt.title(f'{Path(f).name} win{w+1}')
                    plt.colorbar()

        if eyeball:
            plt.suptitle(f'{i+1}/{len(files)}')
            plt.show(block=False)
            ans = input("Keep this frame? [g/b, default=g]: ").strip().lower()
            if ans == "b":
                if verbose:
                    print("  -> rejected")
                for w in range(nwin):
                    stacks[w].pop()
            else:
                keep.append(f)
        else:
            if verbose:
                plt.pause(0.4)

        if verbose or eyeball:
            plt.clf()

    if eyeball:
        with open(f"{filelist}_GOOD", "w") as gf:
            for f in keep:
                gf.write(f + "\n")

    stacks = [np.array(s) for s in stacks]
    combined = [robust_combine(s, method=method) for s in stacks]
    return combined if nwin == 2 else combined[0]

def save_fits(data, outfile, clobber=False):
    if isinstance(data, list):  # ACAM 2 windows
        hdul = fits.HDUList([fits.PrimaryHDU(np.array(data[0])),
                             fits.ImageHDU(np.array(data[1]), name="WIN2")])
        hdul.writeto(outfile, overwrite=clobber)
    else:
        fits.PrimaryHDU(np.array(data)).writeto(outfile, overwrite=clobber)

def main():
    p = argparse.ArgumentParser(description="Stack science frames into a single FITS image.")
    p.add_argument("filelist", help="Text file listing science FITS frames (one per line).")
    p.add_argument("-inst", "--instrument", choices=["EFOSC", "ACAM"], required=True,
                   help="Instrument in use.")
    p.add_argument("-m", "--method", choices=["median", "mean", "sum"], default="median",
                   help="Combination method (default: median). For visualising traces, 'sum' is often best.")
    p.add_argument("-v", "--verbose", action="store_true", help="Show quick-look images while stacking.")
    p.add_argument("-e", "--eyeball", action="store_true", help="Interactively keep/reject frames.")
    p.add_argument("-c", "--clobber", action="store_true", help="Overwrite existing output.")
    args = p.parse_args()

    base = Path(args.filelist).name
    stem = base.split('.')[0]
    outname = f"stack_{args.instrument}_{args.method}_{stem}.fits"

    if args.instrument == "EFOSC":
        combined = combine_efosc(args.filelist, verbose=args.verbose, eyeball=args.eyeball, method=args.method)
    else:
        combined = combine_acam(args.filelist, verbose=args.verbose, eyeball=args.eyeball, method=args.method)

    save_fits(combined, outname, clobber=args.clobber)

    # Quick-look plot
    plt.figure()
    if isinstance(combined, list):  # ACAM 2 windows
        for i, img in enumerate(combined, 1):
            plt.subplot(1, 2, i)
            vmin, vmax = np.percentile(img[np.isfinite(img)], [5, 95])
            plt.imshow(img, vmin=vmin, vmax=vmax)
            plt.title(f"Stack win{i}")
            plt.colorbar()
    else:
        vmin, vmax = np.percentile(combined[np.isfinite(combined)], [5, 95])
        plt.imshow(combined, vmin=vmin, vmax=vmax)
        plt.title("Stacked science")
        plt.colorbar()
    plt.tight_layout()
    plt.show()

    print(f"Wrote {outname}")

if __name__ == "__main__":
    main()
