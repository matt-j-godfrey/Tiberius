#### Author: Adapted from James Kirk's master bias script

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import argparse

plt.rcParams['image.origin'] = 'lower'

parser = argparse.ArgumentParser()
parser.add_argument('arclist', help="""Enter list of arc file names (e.g. made with `ls *ARC*.fits > arc.lis`)""")
parser.add_argument('-v','--verbose', help="""Display each arc frame as it's read.""", action='store_true')
parser.add_argument('-inst','--instrument', help="""Instrument in use: 'EFOSC' or 'ACAM'""", required=True)
parser.add_argument('-c','--clobber', help="""Allow overwriting the output file.""", action='store_true')
parser.add_argument('-e','--eyeball', help="""Manually inspect and sort good/bad arc frames.""", action='store_true')
args = parser.parse_args()

if args.instrument not in ['EFOSC', 'ACAM']:
    raise NameError('Instrument must be either EFOSC or ACAM')

if args.instrument == 'EFOSC':
    nwin = 1
else:
    test_file = np.loadtxt(args.arclist, str)[0]
    test = fits.open(test_file)
    nwin = len(test) - 1

def combine_arcs_1window(filelist, instrument, verbose=False, eyeball=False):
    arc_files = np.loadtxt(filelist, str)
    arc_data = []

    if verbose or eyeball:
        plt.figure()

    if eyeball:
        good_frames = []
        bad_frames = []

    for n, f in enumerate(arc_files):
        hdul = fits.open(f)
        data = hdul[1].data if instrument == 'ACAM' else hdul[0].data
        arc_data.append(data)

        if verbose or eyeball:
            plt.title(f'{n+1}/{len(arc_files)}: {f}')
            plt.imshow(data, vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
            plt.colorbar()
            plt.show(block=False)

            if eyeball:
                choice = input("Keep this frame? [g = good / b = bad]: ")
                if choice == 'g':
                    good_frames.append(f)
                elif choice == 'b':
                    bad_frames.append(f)
                else:
                    print("Invalid choice, assuming good.")
                    good_frames.append(f)
                plt.clf()
            else:
                plt.pause(1.0)
                plt.clf()

        hdul.close()

    if eyeball:
        with open(filelist + '_GOOD', 'w') as gf:
            for g in good_frames:
                gf.write(g + '\n')
        with open(filelist + '_BAD', 'w') as bf:
            for b in bad_frames:
                bf.write(b + '\n')
        print("Saved sorted GOOD/BAD lists. Exiting...")
        raise SystemExit()

    arc_data = np.array(arc_data)
    return np.median(arc_data, axis=0)

def combine_arcs_2windows(filelist, verbose=False):
    arc_files = np.loadtxt(filelist, str)
    arc_data = [[], []]
    if verbose:
        plt.figure()

    for n, f in enumerate(arc_files):
        hdul = fits.open(f)
        for win in range(1, 3):
            data = hdul[win].data
            arc_data[win - 1].append(data)
            if verbose:
                plt.subplot(1, 2, win)
                plt.imshow(data, vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
                plt.title(f'{f[-12:]} win {win}')
                plt.colorbar()
        if verbose:
            plt.pause(1.0)
            plt.clf()
        hdul.close()

    arc_data = np.array(arc_data)
    return np.array([np.median(arc_data[0], axis=0), np.median(arc_data[1], axis=0)])

def save_fits(data, filename, clobber=False):
    fits.PrimaryHDU(data).writeto(filename, overwrite=clobber)

if nwin == 1:
    master_arc = combine_arcs_1window(args.arclist, args.instrument, args.verbose, args.eyeball)
else:
    master_arc = combine_arcs_2windows(args.arclist, args.verbose)

save_fits(master_arc, 'master_arc.fits', args.clobber)

plt.figure()
if nwin == 1:
    plt.imshow(master_arc, vmin=np.percentile(master_arc, 5), vmax=np.percentile(master_arc, 95))
else:
    plt.subplot(121)
    plt.imshow(master_arc[0], vmin=np.percentile(master_arc[0], 5), vmax=np.percentile(master_arc[0], 95))
    plt.subplot(122)
    plt.imshow(master_arc[1], vmin=np.percentile(master_arc[1], 5), vmax=np.percentile(master_arc[1], 95))
plt.colorbar()
plt.title('Master Arc')
plt.show()
