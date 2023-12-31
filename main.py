#!/usr/bin/env python3

import argparse
import lightkurve as lk
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

from remove_noise import get_subsection, remove_noise
from power_spectra import create_power_spectra, average, add_to_lists


def get_time_series(star, output_file):
    # star = "KIC 7799349"
	
    # Search for star
    search_result = lk.search_lightcurve(
        star,
        cadence="short",
        author="Kepler",
        quarter=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
    )
    # Format time series	
    lc = search_result.download_all().stitch()
    lc = (
        lc.remove_nans("flux")
        .remove_outliers()
        .flatten(window_length=501)
        .normalize(unit="ppm")
        .fill_gaps()
    )

    # Plot full timeseries
    fig, ax = plt.subplots()
    lc.plot(ax=ax)
    fig.savefig(output_file)
    return lc


def main(star, output_file, cache=True):
    
    cache_file = Path(f".{star.lower().replace(' ', '_')}.pkl")
    if cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            lc = pickle.load(fh) 
    else:
        lc = get_time_series(star, output_file)
        with open(cache_file, "wb") as fh:
            pickle.dump(lc, fh)
    
    # Obtain flux and time values from time series
    intensity = lc.flux.value
    dt = lc.time.value
    print(intensity, dt)

    # Create 4 day subsections from lightkurve object
    subs = get_subsection(lc, dt)

    # Remove noisy timeseries
    notnoisy = remove_noise(subs)
    print(notnoisy)
    
    # Creates list of power spectra from 4 day timeseries sections###
    #pg_whole = create_power_spectra(subs, notnoisy, min_freq=920, max_freq=1500)

    ###Plotting the power spectra###
    # for i in range(0,len(pgwhole),20):
    #    ax = pgwhole[i].plot()

    # Adding the powers and frequencies from periodogram into a list to average
    #powers, frequencies = add_to_lists(pg_whole)

    # Averaging powers from lightkurve
    #power_avg = average(powers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("star")
    parser.add_argument("output_file")
    args = parser.parse_args()

    main(args.star, args.output_file)
