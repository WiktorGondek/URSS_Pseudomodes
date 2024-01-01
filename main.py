#!/usr/bin/env python3

import argparse
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import sys

from remove_noise import get_subsection, remove_noise


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

def create_power_spectra(sub_ts, not_noisy_ts, min_freq, max_freq):
    """
    Perform FT of least noisy 4 day timeseries using to_periodogram().
    sub_ts are the subsections, not_noisy_ts is list of the index of least noisy sections,
    min_freq and max_freq set the min and max frequencies
    """

    # pg_whole is a list of periodogram objects
    pg_whole = [
        sub_ts[not_noisy_ts[i]].to_periodogram(
            normalization="psd",
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
        )
        for i in range(0, len(not_noisy_ts))
    ]
    return pg_whole

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
    intensity, dt  = lc.flux.value, lc.time.value

    # Create 4 day subsections from lightkurve object
    subs = get_subsection(lc, dt)

    # Remove noisy timeseries
    notnoisy = remove_noise(subs)
    
    # Creates list of power spectra from 4 day timeseries sections###
    pg_whole = create_power_spectra(subs, notnoisy, min_freq=920, max_freq=1500)

    # Adding the powers and frequencies from periodogram into a list to average
    powers, freqs = np.moveaxis([(pg.power.value, pg.frequency.value) for pg in pg_whole], 1, 0)    

    # Averaging powers from lightkurve
    power_avg = np.mean(powers, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("star")
    parser.add_argument("output_file")
    args = parser.parse_args()

    main(args.star, args.output_file)