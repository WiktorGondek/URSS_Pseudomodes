#!/usr/bin/env python3

import argparse
from astropy.convolution import Box1DKernel, convolve
from astropy.timeseries import LombScargle
import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from numpy.polynomial import legendre as L
from pathlib import Path
import pickle
from scipy.signal import find_peaks
import sys

from plotting import plot_pseudo_range, plot_legendre, plot_power_spectrum
from pseudo_range import pseudo_range, legendre_detrend
from remove_noise import get_subsection, remove_noise
from smoothing import remove_edge, plot_smoothed


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


def give_peaks(x, y, height, distance=None):
    peaks = find_peaks(y, height=height, distance=distance)
    peak_vals = [x[i] for i in peaks[0]]
    peak_heights = [i for i in peaks[1]["peak_heights"]]

    return peak_vals, peak_heights


def main(star, output_file, cache=True):
    # Load file from cache if it exists
    cache_file = Path(f".{star.lower().replace(' ', '_')}.pkl")
    if cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            lc = pickle.load(fh)
    else:
        lc = get_time_series(star, output_file)
        with open(cache_file, "wb") as fh:
            pickle.dump(lc, fh)

    # Obtain flux and time values from time series
    dt, intensity = lc.time.value, lc.flux.value

    # Create 4 day subsections from lightkurve object
    subs = get_subsection(lc, dt)

    # Remove noisy timeseries
    notnoisy = remove_noise(subs)

    # Creates list of power spectra from 4 day timeseries sections###
    pg_whole = create_power_spectra(subs, notnoisy, min_freq=920, max_freq=1500)

    # Adding the powers and frequencies from periodogram into a list to average
    freqs, powers = np.moveaxis(
        [(pg.frequency.value, pg.power.value) for pg in pg_whole], 1, 0
    )

    # Averaging powers from lightkurve
    power_avg = np.mean(powers, axis=0)

    # Smoothing power spectrum with 1d box kernel
    width = 5
    box = Box1DKernel(width)
    smoothed_ps = convolve(power_avg, box, boundary="wrap", nan_treatment="interpolate")

    # Removing edges from power spectrum and frequency array
    ps_edge, freq_edge = remove_edge(smoothed_ps, freqs[0], percent=0.04)

    # Find peaks of power spectrum
    peak_vals, peak_heights = give_peaks(
        freq_edge, ps_edge, height=np.mean(ps_edge), distance=5
    )

    # Plot smoothed power spectrum with peaks
    plot_smoothed(
        freqs[0], power_avg, smoothed_ps, freq_edge, ps_edge, peak_vals, peak_heights
    )

    dnu = 33.77942877988895

    # Defining the pseudomode range
    p_dict = pseudo_range(freq_edge, ps_edge, peak_vals, dnu, init_int=5, fin_int=7)
    pseudo_freqs, pseudo_power, first_peak, integer = (
        p_dict["freqs"],
        p_dict["power"],
        p_dict["first_peak"],
        p_dict["integer"],
    )

    # Plotting the pseudomode range
    plot_pseudo_range(freq_edge, ps_edge, pseudo_freqs, pseudo_power, first_peak)

    # Detrended pseudomode range
    leg_fit = [
        legendre_detrend(pseudo_freqs[i], pseudo_power[i], order=2)
        for i in range(len(integer))
    ]
    # Rescaling the frequencies into Hz from muHz to be able to
    # FT using LombScargle
    freq_rescale = [pseudo_freqs[i] * 1e-6 for i in range(len(integer))]

    # Plot legendre
    legendre_plots = [
        plot_legendre(
            pseudo_freqs[i],
            pseudo_power[i],
            leg_fit[i]["leg_val"],
            leg_fit[i]["detrended"],
            2,
            integer[i],
        )
        for i in range(len(integer))
    ]

    # FT of power spectra
    power_spectrum = [
        LombScargle(freq_rescale[i], leg_fit[i]["detrended"]).autopower(
            samples_per_peak=1, nyquist_factor=1
        )
        for i in range(len(integer))
    ]

    # Frequencies and powers of the FT of power spectrum
    ps_freqs = [i[0] for i in power_spectrum]
    ps_powers = [i[1] for i in power_spectrum]

    # Find peaks of FT
    ps_peaks = [
        give_peaks(ps_freqs[i], ps_powers[i], height=0.2) for i in range(len(integer))
    ]

    # Peaks of FT for each integer spacing
    ps_peak_freqs = [i[0] for i in ps_peaks]
    ps_peak_powers = [i[1] for i in ps_peaks]

    # Plot FT of power spectrum
    plot_power_spectrum(ps_freqs, ps_powers, ps_peak_freqs, ps_peak_powers, integer)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("star")
    parser.add_argument("output_file")
    args = parser.parse_args()

    main(args.star, args.output_file)
