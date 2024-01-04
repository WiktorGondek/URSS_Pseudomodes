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

from autocorrelation import get_values, return_ac_freqs, ac_linear_fit
from plotting import (
    plot_time_series,
    plot_smoothed,
    plot_pseudo_range,
    plot_legendre,
    plot_power_spectrum,
    plot_autocorrelation,
    plot_ac_linear_fit,
)
from pseudo_range import pseudo_range, legendre_detrend
from remove_noise import get_subsection, remove_noise
from smoothing import remove_edge, give_peaks


def get_time_series(star):
    """
    Get time series for a given star using lightkurve to search for the star.
    """
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


def get_ft_peak_freqs(peaks_list):
    """
    Get the frequency of peaks from FT of power spectra. peaks_list is the list
    of peaks.
    """
    peak_freqs = [1 / peak for peak in peaks_list]

    return peak_freqs


def main(star, p_mode_spacing, output_dir, cache=True):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Load file from cache if it exists
    cache_file = Path(f".{star.lower().replace(' ', '_')}.pkl")
    if cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            lc = pickle.load(fh)
    else:
        lc = get_time_series(star)
        with open(cache_file, "wb") as fh:
            pickle.dump(lc, fh)

    # Plot time series
    plot_time_series(lc, star, output_dir)

    # Obtain flux and time values from time series
    dt, intensity = lc.time.value, lc.flux.value

    # Create 4 day subsections from lightkurve object
    subs = get_subsection(lc, dt)

    # Remove noisy timeseries
    not_noisy = remove_noise(subs)

    # Creates list of power spectra from 4 day timeseries sections
    pg_whole = create_power_spectra(subs, not_noisy, min_freq=920, max_freq=1500)

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
        freqs[0],
        power_avg,
        smoothed_ps,
        freq_edge,
        ps_edge,
        peak_vals,
        peak_heights,
        star,
        output_dir,
    )

    # Defining the pseudomode range
    p_dict = pseudo_range(
        freq_edge, ps_edge, peak_vals, p_mode_spacing, init_int=5, fin_int=7
    )
    pseudo_freqs, pseudo_power, first_peak, integer, difference = (
        p_dict["freqs"],
        p_dict["power"],
        p_dict["first_peak"],
        p_dict["integer"],
        p_dict["diff"],
    )

    # Plotting the pseudomode range
    plot_pseudo_range(
        freq_edge, ps_edge, pseudo_freqs, pseudo_power, first_peak, star, output_dir
    )

    # Defining list of power spectra
    ft_power_spectra = []
    # Defining list of detrended powers
    detrended = []

    for i in range(len(integer)):
        # Performing Legendre detrend on each pseudo frequency range
        leg_fit = legendre_detrend(pseudo_freqs[i], pseudo_power[i], order=2)

        detrended.append(leg_fit["detrended"])

        # Plotting Legendre fits
        plot_legendre(
            pseudo_freqs[i],
            pseudo_power[i],
            leg_fit["leg_val"],
            leg_fit["detrended"],
            2,
            integer[i],
            star,
            output_dir,
        )

        # Performing FT of power spectra
        ft_power_spectra.append(
            LombScargle(pseudo_freqs[i], leg_fit["detrended"]).autopower(
                samples_per_peak=1, nyquist_factor=1
            )
        )

    # Obtaining frequencies and powers from FT of power spectra
    ps_freqs = [i[0] for i in ft_power_spectra]
    ps_powers = [i[1] for i in ft_power_spectra]

    # Finding peaks
    ps_peaks = [
        give_peaks(ps_freqs[i], ps_powers[i], height=0.2) for i in range(len(integer))
    ]

    # Obtaining peak frequencies and powers
    ps_peak_freqs = [i[0] for i in ps_peaks]
    ps_peak_powers = [i[1] for i in ps_peaks]

    ps_peaks = [get_ft_peak_freqs(peak_list) for peak_list in ps_peak_freqs]

    ps_output = f"{output_dir}/ft_peaks.txt"
    with open(ps_output, "w") as file:
        for i, peak_list in zip(integer, ps_peaks):
            file.write(
                f"integer spacing {i}: "
                + ", ".join(str(peak) for peak in peak_list)
                + f" {chr(956)}Hz"
                + "\n"
            )

    # Plotting power spectrum
    plot_power_spectrum(
        ps_freqs, ps_powers, ps_peak_freqs, ps_peak_powers, integer, star, output_dir
    )

    # Create autocorrelation dictionary
    auto_dict = {}
    for i, powers_arr in zip(integer, detrended):
        auto_dict[f"{i}"] = get_values(powers_arr, difference)

    # Plot autocorrelation
    plot_autocorrelation(auto_dict, star, output_dir)

    # Return and save the autocorrelation peak frequncies and their differences
    # to a file
    return_ac_freqs(auto_dict, f"{output_dir}/ac_peak_freqs.txt")

    # Perform linear fits on autocorrelation data
    auto_dict = ac_linear_fit(auto_dict)

    # Plot linear fits on autocorrelation data
    plot_ac_linear_fit(auto_dict, star, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("star")
    parser.add_argument("p_mode_spacing")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    main(args.star, float(args.p_mode_spacing), args.output_dir)
