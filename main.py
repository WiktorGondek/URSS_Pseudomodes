#!/usr/bin/env python3

import argparse
from astropy.convolution import Box1DKernel, convolve
from astropy.timeseries import LombScargle
import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from numpy.polynomial import legendre as L
import pandas as pd
from pathlib import Path
import pickle
from scipy.signal import find_peaks
import sys

from autocorrelation import get_values
from plotting import (
    plot_time_series,
    plot_smoothed,
    plot_pseudo_range,
    plot_legendre,
    plot_power_spectrum,
)
from pseudo_range import pseudo_range, legendre_detrend
from remove_noise import get_subsection, remove_noise
from smoothing import remove_edge, give_peaks


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


def get_arrays(pseudo_freqs, pseudo_power):
    leg_fit = legendre_detrend(pseudo_freqs, pseudo_power, order=2)

    freq_rescale = pseudo_freqs * 1e-6

    power_spectrum = LombScargle(freq_rescale, leg_fit["detrended"]).autopower(
        samples_per_peak=1, nyquist_factor=1
    )

    ps_freqs = power_spectrum[0]
    ps_powers = power_spectrum[1]

    ps_peaks = give_peaks(ps_freqs, ps_powers, height=0.2)

    array_dict = {
        "leg_fit": leg_fit,
        "freq_rescale": freq_rescale,
        "ps_freqs": ps_freqs,
        "ps_powers": ps_powers,
        "ps_peaks": ps_peaks,
    }

    return array_dict

    # return (leg_fit, freq_rescale, power_spectrum, ps_freqs, ps_powers, ps_peaks)


def main(star, output_dir, cache=True):
    output_dir_name = f"{output_dir}_{star.replace(' ','_')}"
    if not Path(output_dir_name).exists():
        Path(output_dir_name).mkdir(exist_ok=True, parents=True)

    # Load file from cache if it exists
    cache_file = Path(f".{star.lower().replace(' ', '_')}.pkl")
    if cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            lc = pickle.load(fh)
    else:
        lc = get_time_series(star, output_file)
        with open(cache_file, "wb") as fh:
            pickle.dump(lc, fh)

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
    plot_pseudo_range(
        freq_edge, ps_edge, pseudo_freqs, pseudo_power, first_peak, star, output_dir
    )

    # Defining list of power spectra
    ft_power_spectra = []

    for i in range(len(integer)):
        # Performing Legendre detrend on each pseudo frequency range
        leg_fit = legendre_detrend(pseudo_freqs[i], pseudo_power[i], order=2)
        # Rescaling the frequencies to be used in LombScargle
        freq_rescale = pseudo_freqs[i] * 1e-6

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
            LombScargle(freq_rescale, leg_fit["detrended"]).autopower(
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

    # Plotting power spectrum
    plot_power_spectrum(
        ps_freqs, ps_powers, ps_peak_freqs, ps_peak_powers, integer, star, output_dir
    )

    plt.show()


# df = pd.DataFrame(
#    [get_values(val) for val in leg_fit[i]['detrended'],
#    columns=(
#        "auto_corr",
#        "lags",
#        "lags_to_freq",
#        "zero_idx",
#        "ac_peak_vals",
#        "delta_nus",
#    )
# )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("star")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    main(args.star, args.output_dir)
