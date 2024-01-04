#!/usr/bin/env python3

from astropy.convolution import Box1DKernel, convolve
import numpy as np
from scipy.signal import correlate, correlation_lags

from smoothing import give_peaks


def get_ac_freq(peaks_list):
    """Return the frequency value for autocorrelation.
    peakslist is list of peak x-axis values."""
    freq_vals = [peaks_list[i + 1] - peaks_list[i] for i in range(len(peaks_list) - 1)]

    return freq_vals


def get_values(val, difference, smoothing=False):
    """Return dictionary with autocorrelation values. val is the power array to
    perform calculations on, difference is the frequency spacing and smoothing
    allows for the autocorrelation result to be smoothed via convolution.
    """
    width = 5
    box = Box1DKernel(width)
    if smoothing:
        auto_corr = convolve(
            correlate(val, val, mode="full"),
            box,
            boundary="wrap",
            nan_treatment="interpolate",
        )
    else:
        auto_corr = correlate(val, val, mode="full")

    lags = correlation_lags(len(val), len(val))
    lags_to_freq = lags * difference
    zero_idx = np.where(lags_to_freq == 0)[0][0]
    ac_peak_vals, ac_peak_heights = give_peaks(
        lags_to_freq[zero_idx - 1 :],
        auto_corr[zero_idx - 1 :],
        height=-0.2,
        threshold=0.0,
        distance=7,
    )
    delta_nus = get_ac_freq(ac_peak_vals)

    auto_dict = {
        "auto_corr": auto_corr,
        "lags": lags,
        "lags_to_freq": lags_to_freq,
        "zero_idx": zero_idx,
        "ac_peak_vals": ac_peak_vals,
        "ac_peak_heights": ac_peak_heights,
        "delta_nus": delta_nus,
    }

    return auto_dict


def return_ac_freqs(auto_dict, output_file):
    """Return the autocorrelation frequencies and save them to
    output_file. auto_dict is the dictionary of autocorrelation
    values.
    """
    with open(output_file, "w") as file:
        for key in auto_dict:
            int_spacing = f"integer spacing: {key}\n"
            ac_freqs = (
                f"AC peak frequencies: {auto_dict[key]['ac_peak_vals']} {chr(956)}Hz\n"
            )
            ac_freq_diffs = f"AC peak frequency differences: {auto_dict[key]['delta_nus']} {chr(956)}Hz\n\n"

            file.write(int_spacing)
            file.write(ac_freqs)
            file.write(ac_freq_diffs)


def ac_linear_fit(auto_dict):
    """Perform linear fit on autocorrelation, auto_dict is the
    dictionary of autocorrelation values.
    """
    for key, values in auto_dict.items():
        delta_nus = values["delta_nus"]
        acpeakvals = values["ac_peak_vals"]
        if len(delta_nus) <= 1:
            continue

        x = np.linspace(1, len(acpeakvals), len(acpeakvals))
        values["x_values"] = x

        p, cov = np.polyfit(x, acpeakvals, 1, cov=True)
        a = p[0]
        values["gradient"] = a
        b = p[1]
        values["intercept"] = b
        err_bar = np.sqrt(np.diag(cov)[0])
        values["err_bar"] = err_bar

        y_fitted = (a * x) + b
        values["y_fitted"] = y_fitted

    return auto_dict
