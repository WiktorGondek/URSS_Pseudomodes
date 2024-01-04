#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import legendre as L


def plot_time_series(lc, star, output_dir):
    """
    Plot time series of star. lc is the light curve object containing the time
    series, star is the name of the star and output_dir is where the figure is
    saved to.
    """

    fig, ax = plt.subplots()
    lc.plot(ax=ax)
    fig.savefig(f"{output_dir}/time_series.png")


def plot_smoothed(
    freqs,
    pow_avg,
    pow_smooth,
    freq_edge,
    pow_smooth_edge,
    peak_frq,
    peak_heights,
    star,
    output_dir,
):
    """Plot unsmoothed and smoothed power spectra showing peaks.
    freqs, pow_avg are the frequency and power lists,
    pow_smooth is the smoothed power list,
    freqedge, pow_smooth_edge are the smoothed with edges removed frequency and power arrays,
    peak_frq, peak_heights are the frequencies and heights of the peaks in the
    power spectrum. Star is the name of the star and output_dir is where the
    figure is saved to.
    """

    # Plotting the smoothed lightkurve power spectra###
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(freqs, pow_avg, linewidth=0.5)
    ax[0].set_title("Averaged power spectrum")
    ax[1].plot(freqs, pow_smooth, linewidth=0.5)
    ax[1].plot(freq_edge, pow_smooth_edge, "r-", linewidth=0.5)
    ax[1].plot(peak_frq, peak_heights, "x")
    ax[1].set_title("Smoothed averaged power spectrum")
    ax[1].set(
        xlabel=f"Frequency [{chr(956)}Hz]",
        ylabel=f"Power [ppm$^{str({2})}$/{chr(956)}Hz]",
    )
    ax[0].set(ylabel=f"Power [ppm$^{str({2})}$/{chr(956)}Hz]")
    fig.savefig(f"{output_dir}/smoothed.png")


def plot_pseudo_range(
    freqs, power, pseu_freqs, pseu_power, first_peak, star, output_dir
):
    """
    Plot pseudomode range. freqs and power are the smoothed (with edges
    removed) frequency and power arrays, pseu_freqs, pseu_power are the pseudomode
    range frequency and power lists, first_peak is index of where first peak in
    range occurs. star is the name of the star and output_dir is where the
    figure is saved to.
    """

    fig, ax = plt.subplots()
    ax.plot(freqs, power, "r-", linewidth=1)
    ax.plot(freqs[first_peak], power[first_peak], "x")
    ax.set(
        xlabel=f"Frequency [{chr(956)}Hz]",
        ylabel=f"Power [ppm$^{str({2})}$/{chr(956)}Hz]",
    )
    ax.set_title("Pseudomode range")
    for i in reversed(range(len(pseu_freqs))):
        ax.plot(pseu_freqs[i], pseu_power[i], linewidth=0.5)
    fig.savefig(f"{output_dir}/pseudo_range.png")


def plot_legendre(x, y, leg_val, detrended, order, ints, star, output_dir):
    """
    Plotting smoothed spectra in pseudomode range with legendre fits and detrending.
    x, y are x and y data, order is the polynomial order, ints is the list of
    integer p-mode spacings. star is the name of the star and output_dir is
    where the figure is saved to.
    """

    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(x, np.log(y), "r-", linewidth=0.5)
    ax[0].plot(x, leg_val, "--b", label=f"Order: {order}")
    ax[0].legend()
    ax[0].set_ylabel(f"Power [ppm$^{str({2})}$/{chr(956)}Hz]")
    ax[1].plot(x, detrended, "r-", linewidth=0.5, label=ints)
    ax[1].legend()
    ax[1].set_ylabel("Power detrended [A.U]")
    ax[1].set_xlabel(f"Frequency [{chr(956)}Hz]")
    ax[0].set_title("Legendre detrended pseudomode range")
    fig.savefig(f"{output_dir}/legendre_fit_int_{ints}.png")


def plot_power_spectrum(
    freqs, powers, peak_freqs, peak_heights, ints, star, output_dir
):
    """Plots the FT of power spectrum. freqs, powers are the input frequencies
    and powers. peak_freqs and peak_heights are the frequencies and heights of
    the peaks in the spectrum, ints is the list of integer p-mode spacings, star is the
    name of the star and output_dir is where the figure is saved to.
    """

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots()

    for i in range(len(freqs)):
        ax.plot(freqs[i], powers[i], label=ints[i], linewidth=0.7, c=cmap(i))
        ax.plot(peak_freqs[i], peak_heights[i], "x", color="tab:red")
        ax.set(xlabel=f"1/Frequency [{chr(956)}Hz$^{{-1}}$]", ylabel="Power [A.U.]")
        ax.legend()

    fig.savefig(f"{output_dir}/FT_power_spectrum.png")


def plot_autocorrelation(auto_dict, star, output_dir):
    """
    Plots the autocorrelation of the pseudomode range. auto_dict is the
    dictionary of autocorrelation values needed to generate plot.
    """

    fig, ax = plt.subplots()
    for i, a in auto_dict.items():
        lags_to_freq = a["lags_to_freq"]
        autocorr = a["auto_corr"]
        ac_peak_vals = a["ac_peak_vals"]
        ac_peak_heights = a["ac_peak_heights"]

        ax.plot(lags_to_freq, autocorr, linewidth=0.5, label=i)
        ax.plot(ac_peak_vals, ac_peak_heights, "x", color="tab:red")
        ax.set(
            title="Autocorrelation",
            xlabel=f"Lags [{chr(956)}Hz]",
            ylabel="Amplitude [A.U.]",
        )
        ax.legend()

    fig.savefig(f"{output_dir}/autocorrelation.png")


def plot_ac_linear_fit(auto_dict, star, output_dir):
    """Plots a linear fit to the peaks vs frequencies in the autocorrelation
    plot. auto_dict is the dictionary of autocorrelation values needed to
    genreate the plots
    """

    for i, a in auto_dict.items():
        fig, ax = plt.subplots(nrows=2, sharex=False)
        lags_to_freq = a["lags_to_freq"]
        autocorr = a["auto_corr"]
        ac_peak_vals = a["ac_peak_vals"]
        ac_peak_heights = a["ac_peak_heights"]
        zero_idx = a["zero_idx"]
        x = a["x_values"]
        y_fitted = a["y_fitted"]
        gradient = a["gradient"]
        errbar = a["err_bar"]

        ax[0].plot(
            lags_to_freq[zero_idx - 1 :],
            autocorr[zero_idx - 1 :],
            linewidth=0.5,
            label=i,
        )
        ax[0].plot(ac_peak_vals, ac_peak_heights, "x", color="tab:red")
        ax[0].set(xlabel=f"Lags [{chr(956)}Hz]", ylabel="Amplitude")
        ax[0].legend()

        ax[1].scatter(x, ac_peak_vals)
        label = f"Gradient: {round(gradient,1)} u\u00B1 {round(errbar,1)} {chr(956)}Hz"
        ax[1].plot(x, y_fitted, "--g", label=label)
        ax[1].set(xlabel="Peak", ylabel=f"Lags [{chr(956)}Hz]")
        ax[1].legend()
        fig.savefig(f"{output_dir}/ac_linear_fit_int_{i}.png")
