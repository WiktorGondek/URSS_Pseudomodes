#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import legendre as L


def plot_time_series(lc, star):
    fig, ax = plt.subplots()
    lc.plot(ax=ax)
    fig.savefig(f"time_series_{star}.png")
    # fig.savefig(f"{outputdir}/time_series_{star}.png")


def plot_pseudo_range(freq_edge, ps_edge, pseu_freqs, pseu_pow, first_peak):
    """
    Plot pseudomode range. freq_edge and ps_edge are the smoothed (with edges
    removed) frequency and power arrays, pseu_freqs, pseu_pow are the pseudomode
    range frequency and power lists first_peak is index of where first peak in
    range occurs.
    """
    # currentfig = plt.gcf().number + 1
    # fig = plt.figure(currentfig)

    fig, ax = plt.subplots()
    ax.plot(freq_edge, ps_edge, "r-", linewidth=1)
    ax.plot(freq_edge[first_peak], ps_edge[first_peak], "x")
    ax.set_xlabel("Frequency [{}Hz]".format(chr(956)))
    ax.set_ylabel("Power [ppm$^{}$/{}Hz]".format(str({2}), chr(956)))
    ax.set(
        xlabel=f"Frequency [{chr(956)}Hz]",
        ylabel=f"Power [ppm$^{str({2})}$/{chr(956)}Hz]",
    )
    for i in reversed(range(len(pseu_freqs))):
        ax.plot(pseu_freqs[i], pseu_pow[i], linewidth=0.5)

    # plt.plot(freq_edge, ps_edge, "r-", linewidth=1)
    # plt.plot(freq_edge[first_peak], ps_edge[first_peak], "x")
    # plt.xlabel("Frequency [{}Hz]".format(chr(956)))
    # plt.ylabel("Power [ppm$^{}$/{}Hz]".format(str({2}), chr(956)))
    # for i in reversed(range(0, len(pseu_freqs))):
    #    plt.plot(pseu_freqs[i], pseu_pow[i], linewidth=0.5)


def plot_legendre(x, y, leg_val, detrended, order, ints):
    """
    Plotting smoothed spectra in pseudomode range with legendre fits and detrending.
    x,y are x and y data, order is the polynomial order, ints is the list
    integer p-mode spacings, plotting boolean, if True will plot
    """

    fig, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].plot(x, np.log(y), "r-", linewidth=0.5)
    axes[0].plot(x, leg_val, "--b")  # ,label='Order {}'.format(order))
    # axes[0].legend()
    axes[0].set_ylabel("Power [ppm$^{}$/{}Hz]".format(str({2}), chr(956)))
    axes[1].plot(x, detrended, "r-", linewidth=0.5, label=ints)
    axes[1].legend()
    axes[1].set_ylabel("Power detrended")
    # axes[1].set_xlabel("Frequency [{}Hz]".format(chr(956)))
    # axes[0].title("Detrended")
    plt.xlabel("Frequency [{}Hz]".format(chr(956)))
    plt.title("Detrended")


def plot_power_spectrum(ps_freqs, ps_powers, peak_freqs, peak_heights, ints):
    currentfig = plt.gcf().number + 1
    cmap = plt.get_cmap("tab10")

    # ps_peaks = find_peaks(powers, height)
    # peaks = [freqs[i] for i in ps_peaks[0]]
    # peak_heights = [i for i in peaks[1]["peak_heights"]]
    for i in range(len(ps_freqs)):
        fig = plt.figure(currentfig)
        plt.plot(ps_freqs[i], ps_powers[i], label=ints[i], linewidth=0.7, c=cmap(i))
        plt.plot(peak_freqs[i], peak_heights[i], "x", color="tab:red")
        # plt.xlabel('1/Frequency [{}Hz00b1]'.format(chr(956)))
        plt.xlabel("1/Frequency [{}Hz$^{}$]".format(chr(956), "{-1}"))
        # plt.xlabel('1/Frequency [{}Hz $\mathregular{^{-1}}$]'.format(chr(956)))
        plt.ylabel("Power")
        plt.legend()
