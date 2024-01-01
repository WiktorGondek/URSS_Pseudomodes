#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import legendre as L


def plot_pseudo_range(freq_edge, ps_edge, pseu_freqs, pseu_pow, first_peak):
    """
    Plot pseudomode range. freq_edge and ps_edge are the smoothed (with edges
    removed) frequency and power arrays, pseu_freqs, pseu_pow are the pseudomode
    range frequency and power lists first_peak is index of where first peak in
    range occurs.
    """
    currentfig = plt.gcf().number + 1
    fig = plt.figure(currentfig)

    plt.plot(freq_edge, ps_edge, "r-", linewidth=1)
    plt.plot(freq_edge[first_peak], ps_edge[first_peak], "x")
    plt.xlabel("Frequency [{}Hz]".format(chr(956)))
    plt.ylabel("Power [ppm$^{}$/{}Hz]".format(str({2}), chr(956)))
    for i in reversed(range(0, len(pseu_freqs))):
        plt.plot(pseu_freqs[i], pseu_pow[i], linewidth=0.5)


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
