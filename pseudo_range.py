#!/usr/bin/env python3

import numpy as np
from numpy.polynomial import legendre as L
import matplotlib.pyplot as plt


def pseudo_range(freq_edge, ps_edge, peak_freqs, delta_nu, init_int, fin_int):
    """
    Define the pseudomode range. freq_edge ,ps_edge are the smoothed with
    edges removed frequency and power arrays, peak_freqs gives the peak
    frequencies, delta_nu is the p-mode spacing, init_int and
    fin_int are the integer multiples of the p-mopde spacing
    """

    # Getting the p-mode frequency spacing as number of frequency data points
    diff = freq_edge[1] - freq_edge[0]  # frequency spacing
    dnuidx = int(delta_nu / diff)  # p-mode spacing frequency in length of index

    # m is the the fraction of p-mode frequency spacing
    m = 0.2
    j = 0
    # Finding where the first peak is in the power spectrum
    first_peak = np.where(freq_edge == peak_freqs[j])[0][0]
    # Start of the pseudomode range
    start = first_peak - (int(m * dnuidx))
    # Loop to ensure that (first peak - fraction of spacing) exists
    if start < 0:
        while j < len(peak_freqs):
            first_peak = np.where(freq_edge == peak_freqs[j])[0][0]
            start = first_peak - (int(m * dnuidx))
            j = j + 1
            if start >= 0:
                break
        if start < 0:
            raise ValueError("No suitable peak found")

    # List of pseudomode range frequency, powers, and integer number of p-mode
    # frequency spacings added from start of the range
    pseudo_freqs = [
        freq_edge[start : start + (i * dnuidx)] for i in range(init_int, fin_int + 1)
    ]
    pseudo_power = [
        ps_edge[start : start + (i * dnuidx)] for i in range(init_int, fin_int + 1)
    ]
    integer = [i for i in range(init_int, fin_int + 1)]

    pseudo_dict = {
        "dnu_idx": dnuidx,
        "first_peak": first_peak,
        "diff": diff,
        "freqs": pseudo_freqs,
        "power": pseudo_power,
        "integer": integer,
    }
    return pseudo_dict


def plotpseudorange(freq_edge, ps_edge, pseu_freqs, pseu_pow, first_peak):
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


#####################Detrending the pseudomoderange############################


def Legendredetrend(x, y, order, ints, plotting):
    """
    Plotting smoothed in pseudomode range with legendre fits and detrending.
    x,y are x and y data, order is the polynomial order, ints is the list
    integer p-mode spacings, plotting boolean, if True will plot
    """

    # Making legendre fit
    leg_fit = L.legfit(x, np.log(y), deg=order)
    leg_val = L.legval(x, leg_fit)
    detrended = np.log(y) - leg_val

    if plotting == True:
        fig, axes = plt.subplots(nrows=2, sharex=True)
        axes[0].plot(x, np.log(y), "r-", linewidth=0.5)
        axes[0].plot(x, leg_val, "--b")  # ,label='Order {}'.format(order))
        # axes[0].legend()
        axes[0].set_ylabel("Power [ppm$^{}$/{}Hz]".format(str({2}), chr(956)))
        axes[1].plot(x, detrended, "r-", linewidth=0.5, label=ints)
        axes[1].legend()
        axes[1].set_ylabel("Power detrended")
        plt.xlabel("Frequency [{}Hz]".format(chr(956)))
        plt.title("Detrended")
    return detrended


# Obtaining the detrended pseudomode range
def detrend(pseu_freqs, pseu_powers, ints, plotting):
    """
    Obtaining the detrended pseudomode range. pseu_freqs,pseu_powers are the
    pseudomode range frequency and power lists, ints is list of integer p-mode
    spacings, plotting noolean, if True will plot
    """
    # Set the polynomial order
    polyorder = 2

    # Detrend the pseudomode range
    detrended = [
        Legendredetrend(pseu_freqs[i], pseu_powers[i], 2, ints[i], plotting=plotting)
        for i in range(len(pseu_freqs))
    ]

    # Rescaling the frequencies into Hz from muHz to be able to FT using LombScargle
    freqrescale = [i * 1e-6 for i in pseu_freqs]

    return detrended, freqrescale
