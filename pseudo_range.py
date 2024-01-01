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


def legendre_detrend(x, y, order):
    """
    Detrend data with Legendre polynomial fit. x, y is data to be input, order
    is the order of legendre polynomial.
    """

    # Making legendre fit
    leg_fit = L.legfit(x, np.log(y), deg=order)
    leg_val = L.legval(x, leg_fit)
    detrended = np.log(y) - leg_val

    detrend_dict = {"leg_fit": leg_fit, "leg_val": leg_val, "detrended": detrended}

    return detrend_dict
