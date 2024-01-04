#!/usr/bin/env python3

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from statistics import mode, mean


def get_subsection(lightcurve, timearr):
    """
    Create 4 day sections from full time series. lightcurve is lightcurve
    object, timearr is the time array from lightcurve object
    """

    # Truncate into 4 day sections into whole_lst
    whole_lst = [
        len(lightcurve.truncate(i, i + 4))
        for i in range(int(timearr[0]), int(timearr[-1]), 4)
    ]

    # Find modal length of list of sections
    mode_len = mode(whole_lst)

    # Split lightcurve into sections of mode_len
    subsec = [lightcurve[j : j + mode_len] for j in range(0, len(lightcurve), mode_len)]

    # Remove final value from array (which is not of the same length?)
    subsec.pop(-1)
    return subsec


def signal_to_noise(Arr, axis=0, ddof=0):
    """
    Obtain the signaltonoise ratio of each time series.  Arr is array of sample
    data, axis is axis along which to operate, ddof is degrees of freedom
    correction for standard deviation
    """

    Arr = np.asanyarray(Arr)
    me = Arr.mean(axis)
    sd = Arr.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, me / sd)


def remove_noise(sections, N=None, plot_noise=False):
    """
    Remove noisy timeseries.  sections is list of sections created from
    get_subsection(), N is number of noisy sections removed. If no N chosen,
    removes upper few from sorted signaltonoise list
    """

    # Get signal to noise ratio values
    snrs = [signal_to_noise(sec.flux.value, axis=0, ddof=0) for sec in sections]

    # Sort the snrs list in order of smallest to largest
    snrs = np.array(snrs)
    snrs_sort = np.sort(snrs)

    percentage = 0.08
    noise_div = percentage * (max(snrs_sort) - min(snrs_sort))

    # Remove upper few from sorted signal_to_noise list
    if N == None:
        not_noisy_lst = [i for i in snrs_sort if i < (mean(snrs_sort) + noise_div)]
        N = len(snrs_sort) - len(not_noisy_lst)

    # print("Number of noisy sections removed: ", N)

    # Find the index of the last N number of noisiest signals
    noisy = [
        np.where(snrs == snrs_sort[i])[0][0]
        for i in range(len(snrs_sort) - N, len(snrs_sort))
    ]

    # Find the index of the signals up to last series - N
    not_noisy = [
        np.where(snrs == snrs_sort[i])[0][0] for i in range(0, len(snrs_sort - N))
    ]

    not_noisy = np.sort(not_noisy)
    return not_noisy
