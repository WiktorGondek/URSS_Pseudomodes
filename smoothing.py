#!/usr/bin/env python3

from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def take_edges(x, percent):
    """
    Remove some percentage of full range from the edges of data,
    x is the list of sample data, percent is the percentage of the length
    of list to remove from each end
    """

    length = int(len(x) * percent)
    newx = x[0 + length : len(x) - length]
    return newx


def remove_edge(convolved, frqs, percent):
    """
    Smooth and remove edges. avgpor is the average power list,
    frqs is the frequency list, edges sets percentage to remove from the ends,
    box_width sets the width of the box kernel for smoothing
    """
    ps_edge = take_edges(convolved, percent)
    freq_edge = take_edges(frqs, percent)
    return ps_edge, freq_edge


def give_peaks(x, y, height, distance=None):
    peaks = find_peaks(y, height=height, distance=distance)
    peak_vals = [x[i] for i in peaks[0]]
    peak_heights = [i for i in peaks[1]["peak_heights"]]

    return peak_vals, peak_heights


def plot_smoothed(
    frq, pow_avg, pow_smooth, frq_edge, pow_smooth_edge, peak_frq, peak_heights
):
    """Plot unsmoothed and smoothed power spectra showing peaks.
    frq,pow_avg are the frequency and power lists,
    pow_smooth is the smoothed power list,
    freqedge, pow_smooth_edge are the smoothed with edges removed frequency and power arrays,
    peak_frq, peak_heights are the frequencies and heights of the peaks in the power spectrum.
    """

    # Plotting the smoothed lightkurve power spectra###
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(frq, pow_avg, linewidth=0.5)
    axs[0].set_title("Averaged power spectrum")
    axs[1].plot(frq, pow_smooth, linewidth=0.5)
    axs[1].plot(frq_edge, pow_smooth_edge, "r-", linewidth=0.5)
    axs[1].plot(peak_frq, peak_heights, "x")
    axs[1].set_title("Smoothed averaged power spectrum")
    plt.xlabel("Frequency [{}Hz]".format(chr(956)))
    fig.supylabel("Power [ppm$^{}$/{}Hz]".format(str({2}), chr(956)))
    # axes[0].set_ylabel('Power [ppm$^{}$/{}Hz]'.format(str({2}),chr(956)))
