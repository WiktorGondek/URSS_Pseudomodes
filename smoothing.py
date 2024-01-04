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
    # Edges of the dataset are taken to remove sharp points from smoothing
    ps_edge = take_edges(convolved, percent)
    freq_edge = take_edges(frqs, percent)
    return ps_edge, freq_edge


def give_peaks(x, y, height, distance=None, threshold=None):
    """
    Return the peaks of a given dataset. x, y is the sample dataset, height and
    distance are arguments given in the find_peaks function.
    """

    peaks = find_peaks(y, height=height, threshold=threshold, distance=distance)
    peak_vals = [x[i] for i in peaks[0]]
    peak_heights = [i for i in peaks[1]["peak_heights"]]

    return peak_vals, peak_heights
