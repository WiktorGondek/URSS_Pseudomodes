#!/usr/bin/env python3


def create_power_spectra(sub_ts, notnoisy_ts, min_freq, max_freq):
    """
    Perform FT of least noisy 4 day timeseries using to_periodogram().
    sub_ts are the subsections, not_noisy_ts is list of the index of least noisy sections,
    min_freq and max_freq set the min and max frequencies
    """

    # pg_whole is a list of periodogram objects
    pg_whole = [
        sub_ts[
            not_noisy_ts[i].to_periodogram(
                normalization="psd",
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
            )
        ]
        for i in range(0, len(not_noisy_ts))
    ]
    return pg_whole


def average(lst):
    """
    Take average
    """
    
    lst_sum = sum(lst)
    length = len(lst)
    lst_avg = lst_sum / length
    return lst_avg


def add_to_lists(ps):
    """
    Add power and frequency arrays to lists from lightcurve object.
    ps is the list of power spectra objects
    """
    pows = [ps[i].power.value for i in range(0, len(ps))]
    freqs = [ps[i].frequency.value for i in range(0, len(ps))]
    # both = np.array([(ps[i].power.value, ps[i].frequency.value) for ps_i in ps])
    return pows, freqs


