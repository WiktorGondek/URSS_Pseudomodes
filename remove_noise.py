#!/usr/bin/env python3


def subsection(lightcurve, timearr):
    """Create 4 day sections from full time series.
    lightcurve is lightcurve object,
    timearr is the time array from lightcurve object"""

    # truncate into 4 day sections into wholelst
    wholelst = []
    for i in range(int(timearr[0]), int(timearr[-1]), 4):
        sublist = lightcurve.truncate(i, i + 4)
        wholelst.append(len(sublist))

    # Find modal length of list of sections
    modelen = mode(wholelst)

    # Split lightcurve into sections of modelen
    subsec = []
    for j in range(0, len(lightcurve), modelen):
        subsec.append(lightcurve[j : j + modelen])
    # Remove final value from array (which is not of the same length?)
    subsec.pop(-1)
    return subsec


def signaltonoise(Arr, axis=0, ddof=0):
    """Obtain the signaltonoise ratio of each time series.
    Arr is array of sample data,
    axis is axis along which to operate,
    ddof is degrees of freedom correction for standard deviation"""

    Arr = np.asanyarray(Arr)
    me = Arr.mean(axis)
    sd = Arr.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, me / sd)


def removenoise(sections, N=None):
    """Remove noisy timeseries.
    sections is list of sections created from subsection(),
    N is number of noisy sections removed. If no N chosen, removes upper few from sorted signaltonoise list
    """

    # Appends signal to noise ratio value to a list called snrs
    snrs = []
    for i in range(len(sections)):
        # print(signaltonoise(subs[i].flux.value,axis=0,ddof=0))
        snrs.append(signaltonoise(sections[i].flux.value, axis=0, ddof=0))

    # Sort the snrs list in order of smallest to largest
    snrs = np.array(snrs)
    snrssort = np.sort(snrs)

    if N == None:
        ##Plotting noise vs time arrays##
        # currentfig = plt.gcf().number + 1
        # fig = plt.figure(currentfig)
        # x = np.linspace(1,len(snrssort),len(snrssort))
        # plt.axhline(mean(snrssort))
        # plt.axhline((mean(snrssort)+(0.08 * (max(snrssort) - min(snrssort)))))
        # plt.scatter(x,snrssort)
        # plt.xlabel('Sorted time arrays')
        # plt.ylabel('Noise')
        notnoisylst = []
        for i in snrssort:
            if i < (mean(snrssort) + (0.05 * (max(snrssort) - min(snrssort)))):
                notnoisylst.append(i)
        N = len(snrssort) - len(notnoisylst)

    print("Number of noisy sections removed: ", N)

    # Find the index of the last N number of noisiest signals
    noisy = []
    for i in range(len(snrssort) - N, len(snrssort)):
        tempidx = np.where(snrs == snrssort[i])[0][0]
        noisy.append(tempidx)

    # Find the index of the signals up to last series - N
    notnoisy = []
    for i in range(0, len(snrssort) - N):
        tempidx1 = np.where(snrs == snrssort[i])[0][0]
        notnoisy.append(tempidx1)
    notnoisy = np.sort(notnoisy)
    return notnoisy


###Obtain flux and time values from time series###
intensity = lc.flux.value
dt = lc.time.value
print(intensity, dt)

###Create 4 day subsections from lightkurve object###
subs = subsection(lc, dt)

###Remove noisy timeseries###
notnoisy = removenoise(subs)
