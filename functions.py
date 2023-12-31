#!/usr/bin/env python3


import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from astropy.convolution import Box1DKernel, convolve
from numpy.polynomial import legendre as L
from astropy import units as u
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks, correlate, correlation_lags
from scipy import stats
from statistics import mode, mean
import sys 

def importstar(star):
    '''Import sun-like star'''
    search_result = lk.search_lightcurve(star,
                                     cadence='short',
                                     author='Kepler',
                                     quarter=(5,6,7,8,9,10,11,12,13,14,15,16,17))
    return search_result


def subsection(lightcurve,timearr):
    '''Create 4 day sections from full time series.
    lightcurve is lightcurve object,
    timearr is the time array from lightcurve object'''
    wholelst = []
    for i in range(int(timearr[0]),int(timearr[-1]),4):
        sublist = lightcurve.truncate(i,i+4)
        wholelst.append(len(sublist))
    modelen = mode(wholelst) #Find modal length of list of sections
    subsec = []
    for j in range(0,len(lightcurve),modelen):
        subsec.append(lightcurve[j:j+modelen]) #Split lightcurve into sections of modelen
    subsec.pop(-1) #Remove final value from array (which is not of the same length?)
    return subsec

def signaltonoise(Arr, axis=0, ddof=0):
    '''Obtain the signaltonoise ratio of each time series.
    Arr is array of sample data,
    axis is axis along which to operate,
    ddof is degrees of freedom correction for standard deviation'''
    Arr = np.asanyarray(Arr)
    me = Arr.mean(axis)
    sd = Arr.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, me/sd)

def removenoise(sections,N=None):
    '''Remove noisy timeseries.
    sections is list of sections created from subsection(),
    N is number of noisy sections removed. If no N chosen, removes upper few from sorted signaltonoise list'''

    ###Appends signal to noise ratio value to a list called snrs###
    snrs = []
    for i in range(len(sections)):
        #print(signaltonoise(subs[i].flux.value,axis=0,ddof=0))
        snrs.append(signaltonoise(sections[i].flux.value,axis=0,ddof=0))

    ###sort the snrs list in order of smallest to largest
    snrs = np.array(snrs)
    snrssort = np.sort(snrs)
    if N == None:
        ##Plotting noise vs time arrays##
        #currentfig = plt.gcf().number + 1
        #fig = plt.figure(currentfig)
        #x = np.linspace(1,len(snrssort),len(snrssort))
        #plt.axhline(mean(snrssort))
        #plt.axhline((mean(snrssort)+(0.08 * (max(snrssort) - min(snrssort)))))
        #plt.scatter(x,snrssort)
        #plt.xlabel('Sorted time arrays')
        #plt.ylabel('Noise')
        notnoisylst = []
        for i in snrssort:
            if i < (mean(snrssort)+(0.05 * (max(snrssort) - min(snrssort)))):
                notnoisylst.append(i)
        N = len(snrssort)-len(notnoisylst)

    print('Number of noisy sections removed: ', N)

    ###Find the index of the last N number of noisiest signals###
    noisy = []
    for i in range(len(snrssort)-N,len(snrssort)):
        tempidx = np.where(snrs == snrssort[i])[0][0]
        noisy.append(tempidx)

    ###Find the index of the signals up to last series - N###
    notnoisy = []
    for i in range(0,len(snrssort)-N):
        tempidx1 = np.where(snrs == snrssort[i])[0][0]
        notnoisy.append(tempidx1)
    notnoisy = np.sort(notnoisy)
    return notnoisy

def PS(subts,notnoisyts,minfreq,maxfreq):
    '''Perform FT of least noist 4 day timeseries using to_periodogram().
    subts are the subsections,
    notnoisyts is list of the index of least noisy sections,
    minfreq and maxfreq set the min and max frequencies'''

    ##pgwhole is a list of periodogram objects##
    pgwhole = []
    for i in range(0,len(notnoisyts)):
        pgnew = subts[notnoisyts[i]].to_periodogram(normalization ='psd',
                                        minimum_frequency=minfreq,
                                        maximum_frequency=maxfreq)
        pgwhole.append(pgnew)
    return pgwhole

def average(lst):
    '''Take average'''
    lstsum = sum(lst)
    length = len(lst)
    lstavg = lstsum / length
    return lstavg

def addtolists(ps):
    '''Add power and frequency arrays to lists from lightcurve object.
    ps is the list of power spectra objects'''
    pows = []
    freqs = []
    for i in range(0,len(ps)):
        pows.append(ps[i].power.value)
        freqs.append(ps[i].frequency.value)
    return pows,freqs

def smooth(power,width):
    '''Smooth power spectrum with 1d box kernel.
    power is the power list, 
    width is width of box kernel'''
    box = Box1DKernel(width)
    conv = convolve(power,box, boundary='wrap', nan_treatment='interpolate')
    return conv

def takeedges(x,percent):
    '''Remove some percentage of full range from the edges of data.
    x is the list of sample data
    percent is the percentage of the length of list to remove from each end'''
    length = int(len(x) * percent)
    newx = x[0 + length: len(x) - length]
    return newx

def removeedge(avgpwr,frqs,edges,box_width):
    '''Smooth and remove edges.
    avgpor is the average power list,
    frqs is the frequency list,
    edges sets percentage to remove from the ends,
    box_width sets the width of the box kernel for smoothing'''
    z = smooth(avgpwr,box_width)
    zedge = takeedges(z,edges)
    freqedge = takeedges(frqs,edges)
    return z,zedge,freqedge

def peakvalue(x,y,height,tshold=None,dst=None,prmnce=None,wdth=None):
    '''Return the frequency of peak value.
    [0] is index, [1] is the x value, [2] is the y value'''
    xpeakidx = []
    xpeakval = []
    xpeakheight = []
    peaks = find_peaks(y,height,threshold=tshold,distance=dst,prominence=prmnce,width=wdth)
    for i in peaks:
        xpeakidx.append(i)
    for j in xpeakidx[0]:
        xpeakval.append(x[j])
    for k in peaks[1]['peak_heights']:
        xpeakheight.append(k)
    #xpeakheightsort = np.sort(xpeakheight)
    return [xpeakidx[0],xpeakval,xpeakheight]#xpeakheightsort[0]]

def Plotsmoothed(frq,powavg,powsmooth,frqedge,powsmoothedge,pspeakfrq,pspeakheights):
    '''Plot unsmoothed and smoothed power spectra showing peaks.
    frq,powavg are the frequency and power lists,
    powsmooth is the smoothed power list,
    freqedge, powsmoothedge are the smoothed with edges removed frequency and power arrays,
    pspeakfrq, pspeakheights are the frequencies and heights of the peaks in the power spectrum.'''
    ###Plotting the smoothed lightkurve power spectra###    
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(frq,powavg,linewidth=0.5)
    #axs[0].set_title('Unsmoothed averaged lk power spectrum')
    axs[1].plot(frq,powsmooth,linewidth=0.5)
    axs[1].plot(frqedge,powsmoothedge,'r-',linewidth=0.5)
    axs[1].plot(pspeakfrq,pspeakheights,'x')
    #axs[1].set_title('Smoothed averaged lk power spectrum')
    plt.xlabel('Frequency [{}Hz]'.format(chr(956)))
    fig.supylabel('Power [ppm$^{}$/{}Hz]'.format(str({2}),chr(956)))
    #axes[0].set_ylabel('Power [ppm$^{}$/{}Hz]'.format(str({2}),chr(956)))

def pseudorange(frqedge,powsmoothedge,pspeakfrq,deltanu,initialint,finalint):
    '''Define the pseudomode range.
    freqedge,powsmoothedge are the smoothed with edges removed frequency and power arrays,
    pspeakfrq gives the peak index array and peak frequencyes,
    initialint and finalint are the integer for multiples of the p-mopde spacing'''
     
    ###Getting the p-mode frequency spacing as number of frequency data points###
    diff = frqedge[1] - frqedge[0] #frequency spacing
    dnuidx = int(deltanu/diff) #p-mode spacing frequency in length of index

    ###m will be the fraction of p-mode frequency spacing###
    m = 0.2
    j = 0
    ###Finding where the first peak is in the power spectrum###
    firstpeak = np.where(frqedge == pspeakfrq[1][j])[0][0]
    ###Start of the pseudomode range###
    start = firstpeak - (int(m*dnuidx))
    ###Loop to ensure that (first peak - fraction of spacing) exists###
    if start < 0:
# M: you risk getting out of your array doing so. If no j satisfies the
# the condition firstpeak - (int(m*dnuidx)) >=0, you'll exceed the table
# dimension. Maybe, change while j>=0 by while j<len(pspeakfrq[1]) and 
# add a check after the loop. e.g. if start < 0: sys.exit('No suitable 
# peak found')
        while j >= 0:
            firstpeak = np.where(frqedge == pspeakfrq[1][j])[0][0]
            start = firstpeak - (int(m*dnuidx))
            j = j + 1
            if start >= 0:
                break
    ###List of psueodomode range frequency, powers, and the integer number of p-mode frequency spacings added from start of the range###
    pseufreqs = []
    pseupower = []
    integer = []
    ###Range is defined as the number of integer p-mode spacing added from start###
    for i in range(initialint,finalint+1):
        tempfreqs = frqedge[start:start+(i*dnuidx)]
        temppower = powsmoothedge[start:start+(i*dnuidx)]
        pseufreqs.append(tempfreqs)
        pseupower.append(temppower)
        integer.append(i)
    return dnuidx,firstpeak,diff,pseufreqs,pseupower,integer

def plotpseudorange(frqedge,powsmoothedge,pseufrqs,pseupow,firstpeak):
    '''Plot pseudomode range.
    freqedge and powsmoothedge are the smoothed with edges removed frequency and power arrays,
    pseufrqs,pseupow are the pseudomode range frequency and power lists,
    firstpeak is index of where first peak in range occurs'''
    currentfig = plt.gcf().number + 1
    fig = plt.figure(currentfig)
    plt.plot(frqedge,powsmoothedge,'r-',linewidth=0.5)
    plt.plot(frqedge[firstpeak],powsmoothedge[firstpeak],'x')
    plt.xlabel('Frequency [{}Hz]'.format(chr(956)))
    plt.ylabel('Power [ppm$^{}$/{}Hz]'.format(str({2}),chr(956)))
    for i in reversed(range(0,len(pseufrqs))):
        plt.plot(pseufrqs[i],pseupow[i],linewidth=0.5)

def Legendredetrend(x,y,order,ints,plotting):
    '''Plotting smoothed in pseudomode range with legendre fits and detrending.
    x,y are x and y data,
    order is the polynomial order,
    ints is the list integer p-mode spacings,
    plotting boolean, if True will plot'''
    ##Making legendre fit##
    leg_fit = L.legfit(x, np.log(y), deg = order)
    leg_val = L.legval(x,leg_fit)
    detrended = np.log(y) - leg_val
    if plotting == True:
        fig, axes = plt.subplots(nrows = 2, sharex=True)
        axes[0].plot(x, np.log(y),"r-",linewidth=0.5)
        axes[0].plot(x, leg_val, "--b")#,label='Order {}'.format(order))
        #axes[0].legend()
        axes[0].set_ylabel('Power [ppm$^{}$/{}Hz]'.format(str({2}),chr(956)))
        axes[1].plot(x,detrended, "r-", linewidth = 0.5,label=ints)
        axes[1].legend()
        axes[1].set_ylabel('Power detrended')
        plt.xlabel('Frequency [{}Hz]'.format(chr(956)))
        #plt.title('Detrended')
    return detrended

###Obtaining the detrended pseudomode range###
def Detrend(pseufrqs,pseupows,ints,plotting):
    '''Obtaining the detrended pseudomode range.
    pseufrqs,pseupows are the pseudomode range frequency and power lists,
    ints is list of integer p-mode spacings,
    plotting noolean, if True will plot'''
    polyorder = 2
    detrendedlk = []
    freqrescale = []
    for i in range(0,len(pseufrqs)):
        detrendedlk.append(Legendredetrend(pseufrqs[i],pseupows[i],2,ints[i],plotting=plotting))
        
        ###Rescaling the frequencies into Hz from muHz to be able to FT using LombScargle###
        freqrescale.append(pseufrqs[i] * 1e-6)
    ###Additional detrending with different order if needed###
    #for i in range(5,len(pseufrqs)):
    #    detrendedlk.append(Legendredetrend(pseufreqs[i],pseupower[i],5,ints[i],plotting=True))
    #    freqrescale.append(pseufreqs[i] * 1e-6) 
    return detrendedlk,freqrescale

def PSxPS(detrended,freqrs):
    '''Perform FT on pseudomode range.
    detrended is the detrended power list,
    freqrs is the rescaled frequency list in units of Hz'''
    pspsfreqs = []
    pspspows = []
    ###FT of power spectrum in pseudomode region using LombScargle###
    for i in range(0,len(detrended)):
        pspsfreq, pspspow = LombScargle(freqrs[i], detrended[i]).autopower(samples_per_peak=1,nyquist_factor=1)
        pspsfreqs.append(pspsfreq)
        pspspows.append(pspspow)
        print(pspsfreqs[i][1]-pspsfreqs[i][0])
        print(pspsfreqs[i][5]-pspsfreqs[i][4])
    return pspsfreqs,pspspows

def PSxPSpeaks(psxpsfrqs,psxpspws,ints,height,plotting):
    '''Plotting PSxPS and finding peaks.
    psxpsfrqs,psxpspws are the frequencies and powers of PSxPS,
    ints is list of integer p-mode spacings,
    height to set, above this value peaks will be found.'''
    pspspeaks = []
    pspsheights = []
    integers = []
    currentfig = plt.gcf().number + 1
# M: Nice! I usually define colors I like and loop over them using the
# modulo function to remain within the range. e.g.:
# colors=['r','g','b'], c=colors[i%len(colors)]
    #colors1=['tab:blue','tab:green','tab:orange','tab:purple',
    #        'tab:pink','tab:olive','tab:cyan','tab:brown','tab:gray'] 
    #colors=['b','g','c','m','y']
    #color = iter(cm.rainbow(np.linspace(0,1,len(psxpsfrqs))))
    cmap = plt.get_cmap("tab10")
    for i in range(len(psxpsfrqs)):
        #c=colors[i%len(colors)]
        #c = next(colors)
        pspspeak = peakvalue(psxpsfrqs[i],psxpspws[i],height)
        for j in pspspeak[1]:
            pspspeaks.append(j)
            integers.append(ints[i])
        for k in pspspeak[2]:
            pspsheights.append(k)
        if plotting == True:
            fig = plt.figure(currentfig)
            plt.plot(psxpsfrqs[i],psxpspws[i],label=ints[i],linewidth=0.7,c=cmap(i))
            plt.plot(pspspeaks,pspsheights,'x', color='tab:red')
            #plt.xlabel('1/Frequency [{}Hz00b1]'.format(chr(956)))
            plt.xlabel('1/Frequency [{}Hz$^{}$]'.format(chr(956),'{-1}'))
            #plt.xlabel('1/Frequency [{}Hz $\mathregular{^{-1}}$]'.format(chr(956)))
            plt.ylabel('Power')
            plt.legend()
            #plt.title('PSxPS')
    print(pspspeaks)
    return pspspeaks,integers

###Get frequency from PSxPS###
def getpspspeakfreq(peakslist):
    '''Get frequency from PSxPS.
    peakslist is list of peak x-axis values'''
    peakfreqs = []
    for i in range(len(peakslist)):
        peakfreq = (1/peakslist[i]) 
        peakfreqs.append(peakfreq)
    return peakfreqs

###Function to return the frequency value for autocorrelation###
def getacfreq(peakslist):
    '''Return the frequency value for autocorrelation.
    peakslist is list of peak x-axis values.'''
    freqvals = []
    diff = []
    for i in range(0,len(peakslist)-1):
        diff.append(peakslist[i+1]-peakslist[i])
    for j in diff:
        freqvals.append(j)
    return freqvals

###Autocorrelation###
def autocorrelation(detrended,difference,ints,height,tshold,dst,together,smoothing,plottingac,plottingfit):
    '''detrended is the detrended power list,
    difference is the frequency spacing,
    ints is list of integer p-mode spacings,
    height to set, above this value peaks will be found,
    tshold,dst are threshold and distance for peaks,
    together boolean, if True will plot all autocorrelation plots together,
    smoothing boolean, if True will smooth the autocorrelation plot,
    plottingac boolean, if True will plot the autocorrelation plots,
    plottingfit boolean, if True will plot the linear fit to autocorrelation plots.'''

    autocorr = []
    lags = []
    lagstofreq = []
    zeroidx = []
    acpeakidxs = []
    acpeakvals = []
    acpeakheights = []
    delta_nus = []
    currentfig = plt.gcf().number
    ###Creating autocorrelation###
    for i in range(len(detrended)):
        if smoothing == True:
            autocorr.append(smooth(correlate(detrended[i],detrended[i],mode='full'),5))
        else:
            autocorr.append(correlate(detrended[i],detrended[i],mode='full'))
        lags.append(correlation_lags(len(detrended[i]),len(detrended[i])))
        lagstofreq.append(lags[i] * difference)
        zeroidx.append(np.where(lagstofreq[i]==0)[0][0])
        acpeaks = peakvalue(lagstofreq[i][zeroidx[i]-1:],autocorr[i][zeroidx[i]-1:],height,tshold,dst)
        acpeakidxs.append(acpeaks[0])
        acpeakvals.append(acpeaks[1])
        acpeakheights.append(acpeaks[2])
        delta_nus.append(getacfreq(acpeakvals[i]))
    if plottingac == True:
        if together == True:
            for i in range(len(detrended)):
                plt.figure(currentfig+1)
                plt.plot(lagstofreq[i],autocorr[i],linewidth=0.5,label=ints[i])
                plt.plot(acpeakvals[i],acpeakheights[i],'x',color='tab:red')
                #plt.title('Autocorrelation')
                plt.xlabel('Lags [{}Hz]'.format(chr(956)))
                plt.ylabel('Amplitude')    
                plt.legend()
        else:
            for i in range(0,len(detrended)):
                fig = plt.figure(currentfig + 1 + i)
                plt.plot(lagstofreq[i],autocorr[i],linewidth=0.5,label=ints[i])
                plt.plot(acpeakvals[i],acpeakheights[i],'x',color='tab:red',linewidth=0.5)
                #plt.title('Autocorrelation')
                plt.xlabel('Lags [{}Hz]'.format(chr(956)))
                plt.ylabel('Amplitude')       
                plt.legend()
    ###Printing the peak frequencies and frequency differences###        
    for i in range(0,len(detrended)):
        print('integer: ', integer[i]) 
        print('AC peak frequencies: ', acpeakvals[i])
        print('AC peak frequency differences: ', delta_nus[i])
        print('\n')
    delta_nusavg = []
    gradients = []
    gradientserr = []
    ###Making linear fit to data###
    for i in range(0,len(delta_nus)):
        if len(delta_nus[i]) == 1:
            continue
        else:
            x = np.linspace(1,len(acpeakvals[i]),len(acpeakvals[i]))     
            p, cov = np.polyfit(x,acpeakvals[i],1,cov=True)
            a = p[0]
            b = p[1]
            errbar = np.sqrt(np.diag(cov)[0])
            gradients.append(a)
            gradientserr.append(errbar)
            print("integer: {}, gradient: {} +/- {}".format(ints[i],a,errbar))
            y_fitted = (a * x) + b
            if plottingfit == True:
                fig, axes = plt.subplots(nrows = 2, sharex=False)
                axes[0].plot(lagstofreq[i][zeroidx[i]-1:],autocorr[i][zeroidx[i]-1:],linewidth=0.5,label=ints[i]) 
                axes[0].plot(acpeakvals[i],acpeakheights[i],'x',color='tab:red') 
                axes[0].set_xlabel('Lags [{}Hz]'.format(chr(956)))
                axes[0].set_ylabel('Amplitude')
                axes[0].legend()
                axes[1].scatter(x,acpeakvals[i])
                label = "Gradient: {} {} {} {}Hz".format(round(p[0],1),u"\u00B1",round(errbar,1),chr(956))
                axes[1].plot(x,y_fitted,'--g',label=label)
                axes[1].set_xlabel('Peak')
                axes[1].set_ylabel('Lags [{}Hz]'.format(chr(956)))
                axes[1].legend()
            delta_nusavg.append(average(delta_nus[i]))
    gradientsavg = average(gradients)
    ###Printing averages values###
    print('Averaged AC peak frequency differences: ', delta_nusavg)
    print('Averaged AC peak frequency gradient:', gradientsavg)
    print('\n')
    return gradients,gradientserr
 
