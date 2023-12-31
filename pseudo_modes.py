#!/usr/bin/env python3

###Splitting time series into 4 day subsections, removing noisy data, and using lightkurve to get power spectrum###

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

# M: Suggestions
import sys # Access to system commands etc

###Setting matplotlib plot parameters###
plt.rcParams['figure.figsize'] = (7,5)
plt.rcParams['figure.dpi'] = 180
plt.rcParams['figure.autolayout'] = True

#############################################Reading textfile and importing#####################################


# M: start: To add flexibility call the script with star ID as an
# argument
if len(sys.argv)<2 :
    sys.exit('Use: {:s} <star_id>'.format(sys.argv[0]))
star = sys.argv[1]
#M: end

####Reading textfile####
###Reading text file to get the p-mode frequency spacing for this star###
# M: start: Wrapping up with a try statement avoid trying to open a file 
# that doesn't exist. Not necessary but usefull when you run your script
# automatically with large numbers of files.
try:
    with open('pmodespacing') as f:
        lines = f.readlines()
#M: ident dnu so that it is in the with section (if opening failed, won't
# try to access unattributed lines variable)
###p-mode frequency spacing###
        dnu = float(lines[0])
except IOError as error:
    sys.exit('file: {:s} does not exist.'.format('pmodespacing'))
# M: end

search_result = importstar(star)
#search_result = importstar('KIC 7799349')

###Returns stiched normalized timeseries from quarters listed above###
lc = search_result.download_all().stitch()
lc = lc.remove_nans('flux').remove_outliers().flatten(window_length = 501).normalize(unit='ppm').fill_gaps()

###Plot full timeseries###
lc.plot()


################################Spltting timeseries and removing noise###########################################
####Splitting timeseries into 4 day sections####

###Obtain flux and time values from time series###
intensity = lc.flux.value
dt = lc.time.value
            
###Create 4 day subsections from lightkurve object###
subs = subsection(lc,dt)

###Remove noisy timeseries###
notnoisy = removenoise(subs)

#####################################Power spectrum##############################################

###Creates list of power spectra from 4 day timeseries sections###
pgwhole = PS(subs,notnoisy,minfreq=920,maxfreq=1500)

###Plotting the power spectra###
#for i in range(0,len(pgwhole),20):
#    ax = pgwhole[i].plot()

###Adding the powers and frequencies from periodogram into a list to average###
powers, frequencies = addtolists(pgwhole)

###Averaging powers from lightkurve###
poweravg = average(powers)

###Plotting the averaged power spectrum###
# M: Interesting approach, I usually define figures beforehand, i.e.:
# fig,ax = plt.subplots(nrows=1,ncols=1) and so on for each fig. Your
# approach seems a bit more flexible depending on the cases.
currentfig = plt.gcf().number + 1
fig = plt.figure(currentfig)
plt.plot(frequencies[0],poweravg,linewidth=0.5)
plt.xlabel('Frequency [{}Hz]'.format(chr(956)))
plt.ylabel('Power [ppm2/{}Hz]'.format(chr(956)))

################################Smoothing#####################################


# M: for a bit more flexibility, you can define your figure outside
# of the function and pass the figure and axis as argument. This allows
# you to further customise it outside of the function, i.e.
# - def Plotsmoothed(fig,ax,frq,powavg,powsmooth,frqedge,powsmoothedge,pspeakfrq,pspeakheights):
# - create fig outside func: fig,ax = ...
# - the call: Pltsmoothed(fig,ax,...)
# - fiddle with the legend (for example):
#   handles, labels = ax.get_legend_handles_labels()
#   labels = labels.append('New label')
#   handles = handles.append(Line2D ...)
#   ax.legend(handles,labels)
###Smoothed and edges removed###
z,zedge,freqedge = removeedge(poweravg,frequencies[0],edges=0.04,box_width=5)

###Obtain the peaks from the power spectrum###
pspeaks = peakvalue(freqedge,zedge,height=average(zedge),dst=5)

###Plot the unsmoothed and smoothed power spectrum###
Plotsmoothed(frequencies[0],poweravg,z,freqedge,zedge,pspeaks[1],pspeaks[2])


#########################Pseudomoderange#######################################

###Defining the pseudomode range###
dnuidx,firstpeak,diff,pseufreqs,pseupower,integer = pseudorange(freqedge,zedge,pspeaks,dnu,initialint=5,finalint=7)


###Plotting the pseudomode range###
plotpseudorange(freqedge,zedge,pseufreqs,pseupower,firstpeak)


#####################Detrending the pseudomoderange############################

###Detrended pseudomode range###
detrendedlk,freqrescale = Detrend(pseufreqs,pseupower,integer,plotting=True)

################################################PSxPS#########################################################

###PSxPS###
pspsfreqs,pspspows = PSxPS(detrendedlk,freqrescale)
pspspeaks,integers = PSxPSpeaks(pspsfreqs,pspspows,integer,height=0.2,plotting=True)
pspspeakfreqs = getpspspeakfreq(pspspeaks)
peaksandidx = list(zip(integers,pspspeakfreqs))
print('PSxPS peak frequency: ', *peaksandidx)
print('\n')


################################Autocorrelation################################################################

   

autocorrelation(detrendedlk,diff,integer,height=-0.2,tshold=0.0,dst=7,together=True,smoothing=True,plottingac=True,plottingfit=True)


plt.show()
