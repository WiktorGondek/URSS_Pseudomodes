#!/usr/bin/env python3 


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


# M: for a bit more flexibility, you can define your figure outside
# of the function and pass the figure and axis as argument. This allows
# you to further customise it outside of the function, i.e.
# - def Plotsmoothed(fig,ax,frq,powavg,powsmooth,frqedge,powsmoothedge,pspeakfrq,pspeakheights):
# - create fig outside func: fig,ax = ...
# - the call: Pltsmoothed(fig,ax,...)
