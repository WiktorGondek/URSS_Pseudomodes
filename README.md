
# Detecting high-frequency modes in solar-like stars

This is the work I made for my project as a student researcher at the
University of Warwick at the Centre for Fusion, Space, and Astrophysics. The
work aims to narrow down the pseudomode range of solar-like star and determine
its frequency via (i) Performing a FT of the power spectrum in the pseudomode
range and (ii) Calculating the autocorrelation.

## Setup

To install the required python packages:
```
pip install -r requirements.txt
```

## Example 

To get data and plots for a given star, run: 
```
./main.py 'KIC 7799349' 33.779 output_dir
```
which will generate and save plots from the star `'KIC 7799349'` with p-mode
spacing `33.779` in the `output_dir`.

