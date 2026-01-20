#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlotRevisedNsoae4.py [2025.08.12ff]

Purpose: Using human SOAE peak freqs as determined via EXplotSOAEwfP13B.py
(see notes below on params), this code calculates various related quantities
(e.g. Nsoae) and also create a "Fig.4" panels for the coherence paper that shows how
Nsoae estimates can change for peak-picking from standard spectrally-avgd.
magnitudes versus xi-adjusted phase time-avgd. mags (as per EXplotSOAEwfP13B.py)
    
-------
NOTES

o [2025.08.25: also plots human Nsoae versus Nxi

o [2025.08.13ff] For v.4, using a slightly modified that allowed me to plot the entire raw
wf and visually identify if/when there were large artifacts (e.g., a cough). In 
cases where things were localized to a segment of time, I made shorter waveforms
that excluded those regions so to have a more artifact-free waveform, even if
shorter. These have the suffix "short". I analyzed those in a similar fashion 
to see if there were any obvious changes in SOAE peaks (as described below). Overall,
this re-analysis had a subtle (though non-trivial) effect, nothing major though. 
In general, getting rid of artifacts seems to = good thing to do though.

o v.4 tweaks SOAE peak freq. vals to be a bit more conservative and 
verifiable via the "FIG.21" in EXplotSOAEwfP13B.py, as well as display
some basic stats back

o To create a string that can then be used to point towards a variable, 
use the eval command as follows:
    > # eval('name'+str(some#)+'anythingElse')
o Converting SOAE waveform files from .txt to .mat (i.e., from ascii to
 binary to reduce memory sans compression) is done via Matlab as follows:
    >  wf= load('filename.txt');  
    > save('filename','wf')                                                



Human subjects/waveforms used (all should be .mat files):
    > human_TH14RearwaveformSOAE
    > human_RRrearSOAEwf1
    > human_TH13RearwaveformSOAE
    > human_KClearSOAEwf2
    > human_AP7RearwaveformSOAE
    > human_coNW_fgF090728R
    > human_TH21RearwaveformSOAE
    > human_AVGrearSOAEwf2
    > human_FMlearSOAEwfA01
    > human_JBrearSOAEwf2
    > human_LSrearSOAEwf1   [NOTE: used by Seth for colossogram analysis]
    > human_JIrearSOAEwf2 
    
NOTE: Subject human_AWrearSOAEwf2 could be included


Created on Tue Jun 17 17:04:16 2025
@author: pumpkin
Christopher Bergevin 
cberge@yorku.edu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from helper_funcs import get_human_peak_freqs


# ======================================================
binsN= 37    # number of bins for Nsoae histogram {27}

N = 1000   # number of times to bootstrap re xi-adjusted power law fit {25?}
ratioBinCNT= 200   # numb. of bins for freq. ratio histogram

fName= '../Fig0X - N_xi and tuning comps/N_xi Fitted Parameters (rho=0.7, PW=True)E.xlsx'
fact= 1/(4*np.pi);  # scaling factor for all Nxi {1/4pi}
# ======================================================

# -----------------------------
# function to extract Nsoae and geomean freq from SOAE peak array
def computeNsoae(arr):
    tempARR= arr
    cnt= 0
    M= len(tempARR)  # numb. of peaks
    Mint= M-1   # only consider adjacent peak pairs
    nsoae= np.empty([Mint])
    geofreq= np.empty([Mint])
    # --- only adjacent neighboring pairs
    for nn in range(0,M-1):
        fL= tempARR[nn]*1000 # pick off lowest freq. yet to analyze  [Hz]
        fH= tempARR[nn+1]*1000  # higher freq. of the pair [Hz]
        freqGM= np.sqrt(fH*fL)  # geometric mean freq.
        freqDiff= fH-fL
        geofreq[cnt]= freqGM  # stored geometric mean freq. [Hz]
        nsoae[cnt]= freqGM/freqDiff # stored Nsoae
        cnt= cnt+1
        
    return geofreq, nsoae

# =======================================================================
# ===Human SOAE peak freqs.


Scnt= 0   # counter for # of subjects
# =======================   =======================  =======================
# subj.1 = human_TH14RearwaveformSOAEshort *
# [original 120 s waveform had several artifacts before 52.5 s mark, so
# made shorter wf with suffix "short" of last 67.5 s]; still a few artifacts, 
# but more minor] --> analysis of shorter wf (sans artifacts) revealed some
# new supra-thresh. peaks (e.g., 433 Hz) and some changes (e.g., 0.662-->0.706),
# but these changes were fairly minor (e.g., apparent sub-thresh. peaks


# Npts= 256*12
# xiS= 256*2.6 & magtempfact=1.0
# --> using above, only specifying peaks w/ Cxi>0.2 (seems reasonable, but
# also likely conservative). 
Scnt += 1  # increment the subject counter
fS1, fT1 = get_human_peak_freqs("human_TH14RearwaveformSOAEshort")
# -----------
gmS1, nS1= computeNsoae(fS1)
gmT1, nT1= computeNsoae(fT1)
ratioT1= np.array(fT1[1:])/np.array(fT1[:-1])
# =======================   =======================  =======================
# subj.2 = human_RRrearSOAEwf1 *
# [long wavefrom is 60 s long and has some  artifacts in last ~20 s past 
# 39 s mark]
# --> made a shorter vers. (only first 39 s) with suffix "short"; this allows 
# for some new peaks to be detectable across entire spectrum; also made a few 
# from the longer vers. become slightly sub-thresh.

# Npts= 256*12
# xiS= 256*2.6 & magtempfact=1.0
# --> using above, only specifying peaks w/ Cxi>0.2 (seems reasonable, but
# also likely conservative). 
# *** NOTE *** There are certainly more peaks that could be specified for this ear
# if one was not being as conservative....
Scnt += 1  # increment the subject counter
fS2, fT2 = get_human_peak_freqs("human_RRrearSOAEwf1short")
# -----------
gmS2, nS2= computeNsoae(fS2)
gmT2, nT2= computeNsoae(fT2)
ratioT2= np.array(fT2[1:])/np.array(fT2[:-1])
# =======================   =======================  =======================
# subj.3 = human_TH13RearwaveformSOAE *
# [long wavefrom has some large artifacts in last ~26 s past 94 s mark]
# --> made a shorter vers. (only first 94 s) with suffix "short"; this helps
# clean things up but does not drastically affect things so to change the vals below
# Npts= 256*12
# xiS= 256*2.6 & magTempFact=1.0 (not 2.0; noisy??)
# o using above, only specifying peaks w/ Cxi>0.2 
# --> seems quite conservative

Scnt += 1  # increment the subject counter
fS3, fT3 = get_human_peak_freqs("human_TH13RearwaveformSOAEshort")
# -----------
gmS3, nS3= computeNsoae(fS3)
gmT3, nT3= computeNsoae(fT3)
ratioT3= np.array(fT3[1:])/np.array(fT3[:-1])
# =======================   =======================  =======================
# subj.4 = human_KClearSOAEwf2 *
# [long wavefrom has ~4 s beats; due to respiration? --> no easy way at moment
# to parse up into a shorter wf sans artifacts]]
# Npts= 256*12
# xiS= 256*2.6 & magtempfact=1.0
# o using above, only specifying peaks w/ Cxi>0.2 
# --> seems conservative

# NOTE: A few extra borderline peaks around 8.9-9.1 kHz and 10.4-11 kHz
# --> these have been excluded below)
Scnt += 1  # increment the subject counter
fS4, fT4 = get_human_peak_freqs("human_KClearSOAEwf2")
# -----------
gmS4, nS4= computeNsoae(fS4)
gmT4, nT4= computeNsoae(fT4)
ratioT4= np.array(fT4[1:])/np.array(fT4[:-1])
# =======================   =======================  =======================
# subj.5 = human_AP7RearwaveformSOAE *
# [long wavefrom has large artifact early on and some smaller ones a bit later
# on; beyond 39.3 s of 120 s waveform seems relatively clean, so creating
# ~80 s "short" wf to be sans artifact] --> had no effect on changing freqs.
# as specified below

# Npts= 256*12
# xiS= 256*2.6 & magTempFact=1.0 (not 2.0; noisy at low freqs?)
# o using above, only specifying peaks w/ Cxi>0.2 
# --> seems like higher freqs. have SOAE-like rippling... (not incl. below)

# NOTE: tweaked a bit w/ xiS=256*2.8-3.0
# NOTE II: the 0.741 peak is borderline re 2 dB, but confidently seems legit
# NOTE III: the 2.459 peak is small but seems legit
# NOTE IV: the 3.115 peak is small/wide
Scnt += 1  # increment the subject counter
fS5, fT5 = get_human_peak_freqs("human_AP7RearwaveformSOAEshort")
# -----------
gmS5, nS5= computeNsoae(fS5)
gmT5, nT5= computeNsoae(fT5)
ratioT5= np.array(fT5[1:])/np.array(fT5[:-1])
# =======================   =======================  =======================
# subj.6 = human_coNW_fgF090728R *
# [long waveform revealed artifacts at ~4 s intervals (respiration?) and a large
# artifact around 62.6 s; not going to create a "short" waveform, but will not
# this 120 s waveform is just a short part of the much longer (30 min?) raw 
# waveform from Northwestern, so subject human_coNW_fg can certainly be 
# revisited...]

# Npts= 256*12
# xiS= 256*2.6 & magTempFact=1.0
# o using above, only specifying peaks w/ Cxi>0.2 
# --> this case seems fairly straightforward...
# NOTE: there are subthres. peaks about ~8-9 kHz (not incl)
Scnt += 1  # increment the subject counter
fS6, fT6 = get_human_peak_freqs("human_coNW_fgF090728R")

# -----------
gmS6, nS6= computeNsoae(fS6)
gmT6, nT6= computeNsoae(fT6)
ratioT6= np.array(fT6[1:])/np.array(fT6[:-1])
# =======================   =======================  =======================
# subj.7 = human_TH21RearwaveformSOAE *
# [the long 120 s waveform is uniformly noisy, so no point to parse down
# into a shorter artifact-free version]

# Npts= 256*12
# xiS= 256*2.6
# o using above, only specifying peaks w/ Cxi>0.2 
# --> fairly straightforward. However, dropping  xiS to 256*2.2 revealed another 
# peak at 0.919, but seems spurious (so not including)

Scnt += 1  # increment the subject counter
fS7, fT7 = get_human_peak_freqs("human_TH21RearwaveformSOAE")
# -----------
gmS7, nS7= computeNsoae(fS7)
gmT7, nT7= computeNsoae(fT7)
ratioT7= np.array(fT7[1:])/np.array(fT7[:-1])
# =======================   =======================  =======================
# subj.8 = human_AVGrearSOAEwf2 *
# [this is a fairly short wf at 30 s; a few artifacts are apparent, but 
# nothing so that a shorter wf to preclude such would help]

# Npts= 256*12
# xiS= 256*2.6 AND magTempFact=1.0 

# o using above, only specifying peaks w/ Cxi>0.2 
# --> This one is a bit tricky because there are numerous supra-thresh Cxi
# vals for higher freqs above the (large) 6.55 kHz peak (e.g., at
# [7.318,7.617,7.867,8.859,9.160,9.246,9.520,9.720]). But given that there
# is no obvious correlating mag. peaks AND the wf is only 30 s, excluding 
# these to stay on conservative side of things. Also leaving out some other 
# peaks (e.g., 5.067,5.613) as they seem sensitive to choice of xiS.
# --> In short, there is likely other viable SOAE peaks from this subject,
# but the short nature of the waveform makes it harder to extract out the
# smaller ones (e.g., some small but other possible peaks: 7.05,7.16,7.24,7.32)
# --> It appears that wf1 for this subject generally yields similar peaks, 
# though a few noted below are absent and wf1 shows additional peaks at lower 
# and higher freqs (e.g., in the 6.5-8kHz range)

# Fortunately this subject has several right ear (as well as left ear that also
# had decent SOAE activity) waveforms collected at different days:
# [though I am not sure which one "wf2" is!]
# 07.09.09 (AVGrearSOAEwf1) --> 30 s wf
# 07.17.09 (AVGrearSOAEwf1) --> 30 s wf
# 07.21.09 (AVGrearSOAEwf1) --> 15 s wf? 30 s at lower SR?
# 05.25.10 (AVGrearSOAEwf1) --> 30 s wf

Scnt += 1  # increment the subject counter
fS8, fT8 = get_human_peak_freqs("human_AVGrearSOAEwf2")

# -----------
gmS8, nS8= computeNsoae(fS8)
gmT8, nT8= computeNsoae(fT8)
ratioT8= np.array(fT8[1:])/np.array(fT8[:-1])
# =======================   =======================  =======================
# subj.9 = human_FMlearSOAEwfA01 *
# [overall, the long 120 s waveform is fairly noisy, though no well
# localized temporal artifacts that would suggest a shorter waveform
# would be useful]

# Npts= 256*12
# xiS= 256*2.6 AND magTempFact=1.0 
# o using above, only specifying peaks w/ Cxi>0.2 
# --> fairly straightforward...

Scnt += 1  # increment the subject counter
fS9, fT9 = get_human_peak_freqs("human_FMlearSOAEwfA01")

# -----------
gmS9, nS9= computeNsoae(fS9)
gmT9, nT9= computeNsoae(fT9)
# -- calc. freq. ratios (higher/lower)
ratioT9= np.array(fT9[1:])/np.array(fT9[:-1])
# =======================   =======================  =======================
# subj.10 = human_JBrearSOAEwf2 *
# [this was a 60 s long waveform, with some non-trivial artifacts in the 
# final 28 s; creating a 32 s "short" waveform to have a more artifact-free wf]

# NOTE: There is another SOAE waveform from this individual 
# (/OAE Data/Human (UofA S&A via Wiggio)/06.02.10/JBrearSOAEwf1.txt)
# --> THat waveform had a large artifact in the last 5 s of the 60 s recording,
# plus other artifacts throughout. I chopped off the last 22 s to create a 38 s 
# "wf1" (no short suffix), though note that it is still relatively more noisy.
# Crosschecking re wf2 vals as below, the larger peaks match up but many of the
# smaller peaks do not, further justifying a conservative approach (and makes me
# wonder if I need to be even more conservative....)

# Npts= 256*12 (re entire long 60 s wf)
# xiS= 256*2.6 AND magTempFact=1.0 
# o using above, only specifying peaks w/ Cxi>0.2 
# --> this is a tough one as there are lots of small ripple-like peaks that 
# seem awfully tempting to include; trying to find the right balance...

# o ** NOTE **: there is a nice tight cluster of peaks around 0.6-0.9 kHz
# > the 5.031 peak comes when xiS is bumped to 256*3.3
# > the "short" vers. reveals some slight changes to weave in:
# * some peaks are now supra-thresh: 0.732,1,136,2.357,
# 4.148,4.194,4.279,4.379,4.596,5.543,6.318,6.462,6.979,7.064,7.122
# --> will incl. some (the logic being that any noise due to artifact 
# is now "unmasked") but not all (e.g., 7.122) as some do not appear robust to
# changes in xiS (or have clear mag peaks) and thus I defer to a conservative est.                  
# * some are now sub-thresh (0.604,5.03) --> will still be incl.
# * some slightly freq-shifted (0.906 --> 0.890) --> will change

# --
Scnt += 1  # increment the subject counter
fS10, fT10 = get_human_peak_freqs("human_JBrearSOAEwf2short")
# -----------
gmS10, nS10= computeNsoae(fS10)
gmT10, nT10= computeNsoae(fT10)
# -- calc. freq. ratios (higher/lower)
ratioT10= np.array(fT10[1:])/np.array(fT10[:-1])


# =======================   =======================  =======================
# subj.11 = human_LSrearSOAEwf1 *
# [this is a 60 s waveform; large artifact around 19.2 s, so creating a "short"
# vers. 40.5 s long (i.e., excluding the first 19.5 s)]

# NOTE: This subject appears to have a second SOAE waveform (wf2), though it
# has a large artifact ~16.3 s through. So will create a "short" vers. of
# wf2 that excludes the first 20 s (i.e., it is 40 s long)
# --> while the large peaks are similar between wf1 and wf2, there are some 
# interesting diffs
# > some peaks present in wf1 are not present/prominentin wf2: 0.288,
# 0.650,1.766,2.111
# > some peaks are present in wf2 that are not in wf1: 1.136,2.687,2.824,
# 3.65,5.258,6.219
# --> for sanity's sake, these additional freqs. will not be included below
# and only wf1 wals will be used

# Npts= 256*12 (re entire long 60 s wf)
# xiS= 256*2.6 AND magTempFact=1.0 
# o using above, only specifying peaks w/ Cxi>0.2 
# --> another tricky one as lots of subthresh. rippling (e.g., about 4-6 kHz) 
# that is even better when using the "short" waveform. Nonetheless, will
# exclude those. See also note re wf2 above.

# --
Scnt += 1  # increment the subject counter
fS11, fT11 = get_human_peak_freqs("human_LSrearSOAEwf1short")
# -----------
gmS11, nS11= computeNsoae(fS11)
gmT11, nT11= computeNsoae(fT11)
# -- calc. freq. ratios (higher/lower)
ratioT11= np.array(fT11[1:])/np.array(fT11[:-1])


# =======================   =======================  =======================
# subj.12 = human_JIrearSOAEwf2short *
# [this ~97s long waveform has some large artifacts starting around the 
# 59 s mark; will create a "short" version that just takes that intial 
# part to reduce spurious transients]

# NOTE: There is another waveform (wf1) recorded for this subject. There
# were some early transients from this 120 s waveform, confined to the first
# 14 s. So will create a slightly shortened wf1 that is 106 s long (without
# a "short" suffix). Overall, wf1 peaks are consistent w/ wf2short peaks
# as specified below, though a few of the smaller peaks are subtrhesh. in
# wf1

# Npts= 256*12 (re entire long 60 s wf)
# xiS= 256*2.6 AND magTempFact=1.0 
# o using above, only specifying peaks w/ Cxi>0.2 
# --> fairly straightforward, esp. re crosschecking between wf2, wf2short and wf1.
# Comparing peaks as determined via Cxi (as specified below) to PSD (i.e.,
# magnitude-centric) peaks via magtempfact=1.0, it seems quite reasonable that we 
# have a conservative underestimate (but one that is reasonably good)

# --
Scnt += 1  # increment the subject counter
fS12, fT12 = get_human_peak_freqs("human_JIrearSOAEwf2short")
# -----------
gmS12, nS12= computeNsoae(fS12)
gmT12, nT12= computeNsoae(fT12)
# -- calc. freq. ratios (higher/lower)
ratioT12= np.array(fT12[1:])/np.array(fT12[:-1])



"""
# =======================   =======================  =======================
# subj.1X = X.mat
# --
Scnt += 1  # increment the subject counter
# ===== standard spectral averaging peak freqs
fS1X= []
# ===== xi-adjusted temporal averaging peak freqs

fT1X= []

# -----------
gmS1X, nS1X= computeNsoae(fS1X)
gmT1X, nT1X= computeNsoae(fT1X)
# -- calc. freq. ratios (higher/lower)
ratioT1X= np.array(fT1X[1:])/np.array(fT1X[:-1])

"""


# =======================================================================
# ------------------------------------------------
# ==== compile #s across subjects [KLUDGE]
# --- standard spectral averaging peak freqs
gmSall= np.concatenate((gmS1,gmS2,gmS3,gmS4,gmS5,gmS6,gmS7,gmS8,gmS9,gmS10,
                        gmS11,gmS12))
nSall= np.concatenate((nS1,nS2,nS3,nS4,nS5,nS6,nS7,nS8,nS9,nS10,nS11,nS12))
# --- xi-adjusted temporal averaging peak freqs
gmTall= np.concatenate((gmT1,gmT2,gmT3,gmT4,gmT5,gmT6,gmT7,gmT8,gmT9,gmT10,
                        gmT11,gmT12))
nTall= np.concatenate((nT1,nT2,nT3,nT4,nT5,nT6,nT7,nT8,nT9,nT10,nT11,nT12))
# -- freq diffs
fDiffsT= gmTall/nTall
# -- freq ratios
fRatiosT= np.concatenate((ratioT1,ratioT2,ratioT3,ratioT4,ratioT5,ratioT6,
                          ratioT7,ratioT8,ratioT9,ratioT10,ratioT11,ratioT12))


# --- send some vals to screen for ref
print(f'# of mag-avgd peaks found = {str(len(gmSall))} ')
print(f'# of C_xi^M peaks found = {str(len(gmTall))} ')
perctIN= 100*(len(gmTall)-len(gmSall) )/len(gmSall)
print(f'% increase in # of peaks found = {perctIN:.2f}% ')


# =======================================================================
# ==== Shera 2003 Nsoae vals
# (previously computed for CB 2012 HR paper; vals. extracted from older analysis
# code: ...//Users/pumpkin/Dropbox/Collaborations B/Oldenburg/Analysis/
#    /2014 MoH owl/Analysis/2012 HR human analysis codesModelPredictN2.m

Shera03n=np.array([1.1686,1.5560,1.9393,2.2290,2.5734,2.9629,3.4175,3.9169,
    4.5260,5.2217,6.0284,6.9145,7.9564,9.1602,10.4685,12.1349,13.7273,
   16.1044,18.6965,21.4918,24.6847,28.2800,32.4947,37.3661,43.2892])
Shera03cnt=np.array([4.2852,6.3118,8.2251,5.3621,11.4314,7.3887,18.3734,10.4826,
   15.4564,12.4530,19.4773,16.6705,21.4195,19.3430,33.3050,47.6042,
   47.6622,69.4051,67.4692,58.4829,24.4698,16.6070,7.3961,2.3141,1.3894])




# ------------------------------------------------
# --- add a small bit of jitter to compiled time avgd. vers.?
# [chiefly no, but helps provide a meas. of robustness]
if (1==0):
    jittL= 0.3
    gmTall= gmTall+ jittL*np.random.randn(len(gmTall))
    nTall= nTall+ jittL*np.random.randn(len(nTall))


# ------------------------------------------------
# ------------------------------------------------
# --- grab a subset of Nsoae vals above a certain min. val.
# (a la Shera 2003 analysis to "the peak of the distribution [...]
# To reduce bias in the fit"
# NOTE: unless there is a good reason to utilize this set of #s, seems 
# wiser to avoid
NsoaeMIN= 0.75*np.mean(nTall)
gmTthresh= gmTall[nTall>NsoaeMIN]
nTthresh= nTall[nTall>NsoaeMIN]


# =======================================================================
# ==== (single) Power Law fit (a la Shera 2003 approach)
# --
def powlawFunc(f,A,c):
    return A*(f**c)

# === do the fits via the  scipy blackbox
# --- standard spectral averaging peak freqs
paramsS, covarS = curve_fit(powlawFunc,gmSall,nSall)
AfitS, cfitS = paramsS
# --- xi-adjusted temporal averaging peak freqs
paramsT, covarT = curve_fit(powlawFunc,gmTall,nTall)
AfitT, cfitT = paramsT
# --- Shera03-style "thresholded" vals.
paramsTthresh, covarTthresh = curve_fit(powlawFunc,gmTthresh,nTthresh)
AfitTthresh, cfitTthresh = paramsTthresh

# ==== compute fit curves to plot
fitF= np.logspace(np.log10(300),np.log10(max(gmTall)), 50)
fitNS= powlawFunc(fitF,AfitS, cfitS)
fitNT= powlawFunc(fitF,AfitT, cfitT)
fitNTthresh= powlawFunc(fitF,AfitTthresh, cfitTthresh)
# --- Shera's fit vals (via Table I in 2003 paper)
fitFshera= np.logspace(np.log10(550),np.log10(7000), 50)
fitShera= powlawFunc(fitFshera/1000,13.7,0.31)


# =======================================================================
# ==== (bootstrapped I) Power Law fit --> amongst pooled datas
# [adapting bits from my EXstatBootstrap2.py code]

nbs= len(nTall)
#nbs= int(np.round(0.9*len(nTall)))
indx= np.arange(nbs)  # create array index 

for n in range(0,N):
    # === grab a resampled array 
    indxBS= np.random.choice(indx,replace=1,size=nbs)
    xT= gmTall[indxBS] # 
    yT= nTall[indxBS] # 
    pTtemp, covarTtemp = curve_fit(powlawFunc,xT,yT,maxfev=2000)
    AfitTtemp, cfitTtemp = pTtemp
    fitTtemp= powlawFunc(fitF,AfitTtemp, cfitTtemp)
    # -- store away fits
    if n==0:
        yDf= fitTtemp  # kludgy
    else:
        yDf= np.vstack((yDf,fitTtemp)) 

# === determine mean and SD (& SE) from bootstrapped loess      
yDfitM= np.mean(yDf,0)  # mean loess fit
yDfitSD= np.std(yDf,axis=0)  # standard deviation
SE= yDfitSD/np.sqrt(N)  # " standard error
CI= 2*SE # confidence intervals (CIs) as +/-95% (i.e., 2*SE)


# =======================================================================
# ==== (bootstrapped II) Power Law fit --> amongst subjects

nbs2= Scnt 
#nbs2= 5 
# NOTE: can make this smaller if you want fewer resmapled subjects
indx2= np.arange(nbs2)  # create array index 
# --
for mm in range(0,nbs2-1):
    # === grab a resampled list of subject
    indxBS2= np.random.choice(indx2,replace=1,size=nbs2)
    xT2= []
    yT2= []
    # --- compile #s together via for loop (better way to do??)
    for nn in range(0,nbs2-1):
        xT2= np.concatenate((xT2,eval('gmT'+str(indxBS2[nn]+1))))
        yT2= np.concatenate((yT2,eval('nT'+str(indxBS2[nn]+1))))
    
    pTtemp2, covarTtemp2 = curve_fit(powlawFunc,xT2,yT2)
    AfitTtemp2, cfitTtemp2 = pTtemp2
    fitTtemp2= powlawFunc(fitF,AfitTtemp2, cfitTtemp2)
    # -- store away fits
    if mm==0:
        yDf2= fitTtemp2  # kludgy
    else:
        yDf2= np.vstack((yDf2,fitTtemp2))

# === determine mean and SD (& SE) from bootstrapped loess      
yDfitM2= np.mean(yDf2,0)  # mean loess fit
yDfitSD2= np.std(yDf2,axis=0)  # standard deviation
SE2= yDfitSD2/np.sqrt(N)  # " standard error
CI2= 2*SE2 # confidence intervals (CIs) as +/-95% (i.e., 2*SE)


# =======================================================================
# ==== create histogram of Nsoae vals (akin to Fig.2 of Shera 2003)
log_bins = np.logspace(np.log10(nTall.min()), np.log10(nTall.max()),binsN) 
log_binsShera = np.logspace(np.log10(Shera03n.min()), np.log10(Shera03n.max()),len(Shera03n)) 
bin_centers = (log_bins[:-1] + log_bins[1:]) / 2
# --
countsT, binsREP = np.histogram(nTall,log_bins)
countsS, binsREP = np.histogram(nSall,log_bins)



# =======================================================================
# -- determine average Nsoae vals. in oct.-wide bins (starting at 0.3 kHz)

freqOct= [300,600,1200,2400,4800,9600]  # oct. range bounds

avgN= []
stdN= []
serrN= []
avgGM= []
for pp in range(0,len(freqOct)-1):
    tmp0= len(nTall[np.where(np.logical_and(gmTall>=freqOct[pp],gmTall<freqOct[pp+1]))])
    tmp1= np.mean(nTall[np.where(np.logical_and(gmTall>=freqOct[pp],gmTall<freqOct[pp+1]))])
    tmp2= np.std(nTall[np.where(np.logical_and(gmTall>=freqOct[pp],gmTall<freqOct[pp+1]))])
    tmp3= np.mean(gmTall[np.where(np.logical_and(gmTall>=freqOct[pp],gmTall<freqOct[pp+1]))])
    avgN.append(tmp1)
    stdN.append(tmp2)
    serrN.append(tmp2/np.sqrt(tmp0))
    avgGM.append(tmp3)
    
    
avgN= np.array(avgN)  # kludge
stdN= np.array(stdN)
serrN= np.array(serrN)
avgGM= np.array(avgGM)

#val1= np.mean(nTall[np.where(np.logical_and(gmTall>=300,gmTall<600))])
#std1= np.std(nTall[np.where(np.logical_and(gmTall>=300,gmTall<600))])

# --- also create the assoc. vers. of the freq. diff. from Shera's 2003
# power law fit


fDiffSheraPLoct= np.log2(fitFshera/fitShera)


# =======================================================================
# read in human Nxi vals.


# --- read in data
df = pd.read_excel(fName)
human_freqs = df[df['Species'] == 'Human']['Frequency'].values
human_Nxi = df[df['Species'] == 'Human']['N_xi'].values

# =======================================================================
# ==== visualize
plt.close("all")
# ------------------------------------------------
# Fig.1 - Nsoae vs freq. (along with various fits/comps)
fig1, ax1 = plt.subplots()
# --- plot one specific individual?
if (1==0):
    fig1= plt.plot(gmS2/1000,nS2,'^',color='cyan',alpha=0.4,ms=10,markerfacecolor='none')
    fig1= plt.plot(gmT2/1000,nT2,'o',color='lime',alpha=1,ms=6,markeredgecolor='none',label='Subj.X')
# --- plot all compiled points
#fig1= plt.plot(gmSall/1000,nSall,'x',color='r',alpha=0.3,ms=5,markerfacecolor='none',label='Spectral Avg.')
fig1= plt.scatter(gmSall/1000,nSall,marker='x', color='darkorange',s=24,alpha=0.5,linewidths=2,label='Spectral Avg.')
fig1= plt.plot(gmTall/1000,nTall,'s',color='royalblue',alpha=0.4,ms=4,markerfacecolor='royalblue',
               markeredgecolor='none',label='Temporal Avg.')
# --- plot power law fits
#fig1= plt.plot(fitF/1000,fitNS,'r--',lw=1,label='Spectral Avg.')
#fig1= plt.plot(fitF/1000,fitNT,'k-',lw=2,alpha=0.3,label='Power fit (all)')
#fig1= plt.plot(fitF/1000,fitNTthresh,'c-',lw=3,label='Thresholded')
fig1= plt.plot(fitFshera/1000,fitShera,'-.',lw=2,color='black',label='Shera (2003)')
# --- plot bootstrapped power law fits (all data pooled for bootstrap)
fig1= plt.plot(fitF/1000,yDfitM,'-',color='royalblue',lw=2,label='Bootstrapped power law fit')
fig1= plt.fill_between(fitF/1000, (yDfitM-yDfitSD), (yDfitM+yDfitSD), 
                       color='royalblue',alpha=0.1)   
# - Subj.-pooled bootstrapped fit?
if (1==0):
    fig1= plt.plot(fitF/1000,yDfitM2,'-',color='magenta',lw=2,label='Subj.-pooled bootstrapped fit')
    fig1= plt.fill_between(fitF/1000, (yDfitM2-yDfitSD2), (yDfitM2+yDfitSD2), 
                           color='magenta',alpha=0.1)
# --- plot mean oct-wide bin vals?
if (1==1):
    #fig1= plt.plot(avgGM/1000,avgN,'^',color='magenta')
    # --- plotting w/ standard error
    fig1= plt.errorbar(avgGM/1000,avgN, yerr=serrN, fmt='d',capsize=4,lw=1,
                       color='blue',alpha=0.7,label='octave-wide averages')
    

# --- bookeeping
ax1.set_xscale('log')
ax1.set_yscale('log')
fig1= plt.xlim([0.27,15])
fig1= plt.ylim([1,300])
fig1= plt.xlabel('Frequency [kHz]',fontsize=12)
fig1= plt.ylabel(r"$N_{SOAE}$",fontsize=12) 
#fig1= plt.title('Human Nsoae: Spec-avg. (triangle) vs xi-adjust. Temp.-avg. (dot)') 
fig1= plt.grid(True, which="both", ls="-", color='0.9')
ax1.set_axisbelow(True)
fig1= plt.legend()

# ------------------------------------------------
# Fig.2 - Histogram of Nsoae
if (1==1):
    fig2, ax2 = plt.subplots()
    # ---
    plt.bar(bin_centers,countsT/np.sum(countsT),alpha=0.6,width=np.diff(log_bins),ec="k", align="edge",label='Temporal avg.')
    plt.bar(bin_centers,countsS/np.sum(countsS),alpha=0.5,width=np.diff(log_bins),ec="k",align="edge",label='Spectral avg.')
    # ----
    plt.plot(Shera03n,Shera03cnt/np.sum(Shera03cnt),'k--',lw=2.5,alpha=0.5,label='Shera (2003)')
    plt.xscale('log') # Still useful to ensure proper display of log ticks
    plt.xlabel(r"$N_{SOAE}$",fontsize=12)
    plt.ylabel("Probability",fontsize=12)
    #plt.title(r'Comparison of $N_{SOAE}$')
    plt.legend()
    plt.show()

# ------------------------------------------------
# Fig.3 - Freq. spacings vs GM freq.
# Purpose: Provide visualization to compare to SSOAE spacings as shown in
# Fig.2 of Bell and Jedrzejczak (2017)
if (1==1):
    fig3, ax3 = plt.subplots(2,1)
    ax3[0].plot(gmTall/1000,np.log2(fRatiosT),'bo',ms=3,alpha=0.4)
    ax3[0].set_ylabel("Freq. Pair Spacing [oct]",fontsize=12)
    ax3[0].set_xlabel("Geometric Mean Frequency [kHz]",fontsize=12)
    ax3[0].grid()
    # --
    n, bins, patches = plt.hist(fRatiosT,bins=ratioBinCNT,label='SOAE spacing',
                                color='orange',edgecolor='black',alpha=0.5)
    maxCNT = n.max()
    ax3[1].plot([1.06,1.06],[0,maxCNT+2],'r--',lw=3,label='SSOAE spacing (B&J2017)')
    ax3[1].set_xlim([1,1.5])
    ax3[1].set_xlabel("Freq. Ratio",fontsize=12)
    ax3[1].set_ylabel("Counts",fontsize=12)
    ax3[1].grid(True, which="both", ls="-", color='0.7')
    ax3[1].set_axisbelow(True)
    ax3[1].legend(loc="upper right")
    
    plt.tight_layout()
    
    
# ------------------------------------------------
# Fig.4 - Compare human Nsoae versus Nxi
# Purpose: 
if (1==1):
    fig4, ax4 = plt.subplots()
    # ---
    fig4= plt.plot(gmTall/1000,nTall,'s',color='royalblue',alpha=0.4,ms=4,markerfacecolor='royalblue',
                   markeredgecolor='none',label=r"$N_{SOAE}$")
    fig4= plt.plot(human_freqs/1000,human_Nxi*fact,'x',color='red',
                   alpha=1,ms=5,markeredgewidth=2,markerfacecolor='none',label=r'$N_{\xi}$')

    ax4.set_xscale('log')
    ax4.set_yscale('log')
    fig4= plt.xlim([0.27,15])
    fig4= plt.ylim([1,300])
    fig4= plt.xlabel('Frequency [kHz]',fontsize=12)
    fig4= plt.ylabel(r"Human $N$ values",fontsize=12) 
    #fig1= plt.title('Human Nsoae: Spec-avg. (triangle) vs xi-adjust. Temp.-avg. (dot)') 
    fig4= plt.grid(True, which="both", ls="-", color='0.9')
    ax4.set_axisbelow(True)
    fig4= plt.legend()

"""
# --------------------
# [v2 Notes]
o v.2 makes the Nsoae computation a function
o Extracting SOAE peak spacing using EXplotSOAEwfP13B.py via normal
spectral averaging and xi-adjusted phase time averaging (which can reveal 
additonal peaks) and manually inputting both sets of #s here
o picking peaks manually from the plots using the criteria that a peak has 
got to have at least a 2 dB SNR

o Code below only plots for adjacent pairs. If you wanted to compute Nsoae 
for all (picked) peaks, loop might be 

#Mint= int(M*(M-1)/2) # total numb. of unique pairs
# --- all pairs
# for nn in range(0,M):
#     fL= tempARR[nn]*1000 # pick off lowest freq. yet to analyze  [Hz]
#     # -- analyze fL re the higher freqs
#     for mm in range(nn+1,M):
#         fH= tempARR[mm]*1000  # higher freq. of the pair [Hz]
#         freqGM= np.sqrt(fH*fL)  # geometric mean freq.
#         freqDiff= fH-fL
#         geofreq[cnt]= freqGM  # stored geometric mean freq. [Hz]
#         nsoae[cnt]= freqGM/freqDiff # stored Nsoae
#         cnt= cnt+1


snippet from RScheckSOAEspacing2.m
% === loop thru freqs for a given ear
    for nn=1:numel(freqs)
        fL= freqs(nn);  % pick off lowest freq. yet to analyze
        % === analyze fL re the higher freqs
        for mm=nn+1:numel(freqs)
            % --- compute relevant vals and store away
            freqGM(end+1)= sqrt(freqs(mm)*fL);  % geometric mean freq.
            fRatio(end+1)= freqs(mm)/fL;  % interpeak ratio re lower peak freq.
            % --- also compute Nsoae val
            if mm==nn+1
                fPeakL(end+1)= fL;     % keep track in lower peak freq
                fDiff(end+1)= (freqs(mm)-fL);  % interpeak freq. diff
                Nsoaef(end+1)= freqGM(end);
                Nsoae(end+1)= freqGM(end)/(fDiff(end));
            end
        end
    end
 """