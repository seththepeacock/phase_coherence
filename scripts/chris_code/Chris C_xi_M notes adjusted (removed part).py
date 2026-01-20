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

"""
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



# =======================================================================
# ------------------------------------------------
# ==== compile #s across subjects [KLUDGE]
# --- standard spectral averaging peak freqs
all_geofreqs_mags= np.concatenate((gmS1,gmS2,gmS3,gmS4,gmS5,gmS6,gmS7,gmS8,gmS9,gmS10,
                        gmS11,gmS12))
all_Nsoae_mags= np.concatenate((nS1,nS2,nS3,nS4,nS5,nS6,nS7,nS8,nS9,nS10,nS11,nS12))
# --- xi-adjusted temporal averaging peak freqs
all_geofreqs_C= np.concatenate((gmT1,gmT2,gmT3,gmT4,gmT5,gmT6,gmT7,gmT8,gmT9,gmT10,
                        gmT11,gmT12))
all_Nsoae_C = np.concatenate((nT1,nT2,nT3,nT4,nT5,nT6,nT7,nT8,nT9,nT10,nT11,nT12))
# -- freq diffs
freq_diffs_C= all_geofreqs_C/all_Nsoae_C
# -- freq ratios
all_freq_ratios_C= np.concatenate((ratioT1,ratioT2,ratioT3,ratioT4,ratioT5,ratioT6,
                          ratioT7,ratioT8,ratioT9,ratioT10,ratioT11,ratioT12))

"""

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

# --- plot one specific individual?
if (1==0):
    fig1= plt.plot(gmS2/1000,nS2,'^',color='cyan',alpha=0.4,ms=10,markerfacecolor='none')
    fig1= plt.plot(gmT2/1000,nT2,'o',color='lime',alpha=1,ms=6,markeredgecolor='none',label='Subj.X')