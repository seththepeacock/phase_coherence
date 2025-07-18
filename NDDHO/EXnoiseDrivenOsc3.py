#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
### EXnoiseDrivenOsc3.py ###

Purpose: Using an interpolated noise waveform, this code numerically integrates
(hard-coded RK4) a noise-driven DDHO, van der Pol, or a Duffing oscillator 

NDDDHO: x" = -gamma*x' - (w0**2)*x + A*Noise
NDvdP:  x" = mu*(1-xn**2)*x' - (w0**2)*x + A*Noise
ND-Duffing: x" = -gamma*x' - (w0**2)*x -beta*x**3 + A*Noise

Several time waveforms are created as follows:
wf1 > noise-driven oscillator w/ ~resonant freq. w_0 
wf2 > Noise stimulus itself to drive wf1; Brownian interp. noise
wf3 > sinusoid at w_0 

+ Allows for additive noise to be added on after the integration via 
addnoise (and scaled via nAmpl). Note that this is *default*
(as it creates a more natural/expt-like wf)
+ This code saves waveforms to file as .npz binary that can
be fed into coherence analysis codes such as EXplotSOAEwfP13.py 
    
----
Notes (older vers. updates at the bottom)
o v3 updates:
    + [2025.07.11] stripped all outdated coherence code and also create a boolean
    to only solve one specified oscillator type (viaoscType)
    + [2025.07.11] kludge fixed a bug that had first noise point abnormally high
    + hard-coding Nxi to be N (i.e., # of noise points is same as # of time points]
    +
    
----  
To Do (2/24/25)
o test that Duffing osc. is working as expected
o normalize noise levels re mu and gamma so to make sure vdP and Duffing 
cases are being sufficiently/comparably noise-driven


Created on Fri Feb 21 10:24:05 2025
@author: pumpkin
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
from scipy.interpolate import CubicSpline
from pathlib import Path
np.complex_ = np.complex128

# ================= [User Params]  =================
oscType= 1   # choose osc. type to analyze 1=DDHO,2=vdP,3=Duffing
# ---
saveWF=1  # save all for wfs to (binary) file?
fileS= './Waveforms/testV10'  # file name root to save data to (as binary '.npz' file)
# ===== osc. params
w0 = 3140     # natural freq (rads/s) for both DDHO, vdP, and Duffing  {3140}
quantF= 0     # boolean to quantize freq re tau-window {0}
gamma = 5.0      # damping term for DDHO and Duffing {20}
mu= 0.1       # negative damping coefficient for vdP {0.1}

beta= 1.0*np.abs(w0)**3   # cubic nonlinearity in Duffing {100*w0**3??}

# --- drive-noise params
A = 5000      # noise amplitude {5000?}
#Nxi= 500000  # number of base points for noise waveform {30000} 
# --- additive-noise params(i.e., add noise to waveform after integ.?)
addNoise= 1    # ** boolean to  {1} 
nAmpl= 0.5   # scaling factor for the noise {0.5?}                              
# --- integration time & spec params
tE = 15        # integration time (assumes tS=0) {15}
SR= 15000     # "sample rate" (i.e., step-size = 1/SR) [Hz] {10000}
Npoints= 8192*1  # of point for "steady-state" waveform to compute  # spec. for visual. {8192*1?}

# --- ICs
x0 = 0.0001       # initial x {0.0001}
y0 = 0.0       # initial y  {0}
tS = 0         # start time  {0} 

# ==================================================

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------- [define ODE system re oscillator + noise]
if (oscType== 1):   # NDDHO
    def fN(rN,t,xiVal):
        xn = rN[0]
        yn = rN[1]
        fxn = yn
        fyn = -gamma*yn - (w0**2)*xn + A*xiVal
        return np.array([fxn,fyn],float)
elif (oscType== 2):   # NDvdP
    def fN(rN,t,xiVal):
        xn = rN[0]
        yn = rN[1]
        fxn = yn
        fyn = mu*(1-xn**2)*yn - (w0**2)*xn + A*xiVal
        return np.array([fxn,fyn],float)
elif (oscType== 3):   # ND-Duffing
    def fN(rN,t,xiVal):
        xn = rN[0]
        yn = rN[1]
        fxn = yn
        fyn = -gamma*yn - (w0**2)*xn -beta*xn**3 + A*xiVal
        return np.array([fxn,fyn],float)
# ----------- [define RK4]
def rk4n(rN,t,h):
    k1 = h*fN(rN,t,xiVal)
    k2 = h*fN(rN+0.5*k1,t+0.5*h,xiVal)
    k3 = h*fN(rN+0.5*k2,t+0.5*h,xiVal)
    k4 = h*fN(rN+k3,t+h,xiVal)
    rN += (k1+2*k2+2*k3+k4)/6
    return rN

# =================================================
# ===== bookkeeping =====
# ----
if (quantF==1):
    f0= w0/(2*np.pi)
    df = SR/Npoints;      # quant. re tau-window (this is what we want!)
    fQ= np.ceil(f0/df)*df;   # quantized natural freq.
    w0= fQ*2*np.pi

# ----
ICs = [x0,y0]   # repackage ICs
Q= w0/gamma   # -- "quality factor" 
N = tE*SR     # num of time points
Nxi= N     # same # of noise points as time points to solve on
h = (tE-tS)/N   # step-size
tpoints = np.arange(tS,tE,h)   # time point array
L= len(tpoints)   # total numb of time points
# --- SS-related #s
VTW = L-(Npoints);  # create offset indx point extracting FFT window
tW= tpoints[L-Npoints:L]  #(shorter/later) time interval for FFT window
tI= tpoints[0:Npoints]  # time window for impulse resp
tSS= Npoints/SR  # time point considered for steady-state
df = SR/Npoints
freq= np.arange(0,(Npoints+1)/2,1)    # create a freq. array (for FFT bin labeling)
freq= SR*freq/Npoints;
w= 2*np.pi*freq  # angular freqs
# --- allocate some memory for sol arrays
xXi= []; yXi = [];
# -- input ICs for solver to get started
rN= [x0,y0]
# ======= create a unique (low-pass) Brownian noise
nBase= np.random.randn(Nxi,1)  # create noise waveform
tXibase= np.linspace(1,N,num=Nxi)/SR  # time array for base noise
nFine= CubicSpline(tXibase,nBase)  
NoiseR= nFine(tpoints)
NoiseR[0]= 0   # kludge fix as that val. was abnormally large (due to interp?)
Noise= np.concatenate(NoiseR,axis=0)  # need to recast array of arrays to single array  
# -- FFT-related stuff
dfCh = SR/len(Noise)
freqCh= np.arange(0,(len(Noise)+1)/2,1)    # create a freq. array (for FFT bin labeling)
freqCh= SR*freqCh/len(Noise)
wCh= 2*np.pi*freqCh

# --- send some vals to screen for ref
print(f'Oscillator Q = {Q} and damping cofficient zeta = {1/(2*Q)}')
if (oscType==1): 
    print('--> solving NDDHO')   
elif (oscType==2): 
    print('--> solving NDvDP')    
elif (oscType==3): 
    print('--> solving ND-Duffing')

# =================================================
# --- main integration loop
indx = 0
for t in tpoints:
    xXi = np.insert(xXi,indx,rN[0])
    yXi = np.insert(yXi,indx,rN[1])
    xiVal= Noise[indx]
    rN = rk4n(rN,t,h)
    indx = indx+1

# ====
sinRef= np.sin(w0*tpoints)   # create a sinusoid at w_0
xXi= xXi.flatten()  # make arrays better behaved
Noise= Noise.flatten()
sinRef= sinRef.flatten()

# ==== add in (additive) noise to waveform (position "x" only)
if (addNoise==1):
    xXi= xXi+ float(np.mean(np.abs(xXi.flatten())))* nAmpl*np.random.randn(len(xXi))
    yXi= yXi+ float(np.mean(np.abs(yXi.flatten())))* nAmpl*np.random.randn(len(yXi))

# ====
# rename/store away the relevant (whole) waveforms
wf1= xXi   # DDHO
wf2= Noise   # Noise
wf3= sinRef   # sinusoid at w_0


# ========================================================
if (saveWF==1):
    if (1==0):
        # ==== save all for wfs as a single npz file
        np.savez(Path(fileS+'.all.npz'), wf1,wf2,wf3)
    else:
        # ==== save each wf as a single npz file
        np.savez(Path(fileS+'.wf1.npz'),wf1)
        np.savez(Path(fileS+'.wf2.npz'),wf2)
        np.savez(Path(fileS+'.wf3.npz'),wf3)


# ================================
# grab steady-state bits and compute SS spectrum
indxE= len(wf1)
wf1S= wf1[indx-Npoints:indx]  # grab last nPts
wf2S= wf2[indx-Npoints:indx]
wf3S= wf3[indx-Npoints:indx]
tpointsS=  tpoints[indx-Npoints:indx]
# --- compute spectrum of each
spec1= rfft(wf1S)
spec2= rfft(wf2S)
spec3= rfft(wf3S)
freqS= np.arange(0,(Npoints+1)/2,1)    # create a freq. array (for FFT bin labeling)
freqS= SR*freqS/Npoints;

# --- create scaling factor for noise and sinusoid to help plotting
Nscale= np.max(wf1S)/np.max(wf2)
Sscale= np.max(wf1S)/np.max(wf3)


# =============================================
# ==== visualize
plt.close("all")

# --- FIG: time waveforms (final SS segment) & SS spectal mags.
if 1==1:
    fig88, ax88 = plt.subplots(2,1)
    ax88[0].plot(tpointsS,wf2S*Nscale,'--',label='Noise (Scaled)',alpha=0.15,color='blue')
    ax88[0].plot(tpointsS,wf3S*Sscale,'r--',label='Sinusoid (Scaled)',alpha=0.08)
    ax88[0].plot(tpointsS,wf1S,'k-',label='DDHO',alpha=0.9,lw=2)
    ax88[0].set_xlabel('Time [s]')  
    ax88[0].set_ylabel('Position x') 
    ax88[0].set_title('(steady-state) Noise-Driven Oscillator')
    ax88[0].grid()
    # ~~ Fig.1B: steady-state spec. mags.
    ax88[1].plot(freqS/1000,20*np.log10(abs(spec1)),'k-',label='Oscillator',alpha=0.9,lw=2)
    ax88[1].plot(freqS/1000,20*np.log10(abs(spec2)),'b-.',label='Noisy Drive',alpha=0.5)
    ax88[1].plot(freqS/1000,20*np.log10(abs(spec3)),'--',label='Sinusoid',alpha=0.5,color='red')
    ax88[1].set_xlabel('Frequency [kHz]')  
    ax88[1].set_ylabel('Magnitude [dB]') 
    ax88[1].set_title('Steady-state Spectral Mags.')
    ax88[1].grid()
    ax88[1].set_xlim([0, 1.5*w0/(1000*np.pi)])
    ax88[1].legend()
    fig88.tight_layout(pad=1.5)


# --- FIG: plot ENTIRE time waveforms 
if 1==1:
    fig1, ax1 = plt.subplots()
    Lb1= plt.plot(tpoints,wf1,label='Position')
    ax1b = ax1.twinx()
    Lb2= ax1b.plot(tpoints,yXi,'r',alpha=0.2,label='Velocity')
    plt.xlabel('Time [s]',fontsize=12)
    plt.ylabel('Position',fontsize=12) 
    plt.title('Entire solved waveforms',fontsize=10,loc='right') 
    plt.grid()
    LbT = Lb1+Lb2  # deal w/ legend given multiple ordinates
    labelF5 = [l.get_label() for l in LbT]
    ax1.legend(LbT,labelF5,loc="best")


"""
----
Notes (older vers. updates at the bottom)
o v2 updated to add in DUffing oscillator calc. Need a large beta since rms(x) 
is small and thus a cubic version would contribute negligibly otherwise
o wf1, wf2 & wf5 obtained via direct RK4 numeric integration, whereas wf3 is the
noise waveform fed into them; wf4 is a sinusoid plus arb. noise
** --> thus it is important that SR is large enough to ensure minimal
 numeric int. error
o uses legacy code of EXplotSOAEwfP10.py to plot the avgd. spectrum and coherence
  (but only wf# identified via wfB)
o this code is derived from legacy code EXcoherenceCanonical2.py (First attempt to 
combine EXplotSOAEwfP10.py and EXnoiseDrivenOsc.py) and includes
elements of EXnoiseDrivenOsc.py
"""
