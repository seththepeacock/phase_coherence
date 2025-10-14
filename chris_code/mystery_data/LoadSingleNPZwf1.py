#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
### LoadSingleNPZwf1.py ###

Purpose: Load in saved waveform data (as binary .npz file) as created by
EXnoiseDrivenOsc1.py

Created on Tue Apr 29 09:08:08 2025

@author: pumpkin
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
fileL= 'mystery1.wf3.npz'  # file name to load in
Npoints= 2560  # of point for plotting a snippet of the waveform
SR=10000  # SR used when creating waveforms
# =========================================================

# ---- load in data
loadData = np.load('mystery1.wf3.npz')
# ----  parse up
wf1= loadData['arr_0']
# ---- bookeeping
N = len(wf1)    # num of time points
tE = N/SR        # time length of waveform 
h = (tE)/N   # step-size
tpoints = np.arange(0,tE,h)   # time point array
# ---- create shorter vers. for visualiz.
indxE= N
wf1S= wf1[indxE-Npoints:indxE]  # grab last nPts
tpointsS=  tpoints[indxE-Npoints:indxE]
# ==== visualize
plt.close("all")
# --- FIG: time waveforms (final SS segment) & SS spectal mags.
if 1==1:
    fig88, ax88 = plt.subplots()
    ax88.plot(tpointsS,wf1S,'k-',label='wf1',alpha=0.9,lw=2)
    ax88.set_xlabel('Time [s]')  
    ax88.set_ylabel('Position x') 
    ax88.set_title('Loaded WF')
    ax88.grid()
    fig88= plt.legend()