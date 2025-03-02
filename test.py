from SOAEpeaks import load_df
import phaseco as pc
from phaseco import *
import numpy as np
import pywt
import matplotlib.pyplot as plt
wf = np.ones(200000)

# Downsample in time
ds_factor = 4
wf_ds = wf[::ds_factor]
fs = int(44100/ds_factor)
# Crop waveform
crop_factor = 50
wf_cropped = wf_ds[0:int(len(wf_ds)/crop_factor)]

xi = 0.0025
tau = 0.05
f_cwt = np.fft.rfftfreq(int(tau*fs), d=1/fs)[1:]

f_cwt, cwt = get_cwt(wf=wf_cropped, fs=fs, fb=100, f=f_cwt)
wav_coherence = pc.get_wavelet_coherence(wf=wf_cropped, f=f_cwt, coefficients=cwt, fs=fs, xi=xi)