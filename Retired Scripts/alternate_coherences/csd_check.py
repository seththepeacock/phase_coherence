import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, welch, csd
from phaseco import *
from tqdm import tqdm
from collections import defaultdict



fs = 2000  # Sampling frequency (Hz)
N = 2**9  # Window length
tauS = N
tau = tauS / fs
L = 2**13  # Signal length per realization
win_type = 'hann'
window = get_window(win_type, tauS)
nrealizations = 10 # Number of independent realizations to average
freq = 100 # If we only want to look at a single frequency bin
xi = 0.1
seg_spacing = 0.1
xiS = round(xi * fs)

x_full = np.random.normal(0, 1, L)
x = x_full[:(L-xiS)]
x_delayed = x_full[xiS:]


# Compute coherence using scipy and phaseco
hop = round(seg_spacing * fs)
noverlap = tauS - hop

SFT = ShortTimeFFT(window, hop, fs, fft_mode='onesided', scale_to=None, phase_shift=None)

# Compute spectrogram: csd uses y, x (note reversed order)
Pxy_scipy_spect = SFT.spectrogram(x_delayed, x, p0=0, p1=(len(x) - noverlap) // hop, k_offset=tauS // 2)

# Apply onesided doubling (if real and return_onesided=True)
if np.isrealobj(x) and SFT.fft_mode == 'onesided':
    Pxy_scipy_spect[1:-1 if SFT.mfft % 2 == 0 else None, :] *= 2

# Average across time segments (axis=1 if time is columns)
Pxy_scipy_spect = np.mean(Pxy_scipy_spect, axis=1)

# Normalize (done already)
Pxy_scipy_spect /= fs * np.sum(window ** 2)



f_scipy, Pxy_scipy = csd(x, x_delayed, fs=fs, window=window, nperseg=tauS, noverlap=noverlap, detrend=False, scaling='density')
# f_phaseco, Pxy_phaseco= get_csd(x, x_delayed, fs=fs, tauS=tauS, seg_spacing=seg_spacing, win_type=win_type)
# f_manual, Pxy_manual = csd_manual(x, x_delayed, fs=fs, nperseg=tauS, noverlap=noverlap, window=win_type)


plt.scatter(f_scipy, np.abs(Pxy_scipy), label='scipy')
# plt.scatter(f_phaseco, np.abs(Pxy_phaseco), label='phaseco', s=10, zorder=2)
plt.scatter(f_scipy, np.abs(Pxy_scipy_spect), label='scipy_spect', s=20, zorder=1)

# plt.scatter(f_manual, np.abs(Pxy_manual))
plt.show()

# from scipy.signal import csd
# import matplotlib.pyplot as plt
# import numpy as np
# from phaseco import *

# fs = 1000
# t = np.arange(0, 1.0, 1/fs)
# x = np.sin(2*np.pi*100*t) + 0.5*np.random.randn(len(t))
# y = np.sin(2*np.pi*100*t + 0.3) + 0.5*np.random.randn(len(t))
# nperseg = 256   
# noverlap = 128
# # Compare your function to scipy's"
# f1, Pxy_manual = get_csd(x, y, fs=fs, nperseg=nperseg, window='hann')
# f2, Pxy_scipy = csd(x, y, fs=fs, nperseg=nperseg, window='hann', noverlap=noverlap, nfft=nperseg,
#                     scaling='density', detrend=False)


# tau = nperseg / fs
# # print(1/tau)
# print(fs/tau)
# # Plot
# # plt.figure()
# plt.scatter(f1, np.abs(Pxy_manual), label="manual")
# plt.scatter(f2, np.abs(Pxy_scipy), label="scipy")

# plt.legend()
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("CSD magnitude")
# plt.title("Manual vs SciPy CSD")
# plt.show()
