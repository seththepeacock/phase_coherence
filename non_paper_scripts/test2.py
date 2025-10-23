from nddho_generator import nddho_generator
import phaseco as pc
import matplotlib.pyplot as plt
from scipy.signal import correlate, get_window, stft
from scipy.optimize import curve_fit
import numpy as np
import time
from helper_funcs import *
os.chdir(r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence")


wf = np.random.normal(0, 1, int(0.5*44100))
xi = 10000
fs = 44100
f0 = 3000
tau = 2**12
hop = 1
win_meth = {'method':'static', 'win_type':'flattop'}
win = pc.get_win(win_meth, tau, xi)[0]
pw=True

start = time.time()
f = rfftfreq(tau, 1/fs)
f0_idx = np.argmin(np.abs(f0-f))
f0_exact = f[f0_idx]


print("Calculating old way 2")
t, f, stft_old = get_stft(wf, fs, f0s=None, win=win, tau=tau, hop=hop, realfft=True)
# print(stft_old.shape)
wf_old = stft_old[:, f0_idx]


print("Calculating new way")
omega_0_norm = f0_exact * 2*np.pi / fs
n = np.arange(len(win))
kernel = win * np.exp(1j * omega_0_norm * n)
wf_new = convolve(wf, kernel, mode='valid', method='fft')

f, psd_old = pc.get_welch(wf_old, fs, tau, realfft=False)
f, psd_new = pc.get_welch(wf_new, fs, tau, realfft=False)

plt.scatter(f, 10*np.log10(psd_old), label='Old', s=10)
plt.scatter(f, 10*np.log10(psd_new), label='New', s=2)
plt.legend()
plt.show()

print(np.max(np.abs(wf_new-wf_old)))


