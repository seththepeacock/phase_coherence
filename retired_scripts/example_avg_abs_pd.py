import phaseco as pc
import numpy as np
from scipy.signal import ShortTimeFFT
import matplotlib.pyplot as plt

T = 13
fs = 44100
N = round(T * fs)
times = np.arange(0, 10, 1/fs)

f0 = 239
phi = 0
A = 10000
cosine = A * np.cos(2*np.pi*f0*times + phi)
# noise = np.random.randn(N)

wf = cosine


tau_s = 1.0
hop_s = 0.185

tau = round(tau_s*fs)
hop = round(hop_s*fs)

plt.figure(figsize=(8, 16))
plt.subplot(2, 1, 1)
phase_corr = 1
t, f, stft = pc.get_stft(wf, fs, tau=tau, hop=hop, phase_corr=phase_corr)
f0_idx = np.argmin(np.abs(f-f0))
phases = np.angle(stft[:, f0_idx])
plt.scatter(t, phases)
plt.ylim(-np.pi, np.pi)
plt.title("Corrected")
plt.subplot(2, 1, 2)
phase_corr = 0
t, f, stft = pc.get_stft(wf, fs, tau=tau, hop=hop, phase_corr=phase_corr)
f0_idx = np.argmin(np.abs(f-f0))
phases = np.angle(stft[:, f0_idx])
plt.scatter(t, phases)
plt.ylim(-np.pi, np.pi)
plt.title("Uncorrected")
plt.show()
