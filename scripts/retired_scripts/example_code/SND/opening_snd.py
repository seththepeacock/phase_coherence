import soundfile as sf
from scipy.fft import rfft, rfftfreq
import numpy as np
import matplotlib.pyplot as plt

wf, fs = sf.read('nkF090714.soae.snd')
print(f"Sample Rate: {fs}, Waveform Shape: {wf.shape}")

# Get time axis
t = np.arange(len(wf)) / fs

# Crop to 20-120 seconds since lots of noise at the beginning and end
t_min = 20
t_max = 120
N_min = int(t_min * fs)
N_max = int(t_max * fs)
wf = wf[N_min:N_max]
t = t[N_min:N_max]

# Get the spectrum
mags = np.abs(rfft(wf))
freqs = rfftfreq(len(wf), d=1/fs)

plt.figure(figsize=(12, 6))

# Plot the waveform
plt.subplot(1, 2, 1)
plt.scatter(t, wf, s=1)
plt.title("Waveform")
plt.xlabel("Time (s)")

# Plot the spectrum
plt.subplot(1, 2, 2)
plt.scatter(freqs,10*np.log10(mags), s=1)
plt.title('Magnitudes')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.xlim(0, 5000)
plt.show()