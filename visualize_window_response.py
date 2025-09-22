import matplotlib.pyplot as plt
import scipy.signal as ssig
from scipy.fft import fft, fftfreq
import numpy as np
import phaseco as pc

plt.figure()
plt.subplot(2, 1, 1)
plt.title(r"$W(f)$")
plt.subplot(2, 1, 2)
plt.title(r"$w[n]$")
fs = 44100

dtft_oversample_factor = 16
xi_s = 0.050
xi = round(xi_s*fs)
rho = 0.7




taus = [2**12, 2**13, 2**14]
windows = ['blackman']

freq_bw = 50
for tau in taus:
    for window in windows:
        N = tau * dtft_oversample_factor
        if window!='gauss':
            win = ssig.get_window(window, tau)
        else:
            desired_fwhm = rho * xi
            sigma = desired_fwhm / (2 * np.sqrt(2 * np.log(2)))
            win = ssig.get_window((window, sigma), tau)
        if window == 'flattop':
            bw_omega = (20*np.pi / tau)
            bw_f = bw_omega * (fs / (2*np.pi))
            print(f"Bandwidth = {bw_f} Hz")
        


        f = fftfreq(N, 1/fs)
        WIN = np.abs(fft(win, N))**2
        plt.subplot(2, 1, 1)
        plt.scatter(f, WIN / np.max(WIN), label=rf"{window.capitalize()} ($\tau={(tau/fs*1000):.2f}$ms)", s=5)
        plt.subplot(2, 1, 2)
        plt.scatter(np.arange(-len(win)/2, len(win)/2), win, s=5)
plt.subplot(2, 1, 1)
plt.xlim(-freq_bw, freq_bw)
plt.legend(loc='upper right')
plt.show()