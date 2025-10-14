import matplotlib.pyplot as plt
import scipy.signal as ssig
from scipy.fft import fft, fftfreq
import numpy as np
from N_xi_fit_funcs import *
import phaseco as pc


fs = 44100

dtft_oversample_factor = 32
xi_s = 0.050
xi = round(xi_s*fs)
rho = 0.7
plot= False

# hann with 144ms, Width = 10.006946325302124
# hann with 72ms, Half-Width = 10.005632042884827
# boxcar with 44ms, Half-Width = 10.070031881332397
# boxcar with 88ms, Width = 10.067403316497803


tau_ss = [0.088]
windows = ['boxcar']

taus = np.array(np.round(np.array(tau_ss) * fs), dtype=int)
freq_bw = 50

if plot:
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(r"$W(f)$")
    plt.subplot(2, 1, 2)
    plt.title(r"$w[n]$")

for tau, tau_s in zip(taus, tau_ss):
    for window in windows:
        N = 2**25
        if window!='gauss':
            win = ssig.get_window(window, tau)
        else:
            desired_fwhm = rho * xi
            sigma = desired_fwhm / (2 * np.sqrt(2 * np.log(2)))
            win = ssig.get_window((window, sigma), tau)
        bw = get_win_hpbw({'method':'static', 'win_type':window}, tau, fs, nfft=N)
        print(f"{window} with {tau_s*1000:.0f}ms, Width = {2*bw}")
        # if window == 'flattop':
        #     bw_omega = (20*np.pi / tau)
        #     bw_f = bw_omega * (fs / (2*np.pi))
        #     print(f"Bandwidth = {bw_f} Hz")
        

        if plot:
            f = fftfreq(N, 1/fs)
            WIN = np.abs(fft(win, N))**2
            plt.subplot(2, 1, 1)
            plt.scatter(f, WIN / np.max(WIN), label=rf"{window.capitalize()} ($\tau={(tau/fs*1000):.2f}$ms)", s=5)
            plt.subplot(2, 1, 2)
            plt.scatter(np.arange(-len(win)/2, len(win)/2), win, s=5)
if plot:
    plt.subplot(2, 1, 1)
    plt.xlim(-freq_bw, freq_bw)
    plt.legend(loc='upper right')
    plt.show()