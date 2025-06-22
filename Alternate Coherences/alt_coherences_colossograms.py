import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, coherence
from phaseco import *
from tqdm import tqdm
from collections import defaultdict


# Parameters
fs = 1000  # Sampling frequency (Hz)
N = 2**9  # Window length
tauS = N
tau = tauS / fs
xis = np.linspace(0, 0.1, 6)
# xis = np.array([0])
xiSs = (xis * fs).astype(int)
L = 2**13  # Signal length per realization
f = np.fft.rfftfreq(tauS, 1/fs)

win_type = 'boxcar'
window = get_window(win_type, tauS)
nrealizations = 1 # Number of independent realizations to average
calc_types_to_plot = ['scipy_extra_manual','phaseco_weighted']



for hop in [xis[1]]:
    print(f"Running for L={L}, hop={hop}, nrealizations={nrealizations}")
    hopS = round(hop * fs)
    noverlap = tauS - hop * fs 

    # Store average coherence values across realizations
    coherences = defaultdict(lambda: np.zeros((len(f), len(xis))))

    # Simulate and compute coherence two difference ways

    for xi_idx, xi in enumerate(xis):
        xiS = int(xi*fs)
        
        for _ in range(nrealizations):
            
            t = np.arange(L)/fs
            f0 = 250
            x_full = np.sin(2*np.pi*f0*t) + 2*np.random.randn(L)

            # x_full = np.random.normal(0, 1, L)
            x = x_full[:(L-xiS)]
            x_delayed = x_full[xiS:]
            
            
            tau = tauS / fs
            xi = xiS / fs
            hopS = round(hop * fs)
            noverlap = tauS - hopS
            
            # Get welch
            # Pxx = welch(x, fs=fs, window=win_type, nperseg=tauS, noverlap=noverlap, detrend=False)[1]
            # Pyy = welch(x_delayed, fs=fs, window=win_type, nperseg=tauS, noverlap=noverlap, detrend=False)[1]
            
            
            "Scipy Coherence (equivalent to manual, IFF k_offset = tauS // 2 for Pxx and Pxy)" 
            # f, Cxy = coherence(x, x_delayed, fs=fs, window=window, nperseg=tauS, noverlap=noverlap, detrend=False)
            # N_pd = int(((L-xiS)-noverlap) / (tauS-noverlap)) 
            # if freq:
            #     Cxy = Cxy[freq_idx]
            # coh_vals_scipy_coherence.append(Cxy)
            
            
            
            
            
            
            
            
            # # Phaseco (OG)
            # d = get_coherence(x_full, fs, xi=xi, tauS=tauS, hop = hop, win_type=window, power_weights=None, return_dict=True)
            # f, phasecoherence, N_pd = d['f'], d['coherence'], d['N_pd']
            # Cxy = phasecoherence**2
            # if freq:
            #     Cxy = Cxy[freq_idx]
            # coh_vals_phaseco.append(Cxy)
            
            
            

            "SCIPY EXTRA MANUAL"
            
            # Compute coherence using scipy and phaseco
            SFT = ShortTimeFFT(window, hopS, fs, fft_mode='onesided', scale_to=None, phase_shift=None)
            

            k_offset = tauS // 2
            # k_offset = 0  `   `
            # Compute PSD/CSD: csd uses y, x (note reversed order)
            scipy_stft_full = SFT.stft(x_full, p0=0, p1=(len(x_full) - noverlap) // hopS, k_offset=k_offset)
            # scipy_stft = SFT.stft(x, p0=0, p1=(len(x) - noverlap) // hopS, k_offset=k_offset)
            # scipy_stft_delayed = SFT.stft(x_delayed, p0=0, p1=(len(x_delayed) - noverlap) // hopS, k_offset=k_offset)
            # Pxy = scipy_stft_delayed * np.conj(scipy_stft)
            Pxy = SFT.spectrogram(x_delayed, x, p0=0, p1=(len(x) - noverlap) // hopS, k_offset=k_offset, detr=None)
            Pxx = SFT.spectrogram(x, x, p0=0, p1=(len(x) - noverlap) // hopS, k_offset=k_offset, detr=None)
            Pyy = SFT.spectrogram(x_delayed, x_delayed, p0=0, p1=(len(x_delayed) - noverlap) // hopS, k_offset=k_offset, detr=None)
            
            # print(f"Pxy shape: {Pxy.shape}")
            # print(f"Pxy2 shape: {Pxy2.shape}")

# 
            # print("Pxy shape:", Pxy.shape)
            # print("Pxx shape:", Pxx.shape)
            
            
            
            # Scale
            Pxy[1:-1 if SFT.mfft % 2 == 0 else None, :] *= 2
            Pxy /= fs * np.sum(window ** 2)
            Pxx[1:-1 if SFT.mfft % 2 == 0 else None, :] *= 2
            Pxx /= fs * np.sum(window ** 2)
            Pyy[1:-1 if SFT.mfft % 2 == 0 else None, :] *= 2
            Pyy /= fs * np.sum(window ** 2)
            
            
            # Average across time segments (axis=1 if time is columns)
            Pxy = np.mean(Pxy, axis=-1)
            Pxx = np.mean(Pxx, axis=-1)
            Pyy = np.mean(Pyy, axis=-1)
            
            
            
            Cxy_extra_manual = np.abs(Pxy)**2 / (Pxx * Pyy)
            coherences['scipy_extra_manual'][:, xi_idx] = Cxy_extra_manual
            
            
            # "Scipy Manual (Equivalent to extra manual)"
            # f, Pxy_manual = csd(x, x_delayed, fs=fs, window=win_type, nperseg=tauS, noverlap=noverlap, detrend=False, scaling='density')
            
            # Cxy_manual = np.abs(Pxy_manual)**2 / (Pxx * Pyy)
            # coherences['scipy_manual'][:, xi_idx] = Cxy_manual
            
            # print(np.max(np.abs(Cxy_extra_manual - Cxy_manual)))
            
        
            "Phaseco (Weighted)"
            # Build STFT dict 
            scipy_stft_full = scipy_stft_full.T
            f = SFT.f               
            stft_dict = {
                "tau": tau,
                "tauS": tauS,
                "xi": xi,
                "xiS": xiS,
                "f": f,
                "stft": scipy_stft_full,
                "hop": hop,
                "hopS": hopS,
                "window": window
            }
            power_weights = True
            stft_dict = None
            d = get_coherence(x_full, fs, xiS=xiS, tauS=tauS, hopS=hopS, win_type=win_type, power_weights=power_weights, reuse_stft=stft_dict, return_dict=True)
            f, Pxy, N_pd = d['f'], d['coherence'], d['N_pd']
            if power_weights is not None:
                Cxy = np.abs(Pxy)**2 / (Pxx * Pyy) 
            else:
                Cxy = Pxy
            coherences['phaseco_weighted'][:, xi_idx] = Cxy
    
    # for coherence_type in coherences.keys():
    #     coherences[coherence_type] = coherences[coherence_type][1:-1]
    # f = f[1:-1]
    print(np.sort(np.abs(coherences['phaseco_weighted'] - coherences["scipy_extra_manual"])[:, ], axis=None)[-5:])
    # print(np.max((coherences['phaseco_weighted'][:])))
        

    # Theoretical bias estimate
    def window_autocorr(w, xiS):
        return np.sum(w[:N - xiS] * w[xiS:]) if xiS < N else 0

    R0 = window_autocorr(window, 0)
    R_xiS = np.array([window_autocorr(window, xiS) for xiS in xiSs])
    bias_estimate = (R_xiS / R0) ** 2
    
    plt.figure(figsize=(5*len(calc_types_to_plot), 6))
    for plot_idx, calculation_type in enumerate(calc_types_to_plot):
        plt.subplot(len(calc_types_to_plot), 1, plot_idx + 1)
        plot_colossogram(coherences[calculation_type], f, xis, title=calculation_type)
            
            
    plt.show()
