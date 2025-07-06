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
xis = np.linspace(0, 0.2, 6)
# xis = np.array([0])
xiSs = (xis * fs).astype(int)
L = 2**13  # Signal length per realization

win_type = 'boxcar'
window = get_window(win_type, tauS)
nrealizations = 10 # Number of independent realizations to average
freq = 10 # If we only want to look at a single frequency bin
calc_types_to_plot = ['phaseco', 'phaseco_PW', 'scipy_extra_manual']

if freq: # Get index, if necessary
    freq_idx = np.argmin(np.abs(np.fft.rfftfreq(tauS, 1/fs) - freq))

# for nrealizations in [1, 10, 100]:
#     for hop in (xis[1], xis[1]/2, xis[1]/4, xis[1]/8):
for nrealizations in [10]:

    for hop_s in [xis[1]]:
        hop = round(hop_s * fs)
        print(f"Running for L={L}, hop={hop}, nrealizations={nrealizations}")
        noverlap = tauS - hop * fs 

        # Store average coherence values across realizations
        avg_coherence_vals = defaultdict(lambda: np.zeros(len(xis)))

        # Simulate and compute coherence two difference ways
    
        for xi_idx, xi in enumerate(tqdm(xis)):
            xiS = int(xi*fs)
            coh_vals_phaseco = []
            coh_vals_phaseco_PW = []
            coh_vals_scipy_coherence = []
            coh_vals_scipy_manual = []
            coh_vals_scipy_pxy = []
            coh_vals_scipy_extra_manual = []
            
            for _ in range(nrealizations):
                
                t = np.arange(L)/fs
                f0 = 10
                # x_full = np.sin(2*np.pi*f0*t) + 2*np.random.randn(L)

                # x_full = np.random.normal(0, 1, L)
                # x = x_full[:(L-xiS)]
                # x_delayed = x_full[xiS:]
                x = np.random.normal(0, 1, L-xiS)
                x_delayed = np.random.normal(0, 1, L-xiS)
                
                
                tau = tauS / fs
                xi = xiS / fs
                noverlap = tauS - hop

                
                "Scipy Coherence (equivalent to manual, IFF k_offset = tauS // 2 for Pxx and Pxy)" 
                # f, Cxy = coherence(x, x_delayed, fs=fs, window=window, nperseg=tauS, noverlap=noverlap, detrend=False)
                # N_pd = int(((L-xiS)-noverlap) / (tauS-noverlap)) 
                # if freq:
                #     Cxy = Cxy[freq_idx]
                # coh_vals_scipy_coherence.append(Cxy)
                


                "SCIPY EXTRA MANUAL"
                
                # Compute coherence using scipy and phaseco
                SFT = ShortTimeFFT(window, hop, fs, fft_mode='onesided', scale_to=None, phase_shift=None)
                

                k_offset = tauS // 2
                # k_offset = 0  `   `
                # Compute PSD/CSD: csd uses y, x (note reversed order)
                # scipy_stft_full = SFT.stft(x_full, p0=0, p1=(len(x_full) - noverlap) // hop, k_offset=k_offset)
                # scipy_stft = SFT.stft(x, p0=0, p1=(len(x) - noverlap) // hop, k_offset=k_offset)
                # scipy_stft_delayed = SFT.stft(x_delayed, p0=0, p1=(len(x_delayed) - noverlap) // hop, k_offset=k_offset)
                # Pxy = scipy_stft_delayed * np.conj(scipy_stft)
                Pxy = SFT.spectrogram(x_delayed, x, p0=0, p1=(len(x) - noverlap) // hop, k_offset=k_offset, detr=None)
                Pxx = SFT.spectrogram(x, x, p0=0, p1=(len(x) - noverlap) // hop, k_offset=k_offset, detr=None)
                Pyy = SFT.spectrogram(x_delayed, x_delayed, p0=0, p1=(len(x_delayed) - noverlap) // hop, k_offset=k_offset, detr=None)
                

                
                
                
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
                
                
                
                Cxy = np.abs(Pxy)**2 / (Pxx * Pyy)
                coh_vals_scipy_extra_manual.append(Cxy)
                
                
                # "Scipy Manual"
                # f, Pxy_manual = csd(x, x_delayed, fs=fs, window=win_type, nperseg=tauS, noverlap=noverlap, detrend=False, scaling='density')
                
                # Cxy = np.abs(Pxy_manual)**2 / (Pxx * Pyy)
                # # if freq:
                # #     Cxy = Cxy[freq_idx]
                # coh_vals_scipy_manual.append(Cxy)
                
            
                # "Phaseco (Weighted)"
                # PW = True
                # d = get_coherence(x_full, fs, xiS=xiS, nperseg=tauS, hop=hop, win_type=win_type, PW=PW, reuse_stft=None, return_dict=True)
                # f, Cxy, N_pd = d['f'], d['coherence'], d['N_pd']
                # # if freq:
                # #     Pxy = Pxy[freq_idx]
                # coh_vals_phaseco_PW.append(Cxy)
                
                # "Phaseco (Unweighted)"
                # PW = None
                # d = get_coherence(x_full, fs, xiS=xiS, nperseg=tauS, hop=hop, win_type=win_type, PW=PW, reuse_stft=None, return_dict=True)
                # f, Cxy, N_pd = d['f'], d['coherence'], d['N_pd']
                # # if freq:
                # #     Pxy = Pxy[freq_idx]
                # coh_vals_phaseco.append(Cxy)
                    

                
             
            # print(f"N_pd = {N_pd}, N= {N}, L ={L}, xiS = {xiS}, nperseg = {tauS}, hop={hop}")

            
            # Average over frequency and trials
            # avg_coherence_vals['scipy_coherence'][xi_idx] = np.mean([np.mean(c) for c in coh_vals_scipy_coherence]) 
            # avg_coherence_vals['scipy_manual'][xi_idx] = np.mean([np.mean(c) for c in coh_vals_scipy_manual])
            avg_coherence_vals['scipy_extra_manual'][xi_idx] = np.mean([np.mean(c) for c in coh_vals_scipy_extra_manual])
            avg_coherence_vals['phaseco'][xi_idx] = np.mean([np.mean(c) for c in coh_vals_phaseco])  
            avg_coherence_vals['phaseco_PW'][xi_idx] = np.mean([np.mean(c) for c in coh_vals_phaseco_PW])

        # Theoretical bias estimate
        def window_autocorr(w, xiS):
            return np.sum(w[:N - xiS] * w[xiS:]) if xiS < N else 0

        R0 = window_autocorr(window, 0)
        R_xiS = np.array([window_autocorr(window, xiS) for xiS in xiSs])
        bias_estimate = (R_xiS / R0) ** 2

        # Plotting
        plt.figure(figsize=(10, 6))
        for calculation_type in calc_types_to_plot:
            plt.plot(xis, avg_coherence_vals[calculation_type], 'o-', label=f'Simulated Mean Coherence (White Noise) -- {calculation_type}', linewidth=2)
        plt.plot(xis, bias_estimate, 's--', label='Theoretical Bias $(R_w(\\xi_S)/R_w(0))^2$', linewidth=2)
        plt.xlabel("Lag (xi)")
        plt.ylabel("Coherence")
        # plt.ylim(0, 1)
        title = f"White Noise Coherence Decay - wf_length={L}, nrealizations={nrealizations}, freq={freq}, hop={hop}"
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Alternate Coherences\Figs\{title}.png")
        plt.show()
        



# def get_csd(x, y, fs, tau, hop=None, win_type='boxcar'):
#     window = get_window(win_type, tau)

#     if hop is None:
#         hop = tau # non-overlapping

#     noverlap = tau - hop

#     SFT = ShortTimeFFT(window, hop, fs, fft_mode='onesided', scale_to=None, phase_shift=None)

#     # Compute spectrogram: csd uses y, x (note reversed order)
#     Pxy = SFT.spectrogram(y, x, p0=0, p1=(len(x) - noverlap) // hop, k_offset=tau // 2)

#     # Apply onesided doubling (if real and return_onesided=True)
#     if np.isrealobj(x) and SFT.fft_mode == 'onesided':
#         Pxy[1:-1 if SFT.mfft % 2 == 0 else None, :] *= 2

#     # Average across time segments (axis=1 if time is columns)
#     Pxy = np.mean(Pxy, axis=1)

#     # Normalize (done already)
#     Pxy /= fs * np.sum(window ** 2)
    
#     return SFT.f, Pxy