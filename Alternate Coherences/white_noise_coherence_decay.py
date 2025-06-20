import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd
from phaseco import *
from tqdm import tqdm
from collections import defaultdict


# Parameters
fs = 2000  # Sampling frequency (Hz)
N = 2**9  # Window length
tauS = N
tau = tauS / fs
xis = np.linspace(0, 0.1, 6)
# xis = np.array([0])
xiSs = (xis * fs).astype(int)
L = 2**13  # Signal length per realization
win_type = 'boxcar'
window = get_window(win_type, tauS)
nrealizations = 10 # Number of independent realizations to average
freq = 0 # If we only want to look at a single frequency bin


if freq: # Get index, if necessary
    freq_idx = np.argmin(np.abs(np.fft.rfftfreq(tauS, 1/fs) - freq))

# for nrealizations in [1, 10, 100]:
#     for seg_spacing in (xis[1], xis[1]/2, xis[1]/4, xis[1]/8):
for nrealizations in [10]:

    for seg_spacing in [xis[1]]:
        print(f"Running for L={L}, seg_spacing={seg_spacing}, nrealizations={nrealizations}")


        noverlap = tauS - seg_spacing * fs 

        # Store average coherence values across realizations
        avg_coherence_vals = defaultdict(lambda: np.zeros(len(xis)))

        # Simulate and compute coherence two difference ways
    
        for xi_idx, xi in enumerate(tqdm(xis)):
            xiS = int(xi*fs)
            coh_vals_phaseco = []
            coh_vals_phaseco_weighted = []
            coh_vals_scipy_coherence = []
            coh_vals_scipy_manual = []
            coh_vals_scipy_pxy = []
            coh_vals_scipy_extra_manual = []
            
            for _ in range(nrealizations):
                x_full = np.random.normal(0, 1, L)
                x = x_full[:(L-xiS)]
                x_delayed = x_full[xiS:]

                # Compute coherence using scipy and phaseco
                tau = tauS / fs
                xi = xiS / fs
                seg_spacingS = round(seg_spacing * fs)
                noverlap = tauS - seg_spacingS
                
                # # Scipy
                # f, Cxy = coherence(x, x_delayed, fs=fs, window=window, nperseg=tauS, noverlap=noverlap, detrend=False)
                N_pd = int(((L-xiS)-noverlap) / (tauS-noverlap)) 
                # if freq:
                #     Cxy = Cxy[freq_idx]
                # coh_vals_scipy_coherence.append(Cxy)
                
                Pxx = welch(x, fs=fs, window=win_type, nperseg=tauS, noverlap=noverlap, detrend=False)[1]
                Pyy = welch(x_delayed, fs=fs, window=win_type, nperseg=tauS, noverlap=noverlap, detrend=False)[1]
                
                # # Scipy manual
                f, Pxy = csd(x, x_delayed, fs=fs, window=win_type, nperseg=tauS, noverlap=noverlap, detrend=False, scaling='density')
                Cxy = np.abs(Pxy)**2 / (Pxx * Pyy)
                if freq:
                    Cxy = Cxy[freq_idx]
                coh_vals_scipy_manual.append(Cxy)
                
                
                
                
                # # Phaseco (OG)
                # d = get_coherence(x_full, fs, xi=xi, tauS=tauS, seg_spacing = seg_spacing, win_type=window, power_weights=None, return_dict=True)
                # f, phasecoherence, N_pd = d['f'], d['coherence'], d['N_pd']
                # Cxy = phasecoherence**2
                # if freq:
                #     Cxy = Cxy[freq_idx]
                # coh_vals_phaseco.append(Cxy)
                
                # Phaseco (Weighted)
                d = get_coherence(x_full, fs, xi=xi, tauS=tauS, seg_spacing = seg_spacing, win_type=win_type, power_weights=True, scipy_stft=True, return_dict=True)
                f, Pxy, N_pd = d['f'], d['coherence'], d['N_pd']
                if freq:
                    Pxy = Pxy[freq_idx]
                Cxy = np.abs(Pxy)**2 / (Pxx * Pyy) 
                coh_vals_phaseco_weighted.append(Cxy)
                
                "PROVEN THIS IS EQUIVALENT TO SCIPY"
                # scipy extra manual
                f, Pxy = get_csd(x, x_delayed, fs=fs, tauS=tauS, seg_spacing=seg_spacing, win_type=win_type)
                Cxy = np.abs(Pxy)**2 / (Pxx * Pyy)
    
                if freq:
                    Cxy = Cxy[freq_idx]
                coh_vals_scipy_extra_manual.append(Cxy)
                    

                
            
            # print(f"N_pd = {N_pd}, N= {N}, L ={L}, xiS = {xiS}, nperseg = {tauS}, seg_spacingS={seg_spacingS}")

            
            # Average over frequency and trials
            # avg_coherence_vals['scipy_coherence'][xi_idx] = np.mean([np.mean(c) for c in coh_vals_scipy_coherence]) 
            # avg_coherence_vals['scipy_manual'][xi_idx] = np.mean([np.mean(c) for c in coh_vals_scipy_manual]) 
            avg_coherence_vals['scipy_extra_manual'][xi_idx] = np.mean([np.mean(c) for c in coh_vals_scipy_extra_manual]) 
            # avg_coherence_vals['phaseco'][xi_idx] = np.mean([np.mean(c) for c in coh_vals_phaseco])  
            avg_coherence_vals['phaseco_weighted'][xi_idx] = np.mean([np.mean(c) for c in coh_vals_phaseco_weighted])

        # Theoretical bias estimate
        def window_autocorr(w, xiS):
            return np.sum(w[:N - xiS] * w[xiS:]) if xiS < N else 0

        R0 = window_autocorr(window, 0)
        R_xiS = np.array([window_autocorr(window, xiS) for xiS in xiSs])
        bias_estimate = (R_xiS / R0) ** 2

        # Plotting
        plt.figure(figsize=(10, 6))
        # for calculation_type in ['phaseco', 'phaseco_weighted', 'scipy_coherence', 'scipy_manual']:
        for calculation_type in ['phaseco_weighted', 'scipy_extra_manual']:
            plt.plot(xis, avg_coherence_vals[calculation_type], 'o-', label=f'Simulated Mean Coherence (White Noise) -- {calculation_type}', linewidth=2)
        plt.plot(xis, bias_estimate, 's--', label='Theoretical Bias $(R_w(\\xi_S)/R_w(0))^2$', linewidth=2)
        plt.xlabel("Lag (xi)")
        plt.ylabel("Coherence")
        # plt.ylim(0, 1)
        title = f"White Noise Coherence Decay - wf_length={L}, nrealizations={nrealizations}, freq={freq}, seg_spacing={seg_spacing}, N_pd={N_pd}"
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f"Alternate Coherences\{title}.png")
