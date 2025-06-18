import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, hann
from phaseco import *
from tqdm import tqdm

phaseco = 0

# Parameters
fs = 1000  # Sampling frequency (Hz)
N = 2**9  # Window length
tauS = N
tau = tauS / fs
nrealizations = 1000  # Number of independent realizations to average
xis = np.linspace(0.02, 0.1, 5) 
xiSs = (xis * fs).astype(int)
L = 2**12  # Signal length per realization
window = hann(tauS)
seg_spacing = xis[0]
noverlap = tauS - seg_spacing * fs 

# Store average coherence values across realizations
avg_coherence_vals = np.zeros((len(xis), 2))

# Simulate and compute coherence two difference ways
for phaseco in [0, 1]:
    for xi_idx, xi in enumerate(tqdm(xis)):
        xiS = int(xi*fs)
        coh_vals = []
        
        for _ in range(nrealizations):
            x_full = np.random.normal(0, 1, L + xiS)
            x_delayed = x_full[xiS:]
            x = x_full[:L]
            
            if phaseco:
                tau = tauS / fs
                xi = xiS / fs
                seg_spacingS = seg_spacing * fs
                d = get_coherence(x_full, fs, xi=xi, tauS=tauS, seg_spacing = seg_spacing, win_type=window, return_dict=True)
                f, Cxy, N_pd = d['f'], d['coherence'], d['N_pd']
                
                
                coh_vals.append(Cxy**2)
            else: 
                seg_spacingS = int(seg_spacing * fs)
                noverlap = tauS - seg_spacingS
                f, Cxy = coherence(x, x_delayed, fs=fs, window=window, nperseg=tauS, noverlap=noverlap)
                N_pd = int((L-noverlap) / (tauS-noverlap)) # equivalent to N_segs for this method
                coh_vals.append(Cxy)
        
        print(f"N_pd = {N_pd}, N= {N}, L ={L}, xiS = {xiS}, nperseg = {tauS}, seg_spacingS={seg_spacingS}")

        
        
        avg_coherence_vals[xi_idx, phaseco] = np.mean([np.mean(c) for c in coh_vals])  # Average over frequency and trials

# Theoretical bias estimate
def window_autocorr(w, xiS):
    return np.sum(w[:N - xiS] * w[xiS:]) if xiS < N else 0

R0 = window_autocorr(window, 0)
R_xiS = np.array([window_autocorr(window, xiS) for xiS in xiSs])
bias_estimate = (R_xiS / R0) ** 2

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(xis, avg_coherence_vals[:, 1], 'o-', label='Simulated Mean Coherence (White Noise) -- PHASECO', linewidth=2)
plt.plot(xis, avg_coherence_vals[:, 0], 'o-', label='Simulated Mean Coherence (White Noise) -- SCIPY', linewidth=2)
plt.plot(xis, bias_estimate, 's--', label='Theoretical Bias $(R_w(\\xi_S)/R_w(0))^2$', linewidth=2)
plt.xlabel("Lag (xi)")
plt.ylabel("Coherence")
plt.title("White Noise: Simulated vs Theoretical Self-Coherence Bias")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
