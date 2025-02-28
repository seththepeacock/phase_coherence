# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Import funcs file
from funcs import *

# Generate random noise
t_max = 10
sr = 44100
t = np.arange(0, t_max, 1/sr)
N = len(t)
wf = np.random.normal(loc=0, scale=1000, size=N)

# Let's say we're making lots of plots with the same tau/xi; we can calculate the STFT in advance so it doesn't have to re-FFT everything over and over
tau = 0.05
xi = tau # For this example, tau and xi are the same (C_tau)

# Get STFT
f, stft = get_stft(wf, sr, tau=tau, xi=xi)

# Get coherence, passing in STFT
f, C_tau = get_coherence(wf, sr, tau=tau, xi=xi, ref_type="next_seg", reuse_stft=(f, stft))

# Now let's get the magnitude spectrum - still reusing the STFT from before!
f, mag_spec = get_welch(wf, sr, tau=tau, xi=xi, scaling='mags', reuse_stft=(f, stft)) 

# And C_theta; note ref_type can be "next_freq" or "both_freqs" for referencing to either side - I think Chris' code does the latter so we'll use that
# Either way, frequency axis is slightly modified so we'll redefine f (both_freqs loses the first and last bin)
f_theta, C_theta = get_coherence(wf, sr, tau=tau, xi=xi, ref_type="both_freqs", reuse_stft=(f, stft))

# Now let's get a C_xi; we can't reuse the STFT since we're changing xi, but that's okay (the F in FFT is there for a reason!)
xi2 = 0.0025
f, C_xi = get_coherence(wf, sr, tau=tau, xi=xi2, ref_type="next_seg")


# Plot
fig = plt.figure(figsize=(12,6))
plt.title(r"White Noise Measures, "+ r"$\tau$ = " + f"{tau*1000:.2f}ms", fontsize=20)
plt.ylabel("Vector Strength", fontsize=18)
plt.xlabel("Frequency (kHz)", fontsize=18)
plt.scatter(f/1000, C_tau, marker='+', label=r"$C_{\tau}$ ($\xi$ = $\tau$)")
plt.scatter(f_theta/1000, C_theta, marker='1', label=r"$C_{\theta}$ ($\xi$ = $\tau$)")
plt.scatter(f/1000, C_xi, marker='D', s=10, label=r"$C_{\xi}$ ($\xi$ = " + f"{xi2*1000:.2f}ms)")
plt.xlim(0, 6)
plt.ylim(0, 1)
ax2 = plt.twinx()
ax2.scatter(f/1000, 10*np.log10(mag_spec), marker='.', color='r', label="Magnitude Spectrum")
ax2.set_ylabel("Magnitude (dB)", fontsize=18)
fig.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.show()


# With "return_dict" set to True, we get back a dictionary:
tau = 0.05
xi = 0.0025
coherence_dict = get_coherence(wf, sr, tau=tau, xi=xi, ref_type="next_seg", reuse_stft=(f, stft), return_dict=True)

# which we of course can get the freq axis and coherence from:
f = coherence_dict["freq_ax"]
C_xi = coherence_dict["coherence"]

# But also other goodies like the phase diffs themselves:
phase_diffs = coherence_dict["phase_diffs"] # a 2D array with shape (num_segs, num_freq_bins)

# Or the average of the absolute value of the phase diffs
avg_abs_phase_diffs = coherence_dict["<|phase_diffs|>"]

# Or the angle of the vector made by averaging over the phase-diff-unit-vectors (the one which we take the magnitude of for vector strength)
avg_vector_angle = coherence_dict["avg_vector_angle"]

# Or the number of segments used in the STFT
N_segs = coherence_dict["N_segs"]

