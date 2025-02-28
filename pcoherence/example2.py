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
wf = np.random.normal(loc=0, scale=1, size=N)

# Let's say we're making lots of plots with the same tau/xi; we can calculate the STFT in advance so it doesn't have to re-FFT everything over and over
tau = 0.05
xi = tau # For this example, tau and xi are the same (C_tau)

# Get STFT
f, stft = get_stft(wf, sr, tau=tau, xi=xi)

# Get coherence, passing in STFT
f, c_tau = get_coherence(wf, sr, tau=tau, xi=xi, ref_type="next_seg", reuse_stft=(f, stft))

# Now let's get the magnitude spectrum - reusing the STFT from before!
f, mag_spec = get_welch(wf, sr, tau=tau, xi=xi, reuse_stft=(f, stft)) 

# And C_theta; note ref_type can be "next_freq" or "both_freqs" for referencing to either side (which I believe is what Chris does)
f, C_theta = get_coherence(wf, sr, tau=tau, xi=xi, ref_type="both_freqs", reuse_stft=(f, stft))

# Now let's get a C_xi; we can't reuse the STFT since we're changing xi, but that's okay (the F in FFT is there for a reason)
f, C_xi = get_coherence(wf, sr, tau=tau, xi=xi, ref_type="next_seg")


# Plot
plt.figure(figsize=(12,6))
plt.title(r"$C_{\xi}$ for White Noise", fontsize=20)
plt.ylabel("Vector Strength", fontsize=18)
plt.xlabel("Frequency (kHz)", fontsize=18)
plt.plot(f/1000, coherence, label=r"$\xi$ = " + f"{xi*1000:.2f}ms, "+ r"$\tau$ = " + f"{tau*1000:.2f}ms")
plt.legend(fontsize=24)
plt.xlim(0, 6)
plt.tight_layout()
plt.show()





# With "return_dict" set to True, we get back a dictionary:
coherence_dict = get_coherence(wf, sr, tau=tau, xi=xi, ref_type="next_seg", reuse_stft=(f, stft), return_dict=True)

# which we of course can get the freq axis and coherence from:
f = coherence_dict["freq_ax"]
C_tau = coherence_dict["coherence"]

# But also other goodies like the phase diffs:
phase_diffs = coherence_dict["phase_diffs"] # a 2D array with shape (num_segs, num_freq_bins)

# Or the average of the absolute value of the phase diffs
avg_abs_phase_diffs = coherence_dict["|<phase_diffs>|"]

# Or the angle of the vector made by averaging over the phase-diff-unit-vectors (the one which we take the magnitude of for vector strength)
avg_vector_angle = coherence_dict["avg_vector_angle"]

# Or the number of segments used in the STFT
N_segs = coherence_dict["N_segs"]
