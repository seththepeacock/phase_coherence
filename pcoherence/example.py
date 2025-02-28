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

# Get coherence, referencing each phase to that of the next segment for C_xi
tau = 0.05
xi = 0.0025
f, C_xi = get_coherence(wf, sr, tau=tau, xi=xi, ref_type="next_seg")

# Plot
plt.figure(figsize=(12,6))
plt.title(r"$C_{\xi}$ for White Noise", fontsize=20)
plt.ylabel("Vector Strength", fontsize=18)
plt.xlabel("Frequency (kHz)", fontsize=18)
f_khz = f/1000
plt.plot(f_khz, C_xi, label=r"$\xi$ = " + f"{xi*1000:.2f}ms, "+ r"$\tau$ = " + f"{tau*1000:.2f}ms")
plt.legend(fontsize=24)
plt.xlim(0, 6)
plt.tight_layout()
plt.show()


