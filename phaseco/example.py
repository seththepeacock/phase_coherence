import phaseco as pc
import numpy as np

fs = 44100
wf = np.rand()
tau_s = 10 # 10 seconds
tau = round(tau_s * fs)
xi = 8

f, coherence = pc.get_coherence(wf, fs, xi, tau)
