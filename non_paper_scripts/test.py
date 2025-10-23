from nddho_generator import nddho_generator
import phaseco as pc
import matplotlib.pyplot as plt
from scipy.signal import correlate, get_window
from scipy.optimize import curve_fit
import numpy as np
import time
from helper_funcs import *
os.chdir(r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence")


wf = np.random.normal(0, 1, int(0.5*44100))
xis = {'xi_min_s': 0.005, 'xi_max_s':0.05, 'delta_xi_s': 0.005}
fs = 44100
f0s = [1000]
tau = 2**14
hop = 1
win_meth = {'method':'static', 'win_type':'flattop'}

start = time.time()

colossogram_old = get_colossogram(wf, fs, f0s=f0s, xis=xis, win_meth=win_meth, tau=tau, hop=hop, pw=False, nbacf=False)[-1]


stop = time.time()
print(f"Old method: {stop-start:.3f}")

start = time.time()


xis_s, f, colossogram_new = get_colossogram(wf, fs, f0s=f0s, xis=xis, win_meth=win_meth, tau=tau, hop=hop, pw=False, nbacf=True)

stop = time.time()
print(f"New method: {stop-start:.3f}")

plt.scatter(xis_s, colossogram_new, s=10)
plt.scatter(xis_s, colossogram_old, s=2)
plt.show()


print(np.max(np.abs(colossogram_new-colossogram_old)))


