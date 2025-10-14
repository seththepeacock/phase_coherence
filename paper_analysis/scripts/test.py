from nddho_generator import nddho_generator
import phaseco as pc
import matplotlib.pyplot as plt
from scipy.signal import correlate, get_window
from scipy.optimize import curve_fit
import numpy as np
import time
from helper_funcs import *
os.chdir(r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence")


wf = np.random.normal(0, 1, 10*44100)
xis = {'xi_min_s': 0.005, 'xi_max_s':0.01, 'delta_xi_s': 0.005}
fs = 44100
f0s = None
tau = 2**14
hop = 0.01


start = time.time()

stft_old = get_stft(wf, fs, tau, hop=hop)[-1]


stop = time.time()
print(f"Old method: {stop-start:.3f}")

start = time.time()


stft_new = get_stft_new(wf, fs, tau, hop=hop)[-1]

stop = time.time()
print(f"New method: {stop-start:.3f}")


print(np.max(np.abs(stft_new-stft_old)))


