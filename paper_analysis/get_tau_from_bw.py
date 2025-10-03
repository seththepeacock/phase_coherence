import phaseco as pc
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq
import numpy as np

from N_xi_fit_funcs import *

win_type = 'flattop'
nfft = 2**25
for hpbw in [20]:
    print(f"HPBW={hpbw}")
    for fs in [44100, 48000]:
        print(f"fs={fs} -- [{get_tau_from_bw(hpbw, win_type, fs, nfft, verbose=False)[0]}]")

# print(get_hpbw(win_type='flattop', tau=int(round((0.025*fs))), fs=44100))