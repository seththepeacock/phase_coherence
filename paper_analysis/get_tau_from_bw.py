import phaseco as pc
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq
import numpy as np

from N_xi_fit_funcs import *

fs = 48000
win_type = 'flattop'
hpbw = 50
nfft = 2**25
print(get_tau_from_bw(hpbw, win_type, fs, nfft, verbose=True))

# print(get_hpbw(win_type='hann', tau=int(round((0.025*fs))), fs=44100))