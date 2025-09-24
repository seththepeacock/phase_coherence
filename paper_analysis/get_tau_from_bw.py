import phaseco as pc
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq
import numpy as np

def get_hpbw(win_type, tau, fs, nfft=None):
    if nfft is None:
        nfft = tau * 8
    win = get_window(win_type, tau)
    win_psd = np.abs(rfft(win, nfft)) ** 2
    target = np.max(win_psd) / 2

    idx = np.where(win_psd <= target)[0][0]
    hpbw = rfftfreq(nfft, 1 / fs)[idx] * 2
    return hpbw

def get_tau_from_hpbw(hpbw, win_type, fs, nfft=2**20, verbose=False):
    # Get the tau that leads to a window with hpbw closest to the target

    # Exponential search for an upper bound
    lo = 0
    hi = 1
    if verbose:
        print(f"Initializing exponential search for upper bound;")
        print(f"Lower bound is xi={lo}")
        print(f"Testing {hi}:")
    while get_hpbw(win_type, tau=hi, fs=fs, nfft=nfft) < hpbw:
        lo = hi
        hi *= 2
        if verbose:
            print(f"Tested {lo}, that was too small!")
            print(f"Testing {hi}:")
    if verbose:
        print(f"Found upper bound: {hi}")
        print(f"Initializing binary search")
    # Binary search between lo and hi until they are neighbors
    while hi - lo > 1:
        mid = (lo + hi + 1) // 2
        if verbose:
            print(f"[{lo}, {hi}] --- testing {mid}")
        mid_hpwb = get_hpbw(win_type, tau=mid, fs=fs, nfft=nfft)
        if mid_hpwb == hpbw:
            return mid_hpwb
        elif mid_hpwb < hpbw:
            lo = mid
        else:
            hi = mid
    if verbose:
        print(f"Now we're down to [{lo}, {hi}]")
    lo_hpwb = get_hpbw(win_type, tau=lo, fs=fs, nfft=nfft)
    hi_hpwb = get_hpbw(win_type, tau=hi, fs=fs, nfft=nfft)
    # Note we know that lo is strictly lower and hi is strictly higher
    if hi_hpwb-hpbw < hpbw - lo_hpwb:
        return hi_hpwb
    else:
        return lo_hpwb

fs = 44100
win_type = 'flattop'
hpbw = 20
nfft = 2**20
print(get_tau_from_hpbw(hpbw, win_type, fs, nfft))
