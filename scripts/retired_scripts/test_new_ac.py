import numpy as np
from scipy.signal import get_window
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
import phaseco as pc
from N_xi_fit_funcs import *



def get_ac_from_stft_og(stft, pw, xi_nsegs, wa=False, return_pd=False):

    pd_dict = {}  # This will pass through empty if not return_pd

    # Powerweighted (C_xi)
    if pw:
        # Calculate coherence
        if wa:
            Pxy = np.mean(stft_xi * np.conj(stft_0), 0)
            avg_weights = np.mean(np.abs(stft_xi) * np.abs(stft_0), 0)
            return Pxy / avg_weights
        else:
            Pxy = np.mean(stft_xi * np.conj(stft_0), 0)
            Pxx = np.mean(pc.magsq(stft_0), 0)
            Pyy = np.mean(pc.magsq(stft_xi), 0)
            autocoherence = pc.magsq(Pxy) / (Pxx * Pyy)
            if return_pd:
                pds = np.angle(Pxy)
                avg_pd = np.angle(np.mean(np.exp(1j * pds), 0, dtype=complex))

    # Non powerweighted (C_xi^phi)
    else:

        phases = np.angle(stft)
        phases_0 = phases[0:-xi_nsegs]
        phases_xi = phases[xi_nsegs:]
        # Calculate coherence
        pds = phases_xi - phases_0
        if return_pd:
            autocoherence, avg_pd = pc.get_avg_vector(pds, return_angle=return_pd)
        else:
            autocoherence = pc.get_avg_vector(
                pds, return_angle=return_pd
            )  # This will either be one or two arguments according to return_angle

    # Calculate pd things if requested
    if return_pd:
        pd_dict["pds"] = pds
        pd_dict["avg_pd"] = avg_pd
        pd_dict["avg_abs_pd"] = pc.get_avg_abs_pd(pds, ref_type="time")

    return autocoherence, pd_dict  # Latter two arguments are possibly None


def get_ac_from_stft(stft_0, stft_xi, pw, wa=False, return_pd=False):
    pd_dict = {}  # This will pass through empty if not return_pd

    # Universals
    xy = stft_xi * np.conj(stft_0)
    # Powerweighted (C_xi)
    if pw:
        # Calculate coherence
        Pxy = np.mean(xy, 0)
        if wa:
            avg_weights = np.mean(np.abs(stft_xi) * np.abs(stft_0), 0)
            autocoherence = Pxy / avg_weights
        else:
            Pxx = np.mean(pc.magsq(stft_0), 0)
            Pyy = np.mean(pc.magsq(stft_xi), 0)
            autocoherence = pc.magsq(Pxy) / (Pxx * Pyy)
            if return_pd:
                pds = np.angle(Pxy)
                avg_pd = np.angle(np.mean(np.exp(1j * pds), 0, dtype=complex))

    # Non powerweighted (C_xi^phi)
    else:
        # Normalize for unit vectors
        xy_norm = xy / np.abs(xy)
        # Get average unit vector
        avg_xy_norm = np.mean(xy_norm, axis=0)
        # Take vector strength for autocoherence 
        autocoherence = np.abs(avg_xy_norm)
        if return_pd:
            # Calculate the angle of the average unit vector
            avg_pd = np.angle(avg_xy_norm)

    # Add various pd things if requested
    if return_pd:
        pd_dict["pds"] = pds
        pd_dict["avg_pd"] = avg_pd
        # Calculate phase diffs
        pds = np.angle(xy)
        pd_dict["avg_abs_pd"] = pc.get_avg_abs_pd(pds, ref_type="time")

    return autocoherence, pd_dict  # Latter two arguments are possibly None



wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
    species="Anole",
    wf_idx=0,
)

stft = pc.get_stft(wf=wf,
                fs=fs,
                tau=8192,
                hop=1000,
                nfft=None,
                win='hann',
)[-1]
xi_nsegs = 10

start = time.time()
ac_og, _ = get_ac_from_stft_og(stft, pw=False, xi_nsegs=xi_nsegs)
stop = time.time()
print(f"OG Method took {(stop-start):.2f}s")

start = time.time()
stft_0 = stft[:-xi_nsegs]
stft_xi = stft[xi_nsegs:]
ac, _ = get_ac_from_stft(stft_0, stft_xi, pw=False)
stop = time.time()
print(f"New Method took {(stop-start):.2f}s")

start = time.time()
stft_0 = stft[:-xi_nsegs]
stft_xi = stft[xi_nsegs:]
ac_pw, _ = get_ac_from_stft(stft_0, stft_xi, pw=True)
stop = time.time()
print(f"PW Method took {(stop-start):.2f}s")


print(f"Max diff is {np.max(np.abs(ac_og-ac))}")