import numpy as np

from phaseco.funcs import get_stft
from phaseco.helper_funcs import *
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq, fftshift
import pywt
from tqdm import tqdm


"""
OLD FUNCTIONS
"""


def asym_coherence(wf, fs, tau, xi, fwhm=None, rho=None, N_segs=None):
    """Gets the coherence using an asymmetric window (likely will eventually be assimilated as an option in get_coherence())
    Parameters
    ------------
        wf: array
          waveform input array
        fs: int
          sampling rate of waveform
        tau: int
          length (in samples) of each window
        xi: float
          amount (in time) between the start points of adjacent segments
        fwhm: double, Optional
          FWHM of the Gaussian window (in seconds), need either this or rho
        rho: double, Optional
          Applies a Gaussian window whose FWHM is rho*xi, need either this or fwhm
    """

    # Get SIGMA for the gaussian part of the window either via rho (dynamic FWHM = rho*xi) or fixed FWHM
    if fwhm is None and rho is None:
        raise ValueError("You must input either FWHM or rho!")
    elif fwhm is not None:
        sigma = get_sigma(fwhm=fwhm, fs=fs)
    elif rho is not None:
        sigma = get_sigma(fwhm=rho * xi, fs=fs)

    # Generate gaussian
    gaussian = get_window(("gauss", sigma), tau)
    left_window = np.ones(int(tau))
    right_window = np.ones(int(tau))
    left_window[int(tau / 2) :] = gaussian[
        int(tau / 2) :
    ]  # Left window starts with ones and ends with gaussian (gaussian on overlapping side)
    right_window[0 : int(tau / 2)] = gaussian[0 : int(tau / 2)]  # Vice versa

    # Get STFTs for each side
    f, left_stft = get_stft(
        wf=wf, fs=fs, tau=tau, hop=xi, N_segs=N_segs, win=left_window
    )
    f, right_stft = get_stft(
        wf=wf, fs=fs, tau=tau, hop=xi, N_segs=N_segs, win=right_window
    )
    # Extract angles
    left_phases = np.angle(left_stft)
    right_phases = np.angle(right_stft)
    # Calc phase diffs
    N_bins = len(f)  # Number of frequency bins
    phase_diffs = np.zeros((N_segs - 1, N_bins))
    for win in range(N_segs - 1):
        # Take the difference between the phases, with the windows aligned as to minimize shared samples
        phase_diffs[win] = right_phases[win + 1] - left_phases[win]

    coherence, avg_pd = get_avg_vector(phase_diffs)  # Get vector strength

    return f, coherence


def cwt(wf, fs, fb, f):
    """Returns the CWT coefficients of the given waveform with a complex morelet wavelet
    Parameters
    ------------
        wf: array
            waveform input array
        fs: int
            sampling rate of waveform
        fb: float
            bandwidth of wavelet in time; bigger bandwidth = better frequency resolution but less time resolution (similar to tau)
        f: array
            frequency axis
    """
    # Perform a Continuous Wavelet Transform (CWT)
    dt = 1 / fs
    # Define wavelet and get its center frequency (for scale-frequency conversion)
    wavelet_string = f"cmor{fb}-1.0"
    wavelet = pywt.ContinuousWavelet(wavelet_string)
    fc = pywt.central_frequency(
        wavelet
    )  # This will always be 1.0 because we set it that way
    # Convert frequencies to scales
    scales = fc / (f * dt)
    coefficients, f_cwt = pywt.cwt(
        wf, scales, wavelet, method="fft", sampling_period=dt
    )
    return coefficients.T


def wavelet_coherence(wf, fs, f, fb=1.0, xi):
    """Returns the wavelet coherence of the given waveform with a complex morelet wavelet
    Parameters
    ------------
        wf: array
            waveform input array
        fs: int
            sampling rate of waveform
        f: array
            frequency axis
        fb: float, Optional
            bandwidth of wavelet in time; bigger bandwidth = better frequency resolution but less time resolution (similar to tau)
        xi: float, Optional
            length (in time) between the start of successive segments
    """
    cwt = cwt(wf=wf, fs=fs, fb=fb, f=f)
    # get phases
    phases = np.angle(cwt)

    N_segs = int(len(wf) / xi) - 1

    # initialize array for phase diffs
    phase_diffs = np.zeros((N_segs, len(f)))

    # calc phase diffs
    for seg in range(N_segs):
        phase_diffs[seg, :] = phases[int(seg * xi)] - phases[int((seg + 1) * xi)]

    wav_coherence, _ = get_avg_vector(phase_diffs)

    return wav_coherence