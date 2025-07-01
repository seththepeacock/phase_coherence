import numpy as np



"""
HELPER FUNCTIONS
"""


def get_avg_vector(phase_diffs, PW=None, Pxx=None, Pyy=None):
    """Returns magnitude, phase of vector made by averaging over unit vectors with angles given by input phases

    Parameters
    ------------
        phase_diffs: array
          array of phase differences (N_pd, N_bins)
    """        
    Zs = np.exp(1j * phase_diffs)
    if PW is not None:
        Zs = PW * Zs
    avg_vector = np.mean(Zs, axis=0, dtype=complex)
    vec_strength = np.abs(avg_vector)
    
    if PW is not None:
        if Pxx is None or Pyy is None:
            raise ValueError("Pxx and Pyy must be provided if PW is provided")
        vec_strength = vec_strength**2 / (Pxx * Pyy)
    

    # finally, output the averaged vector's vector strength and angle with x axis (each a 1D array along the frequency axis)
    return vec_strength, np.angle(avg_vector)








def spectral_filter(wf, fs, cutoff_freq, type="hp"):
    """Filters waveform by zeroing out frequencies above/below cutoff frequency

    Parameters
    ------------
        wf: array
          waveform input array
        fs: int
          sample rate of waveform
        cutoff_freq: float
          cutoff frequency for filtering
        type: str, Optional
          Either 'hp' for high-pass or 'lp' for low-pass
    """
    fft_coefficients = np.fft.rfft(wf)
    frequencies = np.fft.rfftfreq(len(wf), d=1 / fs)

    if type == "hp":
        # Zero out coefficients from 0 Hz to cutoff_frequency Hz
        fft_coefficients[frequencies <= cutoff_freq] = 0
    elif type == "lp":
        # Zero out coefficients from cutoff_frequency Hz to Nyquist frequency
        fft_coefficients[frequencies >= cutoff_freq] = 0

    # Compute the inverse real-valued FFT (irfft)
    filtered_wf = np.fft.irfft(
        fft_coefficients, n=len(wf)
    )  # Ensure output length matches input

    return filtered_wf


def get_SIGMA(fwhm):
    """Gets SIGMA for (SciPy get_window) as a function of what you want the Gaussian FWHM to be (in samples)

    Parameters
    ------------
        fwhm: float
          Desired FWHM of the Gaussian window (in samples)
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma



# def param_or_PARAM(fs, param, PARAM, name):
#     if PARAM is None:
#         if param is None:
#             raise ValueError(f"We didn't get {name} in either seconds or samples!")
#         else:
#             PARAM = round(param * fs)
#     else:
#         # check if PARAM is an int
#         if not isinstance(PARAM, int):
#             raise ValueError(f"{name}S must be an integer!")
#         if param is not None:
#             # Here both param and PARAM have been passed in; check if they're equivalent  
#             if PARAM != round(param * fs):
#                 raise ValueError(
#                     f"You gave both {name}={param} and {name}S={PARAM}, and they're not equivalent for fs={fs}... which do we use?"
#                 )
#         else:
#             param = PARAM / fs
#     return param, PARAM


def check_xi_in_num_segs(xi, hop, name):
    xi_in_num_segs = xi / hop
    if np.abs(round(xi_in_num_segs) - (xi_in_num_segs)) > 1e-12:
        raise Exception(
            f"For {name}: this xi corresponds to going {xi_in_num_segs} segs away, needs to be an integer! Change XI={xi} or HOP={hop}."
        )
    return round(xi_in_num_segs)