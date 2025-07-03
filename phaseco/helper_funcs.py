import numpy as np
from scipy.signal import get_window


"""
HELPER FUNCTIONS
"""


def get_avg_vector(phase_diffs):
    """Returns magnitude, phase of vector made by averaging over unit vectors with angles given by input phases

    Parameters
    ------------
        phase_diffs: array
          array of phase differences (N_pd, N_bins)
    """        
    Zs = np.exp(1j * phase_diffs)
    avg_vector = np.mean(Zs, axis=0, dtype=complex)
    vec_strength = np.abs(avg_vector)
    

    # finally, output the averaged vector's vector strength and angle with x axis (each a 1D array along the frequency axis)
    return vec_strength, np.angle(avg_vector)

def get_phaseco_win(dyn_win, tau, xi, static_win=None, ref_type="next_seg"):
        if dyn_win is not None:
            if ref_type != "next_seg":
                raise ValueError(
                    f"You passed in a dynamic windowing method but you're using a {ref_type} reference; these were designed for next_seg!"
                )
            if static_win is not None:
                raise ValueError(
                    "You passed in a dynamic windowing method and a static window; which do we use???"
                )
            # Unpack dyn_win
            match dyn_win[0]:
                case "rho":
                    rho, snapping_rhortle = dyn_win[1], dyn_win[2]
                    if snapping_rhortle and xi > tau:
                        win = None
                    else:
                        desired_fwhm = rho * xi
                        sigma = desired_fwhm / (2 * np.sqrt(2 * np.log(2)))
                        win = get_window(("gaussian", sigma), tau)
                case "eta":
                    eta, win_type = dyn_win[1], dyn_win[2]
                    tau = get_tau_from_eta(tau, xi, eta, win_type)
                    raise ValueError("Haven't implemented eta dynamic windowing yet!")
                    
                case _:
                    raise ValueError(
                        "Dynamic windowing method must be either 'rho' or 'eta'!"
                    )
        elif static_win is not None:
            # Here we just use the static window since dyn_win was None
            if isinstance(static_win, str) or (
                isinstance(static_win, tuple) and isinstance(static_win[0], str)
            ):
                # Get window function (boxcar is no window)
                win = get_window(static_win, tau)
            else:  # Here we assume static_win was just an array of window coefficients
                win = static_win
                if len(win) != tau:
                    raise Exception(
                        f"Your window must be the same length as tau (={tau})!"
                    )
        else:
            # Neither was passed, so we just set win to None and get_stft will use a boxcar
            win = None

        return win, tau # Note that unless explicitly changed via eta windowing, tau just passes through

def get_tau_from_eta(tau, xi, eta, win_type):
    """Returns the minimum tau such that the expected coherence for white noise for this window is less than eta

    Parameters
    ------------
    """ 
    for test_tau in range(tau):
        win = get_window(win_type, tau)
        R_w_0 = get_win_autocorr(win, 0)
        R_w_xi = get_win_autocorr(win, xi)
        



def get_win_autocorr(win, xi):
    win_0 = win[0:xi]
    win_delayed = win[xi:]
    return np.sum(win_0 * win_delayed)
    