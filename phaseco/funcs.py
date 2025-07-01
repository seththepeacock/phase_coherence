import numpy as np
from .helper_funcs import *
from scipy.signal import get_window, ShortTimeFFT
from scipy.fft import rfft, rfftfreq, fftshift
import pywt
import warnings
from tqdm import tqdm





"""
PRIMARY FUNCTIONS
"""


def get_stft(
    wf,
    fs,
    tau=None,
    hop=None,
    xi=None,
    rho=None,
    win_type="boxcar",
    N_segs=None,
    fftshift_segs=False,
    return_dict=False,
):
    """Returns the segmented fft and associated freq ax of the given waveform

    Parameters
    ------------
        wf: array
          waveform input array
        fs: int
          sample rate of waveform
        tau: float
          length (in time) of each window
        hop: float
          length (in time) between the start of successive segments
        rho: double, Optional
          Applies a Gaussian window with whose FWHM is rho*xi
        win_type: String, Optional
          Window to apply before the FFT
        N_segs: int, Optional
          If this isn't passed, then just get the maximum number of segments of the given size
        cutoff_freq: double, Optional
          HPFs the total waveform by zeroing out frequencies above this frequency
        fftshift_segs: bool, Optional
          Shifts each time-domain window of the fft with fftshift() to center your window in time and make it zero-phase (no effect on coherence)
        return_dict: bool, Optional
          Returns a dict with:
          d["f"] = f
          d["stft"] = stft
          d["segmented_wf"] = segmented_wf
          d["seg_start_indices"] = seg_start_indices
    """
    
    
        
    
    # Handle C_tau case
    if rho == -1:
        if xi is None:
            raise ValueError(
                "You must input xi when rho=-1!"
            )
        og_tau = tau
        tau = min(xi, og_tau)

    # First, get the last index of the waveform
    final_wf_index = len(wf) - 1

    # next, we get what we would be the largest potential seg_start_index
    latest_potential_seg_start_index = final_wf_index - (
        tau - 1
    )  # start at the final_wf_index. we need to collect tau points. this final index is our first one, and then we need tau - 1 more.
    seg_start_indices = np.arange(0, latest_potential_seg_start_index + 1, hop)
    # the + 1 here is because np.arange won't ever include the "stop" argument in the output array... but it could include (stop - 1) which is just our final_seg_start_index!

    # if number of segments is passed in, we make sure it's less than the length of seg_start_indices
    if N_segs is not None:
        max_N_segs = len(seg_start_indices)
        if N_segs > max_N_segs:
            raise Exception(
                f"That's more segments than we can manage - you want {N_segs}, but we can only do {max_N_segs}!"
            )
    else:
        # if no N_segs is passed in, we'll just use the max number of segments
        N_segs = len(seg_start_indices)
        if N_segs != int((len(wf) - tau) / hop) + 1:
            print(
                f"Hm THATS WEIRD - N_SEGS METHOD 1 gives {N_segs} while other method gives {int((len(wf) - tau) / hop) + 1}"
            )

    # Check how win_type has been passed in and get window accordingly
    if isinstance(win_type, str) or (
        isinstance(win_type, tuple) and isinstance(win_type[0], str)
    ):
        # Get window function (boxcar is no window)
        window = get_window(win_type, tau)
        # Do rho windowing if rho was passed in (and is not -1)
        if rho is not None and rho != -1:
            if xiS is None:
                raise ValueError(
                    "If you want to do rho-windowing, you need to input xiS!"
                )
            fwhmS = rho * xi
            SIGMA = get_SIGMA(fwhmS)  # rho is the proportion of xi which we want the gaussian window's FWHM to be
            gaussian = get_window(("gauss", SIGMA), tau)
            window = gaussian * window  # In normal use, window will just be a boxcar, but...
            if win_type != "boxcar":  # Throw a warning if you're going to double window
                warnings.warn(
                    f"You're double-windowing with both a Gaussian and a {win_type}!"
                )
    else:  # Allow for passing in a custom window
        window = win_type
        if len(window) != tau:
            raise Exception(
                f"Your custom window must be the same length as tau ({tau})!"
            )

    # Check if we should be windowing or if unnecessary because they're all ones
    do_windowing = np.any(window != 1)

    # Get segmented waveform matrix
    
    # First handle the weird case where the chunk of waveform is just hop(=xi) and we must zero pad to make up the difference
    if rho == -1:
        segmented_wf = np.zeros((N_segs, og_tau))
        for k in range(N_segs):
            # grab the waveform in this segment
            seg_start = seg_start_indices[k]
            seg_end = seg_start + tau
            seg = wf[seg_start:seg_end]
            if do_windowing:
                seg = seg * window
            if (
                fftshift_segs
            ):  # optionally swap the halves of the waveform to effectively center it in time
                seg = fftshift(seg)
                seg = np.pad(seg, pad_width=((og_tau - tau) / 2)) # In this case, pad on both sides
            else:
                seg = np.pad(
                    seg, pad_width=(0, og_tau - tau)
                )  # Just pad zeros to the end until we get to our original og_tau
            segmented_wf[k, :] = seg
        # Finally, get frequency axis
        f = rfftfreq(og_tau, 1/fs)


    # Now handle the standard case
    else:
        segmented_wf = np.zeros((N_segs, tau))
        for k in range(N_segs):
            seg_start = seg_start_indices[k]
            seg_end = seg_start + tau
            # grab the waveform in this segment
            seg = wf[seg_start:seg_end]
            if do_windowing:
                seg = seg * window
            if (
                fftshift_segs
            ):  # optionally swap the halves of the waveform to effectively center it in time
                seg = fftshift(seg)
            segmented_wf[k, :] = seg
        # Finally, get frequency axis
        f = rfftfreq(tau, 1/fs)
    
    # Now we do the ffts!   

    # initialize segmented fft array
    N_bins = len(f)
    stft = np.zeros((N_segs, N_bins), dtype=complex)

    # get ffts
    for k in range(N_segs):
        stft[k, :] = rfft(segmented_wf[k, :])

    if return_dict:

        return {
            "f": f,
            "stft": stft,
            "seg_start_indices": seg_start_indices,
            "segmented_wf": segmented_wf,
            "hop": hop,
            "fs": fs,
            "xi": xi,
            "tau": tau,
            "hop": hop,
            "window": window,
            "fs": fs,
        }

    else:
        return f, stft


def get_coherence(
    wf,
    fs,
    xi=None,
    tau=None,
    hop=None,
    rho=None,
    win_type="boxcar",
    pw=None,
    N_segs=None,
    reuse_stft=None,
    ref_type="next_seg",
    freq_bin_hop=1,
    return_avg_abs_phase_diffs=False,
    return_dict=False
):
    """Gets the phase coherence of the given waveform with the given window size

    Parameters
    ------------
        wf: array
          waveform input array
        fs:
          sample rate of waveform
        tau: float
          length (in samples) of each segment
        hop: float
          length (in samples) between the start of successive segments
        xi: float
          length (in samples) between phase references
        win_type: String, Optional
          Window to apply before the FFT
        N_segs: int, Optional
          If this isn't passed, then just get the maximum number of segments of the given size
        reuse_stft: tuple, Optional
          If you want to avoid recalculating the STFT, pass it in here as a dictionary
        ref_type: str, Optional
          Either "next_seg" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
        cutoff_freq: double, Optional
          HPFs the total waveform by zeroing out frequencies above this frequency
        freq_bin_hop: int, Optional
          How many bins over to reference phase against for next_freq
        return_dict: bool, Optional
          Defaults to only returning the coherence; if this is enabled, then a dictionary is returned with keys:
          d["f"] = f
          d["coherence"] = coherence
          d["phases"] = phases
          d["phase_diffs"] = phase_diffs
          d["<|phase_diffs|>"] = avg_abs_phase_diffs
          d["avg_vector_angle"] = avg_vector_angle
          d["N_segs"] = N_segs
          d["stft"] = stft
    """


    
    
    # if nothing was passed into reuse_stft then we need to recalculate it
    if reuse_stft is None:
        stft_dict = get_stft(
            wf=wf,
            fs=fs,
            tau=tau,
            xi=xi,
            hop=hop,
            N_segs=N_segs,
            rho=rho,
            win_type=win_type,
            return_dict=True,
        )
    else:
        # If a stft is passed in, we just use it
        stft_dict = reuse_stft

    # Retrieve necessary items from dictionary
    f = stft_dict["f"]
    stft = stft_dict["stft"]
    tau = stft_dict["tau"]
    hop = stft_dict["hop"]
    xi = stft_dict["xi"]
    window = stft_dict["window"]

    # Make sure these are valid
    if stft.shape[1] != f.shape[0]:
        raise Exception("STFT and frequency axis don't match!")


    # calculate necessary params from the stft
    N_segs, N_bins = np.shape(stft)

    # get phases
    phases = np.angle(stft)
    
        

    # Unwrap phases only if needed (it's slow!)
    # This won't affect coherence (except maybe for the case below) but is necessary for <|phase diffs|>
    if return_avg_abs_phase_diffs:
        unwrapping_axis = 0 if ref_type == "next_seg" else 1
        # Unwrap phases along the segment axis (0) or frequency bin axis (1)
        phases = np.unwrap(phases, axis=unwrapping_axis)

    # we can reference each phase against the phase of the same frequency in the next window:
    if ref_type == "next_seg":
        # Make sure we can reference this xi in this STFT; xi should be an integer number of segs away
        xi_in_num_segs = check_xi_in_num_segs(
            xiS, hop, "xi_in_num_segs"
        )  # Note that in classic case, xi=hop and this is xi/hop=1!

        # initialize array for phase diffs; we won't be able to get it for the final few segs though
        N_pd = N_segs - xi_in_num_segs
        phase_diffs = np.zeros((N_pd, N_bins))

        # Optionally calculate the weights for the average vector
        if pw:
            pw = np.zeros((N_pd, N_bins))
            mags = np.abs(stft)
            for seg in range(N_pd):
                pw[seg] = (mags[seg + xi_in_num_segs] * mags[seg])
            pw[:, 1:-1 if tau % 2 == 0 else None] *= 2
            # Scale!
            scaling_factor = np.sum(window**2)*fs   
            pw = pw / scaling_factor
            
            
            stft_squared = (np.abs(stft))**2
            if xi_in_num_segs == 0:
                Pxx = np.mean(stft_squared, axis=0) / scaling_factor
                Pyy = Pxx.copy()
            else:
                Pxx = np.mean(stft_squared[0:-xi_in_num_segs], axis=0) / scaling_factor
                Pyy = np.mean(stft_squared[xi_in_num_segs:], axis=0) / scaling_factor
            
            Pxx[1:-1 if tau % 2 == 0 else None] *= 2
            Pyy[1:-1 if tau % 2 == 0 else None] *= 2
        else:
            Pxx = None
            Pyy = None
        

        # calc phase diffs
        for seg in range(N_pd):
            # take the difference between the phases in this current segment and the one xi seconds away
            phase_diffs[seg] = phases[seg + xi_in_num_segs] - phases[seg]
            # phase_diffs[seg] = phases[seg] - phases[seg + xi_in_num_segs]
                

        coherence, avg_vector_angle = get_avg_vector(phase_diffs, PW=pw, Pxx=Pxx, Pyy=Pyy)

    # or we can reference it against the phase of the next frequency in the same window:
    elif ref_type == "next_freq":

        # initialize array for phase diffs; -freq_bin_hop is because we won't be able to get it for the #(freq_bin_hop) freqs
        phase_diffs = np.zeros((N_segs, N_bins - freq_bin_hop))
        # we'll also need to take the last #(freq_bin_hop) bins off the f
        f = f[0:-freq_bin_hop]

        # calc phase diffs
        for seg in range(N_segs):
            for freq_bin in range(N_bins - freq_bin_hop):
                phase_diffs[seg, freq_bin] = (
                    phases[seg, freq_bin + freq_bin_hop] - phases[seg, freq_bin]
                )

        # get final coherence
        coherence, avg_vector_angle = get_avg_vector(phase_diffs, PW=pw)

        # Since this references each frequency bin to its adjacent neighbor, we'll plot them w.r.t. the average frequency
        # this corresponds to shifting everything over half a bin width
        # this corresponds to shifting everything over half a bin width
        bin_width = f[1] - f[0]
        f = f + (bin_width / 2)

    # or we can reference it against the phase of both the lower and higher frequencies in the same window
    elif ref_type == "both_freqs":

        # initialize arrays
        # even though we only lose ONE freq point with lower and one with higher, we want to get all the points we can get from BOTH so we do - 2
        pd_low = np.zeros((N_segs, N_bins - 2))
        pd_high = np.zeros((N_segs, N_bins - 2))
        # take the first and last bin off the freq ax
        f = f[1:-1]

        # calc phase diffs
        for seg in range(N_segs):
            for freq_bin in range(1, N_bins - 1):
                # the - 1 is so that we start our phase_diffs arrays at 0 and put in N_bins-2 points.
                # These will correspond to our new frequency axis.
                pd_low[seg, freq_bin - 1] = (
                    phases[seg, freq_bin] - phases[seg, freq_bin - 1]
                )
                pd_high[seg, freq_bin - 1] = (
                    phases[seg, freq_bin + 1] - phases[seg, freq_bin]
                )
        coherence_low, _ = get_avg_vector(pd_low)
        coherence_high, _ = get_avg_vector(pd_high)
        # average the coherences you would get from either of these
        coherence = (coherence_low + coherence_high) / 2
        # set the phase diffs to one of these (could've also been pd_high)
        phase_diffs = pd_low

    else:
        raise Exception("You didn't input a valid ref_type!")

    if not return_dict:
        return f, coherence

    else:  # Return full dictionary
        d = {
            "coherence": coherence,
            "phases": phases,
            "phase_diffs": phase_diffs,
            "avg_vector_angle": avg_vector_angle,
            "N_pd": N_pd,
            "N_segs": N_segs,
            "f": f,
            "stft": stft,
            "stft_dict": stft_dict,
            "tau": tau,
            "hop" : hop,
            "xi" : xi,
            "fs": fs,
            "PW": pw,
        }
        if return_avg_abs_phase_diffs:
            # get <|phase diffs|> (note we're taking mean w.r.t. PD axis 0, not frequency axis)
            d["avg_abs_phase_diffs"] = np.mean(np.abs(phase_diffs), 0)
        return d

def get_colossogram_coherences(
    wf,
    fs,
    xi_min,
    xi_max,
    delta_xi,
    tau,
    dyn_win=("rho", 0.7, False),
    pw=None,
    const_N_pd=1,
    dense_stft=1,
    global_xi_max_s=None,
    return_dict=False,
):
    # Unpack dyn_win
    match dyn_win[0]:
        case "rho":
            dyn_win_type, rho, snapping_rhortle = dyn_win
        case _:
            raise ValueError("Haven't implemented any non-rho windowing yet!")
    


    # Calculate xi array and N_bins
    num_xis = int((xi_max - xi_min) / delta_xi) + 1
    xis = np.linspace(xi_min, xi_max, num_xis)
    f = np.array(rfftfreq(tau*fs, 1 / fs))
    N_bins = len(f)
    # Initialize coherences array
    coherences = np.zeros((N_bins, len(xiSs)))

    if dense_stft:
        # Set the STFT hop according to the minimum xi in this colossogram
        hop = xi_min

        # Make sure all xis correspond to an integer*hop; since hop=xi_min, this is equivalent to making sure hop=delta_xi
        if hop != delta_xi:
            raise Exception(
                "hop(=xi_min) must be equal to delta_xi or else our xi values won't correspond to referencing to an integer number of segs away!"
            )

    "Get the minimum N_pd"
    # Set the max xi that will determine this minimum number of phase diffs
    # (either max xi within this colossogram, or a global one so it's constant across all colossograms in comparison)
    if global_xi_max_s is None:
        global_xi_max = xi_max 
    elif not const_N_pd:
        raise Exception(
            "Why did you pass a global max xi if you're not holding N_pd constant?"
        )
    else:
        global_xi_max = global_xi_max_s * fs # Note we wanted to pass in global_xi_max in secs so it can be consistent across samplerates

    # Calculate the maxmium/minimum hop which will determine the minimum/maximum N_pd
    if dense_stft:
        # There's only one seg spacing in this case!
        max_hop = hop
        min_hop = hop
    else:
        # hop=xi, but we know that the minimum N_pd is going to be when hop is maximum so:
        max_hop = global_xi_max
        min_hop = xi_min


    # Get the number of segments we have to shift to go global_xi_max seconds over in both the minimal/maximal N_pd cases
    global_xi_max_in_num_segs = check_xi_in_num_segs(
        global_xi_max, max_hop, "global_xi_max_in_num_segs"
    )
    xi_min_in_num_segs = check_xi_in_num_segs(
        xi_min, min_hop, "xi_min_in_num_segs"
    )

    # Get the number of phase diffs (we can do this outside xi loop since it's constant)
    # There are int((len(wf)-tau)/hop)+1 full tau-segments.
    # But the last global_xi_max_in_num_segs (=1 in non-dense_stft-case, since global_xi_max=max_hop) ones won't have a reference.
    N_pd_min = int((len(wf) - tau) / max_hop) + 1 - global_xi_max_in_num_segs
    N_pd_max = int((len(wf) - tau) / min_hop) + 1 - xi_min_in_num_segs

    if const_N_pd:
        # If we're holding it constant, we hold it to the minimum
        N_pd = N_pd_min
        # Even though the *potential* N_pd_max is bigger, we just use N_pd_min all the way so this is also the max
        # Even though the *potential* N_pd_max is bigger, we just use N_pd_min all the way so this is also the max
        N_pd_max = N_pd_min

    # Loop through xis and calculate coherences
    for i, xi in enumerate(tqdm(xis)):
        # If we're above the threshold where dynamic windowing is needed, shut er off!
        if snapping_rhortle and xiS > nperseg:
            rho = None
        # Set hop if not already done above
        if not dense_stft:
            hop = xi

        # Get current xi in terms of number of segments
        current_xi_in_num_segs = check_xi_in_num_segs(
            xiS, hop, f"current_xi_in_num_segs for xiS={xiS}"
        )

        # Calculate N_pd (assuming we're not holding it constant, in which case it was already done outside of loop)
        # Calculate N_pd (assuming we're not holding it constant, in which case it was already done outside of loop)
        if not const_N_pd:
            # This is just as many segments as we possibly can
            # There are int((len(wf)-tau)/hopS)+1 full tau-segments, but the last current_xi_in_num_segs ones won't have a reference
            N_pd = (
                int((len(wf) - tau) / hop) + 1 - int(current_xi_in_num_segs)
            )

        # Get N_segs; We only need as many segments as there are PDs, plus the last one needs to be able to reference
        N_segs = N_pd + current_xi_in_num_segs
        coherences[:, i] = get_coherence(
            wf=wf,
            fs=fs,
            tau=tau,
            xi=xi,
            hop=hop,
            dyn_win=dyn_win,
            N_segs=N_segs,
            pw=pw
        )[1]

    # Convert to xis_s
    xis_s = xis / fs

    if return_dict:
        return {
            "xis": xis,
            "xis_s" : xis_s,
            "f": f,
            "coherences": coherences,
            "tau": tau,
            "fs": fs,
            "N_pd_min": N_pd_min,
            "N_pd_max": N_pd_max,
            "hop": hop,
            "snapping_rhortle": snapping_rhortle,
            "global_xi_max": global_xi_max
        }
    else:
        return xis_s, f, coherences
    


def get_asym_coherence(wf, fs, tau, xi, fwhm=None, rho=None, N_segs=None):
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
        SIGMA = get_SIGMA(fwhm=fwhm, fs=fs)
    elif rho is not None:
        SIGMA = get_SIGMA(fwhm=rho * xi, fs=fs)

    # Generate gaussian
    gaussian = get_window(("gauss", SIGMA), tau)
    left_window = np.ones(int(tau))
    right_window = np.ones(int(tau))
    left_window[int(tau / 2) :] = gaussian[
        int(tau / 2) :
    ]  # Left window starts with ones and ends with gaussian (gaussian on overlapping side)
    right_window[0 : int(tau / 2)] = gaussian[0 : int(tau / 2)]  # Vice versa

    # Get STFTs for each side
    f, left_stft = get_stft(
        wf=wf, fs=fs, tau=tau, hop=xi, N_segs=N_segs, win_type=left_window
    )
    f, right_stft = get_stft(
        wf=wf, fs=fs, tau=tau, hop=xi, N_segs=N_segs, win_type=right_window
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

    coherence, avg_vector_angle = get_avg_vector(phase_diffs)  # Get vector strength

    return f, coherence


def get_welch(
    wf,
    fs,
    tau=None,
    hop=None,
    N_segs=None,
    win_type="boxcar",
    scaling="density",
    reuse_stft=None,
    return_dict=False,
):
    """Gets the Welch averaged power of the given waveform with the given window size

    Parameters
    ------------
        wf: array
          waveform input array
        fs: int
          sample rate of waveform
        tau: float
          length (in samples) of each window; used in get_stft and to calculate normalizing factor
        win_type: String, Optional
          Window to apply before the FFT
        N_segs: int, Optional
          Used in get_stft;
            if this isn't passed, then just gets the maximum number of segments of the given size
        scaling: String, Optional
          "mags" (magnitudes) or "density" (PSD) or "spectrum" (power spectrum)
        reuse_stft: tuple, Optional
          If you want to avoid recalculating the segmented fft, pass it in here along with the frequency axis as (f, stft)
        return_dict: bool, Optional
          Returns a dict with:
          d["f"] = f
          d["spectrum"] = spectrum
          d["segmented_spectrum"] = segmented_spectrum
    """
    # if nothing was passed into reuse_stft then we need to recalculate it
    if reuse_stft is None:
        f, stft = get_stft(
            wf=wf,
            fs=fs,
            tau=tau,
            hop=hop,
            N_segs=N_segs,
            win_type=win_type,
        )
    else:
        f, stft = reuse_stft
        # Make sure these are valid
        if stft.shape[1] != f.shape[0]:
            raise Exception("STFT and frequency axis don't match!")

    # calculate necessary params from the stft
    N_segs, N_bins = np.shape(stft)


    # initialize array
    segmented_spectrum = np.zeros((N_segs, N_bins))

    # get spectrum for each window
    for win in range(N_segs):
        segmented_spectrum[win, :] = (np.abs(stft[win, :])) ** 2

    # average over all segments (in power)
    spectrum = np.mean(segmented_spectrum, 0)

    window = get_window(win_type, tau)
    S1 = np.sum(window)
    S2 = np.sum(window**2)
    ENBW = tau * S2 / S1**2 # In samples
    U = S2 / tau
    if scaling == "mags":
        spectrum = np.sqrt(spectrum)
        normalizing_factor = 1 / S1

    elif scaling == "spectrum":
        normalizing_factor = 1 / S1**2

    elif scaling == "density":
        # Start with standard periodogram/spectrum scaling, then divide by bin width (times ENBW in samples)
        bin_width = 1 / tau
        bin_width = fs / tau
        normalizing_factor = 1 / S1**2
        normalizing_factor = normalizing_factor /  (ENBW * bin_width)
    

    else:
        raise Exception("Scaling must be 'mags', 'density', or 'spectrum'!")

    # Normalize; since this is an rfft, we should multiply by 2
    spectrum = spectrum * 2 * normalizing_factor
    # Except DC bin should NOT be scaled by 2
    spectrum[0] = spectrum[0] / 2
    # Nyquist bin shouldn't either (note this bin only exists if tau is even)
    if tau % 2 == 0:
        spectrum[-1] = spectrum[-1] / 2

    if not return_dict:
        return f, spectrum
    else:
        return {"f": f, "spectrum": spectrum, "segmented_spectrum": segmented_spectrum}

def get_csd(x, y, fs, tau, hop=None, win_type='boxcar'):
    window = get_window(win_type, tau)

    if hop is None:
        hop = tau # non-overlapping

    noverlap = tau - hop

    SFT = ShortTimeFFT(window, hop, fs, fft_mode='onesided', scale_to=None, phase_shift=None)

    # Compute spectrogram: csd uses y, x (note reversed order)
    Pxy = SFT.spectrogram(y, x, p0=0, p1=(len(x) - noverlap) // hop, k_offset=tau // 2)

    # Apply onesided doubling (if real and return_onesided=True)
    if np.isrealobj(x) and SFT.fft_mode == 'onesided':
        Pxy[1:-1 if SFT.mfft % 2 == 0 else None, :] *= 2

    # Average across time segments (axis=1 if time is columns)
    Pxy = np.mean(Pxy, axis=1)

    # Normalize (done already)
    Pxy /= fs * np.sum(window ** 2)
    
    return SFT.f, Pxy


def get_cwt(wf, fs, fb, f):
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


def get_wavelet_coherence(wf, fs, f, fb=1.0, cwt=None, xi_s=0.0025):
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
        cwt: array, Optional
            CWT coefficients of waveform if already calculated
        xi: float, Optional
            length (in time) between the start of successive segments
    """
    if cwt is None:
        cwt = get_cwt(wf=wf, fs=fs, fb=fb, f=f)
    # get phases
    phases = np.angle(cwt)
    xi = round(xi_s * fs)

    N_segs = int(len(wf) / xi) - 1

    # initialize array for phase diffs
    phase_diffs = np.zeros((N_segs, len(f)))

    # calc phase diffs
    for seg in range(N_segs):
        phase_diffs[seg, :] = phases[int(seg * xi)] - phases[int((seg + 1) * xi)]

    wav_coherence, _ = get_avg_vector(phase_diffs)

    return wav_coherence
