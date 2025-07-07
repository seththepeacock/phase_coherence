import numpy as np
from phaseco.helper_funcs import *
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq, fftshift
from tqdm import tqdm
import phaseco as pc


"""
PRIMARY USER-FACING FUNCTIONS
"""


def get_stft(
    wf,
    fs,
    tau,
    nfft=None,
    hop=None,
    win=None,
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
            length (in samples) of each segment
        nfft: int, optional
            length (in samples) of FFT; if nfft != tau, segments are zero padded to make up the difference
        hop: float, optional
            length (in samples) between the start of successive segments (defaults to tau //2, 50% overlap)
        win: str or array, optional
            Window to apply before the FFT, either array of coefficents (length tau) or string for SciPy get_window()
        N_segs: int, optional
            Limits number of segments to extract from waveform
        fftshift_segs: bool, optional
            Shifts each time-domain window of the fft with fftshift() to center your window in time and make it zero-phase (has no effect on coherence)
        return_dict: bool, optional
            Returns a dict with keys 'f', 'stft', 'seg_start_indices', 'segmented_wf', 'hop', 'fs', 'tau', 'hop', 'win', 'fs'
    Returns
    -------
    f : numpy.ndarray
        Frequency axis.
    stft : numpy.ndarray
        Short-time Fourier transform with dimensions (segments, frequency bins).
    """

    # Handle defaults
    if hop is None:
        hop = tau // 2
    if nfft is None:
        nfft = tau

    if tau > len(wf):
        raise ValueError(f"tau={tau} > len(wf)={len(wf)}; choose a smaller tau!")

    # Check validity of parameters
    if nfft < tau:
        raise ValueError(f"nfft={nfft} < tau={tau}, should be >=")

    
    # Calculate the seg_start_indices

    # First, get the last index of the waveform
    final_wf_idx = len(wf) - 1

    # next, we get what we would be the largest potential seg_start_index
    last_potential_seg_start_idx = final_wf_idx - (
        tau - 1
    )  # start at the final_wf_index. we need to collect tau points. this final index is our first one, and then we need tau - 1 more.
    seg_start_indices = np.arange(0, last_potential_seg_start_idx + 1, hop)
    # + 1 is because highest index np.arange includes is (stop - 1), and we want it to include up to last_potential_seg_start_idx

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
                f"Hm that's strange - the first N_segs calculation gives {N_segs} while other method gives {int((len(wf) - tau) / hop) + 1}"
            )

    # Check if a win has been passed in and set do_windowing based on if it's nontrivial
    if win is None:
        do_windowing = False
    else:
        # in normal get_coherence usage, win will just be an array of window coefficients;
        # this logic allows for passing in a string to get the window via SciPy get_window
        if isinstance(win, str) or (isinstance(win, tuple) and isinstance(win[0], str)):
            # Get window function
            win = get_window(win, tau)
        # Set do_windowing = True unless it's just a boxcar (all 1s)
        do_windowing = np.any(win != 1)

    # Get segmented waveform matrix

    # Detect if we need to zero pad or not
    zpad = True if nfft != tau else False

    segmented_wf = np.empty((N_segs, nfft))
    for k in range(N_segs):
        # grab the waveform in this segment
        seg_start = seg_start_indices[k]
        seg_end = seg_start + tau
        seg = wf[seg_start:seg_end]
        if do_windowing:
            seg = seg * win
        if (
            fftshift_segs
        ):  # optionally swap the halves of the waveform to effectively center it in time
            seg = fftshift(seg)
            if zpad:
                seg = np.pad(
                    seg, pad_width=((nfft - tau) / 2)
                )  # In this case, pad on both sides
        else:
            if zpad:
                seg = np.pad(
                    seg, pad_width=(0, nfft - tau)
                )  # Just pad zeros to the end until we get to our desired nfft
        segmented_wf[k, :] = seg
    # Finally, get frequency axis
    f = rfftfreq(nfft, 1 / fs)

    # Now we do the ffts!

    # initialize segmented fft array
    N_bins = len(f)
    stft = np.empty((N_segs, N_bins), dtype=complex)

    # IMPLEMENT rfft with different nfft instead of zpadding since probably more efficient1
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
            "tau": tau,
            "hop": hop,
            "win": win,
            "fs": fs,
        }

    else:
        return f, stft


def get_coherence(
    wf,
    fs,
    xi,
    pw, 
    tau,
    nfft=None,
    hop=None,
    win_meth={"method": "rho", "rho": 0.7},
    N_pd=None,
    ref_type="next_seg",
    freq_bin_hop=1,
    return_avg_abs_pd=False,
    return_dict=False,
):
    """Gets the phase coherence of the waveform against a copy of the waveform advanced xi samples

    Parameters
    ------------
        wf: array
            waveform input array
        fs: float
            sample rate of waveform
        xi: int
            length (in samples) to advance copy of signal for phase reference
        pw: bool
            weights the vector strength average by the magnitude of each segment * magnitude of xi advanced segment
        tau: int
            length (in samples) of each segment, used in get_stft()
        nfft: int
            length of fft, does zero padding if nfft > tau, used in get_stft()
        hop: int
            length between the start of successive segments, used in get_stft()
        win_meth: dict, optional
            windowing method dictionary; see get_pc_win() for details
        N_pd: int, optional
            number of phase differences in vector strength, converted to N_segs for get_stft()
        ref_type: str, optional
            Either "next_seg" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
        freq_bin_hop: int, optional
            How many bins over to reference phase against for next_freq
        return_avg_abs_pd: bool, optional
            Calculates <|phase diffs|> and adds to output dictionary
        return_dict: bool, optional
            Defaults to only returning (f, coherence); if this is enabled, then a dictionary is returned with keys:
          'coherence', 'phase_diffs', 'avg_pd', 'N_pd', 'N_segs', 'f', 'stft', 'tau', 'hop', 'xi', 'fs', 'pw', 'avg_abs_pd'
    """

    # Handle defaults
    if hop is None:
        hop = tau  # Zero overlap (at least, outside of xi referencing considerations)
    if nfft is None:
        nfft = (
            tau  # No zero padding (at least, outside of eta windowing considerations)
        )

    # Get window (and possibly redfine tau if doing eta windowing)
    win, tau = get_win_pc(win_meth, tau, xi, ref_type)

    # we can reference each phase against the phase of the same frequency in the next window:
    if ref_type == "next_seg":
        # First, check if we can get away with a single STFT; this only works if each xi is an integer number of segment hops away
        xi_nsegs = round(xi / hop)
        if np.abs(xi_nsegs - (xi / hop)) < 1e-12:
            # Yes we can! Calculate this single stft:
            N_segs = N_pd + xi_nsegs
            f, stft = get_stft(wf, fs, tau, hop, xi, nfft, win, N_segs)
            N_bins = len(f)
            # First, do the pw case
            if pw:
                xy = np.empty((N_pd, N_bins), dtype=complex)
                # IMPLEMENT vectorized way to do this?
                for k in range(N_pd):
                    xy[k] = np.conj(stft[k]) * stft[k + xi_nsegs]
                Pxy = np.mean(xy, 0)
                powers = stft.real**2 + stft.imag**2
                # IMPLEMENT quicker way to do this since most of the mean is shared?
                Pxx = np.mean(powers[0:xi_nsegs], 0)
                Pyy = np.mean(powers[xi_nsegs:], 0)
                coherence = (Pxy.real**2 + Pxy.imag**2) / (Pxx * Pyy)
                avg_pd = np.angle(Pxy)

            # Now the unweighted way
            else:
                phases = np.angle(stft)
                phase_diffs = np.zeros((N_pd, N_bins))
                # calc phase diffs
                for seg in range(N_pd):
                    # take the difference between the phases in this current segment and the one xi seconds away
                    phase_diffs[seg] = (
                        phases[seg + xi_nsegs] - phases[seg]
                    )  # minus sign <=> conj
                    # IMPLEMENT vectorized way to do this^^?
                coherence, avg_pd = get_avg_vector(phase_diffs)
        else:
            # In this case, xi is not an integer number of hops away, so we need two stfts each with N_pd segments
            f, stft_undelayed = get_stft(
                wf=wf[0:-xi], fs=fs, tau=tau, hop=hop, nfft=nfft, win=win, N_segs=N_pd
            )
            stft_delayed = get_stft(wf=wf[xi:], fs=fs, tau=tau, hop=hop, nfft=nfft, win=win, N_segs=N_pd)[1]
            N_bins = len(f)
            # First, do the pw case
            if pw:
                xy = np.conj(stft_undelayed) * stft_delayed
                Pxy = np.mean(xy, 0)
                powers_undelayed = stft_undelayed.real**2 + stft_undelayed.imag**2
                powers_delayed = stft_delayed.real**2 + stft_delayed.imag**2
                Pxx = np.mean(powers_undelayed, 0)
                Pyy = np.mean(powers_delayed, 0)
                coherence = (Pxy.real**2 + Pxy.imag**2) / (Pxx * Pyy)
                avg_pd = np.angle(Pxy)

            # Now the unweighted way
            else:
                phases_undelayed = np.angle(stft_undelayed)
                phases_delayed = np.angle(stft_delayed)
                # calc phase diffs
                phase_diffs = phases_delayed - phases_undelayed  # minus sign <=> conj
                coherence, avg_pd = get_avg_vector(phase_diffs)

    # or we can reference it against the phase of the next frequency in the same window:
    elif ref_type == "next_freq":
        # get phases and initialize array for phase diffs
        phases = np.angle(get_stft(wf, fs, tau, hop, xi, nfft, win, N_segs)[1])
        phase_diffs = np.zeros(
            (N_segs, N_bins - freq_bin_hop)
        )  # -freq_bin_hop is because we won't be able to get it for the #(freq_bin_hop) freqs
        # we'll also need to take the last #(freq_bin_hop) bins off the f
        f = f[0:-freq_bin_hop]

        # calc phase diffs
        for seg in range(N_segs):
            for freq_bin in range(N_bins - freq_bin_hop):
                phase_diffs[seg, freq_bin] = (
                    phases[seg, freq_bin + freq_bin_hop] - phases[seg, freq_bin]
                )

        # get final coherence
        coherence, avg_pd = get_avg_vector(phase_diffs)

        # Since this references each frequency bin to its adjacent neighbor, we'll plot them w.r.t. the average frequency;
        # this corresponds to shifting everything over half a bin width
        bin_width = f[1] - f[0]
        f = f + (bin_width / 2)

        # IMPLEMENT "next freq power weights"

    # or we can reference it against the phase of both the lower and higher frequencies in the same window
    elif ref_type == "both_freqs":
        # Get phases
        phases = np.angle(
            get_stft(wf, fs, tau, hop, xi, win=win, nfft=nfft, N_segs=N_segs)[1]
        )
        # initialize arrays
        # even though we only lose ONE freq point with lower and one with higher, we want to get all the points we can get from BOTH so we do - 2
        pd_low = np.zeros((N_segs, N_bins - 2))
        pd_high = np.zeros((N_segs, N_bins - 2))
        # take the first and last bin off the freq ax
        f = f[1:-1]

        # calc phase diffs
        for seg in range(N_segs):
            for freq_bin in range(1, N_bins - 1):
                # the - 1 is so that we start our pd_low and pd_high arrays at 0 and put in N_bins-2 points.
                # These will correspond to our new frequency axis.
                pd_low[seg, freq_bin - 1] = (
                    phases[seg, freq_bin] - phases[seg, freq_bin - 1]
                )
                pd_high[seg, freq_bin - 1] = (
                    phases[seg, freq_bin + 1] - phases[seg, freq_bin]
                )
        # set the phase diffs to one of these so we can return (could've also been pd_high)
        phase_diffs = pd_low
        coherence_low, avg_pd = get_avg_vector(pd_low)
        coherence_high, _ = get_avg_vector(pd_high)
        # average the coherences you would get from either of these
        coherence = (coherence_low + coherence_high) / 2

    else:
        raise Exception("You didn't input a valid ref_type!")

    if not return_dict:
        return f, coherence

    else:  # Return full dictionary
        d = {
            "coherence": coherence,
            "phase_diffs": phase_diffs,
            "avg_pd": avg_pd,
            "N_pd": N_pd,
            "N_segs": N_segs,
            "f": f,
            "stft": stft,
            "tau": tau,
            "hop": hop,
            "xi": xi,
            "fs": fs,
            "pw": pw,
        }
        if return_avg_abs_pd:
            phase_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi
            print("CHECK THIS NEW <|phase diffs|> IMPLEMENTATION WORKS AS EXPECTED")
            # get <|phase diffs|> (note we're taking mean w.r.t. PD axis 0, not frequency axis)
            d["avg_abs_pd"] = np.mean(np.abs(phase_diffs), 0)
        return d


def get_win_pc(win_meth, tau, xi, ref_type="next_seg"):
    """
    Gets the window based on the dynamic (or static) windowing method.

    Parameters
    ----------
    win_meth : dict
        Dictionary specifying windowing method parameters. Keys include:

        - 'method' (str): Specifies the windowing method to use. Options are:

            - `'static'`: Use a static window of type `'win_type'`, which should be a string or tuple
              compatible with SciPy's `get_window()`.
              Required keys: `'win_type'`

            - `'rho'`: Use a Gaussian window whose full width at half maximum (FWHM) is `rho * xi`.
              Required keys: `'rho'`, `'snapping_rhortle'`

            - `'eta'`: Use a window of type `'win_type'` with a shortened duration `tau`, such that
              the expected coherence for white noise is ≤ `eta`. Zero-padding is applied up to `tau`
              to maintain the number of frequency bins.
              Required keys: `'eta'`, `'win_type'`

        - 'rho' (float, optional): Controls the FWHM of the Gaussian window when `'method' == 'rho'`.

        - 'eta' (float, optional): Maximum allowable spurious coherence due to overlap in the white noise case
          (used when `'method' == 'eta'`).

        - 'win_type' (str or tuple, optional): Window type to be passed to `scipy.signal.get_window()`
          (used in `'static'` and `'eta'` methods).

        - 'snapping_rhortle' (bool, optional): When `True` and `'method' == 'rho'`, switches from a Gaussian
          window to a fixed boxcar for all `xi > tau`. Defaults to `False`.

    tau : int
        Length (in samples) of each segment, used in `get_stft()`.
    xi : int
        Length (in samples) to advance copy of signal for phase reference.
    ref_type : str
        Type of reference segment. Must be `'next_seg'` when using a dynamic windowing method,
        since that's what the methods were designed for.
    """

    if "method" not in win_meth.keys():
        return ValueError(
            "the 'win_meth' dictionary must contain a 'method' key! See get_win_pc() documentation for details."
        )
    method = win_meth["method"]

    # First, handle dynamic windows
    if method in ["rho", "eta"]:
        # Make sure our ref_type is appropriate
        if ref_type != "next_seg":
            raise ValueError(
                f"You passed in a dynamic windowing method ({method} windowing) but you're using a {ref_type} reference; this was designed for next_seg!"
            )

        if method == "rho":
            rho = win_meth["rho"]
            snapping_rhortle = (
                win_meth["snapping_rhortle"]
                if "snapping_rhortle" in win_meth.keys()
                else False
            )
            if snapping_rhortle and xi > tau:
                win = None
            else:
                desired_fwhm = rho * xi
                sigma = desired_fwhm / (2 * np.sqrt(2 * np.log(2)))
                win = get_window(("gaussian", sigma), tau)

        else:
            # here, method == 'eta'
            raise RuntimeError("Haven't implemented eta dynamic windowing yet!")
            eta = win_meth["eta"]
            win_type = win_meth["win_type"]
            tau = get_tau_from_eta(tau, xi, eta, win_type)

    elif method == "static":
        win_type = win_meth["win_type"]
        win = get_window(win_type, tau)
    else:
        raise ValueError(
            f"method={method} is not a valid windowing method; see get_win_pc() documentation for details."
        )

    return (
        win,
        tau,
    )  # Note that unless explicitly changed via eta windowing, tau just passes through


def colossogram_coherences(
    wf,
    fs,
    xis,
    pw,
    tau,
    nfft=None,
    hop=None,
    win_meth={"method": "rho", "rho": 0.7},
    const_N_pd=True,
    global_xi_max_s=None,
    ref_type="next_seg",
    return_dict=False,
):
    """Gets the phase coherence of the waveform against a copy of the waveform advanced xi samples

    Parameters
    ------------
        wf: array
            waveform input array
        fs: float
            sample rate of waveform
        xis: array or dictionary
            array of xi to calculate the coherence for, the phase reference distances (in samples);

            alternatively, a dictionary with keys `'xi_min'`, `'xi_max'`, `'delta_xi'` which creates this xis array
        pw: bool
            weights the vector strength average by the magnitude of each segment * magnitude of xi advanced segment
        tau: int
            length (in samples) of each segment, used in get_stft()
        nfft: int
            length of fft; implements zero padding if nfft > tau, used in get_stft() (defaults to tau AKA no zero padding)
        hop: int
            length between the start of successive segments, used in get_stft(); defaults to tau // 2
        win_meth: dict, optional
            windowing method dictionary; see get_pc_win() for details
        const_N_pd: bool, optional
            holds the number of phase differences fixed at the minimum N_pd able to be calculated across all xi (e.g. it's set by the maximum xi in the xis array)
        global_xi_max: int, optional
            instead of the N_pd being set by the maximum xi in this xi array, it's set by this value (e.g. if you're comparing across species with different xi_max)
        ref_type: str, optional
            Either "next_seg" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
        return_dict: bool, optional
            Defaults to only returning (xis_s, f, coherence); but if this is enabled, then a dictionary is returned with keys:
            'xis', 'xis_s', 'f', 'coherences', 'tau', 'fs', 'N_pd_min', 'N_pd_max', 'hop', 'win_meth', 'global_xi_max'
    """
    



    # Handle defaults
    if hop is None:
        hop = tau // 2
    # Get xis array (function is to handle possible passing in of dict with keys 'xi_min', 'xi_max', and 'delta_xi')
    xis = get_xis_array(xis, fs, hop)
    xi_min = xis[0]
    xi_max = xis[-1] 
    # ...also prints if we can turbo boost all the coherence calculations by only calculating a single STFT since xi is always an integer number of segs away


    # Get frequency array
    f = np.array(rfftfreq(tau, 1 / fs))
    N_bins = len(f)
    # Initialize coherences array
    coherences = np.zeros((N_bins, len(xis)))

    "Calculate min/max N_pd"
    # Set the max xi that will determine this minimum number of phase diffs
    # (either max xi within this colossogram, or a global one so it's constant across all colossograms in comparison)
    if global_xi_max_s is None:
        global_xi_max = xi_max
    elif not const_N_pd:
        raise Exception(
            "Why did you pass a global max xi if you're not holding N_pd constant?"
        )
    else:
        global_xi_max = (
            global_xi_max_s * fs
        )  # Note we deliberately passed in global_xi_max in secs so it can be consistent across samplerates

    # Get the number of phase diffs (we can do this outside xi loop since it's constant)
    eff_len_max = len(wf) - xi_min
    eff_len_min = len(wf) - global_xi_max

    # There are int((eff_len-tau)/hop)+1 full tau-segments with a xi reference
    N_pd_min = int((eff_len_min - tau) / hop) + 1
    N_pd_max = int((eff_len_max - tau) / hop) + 1

    if const_N_pd:
        # If we're holding it constant, we hold it to the minimum
        N_pd = N_pd_min
        # Even though the *potential* N_pd_max is bigger, we just use N_pd_min all the way so this is also the max
        N_pd_max = N_pd_min  # This way we can return both a min and a max regardless, even if they are equal

    # Loop through xis and calculate coherences
    for i, xi in enumerate(tqdm(xis)):
        # Calculate N_pd (assuming we're not holding it constant, in which case it was already done outside of loop)
        if not const_N_pd:
            # This is just as many segments as we possibly can with the current xi reference
            eff_len = len(wf) - xi
            N_pd = int((eff_len - tau) / hop) + 1

        print(xi)
        coherences[:, i] = get_coherence(
            wf=wf,
            fs=fs,
            tau=tau,
            pw=pw,
            xi=xi,
            nfft=nfft,
            hop=hop,
            win_meth=win_meth,
            N_pd=N_pd,
            ref_type=ref_type,
        )[1]

    # Convert to xis_s
    xis_s = xis / fs

    if return_dict:
        return {
            "xis": xis,
            "xis_s": xis_s,
            "f": f,
            "coherences": coherences,
            "tau": tau,
            "fs": fs,
            "N_pd_min": N_pd_min,
            "N_pd_max": N_pd_max,
            "hop": hop,
            "win_meth": win_meth,
            "global_xi_max": global_xi_max,
        }
    else:
        return xis_s, f, coherences


def welch(
    wf,
    fs,
    tau,
    nfft=None,
    hop=None,
    N_segs=None,
    win=None,
    scaling="density",
    reuse_stft=None,
    return_dict=False,
):
    """Gets the Welch averaged power of the given waveform with the given window size

    Parameters
    ------------
        wf: array
            waveform input array
        fs: float
            sample rate of waveform
        tau: int
            length (in samples) of each window; used in stft() and to calculate normalizing factor
        hop: int
            length between the start of successive segments, used in get_stft()
        N_segs: int, optional
            Used in stft()
        win: str or tuple or array
            Window to apply to each segment before FFT, either str or tuple for SciPy get_window() or an array (length tau) of window coefficents
        scaling : str, optional
            Type of spectral scaling to apply. Options are:
            - 'density': Power Spectral Density (default)
            - 'spectrum': Power spectrum without normalization
        reuse_stft: tuple, optional
            Pass in the stft dictionary here to avoid recalculation
        return_dict: bool, optional
            Returns a dict with:
            d["f"] = f
            d["spectrum"] = spectrum
            d["segmented_spectrum"] = segmented_spectrum
    """
    # Handle 
    
    # if nothing was passed into reuse_stft then we need to recalculate it
    stft_dict = reuse_stft if reuse_stft is not None else get_stft(wf=wf, fs=fs, tau=tau, nfft=nfft, hop=hop, N_segs=N_segs, win=win, return_dict=True)

    f = stft_dict["f"]
    stft = stft_dict["stft"]
    win = stft_dict["win"]

    # calculate necessary params from the stft
    N_segs, N_bins = np.shape(stft)

    # initialize array
    segmented_spectrum = np.zeros((N_segs, N_bins))

    # get spectrum for each window
    for seg in range(N_segs):
        segmented_spectrum[seg, :] = (np.abs(stft[seg, :])) ** 2

    # average over all segments (in power)
    spectrum = np.mean(segmented_spectrum, 0)

    S1 = np.sum(win)
    S2 = np.sum(win**2)
    if scaling == "mags":
        spectrum = np.sqrt(spectrum)
        scaling_factor = 1 / S1

    elif scaling == "spectrum":
        # Note that this is the density scaling except multiplied by the bin width * ENBW (in # bins)
        scaling_factor = 1 / S1**2

    elif scaling == "density":
        scaling_factor = 1 / (fs * S2)

    else:
        raise Exception("Scaling must be 'mags', 'density', or 'spectrum'!")

    # Normalize; since this is an rfft, we should multiply by 2
    spectrum = spectrum * 2 * scaling_factor
    # Except DC bin should NOT be scaled by 2
    spectrum[0] = spectrum[0] / 2
    # Nyquist bin shouldn't either (note this bin only exists if tau is even)
    if tau % 2 == 0:
        spectrum[-1] = spectrum[-1] / 2

    if return_dict:
        return {"f": f, "spectrum": spectrum, "segmented_spectrum": segmented_spectrum}
    else:
        return f, spectrum
