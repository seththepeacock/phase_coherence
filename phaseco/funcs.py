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
    nfft=None,
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

    # Handle defaults
    if nfft is None:
        nfft = tau

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
                f"Hm that's strange - the first N_segs calculation gives {N_segs} while other method gives {int((len(wf) - tau) / hop) + 1}"
            )

    # Check if a win has been passed in and set do_windowing based on if it's nontrivial
    if win is None:
        do_windowing = False
    else:
        # in normal get_coherence usage, win will just be an array of window coefficients; 
        # this logic allows for passing in a string to get the window via SciPy get_window
        if isinstance(win, str) or (
                isinstance(win, tuple) and isinstance(win[0], str)
            ):
                # Get window function
                win = get_window(win, tau)
        # Set do_windowing = True unless it's just a boxcar (all 1s)
        do_windowing = np.any(win != 1)

    # Get segmented waveform matrix
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

    #IMPLEMENT rfft with different nfft instead of zpadding since probably more efficient1
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
    xi=None,
    tau=None,
    nfft=None,
    hop=None,
    dyn_win=("rho", 0.7, False),
    static_win=None,
    pw=None,
    N_pd=None,
    ref_type="next_seg",
    freq_bin_hop=1,
    return_avg_abs_phase_diffs=False,
    return_dict=False,
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
          d["avg_pd"] = avg_pd
          d["N_segs"] = N_segs
          d["stft"] = stft
    """

    # Handle defaults
    if hop is None:
        hop = tau  # Zero overlap (at least, outside of xi referencing considerations)
    if nfft is None:
        nfft = tau # No zero padding (at least, outside of eta windowing considerations)
    
    # Get window (and possibly redfine tau if doing eta windowing)
    win, tau = get_phaseco_win(dyn_win, tau, xi, nfft, static_win, ref_type)

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
                wf[0:-xi], fs, tau, hop, xi, nfft, win, N_segs=N_pd
            )
            stft_delayed = get_stft(wf[xi:], fs, tau, hop, xi, nfft, win, N_segs=N_pd)[
                1
            ]
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
            "phases": phases,
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
        if return_avg_abs_phase_diffs:
            phase_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi
            print("CHECK THIS <|phase diffs|> IMPLEMENTATION WORKS AS EXPECTED")
            # get <|phase diffs|> (note we're taking mean w.r.t. PD axis 0, not frequency axis)
            d["avg_abs_phase_diffs"] = np.mean(np.abs(phase_diffs), 0)
        return d


def get_colossogram_coherences(
    wf,
    fs,
    tau,
    xi_min,
    xi_max,
    delta_xi=None,
    hop=None,
    dyn_win=("rho", 0.7, False),
    pw=None,
    const_N_pd=1,
    global_xi_max_s=None,
    return_dict=False,
):

    # Handle defaults
    if hop is None:
        hop = xi_min
    if delta_xi is None:
        delta_xi = xi_min
    # Note that these defaults allow us to only calculate a single stft in get_coherence since each xi will be an integer number of steps away!
    if delta_xi == xi_min and xi_min == hop:
        print(
            f"delta_xi = xi_min = hop (= {hop}), so all xis will be an integer number of segs away, so we can turbo-boost the coherences with a single stft!"
        )

    # Calculate xi array and N_bins
    num_xis = int((xi_max - xi_min) / delta_xi) + 1
    xis = np.linspace(xi_min, xi_max, num_xis)
    f = np.array(rfftfreq(tau * fs, 1 / fs))
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
        )  # Note we wanted to pass in global_xi_max in secs so it can be consistent across samplerates

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

        coherences[:, i] = get_coherence(
            wf=wf, fs=fs, tau=tau, xi=xi, hop=hop, dyn_win=dyn_win, N_pd=N_pd, pw=pw
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
            "dyn_win": dyn_win,
            "global_xi_max": global_xi_max,
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
        SIGMA = get_sigma(fwhm=fwhm, fs=fs)
    elif rho is not None:
        SIGMA = get_sigma(fwhm=rho * xi, fs=fs)

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
            win=win_type,
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
    ENBW = tau * S2 / S1**2  # In samples
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
        normalizing_factor = normalizing_factor / (ENBW * bin_width)

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
