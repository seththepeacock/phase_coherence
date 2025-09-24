import numpy as np
import scipy.io as sio
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, kaiserord, firwin, lfilter
import os
import pickle
from phaseco import *
import phaseco as pc
from phaseco.helper_funcs import get_win_meth_str
import time
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm


def load_calc_colossogram(
    wf,
    wf_idx,
    wf_fn,
    wf_len_s,
    wf_pp,
    species,
    fs,
    filter,
    paper_analysis_folder,
    pw,
    tau,
    nfft,
    xi_min_s,
    xi_max_s,
    global_xi_max_s,
    hop,
    win_meth,
    force_recalc_colossogram,
    plot_what_we_got,
    only_calc_new_coherences,
    const_N_pd,
    scale,
    N_bs,
    f0s,
    wa,
):
    filter_str = get_filter_str(filter)

    win_meth_str = pc.get_win_meth_str(win_meth)

    # Build strings

    # Convert for strings
    tau_ms = 1000 * tau / fs
    hop_ms = 1000 * hop / fs if hop > 1 else 1000 * round(hop * tau) / fs
    N_bs_str = "" if N_bs == 0 else f"N_bs={N_bs}, "
    pw_str = f"{pw}" if not wa or not pw else "WA"
    const_N_pd_str = "" if const_N_pd else "N_pd=max, "
    f0s_str = (
        ""
        if f0s is None
        else f"f0s={np.array2string(f0s, formatter={'float' : lambda x: "%.0f" % x})}, "
    )
    nfft_str = "" if nfft is None else f"nfft={nfft}, "
    delta_xi_str = "" if xi_min_s == 0.001 else f"delta_xi={xi_min_s*1000:.0f}ms, "
    hop_str = f"hop={hop_ms:.0f}ms"
    fn_id = rf"{species} {wf_idx}, PW={pw_str}, {win_meth_str}, {hop_str}, tau={tau_ms:.0f}ms, {filter_str}, xi_max={xi_max_s*1000:.0f}ms, {delta_xi_str}{nfft_str}{f0s_str}{const_N_pd_str}{N_bs_str}wf_len={wf_len_s}s, wf={wf_fn.split('.')[0]}"
    pkl_fn = f"{fn_id} (Colossogram)"
    print(f"Processing {fn_id}")
    # Convert to samples
    xi_min = round(xi_min_s * fs)
    xi_max = round(xi_max_s * fs)

    # First, try to load in the new way
    pkl_folder = paper_analysis_folder + "Pickles/"
    pkl_fp = pkl_folder + pkl_fn + ".pkl"
    os.makedirs(pkl_folder, exist_ok=True)

    if os.path.exists(pkl_fp) and not force_recalc_colossogram:
        with open(pkl_folder + pkl_fn + ".pkl", "rb") as file:
            (colossogram_dict) = pickle.load(file)
        if only_calc_new_coherences:
            colossogram_dict["only_calc_new_coherences"] = 1
        colossogram_dict["fn_id"] = fn_id
        with open(pkl_fp, "wb") as file:
            pickle.dump(colossogram_dict, file)

    else:
        # Now, we know they don't exist as pickles new or old, so we recalculate
        if (
            plot_what_we_got
        ):  # Unless plot_what we got, in which case we just end the func here
            return {"plot_what_we_got": 1}, wf_pp

        # First, process the wf (unless it's already processed)
        if wf_pp is None:
            if species in ["Anole", "Human"] and scale:  # Scale wf
                wf = scale_wf(wf)

            # Crop wf
            wf_cropped = crop_wf(wf, fs, wf_len_s)

            # Apply filter
            wf_pp = filter_wf(wf_cropped, fs, filter, species)
        # If it's already been processed and passed in, just use it
        else:
            print("Calculating colossogram with prefiltered waveform!")

        # Then get colossogram!
        colossogram_dict = pc.get_colossogram(
            wf_pp,
            fs,
            xis={"xi_min": xi_min, "xi_max": xi_max, "delta_xi": xi_min},
            hop=hop,
            tau=tau,
            nfft=nfft,
            win_meth=win_meth,
            pw=pw,
            const_N_pd=const_N_pd,
            global_xi_max_s=global_xi_max_s,
            N_bs=N_bs,
            f0s=f0s,
            return_dict=True,
        )
        # Add some extra keys
        extra_keys = {
            "fn_id": fn_id,
            "win_meth_str": win_meth_str,
            "filter_str": filter_str,
        }
        colossogram_dict.update(extra_keys)

        with open(pkl_folder + pkl_fn + ".pkl", "wb") as file:
            pickle.dump(colossogram_dict, file)
    # We now have colossogram_dict either from a saved pickle (new or old) or from the calculation; return it!
    return colossogram_dict, wf_pp


def scale_wf_long_way(wf):
    # First, undo the mic amplifier gain
    gain = 40  # dB
    wf = wf * 10 ** (-gain / 20)
    gain = 40  # dB
    wf = wf * 10 ** (-gain / 20)
    # Then account for the calibration factor
    cal_factor = 0.84
    cal_factor = 0.84
    wf = wf / cal_factor
    # The waveform is now in units of volts, where 1 micro volt = 0dB SPL = 20 micropascals
    # Let's rescale so that now 1 waveform unit (*volt*) = 0dB SPL = 20 micropascals
    wf = wf * 1e6
    # Now, 20*log10(dft_mags(wf)) would directly be in dB SPL.

    # Finally, (optional), we'll just convert it directly to pascals by multiplying by 20 micropascals:
    wf_pa = wf * 20 * 1e-6
    # Now, using this version, we would have to do 20*np.log10(dft_mags(wf_pa) / (20*1e-6)) to get dB SPL.)

    return wf_pa


def scale_wf(wf):
    # Proven this is equivalent to above
    factor = (20 * 0.01) / 0.84
    return wf * factor


def get_wf(wf_fn=None, species=None, wf_idx=None):
    if wf_fn is None:
        if species is None or wf_idx is None:
            raise ValueError("You must input either fn or species and idx!")
        else:
            wf_fn = get_fn(species, wf_idx)

    # Load wf
    paper_analysis_folder = r"paper_analysis/"
    data_folder = paper_analysis_folder + r"Data/"
    if species == "Tokay":
        wf = sio.loadmat(data_folder + wf_fn)["wf"][0]
    elif species == "V Sim Human":
        wf = sio.loadmat(data_folder + wf_fn)["oae"][:, 0]
    else:
        wf = sio.loadmat(data_folder + wf_fn)["wf"][:, 0]

    # Get fs
    if wf_fn in ["Owl2R1.mat", "Owl7L1.mat"]:
        fs = 48000
    elif species == "V Sim Human":
        fs = 40000
    else:
        fs = 44100

    # Get peak list
    match wf_fn:
        # Vaclav's Human
        case "longMCsoaeL1_20dBdiff100dB_InpN1InpYN0gain85R1rs43.mat":
            seth_good_peak_freqs = [1157, 1244, 1518, 1976]
            seth_bad_peak_freqs = []
        # Anoles
        case "AC6rearSOAEwfB1.mat":  # 0
            seth_good_peak_freqs = [1233, 2153, 3704, 4500,]
            seth_bad_peak_freqs = []
            becky_good_peak_freqs = [1233, 2164, 3709, 4506]
            becky_bad_peak_freqs = []
        case "ACsb4rearSOAEwf1.mat":  # 1
            seth_good_peak_freqs = [964, 2729, 3155, 3946,]
            seth_bad_peak_freqs = [
                3025,
            ]
            becky_good_peak_freqs = [964, 3155, 3951]
            becky_bad_peak_freqs = [
                3025,
            ]
        case "ACsb24rearSOAEwfA1.mat":  # 2
            seth_good_peak_freqs = [
                2175, 2498, 3117, 3478,
            ]
            seth_bad_peak_freqs = [1733, 1814,]
            becky_good_peak_freqs = [2175, 2503, 3112, 3478]
            becky_bad_peak_freqs = [1728, 1814]
        case "ACsb30learSOAEwfA2.mat":  # 3
            seth_good_peak_freqs = [
                1798, 2143,
            ]
            seth_bad_peak_freqs = [2417, 2778, 3047,]
            becky_good_peak_freqs = [
                1798,
                2143,
            ]
            becky_bad_peak_freqs = [2406, 2778]

        # Tokays
        case "tokay_GG1rearSOAEwf.mat":  # 0
            seth_good_peak_freqs = [1572, 1717,]
            seth_bad_peak_freqs = [1184, 3219, 3714,]
        case "tokay_GG2rearSOAEwf.mat":  # 1
            seth_good_peak_freqs = [
               1324, 1567, 2896, 3182,
            ]
            seth_bad_peak_freqs = [3435, 3876,]
        case "tokay_GG3rearSOAEwf.mat":  # 2
            seth_good_peak_freqs = [
                1109, 1330, 2821, 3144,
            ]
            seth_bad_peak_freqs = [1620, 2272,]
        case "tokay_GG4rearSOAEwf.mat":  # 3
            seth_good_peak_freqs = [1104, 2288, 3160,]
            seth_bad_peak_freqs = [2848,]

        # Owls
        case "Owl2R1.mat":  # 0
            seth_good_peak_freqs = [
                8010, 8432,
            ]
            seth_bad_peak_freqs = [
                4354, 5572, 5947, 7102, 7453, 9029, 9586,
            ]
            becky_good_peak_freqs = [8016, 8450]
            becky_bad_peak_freqs = [4342, 5578, 5953, 7090, 7453, 9035, 9574]
        case "Owl7L1.mat":  # 1
            seth_good_peak_freqs = [
                7893, 7500, 8836,
            ]
            seth_bad_peak_freqs = [
                6141, 6838, 8443, 9258, 9791,
            ]
            becky_good_peak_freqs = [7922, 7535, 8854]
            becky_bad_peak_freqs = [6164, 6896, 8426, 9252, 9779]
        case "TAG6rearSOAEwf1.mat":  # 2
            seth_good_peak_freqs = [
               6035, 8096, 8484, 9868,
            ]
            seth_bad_peak_freqs = [5626,]
            becky_good_peak_freqs = [6029, 8102, 8489, 9857]
            becky_bad_peak_freqs = [5626]
        case "TAG9rearSOAEwf2.mat":  # 3
            seth_good_peak_freqs = [
                6966,
            ]
            seth_bad_peak_freqs = [
                4926, 6589, 7429, 9760,
            ]
            becky_good_peak_freqs = [6977]
            becky_bad_peak_freqs = [3461, 4613, 4920, 6164, 7445, 9846, 10270]
        case "owl_TAG4learSOAEwf1.mat":  # 4
            seth_good_peak_freqs = [
                5766, 7181, 9636,
            ]
            seth_bad_peak_freqs = [4947, 8446, 8834,]
            becky_good_peak_freqs = [5771, 7176, 9631]
            becky_bad_peak_freqs = [4958, 8463, 8839]

        # Humans
        case "ALrearSOAEwf1.mat":  # 0
            seth_good_peak_freqs = [
                2805, 2945, 3865,
            ]
            seth_bad_peak_freqs = [
                904, 980, 2665, 3219,
            ]
            becky_good_peak_freqs = [2805, 2945, 3865]
            becky_bad_peak_freqs = [904, 980, 2659, 3219]
        case "JIrearSOAEwf2.mat":  # 1
            seth_good_peak_freqs = [
                2810, 2342, 4048, 5841,
            ]
            seth_bad_peak_freqs = [3402, 8312, 8678,]
            becky_good_peak_freqs = [2342, 4048, 5841]
            becky_bad_peak_freqs = [3402, 8312, 8678]
        case "LSrearSOAEwf1.mat":  # 2
            seth_good_peak_freqs = [
                732, 985, 1637, 2229,
            ]
            seth_bad_peak_freqs = [
                985, 1637, 3122,
            ]
            becky_good_peak_freqs = [732, 2230]
            becky_bad_peak_freqs = [985, 1637, 3122]
        case "TH13RearwaveformSOAE.mat":  # 3
            seth_good_peak_freqs = [
                904, 1518, 1674, 2040,
            ]
            seth_bad_peak_freqs = [
                2283, 2697,
            ]
            becky_good_peak_freqs = [904, 1518, 2040]
            becky_bad_peak_freqs = [2697]
    good_peak_freqs = seth_good_peak_freqs
    bad_peak_freqs = seth_bad_peak_freqs
    return wf, wf_fn, fs, np.array(good_peak_freqs), np.array(bad_peak_freqs)


# def crop_wf(wf, fs, wf_len_s, species):
#     if species == "Tokay":
#         wf_len = round(wf_len_s * fs)
#         og_length = len(wf)
#         if og_length < wf_len:
#             raise ValueError(f"Waveform is less than {wf_len_s}s long!")
#         # Start index for the middle chunk
#         start = max(0, (og_length - wf_len_s) // 2)
#         wf = wf[start : start + wf_len]
#     else:  # Just keeping it this way for consistency (it shouldn't matter), will do oeverything the Tokay way if/when we do the final recalc
#         wf = wf[: int(wf_len_s * fs)]
#     return wf


def filter_wf(wf, fs, filter_meth, species):
    if filter_meth is not None and species != "V Sim Human":
        match filter_meth["type"]:
            case "spectral":
                wf = spectral_filter(wf, fs, filter_meth["cf"], type="hp")
            case "kaiser":
                wf = kaiser_filter(
                    wf, fs, filter_meth["cf"], filter_meth["df"], filter_meth["rip"]
                )
            case _:
                raise ValueError(f"{filter_meth['type']} is not a valid HPF type!")
    return wf


def crop_wf(wf, fs, wf_len_s):
    desired_wf_len = round(wf_len_s * fs)
    og_wf_len = len(wf)
    if og_wf_len < desired_wf_len:
        raise ValueError(f"Waveform is less than {wf_len_s}s long!")
    # Start index for the middle chunk
    start = max(0, (og_wf_len - desired_wf_len) // 2)
    wf_cropped = wf[start : start + desired_wf_len]

    return wf_cropped


def get_fn(species, idx):
    match species:
        case "Anole":
            match idx:
                case 0:
                    return "AC6rearSOAEwfB1.mat"
                case 1:
                    return "ACsb4rearSOAEwf1.mat"
                case 2:
                    return "ACsb24rearSOAEwfA1.mat"
                case 3:
                    return "ACsb30learSOAEwfA2.mat"
        case "Tokay":
            return f"tokay_GG{idx+1}rearSOAEwf.mat"

        case "Human":
            match idx:
                case 0:
                    return "ALrearSOAEwf1.mat"
                case 1:
                    return "JIrearSOAEwf2.mat"
                case 2:
                    return "LSrearSOAEwf1.mat"
                case 3:
                    return "TH13RearwaveformSOAE.mat"
        case "Owl":
            match idx:
                case 0:
                    return "Owl2R1.mat"
                case 1:
                    return "Owl7L1.mat"
                case 2:
                    return "TAG6rearSOAEwf1.mat"
                case 3:
                    return "TAG9rearSOAEwf2.mat"
                case 4:
                    return "owl_TAG4learSOAEwf1.mat"
        case "V Sim Human":
            return "longMCsoaeL1_20dBdiff100dB_InpN1InpYN0gain85R1rs43.mat"
        case _:
            raise ValueError(
                "Species must be 'Anole', 'Human', 'Tokay', or 'Owl' (or 'V Sim Human')!"
            )


def get_spreadsheet_df(wf_fn, species):
    df = pd.read_excel(
        r"paper_analysis/Data/2024.07analysisSpreadsheetV8_RW.xlsx",
        sheet_name=species if species != "Anole" else "Anolis",
    )
    if (
        wf_fn == "TAG9rearSOAEwf2.mat"
    ):  # This one has trailing whitespace in Becky's excel sheet
        wf_fn += " "
    if wf_fn == "owl_TAG4learSOAEwf1.mat":
        wf_fn = "TAG4learSOAEwf1.mat"

    return df[df["rootWF"].str.split(r"/").str[-1] == wf_fn].copy()


def get_params_from_df(df, peak_freq):
    df = df[df["CF"] == peak_freq]
    if len(df) == 0:
        raise ValueError("Dataframe is empty...")
    row = df.iloc[0]
    SNRfit = row["SNRfit"]
    fwhm = row["FWHM"]

    return SNRfit, fwhm


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


def kaiser_filter(wf, fs, cf=300, df=50, rip=100):
    """
    Apply an FIR, linear phase filter designed with a Kaiser window.

    Parameters:
        wf (array): Input waveform.
        fs (float): Sampling rate (Hz).
        cf (float or tuple of floats): Cutoff freq (Hz); Two frequencies = BP, one freq = HP
        df (float): Transition bandwidth (Hz).
        rip (float): Max allowed ripple in dB; that is, abs(A(w) - D(w))) < 10**(-ripple/20)

    Returns:
        array: Filtered waveform.
    """
    print(f"Filtering wf with cf={cf}Hz, df={df}Hz, rip={rip}dB")
    start = time.time()
    # Compute filter parameters
    numtaps, beta = kaiserord(rip, df / (0.5 * fs))

    if numtaps % 2 == 0:
        numtaps += 1  # Make it odd for a HPF

    # Design the high-pass FIR filter
    taps = firwin(
        numtaps,
        cf,
        window=("kaiser", beta),
        fs=fs,
        pass_zero=False,  # Neither HPF or BPF want zero
    )

    # Apply filtering
    filtered_wf = lfilter(
        taps, [1.0], wf
    )  # b, the denominator, is 1 for no poles, only zeros = FIR
    stop = time.time()
    print(f"Filtering took {stop-start:.3f}s")
    return filtered_wf


def get_filter_str(filter_meth):
    if filter_meth is None:
        return "filter=None"
    match filter_meth["type"]:
        case "kaiser":
            if not isinstance(filter_meth["cf"], tuple):
                filter_str = rf"HPF=({filter_meth['cf']}Hz cf, {filter_meth['df']}Hz df, {filter_meth['rip']}dB rip)"
            else:
                filter_str = rf"BPF=({filter_meth['cf']}Hz cf, {filter_meth['df']}Hz df, {filter_meth['rip']}dB rip)"
        case "spectral":
            filter_str = rf"HPF=({filter_meth['cf']}Hz)"
    return filter_str


def get_win_hpbw(win_meth, tau, fs, nfft=None):
    if nfft is None:
        nfft = tau * 8
    win, _ = pc.get_win(win_meth, tau=tau, xi=None)
    win_psd = np.abs(rfft(win, nfft)) ** 2
    target = np.max(win_psd) / 2

    idx = np.where(win_psd <= target)[0][0]
    f_win = rfftfreq(nfft, 1 / fs)
    hpbw = f_win[idx] * 2
    return hpbw
