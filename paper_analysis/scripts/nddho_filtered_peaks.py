from helper_funcs import *
import phaseco as pc
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, get_window, convolve, correlation_lags
from numpy.fft import fftshift
from scipy.optimize import curve_fit
import os
from tqdm import tqdm
from nddho_generator import nddho_generator

# Directories
root = r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence"
os.chdir(root)
pkl_folder = os.path.join(root, "paper_analysis", "NDDHO", "pickles")
tau_pkl_folder = os.path.join(root, "paper_analysis", "pickles")


win_type = "flattop"
tau_psd = 2**13
win_type_psd = "hann"
hop_psd = 0.5

output_spreadsheet = 0
fit_peak = 0
save_filtered_peaks = 0
show_filtered_peaks = 0
if output_spreadsheet and not fit_peak:
    raise ValueError("Can't output spreadsheet if we're not fitting peaks!")

# Make subfolder
plot_folder = os.path.join("paper_analysis", "NDDHO", "filtered_peaks")
os.makedirs(plot_folder, exist_ok=True)

# Get colors
good_colors = get_colors("good")
bad_colors = get_colors("bad")


f_ds = [1000, 10000, 100]
gammas = [25, 50, 75, 100, 125, 150, 175, 200]
fs = 44100
wf_len_s = 60
iter = 1
color = 'blue'

diffs = []
if output_spreadsheet:
    rows = []
for f_d in f_ds:
    for gamma in gammas:
        print(f"Processing {f_d}Hz, gamma={gamma}")
        # Get species params
        bw = 2*gamma

        # Load/calc waveform

        # NDDHO WF FP
        wf_id = f"gamma={gamma}, f_d={f_d}, len={wf_len_s}, fs={fs}, iter={iter}"
        wf_fn = f"{wf_id} [NDDHO WF].pkl"
        wf_fp = os.path.join(pkl_folder, wf_fn)
        
        if os.path.exists(wf_fp):
            print("Already got this wf, loading!")
            with open(wf_fp, "rb") as file:
                wf = pickle.load(file)
        else:
            print(f"Generating NDDHO {wf_fn}")
            wf, _ = nddho_generator(f_d, gamma=gamma, fs=fs, t_max=wf_len_s)
            with open(wf_fp, "wb") as file:
                pickle.dump(wf, file)


        # Get subject-specific params
        tau = get_precalc_tau_from_bw(bw, fs, win_type, tau_pkl_folder)

        # Get frequency axis
        f = fftfreq(tau_psd, 1 / fs)


        plt.close("all")
        plt.figure(figsize=(15, 10))
        plt.suptitle(
            rf"{f_d}Hz, gamma={gamma}    [$BW_{{\text{{gamma}}}}=2\gamma={bw}$]   [$\tau_\text{{PSD}}$={tau_psd}, {win_type_psd.capitalize()}]   [H={hop_psd}$\tau$]"
        )
        # Filter wf for a filtered wf equivalent to STFT bin with H=1
        f0_max_bin = f[np.argmin(np.abs(f - f_d))]
        win_type = "flattop"
        win = get_window(win_type, tau)
        omega_0_norm = f0_max_bin * 2 * np.pi / fs
        n = np.arange(len(win))
        kernel = win * np.exp(1j * omega_0_norm * n)
        wf_filtered = convolve(wf, kernel, mode="valid", method="fft") / np.sum(
            win
        )
        psd_filt = get_welch(
            wf_filtered,
            fs,
            tau=tau_psd,
            hop=hop_psd,
            win=win_type_psd,
            realfft=False,
        )[1]
        psd = get_welch(
            wf, fs, tau=tau_psd, hop=hop_psd, win=win_type_psd, realfft=False
        )[1]
        f = fftshift(f)
        psd = fftshift(psd)
        psd_filt = fftshift(psd_filt)
        log = 0
        if log:
            psd = 10 * np.log10(psd)
            psd_filt = 10 * np.log10(psd_filt)
            ylabel = "PSD [Log]"
        else:
            ylabel = "PSD"
        xmin, xmax = f0_max_bin - bw * 2, f0_max_bin + bw * 2
        print((xmin, xmax))
        xmin_idx, xmax_idx = np.argmin(np.abs(f - xmin)), np.argmin(
            np.abs(f - xmax)
        )
        f_crop, psd_crop, psd_filt_crop = (
            f[xmin_idx:xmax_idx],
            psd[xmin_idx:xmax_idx],
            psd_filt[xmin_idx:xmax_idx],
        )
        plt.plot(f_crop, psd_filt_crop, label="Filtered", color=color)
        plt.plot(f_crop, psd_crop, label="Unfiltered", color="k")
        print(psd_crop)
        plt.ylabel(ylabel)
        plt.axvline(x=f0_max_bin - bw / 2, color="g")
        plt.axvline(x=f0_max_bin + bw / 2, color="g")
        plt.xlabel("Frequency [Hz]")
        plt.title(f"{f0_max_bin:.0f} Hz")
        # plt.xlim(50, 150)

        if fit_peak:

            L_f0, L_gamma, L_amp, fitted_lorentz = fit_lorentzian(
                f_crop, psd_filt_crop
            )
            plt.plot(
                f_crop,
                fitted_lorentz,
                label=rf"$\gamma={L_gamma:.3g}$, $A={L_amp:.3g}$",
                color=color,
                ls="--",
            )
            print(
                f"Frequency = {f0_max_bin:.0f}, diff = {np.abs(L_f0-f0_max_bin)}"
            )
            diffs.append(np.abs(L_f0 - f0_max_bin))
            # if output_spreadsheet:
            #     row = {
            #         "species": species,
            #         "wf_idx": wf_idx,
            #         "wf_fn": wf_fn,
            #         "f0_max_int": f0_max,
            #         "f0_max_bin": f0_max_bin,
            #         "L_f0": L_f0,
            #         "L_gamma": L_gamma,
            #         "L_amp": L_amp,
            #         "bw":bw,
            #         "win_type":win_type
            #     }
            #     rows.append(row)

            plt.legend(fontsize=6)
            # plot_fp = os.path.join(
            #     plot_folder, f"{species} {wf_idx} [{'Good' if good_peaks else 'Bad'}]"
            # )
        if show_filtered_peaks:
            plt.show()
        # if save_filtered_peaks:
        #     plt.savefig(plot_fp)
# if output_spreadsheet:
#     df = pd.DataFrame(rows)
#     df_fp = os.path.join(plot_folder, 'df.xlsx')
#     df.to_excel(
#         df_fp, index=False
#     )

# diffs = np.array(diffs)
# print(f"Max diff = {np.max(diffs)}, mean={np.mean(diffs)}, std = {np.std(diffs)}")
