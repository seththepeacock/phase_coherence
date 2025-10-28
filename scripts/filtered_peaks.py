# from helper_funcs import *
import phaseco as pc
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, get_window, convolve, correlation_lags
from scipy.optimize import curve_fit
import os
from tqdm import tqdm

# Directories
root = r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence"
os.chdir(root)
pkl_folder = os.path.join(root, "paper_analysis", "pickles")


species_bws = {"Anole": 150, "Human": 50, "Owl": 300, "Tokay": 150}
win_type = "flattop"
tau_psd = 2**14
win_type_psd = "hann"
hop_psd = 0.5

output_spreadsheet = True
fit_peak = True
save_filtered_peaks = True
show_filtered_peaks = 1
if output_spreadsheet and not fit_peak:
    raise ValueError("Can't output spreadsheet if we're not fitting peaks!")

# Make subfolder
plot_folder = os.path.join("paper_analysis", "filtered_peaks")
os.makedirs(plot_folder, exist_ok=True)

# Get colors
good_colors = get_colors("good")
bad_colors = get_colors("bad")

# Set species list
all_species = ["Anole", "Human", "Owl", "Tokay"]
wf_idxs = range(4)
speciess = all_species

diffs = []
if output_spreadsheet:
    rows = []
for species in speciess:
    for wf_idx in wf_idxs:
        print(f"Processing {species} {wf_idx}")
        # Get species params
        bw = species_bws[species]

        # Get and process waveform
        wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
            species=species,
            wf_idx=wf_idx,
        )
        wf_len_s = 60
        wf = crop_wf(wf, fs, wf_len_s)
        wf = scale_wf(wf, species)
        wf -= np.mean(wf)

        # Get subject-specific params
        tau = get_precalc_tau_from_bw(bw, fs, win_type, pkl_folder)

        # Get frequency axis
        f = fftfreq(tau_psd, 1 / fs)

        for peak_freqs, good_peaks, colors in zip(
            [good_peak_freqs, bad_peak_freqs], [True, False], [good_colors, bad_colors]
        ):
            if len(peak_freqs) == 0:
                continue
            plt.close("all")
            plt.figure(figsize=(15, 10))
            plt.suptitle(
                rf"{species} {wf_idx}    [$BW_{{\text{{{species}}}}}$={bw}]   [$\tau_\text{{PSD}}$={tau_psd}, {win_type_psd.capitalize()}]   [H={hop_psd}$\tau$]"
            )
            for f0_max, color, subplot_idx in zip(peak_freqs, colors, [1, 2, 3, 4]):
                plt.subplot(2, 2, subplot_idx)
                # Filter wf for a filtered wf equivalent to STFT bin with H=1
                f0_max_bin = f[np.argmin(np.abs(f - f0_max))]
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
                log = 0
                if log:
                    psd = 10 * np.log10(psd)
                    psd_filt = 10 * np.log10(psd_filt)
                    ylabel = "PSD [Log]"
                else:
                    ylabel = "PSD"
                xmin, xmax = f0_max_bin - bw * 2, f0_max_bin + bw * 2
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
                plt.ylabel(ylabel)
                plt.axvline(x=f0_max_bin - bw / 2, color="g")
                plt.axvline(x=f0_max_bin + bw / 2, color="g")
                plt.xlabel("Frequency [Hz]")
                plt.title(f"{f0_max_bin:.0f} Hz")

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
                    if output_spreadsheet:
                        row = {
                            "species": species,
                            "wf_idx": wf_idx,
                            "wf_fn": wf_fn,
                            "f0_max_int": f0_max,
                            "f0_max_bin": f0_max_bin,
                            "L_f0": L_f0,
                            "L_gamma": L_gamma,
                            "L_amp": L_amp,
                            "bw":bw,
                            "win_type":win_type
                        }
                        rows.append(row)

                plt.legend(fontsize=6)
            plot_fp = os.path.join(
                plot_folder, f"{species} {wf_idx} [{'Good' if good_peaks else 'Bad'}]"
            )
            if show_filtered_peaks:
                plt.show()
            if save_filtered_peaks:
                plt.savefig(plot_fp)
if output_spreadsheet:
    df = pd.DataFrame(rows)
    df_fp = os.path.join(plot_folder, 'df.xlsx')
    df.to_excel(
        df_fp, index=False
    )

diffs = np.array(diffs)
print(f"Max diff = {np.max(diffs)}, mean={np.mean(diffs)}, std = {np.std(diffs)}")
