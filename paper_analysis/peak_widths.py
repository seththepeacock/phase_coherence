from N_xi_fit_funcs import *
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
pkl_folder = os.path.join(root, "pickles")



bws = {'Anole':150, 'Human':50, 'Owl':200, 'Tokay':150}
win_type = 'flattop'
tau_psd = 2**13
win_type_psd = 'hann'
hop_psd = 0.5

fit_peak = True

# Make plot directory
plot_folder = f"Filtered Peaks"
os.makedirs(plot_folder, exist_ok=True)

# Get colors
good_colors = get_colors('good')
bad_colors = get_colors('bad')

# Plot params
log=True

# Set species list
all_species = ['Human','Anole', 'Owl', 'Tokay']
wf_idxs = range(4)
speciess = all_species

diffs = []
for species in speciess:
    for wf_idx in wf_idxs:
        print(f"Processing {species} {wf_idx}")
        # Get species params
        bw = bws[species]
        # Make subfolder
        plot_folder = "filtered_peaks"
        os.makedirs(plot_folder, exist_ok=True)
        

        # Get and process waveform
        wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
            species=species,
            wf_idx=wf_idx,
        )
        wf_len_s = 60
        wf = crop_wf(wf, fs, wf_len_s)
        wf = scale_wf(wf, species)

        # Get subject-specific params
        tau = get_precalc_tau_from_bw(bw, fs, win_type, pkl_folder)


        # Get frequency axis
        f = fftfreq(tau_psd, 1/fs)

        for peak_freqs, good_peaks, colors in zip([good_peak_freqs, bad_peak_freqs], [True, False], [good_colors, bad_colors]):
            if len(peak_freqs)==0:
                continue
            plt.close('all')
            plt.figure(figsize=(15, 10))
            plt.suptitle(rf"{species} {wf_idx}    [$BW_{{\text{{{species}}}}}$={bw}]   [$\tau_\text{{PSD}}$={tau_psd}, {win_type_psd.capitalize()}]   [H={hop_psd}$\tau$]")
            for f0, color, subplot_idx in zip(
                peak_freqs, colors, [1, 2, 3, 4]
            ):
                plt.subplot(2, 2, subplot_idx)
                # Filter wf for a filtered wf equivalent to STFT bin with H=1
                f0_exact = f[np.argmin(np.abs(f-f0))]
                win_type = 'flattop'
                win = get_window(win_type, tau)
                omega_0_norm = f0_exact * 2*np.pi / fs
                n = np.arange(len(win))
                kernel = win * np.exp(1j * omega_0_norm * n)
                wf_filtered = convolve(wf, kernel, mode='valid', method='fft') / np.sum(win)
                psd_filt = get_welch(wf_filtered, fs, tau=tau_psd, hop=hop_psd, win=win_type_psd, realfft=False)[1]
                psd = get_welch(wf, fs, tau=tau_psd, hop=hop_psd, win=win_type_psd, realfft=False)[1]
                log=0
                if log:
                    psd = 10*np.log10(psd)
                    psd_filt = 10*np.log10(psd_filt)
                    ylabel = 'PSD [Log]'
                else:
                    ylabel = 'PSD'
                xmin, xmax = f0_exact-bw*2, f0_exact+bw*2
                xmin_idx, xmax_idx = np.argmin(np.abs(f-xmin)), np.argmin(np.abs(f-xmax))
                f_crop, psd_crop, psd_filt_crop = f[xmin_idx:xmax_idx], psd[xmin_idx:xmax_idx], psd_filt[xmin_idx:xmax_idx]                
                plt.plot(f_crop, psd_filt_crop, label='Filtered', color=color)
                plt.plot(f_crop, psd_crop, label='Unfiltered', color='k')
                plt.ylabel(ylabel)
                plt.axvline(x=f0_exact-bw/2, color='g')
                plt.axvline(x=f0_exact+bw/2, color='g')
                plt.xlabel('Frequency [Hz]')
                plt.title(f"{f0_exact:.0f} Hz")

                if fit_peak:
                    x0, y0, gamma, A, fitted_lorentz = fit_lorentzian(f_crop, psd_filt_crop)
                    plt.plot(f_crop, fitted_lorentz, label=rf"$y_0={y0:.3g}$, $\gamma={gamma:.3g}$, $A={A:.3g}$", color=color, ls='--')
                    print(f"Frequency = {f0_exact:.0f}, diff = {np.abs(x0-f0_exact)}")
                    diffs.append(np.abs(x0-f0_exact))
                plt.legend(fontsize=6)
            plot_fp = os.path.join(plot_folder, f"{species} {wf_idx} [{'Good' if good_peaks else 'Bad'}]")
            # plt.show()
            plt.savefig(plot_fp)

diffs = np.array(diffs)
print(f"Max diff = {np.max(diffs)}, mean={np.mean(diffs)}, std = {np.std(diffs)}")