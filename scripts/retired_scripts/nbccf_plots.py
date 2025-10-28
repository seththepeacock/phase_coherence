from N_xi_fit_funcs import *
import phaseco as pc
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, get_window, convolve, correlation_lags
from scipy.optimize import curve_fit
import os
from tqdm import tqdm

# Directories
os.chdir(r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence")
pkl_folder = r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence\paper_analysis\pickles"

all_species = ['Human','Anole', 'Owl', 'Tokay']

bws = {'Anole':10, 'Human':10, 'Owl':10, 'Tokay':10}
xi_max_mss = {'Anole':500, 'Human':500, 'Owl':500, 'Tokay':500}
delta_f_maxs = {'Anole':30, 'Human':30, 'Owl':30, 'Tokay':30}
delta_f_deltas = {'Anole':10, 'Human':10, 'Owl':10, 'Tokay':10}
win_type = 'flattop'

# Make plot directory
plot_folder = f"NBCCFs"
os.makedirs(plot_folder, exist_ok=True)

for wf_idx in range(4):
    for species in all_species:
        print(f"Processing {species} {wf_idx}")
        # Get params
        bw = bws[species]
        xi_max_ms = xi_max_mss[species]
        delta_f_max = delta_f_maxs[species]
        delta_f_delta = delta_f_deltas[species]

        # Make subfolder
        plot_subfolder = f"BW={bw}, DFM={delta_f_max}, DFD={delta_f_delta}, XM={xi_max_ms}, WT={win_type}"
        os.makedirs(os.path.join(plot_folder, plot_subfolder), exist_ok=True)
        

        # Get waveform
        wf_len_s = 60
        wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
            species=species,
            wf_idx=wf_idx,
        )
        wf = crop_wf(wf, fs, wf_len_s)
        # f0s = [good_peak_freqs[0]]
        f0s = np.concat((np.array([10000, 20000]), good_peak_freqs, bad_peak_freqs))

        for f0 in f0s:
            print(f"Processing {f0:.0f}Hz")
            # Check if we already have it
            fn_id = f"{species} {wf_idx}, f0={f0:.0f}.jpg"
            plot_fp = os.path.join(plot_folder, plot_subfolder, fn_id)
            if os.path.exists(plot_fp):
                print("Already got this one, continuing!")
                continue
            
            # Set parameters
            win_type = 'flattop'
            pw = True
            tau = get_precalc_tau_from_bw(bw, fs, win_type, pkl_folder)
            nfft = 8192
            f = rfftfreq(nfft, 1/fs)
            f0_exact = np.argmin(np.abs(f-f0))

            "Get single narrowband around f0"
            win_type = 'flattop'
            win = get_window(win_type, tau)

            n_array = np.arange(tau)

            # Get the center narrowband frequency
            omega_0_norm = f0_exact * 2*np.pi / fs
            kernel = win * np.exp(1j * omega_0_norm * n_array)
            wf_0 = convolve(wf, kernel, mode='valid', method='fft')
            wf_0 = wf_0 - np.mean(wf_0)
            sigma_0 = np.std(wf_0)
            N = len(wf_0)
            xis = correlation_lags(N, N, mode='full')
            xis_ms = xis / fs * 1000
            mid_idx = len(xis)//2

            # Set up the arrays for all the other ones
            delta_f0s = np.arange(-delta_f_max,delta_f_max+1, delta_f_delta)
            ccfs = np.empty((2*N-1, len(delta_f0s)))
            num_terms = N - np.abs(xis) # number of terms in each CCF calculation

            # Calculate the NBCCFs
            for k, delta_f0 in enumerate(tqdm(delta_f0s)):
                omega_norm = (f0_exact+delta_f0) * 2*np.pi / fs
                kernel = win * np.exp(1j * omega_norm * n_array)
                wf_omega = convolve(wf, kernel, mode='valid', method='fft')
                wf_omega = wf_omega - np.mean(wf_omega)
                sigma = np.std(wf_omega)
                ccf = correlate(wf_0, wf_omega, mode='full', method='fft')
                ccf_norm = np.abs(ccf)/(sigma_0*sigma*num_terms)
                # print(np.max(ccf_norm))
                ccfs[:, k] = ccf_norm


            # Plot
            xi_min_ms = -xi_max_ms
            xi_min_ms_idx = np.argmin(np.abs(xis_ms - xi_min_ms))
            xi_max_ms_idx = np.argmin(np.abs(xis_ms - xi_max_ms))

            print("Plotting...")
            plt.close('all')
            plt.figure()

            # make meshgrid
            xx, yy = np.meshgrid(
                xis_ms[xi_min_ms_idx:xi_max_ms_idx], delta_f0s
            )  

            # plot the heatmap
            vmin = 0
            vmax = 1
            heatmap = plt.pcolormesh(
                xx, yy, ccfs[xi_min_ms_idx:xi_max_ms_idx, :].T, vmin=vmin, vmax=vmax, cmap='magma', shading="nearest"
            )

            # get and set label for cbar
            # cbar_label = r"$C_\xi^P$" if pw else r"$C_\xi$"
            cbar = plt.colorbar(heatmap)
            # cbar.set_label(cbar_label, labelpad=30)

            # set axes labels and titles
            plt.xlabel(rf"$\xi$ [ms]")
            plt.ylabel(r"$\Delta f$ [Hz]")
            id = f"[BW={bw}Hz]   [{win_type}]"
            plt.suptitle(id)
            peak_str = f"Noise" if f0 in [10000, 20000] else "Peak"
            plt.title(f"{species} {wf_idx}, {f0:.0f}Hz {peak_str}")
            plt.savefig(plot_fp, dpi=200)
            # plt.show()


