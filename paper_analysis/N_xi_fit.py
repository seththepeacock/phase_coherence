import numpy as np
from N_xi_fit_funcs import *
import os
from phaseco import *
import matplotlib.pyplot as plt
import pandas as pd
import phaseco as pc


"PARAMETERS"
# Loop parameters
all_species = [
    "Anole",
    "Human",
    "Owl",
    "Tokay",
]
speciess = ["Anole"]
wf_idxs = range(1)
tau_ss = [2**13 / 44100]
win_meths = [
    # {"method": "rho", "rho": 0.7},
    # {"method": "zeta", "zeta": 0.01, "win_type": "hann"},
    # {"method": "zeta", "zeta": 0.01, "win_type": "boxcar"},
    {"method": "static", "win_type": "flattop"},
]


# WF pre-processing parameters

# bpf_center_freq = 3710
# bpf_bandwidth = 250
filter_meth = {
    "type": "kaiser",
    "cf": 300,
    # "cf": (bpf_center_freq-bpf_bandwidth, bpf_center_freq+bpf_bandwidth),
    "df": 50,
    "rip": 100,
}  # cutoff freq (HPF if one value, BPF if two), transition band width, and max allowed ripple (in dB)
wf_len_s = 60  # Will crop waveform to this length (in seconds)
scale = True  # Scale the waveform for dB SPL (shouldn't have an effect outisde of vertical shift on PSD;
# only actually scales if we know the right scaling constant, which is only Anoles and Humans)

# Coherence Parameters
pw = True
wa = False
hop = 0.2  # proportion of tau
xi_min_s = 0.001
delta_xi_s = xi_min_s

nfft = 2**12
# check this is greater than the max tau_s * 44100
while 4800 * max(tau_ss) > nfft:
    nfft = nfft * 2
nfft = None

const_N_pd = 0

# Options for iterating through subjects
force_recalc_colossogram = 0
plot_what_we_got = 1
only_calc_new_coherences = 0


# Fitting Parameters
noise_freqs = [10000, 20000]
N_bs = 0  # Num of bootstraps to do to calculate fit func CI (0 = standard, no bootstrapping)
# mse_thresh = 0.0001 # Decay start is pushed forward xi by xi until MSE thresh falls below this value
mse_thresh = np.inf
trim_step = 1
A_max = np.inf  # 1 or np.inf
A_const = False  # Fixes the amplitude of the decay at 1
start_fit = "frac"
start_fit_frac = 0.9
stop_fit = "frac"  # stops fit when it reaches a fraction of the fit start value
stop_fit_frac = 0.1  # aforementioned fraction
sigma_weighting_power = 0  # < 0 -> less weight on lower coherence part of fit
fit_func = "exp"  # 'exp' or 'gauss'

# Plotting/output parameters
fits_noise_bin = None
output_colossogram = 1
output_peak_picks = 1
output_fits = 1
output_bad_fits = 1
output_spreadsheet = 0
show_plots = 0

# Species specific params

# Maximum xi value
xi_max_ss = {
    # "Anole": 0.1,
    "Anole": 0.05,
    "Owl": 0.1,
    "Human": 0.2,
    # "Human": 1.0,
    "V Sim Human": 0.2,
    "Tokay": 0.1,
}

# Maximum frequency to plot (in khz)
max_khzs = {
    "Anole": 6,
    "Tokay": 6,
    "Human": 10,
    "V Sim Human": 10,
    "Owl": 12,
}

"Loops"
for win_meth in win_meths:
    for tau_s in tau_ss:
        # Initialize list for row dicts for xlsx file
        rows = []
        for species in speciess:
            for wf_idx in wf_idxs:
                wf_pp = None
                if wf_idx == 4 and species != "Owl":
                    continue
                # if species == "V Sim Human" and wf_idx != 0:
                #     continue

                "Get waveform"
                wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
                    species=species,
                    wf_idx=wf_idx,
                )
                f0s = np.concat((good_peak_freqs, bad_peak_freqs, noise_freqs))

                # Convert to samples
                # hop = round(hop_s * fs)
                tau = round(tau_s * fs)

                # Process species-specific params
                decay_start_limit_xi_s = None  # Defaults to 25% of the waveform
                max_khz = max_khzs[species]
                xi_max_s = xi_max_ss[species]
                # noise_floor_bw_factor = noise_floor_bw_factors[species]
                noise_floor_bw_factor = 1
                global_xi_max_s = max(xi_max_ss.values()) if const_N_pd else None

                "Calculate/load things"
                # This will either load it if it's there or calculate it (and pickle it) if not
                paper_analysis_folder = r"paper_analysis/"
                f0s_cgram = None if output_colossogram else f0s
                cgram, wf_pp = load_calc_colossogram(
                    wf,
                    wf_idx,
                    wf_fn,
                    wf_len_s,
                    wf_pp,
                    species,
                    fs,
                    filter_meth,
                    paper_analysis_folder,
                    pw,
                    tau,
                    tau_s,
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
                    f0s_cgram,
                    wa,
                )
                if "plot_what_we_got" in cgram.keys():
                    print("NO PICKLE, SKIPPING")
                    continue
                if "only_calc_new_coherences" in cgram.keys():
                    print("ALREADY CALCULATED, SKIPPING")
                    continue
                # Load everything that wasn't explicitly "saved" in the filename
                colossogram = cgram["colossogram"]
                fn_id = cgram["fn_id"]
                win_meth_str = cgram["win_meth_str"]
                f = cgram["f"]
                xis_s = cgram["xis_s"]
                N_pd_min = cgram["N_pd_min"]
                N_pd_max = cgram["N_pd_max"]
                hop = cgram["hop"]

                try:
                    method_id = cgram["method_id"]
                except:
                    N_pd_str = pc.get_N_pd_str(const_N_pd, N_pd_min, N_pd_max)
                    method_id = rf"[{win_meth_str}]   [$\tau$={tau_s*1000:.2f}ms]   [$\xi_{{\text{{max}}}}={xi_max_s*1000:.0f}$ms]   [Hop={(hop / fs)*1000:.0f}ms]   [{N_pd_str}]"
                try:
                    filter_str = cgram["hpf_str"]
                except:
                    filter_str = cgram["filter_str"]

                # Handle transpose from old way
                if colossogram.shape[0] != xis_s.shape[0]:
                    colossogram = colossogram.T

                # Get peak bin indices now that we have f
                good_peak_idxs = np.argmin(
                    np.abs(f[:, None] - good_peak_freqs[None, :]), axis=0
                )
                bad_peak_idxs = np.argmin(
                    np.abs(f[:, None] - bad_peak_freqs[None, :]), axis=0
                )

                "Make more directories"
                results_folder = (
                    paper_analysis_folder
                    + rf"Results/Results (PW={pw}, {win_meth_str})"
                )
                all_results_folder = paper_analysis_folder + rf"Results/Results (All)"
                os.makedirs(results_folder, exist_ok=True)
                os.makedirs(all_results_folder, exist_ok=True)
                os.makedirs(
                    paper_analysis_folder + r"Additional Figures/", exist_ok=True
                )

                "Plots"
                suptitle = rf"[{species} {wf_idx}]   [{wf_fn}]   [{wf_len_s}s WF]   {method_id}"
                f_khz = f / 1000
                if output_colossogram:
                    print("Plotting Colossogram")
                    plt.close("all")
                    plt.figure(figsize=(15, 5))
                    pc.plot_colossogram(
                        xis_s,
                        f,
                        colossogram,
                        cmap="magma",
                    )
                    plt.title(rf"Colossogram ($\tau={tau_s:.3f}$s)")
                    plt.ylim(0, max_khz)
                    for f0_idx in good_peak_idxs:
                        plt.scatter(
                            xi_min_s * 1000 + (xi_max_s * 1000) / 50,
                            f_khz[f0_idx],
                            c="w",
                            marker=">",
                            label="Peak at " + f"{f[f0_idx]:0f}Hz",
                            alpha=0.5,
                        )
                    plt.title(f"Colossogram", fontsize=18)
                    plt.suptitle(suptitle, fontsize=10)
                    for folder in [results_folder, all_results_folder]:
                        os.makedirs(rf"{folder}/Colossograms", exist_ok=True)
                        plt.savefig(
                            rf"{folder}/Colossograms/{fn_id} (Colossogram).png", dpi=300
                        )
                    if show_plots:
                        plt.show()

                if output_peak_picks:
                    print("Plotting Peak Picks")
                    wf = filter_wf(wf, fs, filter_meth, species)
                    target_xi = xi_max_s / 10
                    xi_idx = np.argmin(np.abs(xis_s - target_xi))
                    coherence_slice = colossogram[xi_idx, :]
                    psd = pc.get_welch(wf=wf, fs=fs, tau=tau)[1]
                    psd_db = 10 * np.log10(psd)
                    plt.close("all")
                    plt.figure(figsize=(11, 8))
                    plt.suptitle(suptitle)
                    # Coherence slice plot
                    plt.subplot(2, 1, 1)
                    plt.title(rf"Colossogram Slice at $\xi={xis_s[xi_idx]:.3f}$")
                    plt.plot(
                        f_khz, coherence_slice, label=r"$C_{\xi}$, $\xi={target_xi}$"
                    )
                    for f0_idx in good_peak_idxs:
                        plt.scatter(f_khz[f0_idx], coherence_slice[f0_idx], c="r")
                    plt.xlabel("Frequency (kHz)")
                    plt.ylabel(r"$C_{\xi}$")
                    plt.xlim(0, max_khz)
                    # PSD plot
                    plt.subplot(2, 1, 2)
                    plt.title(rf"Power Spectral Density")
                    plt.plot(f_khz, psd, label="PSD")
                    # Color chosen peaks based on the STFT filtering
                    
                    if win_meth['method'] == 'static':   
                        start = time.time()
                        bw = get_win_bw(win_meth, tau, fs, oversample = tau*8, win_bw_thresh_db = 3)
                        bin_width = 1/tau_s
                        bw_bins = round(bw / bin_width)
                        stop = time.time()
                        print(f"{stop}-{start}")
                        for f0_idx in f0s:
                            plt.plot(f_khz[f0_idx-bw_bins:f0_idx+bw_bins], psd_db[f0_idx-bw_bins:f0_idx+bw_bins], )
                    # If it's not static, we just mark with dots
                    else:
                        for f0_idx in f0s:
                            plt.scatter(f_khz[f0_idx], 10 * np.log10(psd[f0_idx]), c="r")
                    plt.xlabel("Frequency (kHz)")
                    plt.ylabel("PSD [dB]")
                    plt.legend()
                    plt.xlim(0, max_khz)
                    plt.tight_layout()
                    os.makedirs(f"{results_folder}/Peak Picks/", exist_ok=True)
                    plt.savefig(
                        rf"{results_folder}/Peak Picks/{fn_id} (Peak Picks).png",
                        dpi=300,
                    )
                    if show_plots:
                        plt.show()

                "FITTING"
                if output_fits:
                    print(rf"Fitting {wf_fn}")
                    # Get becky's dataframe
                    if species not in ["Tokay", "V Sim Human"]:
                        df = get_spreadsheet_df(wf_fn, species)

                    p0 = [1, 0.5]
                    bounds = ([0, 0], [np.inf, A_max])  # [T, amp]

                    for peak_freqs, peak_idxs, good_peaks in zip(
                        [good_peak_freqs, bad_peak_freqs],
                        [good_peak_idxs, bad_peak_idxs],
                        [True, False],
                    ):
                        # If there are no peaks, continue
                        if len(peak_idxs) == 0:
                            if good_peaks:
                                print("WARNING: No good peaks were picked!")
                            # Could also just be that there are no bad peaks! either way...
                            continue
                        if not good_peaks and not output_bad_fits:
                            continue

                        if not good_peaks:
                            print("Now fitting bad peaks...")

                        plt.close("all")
                        plt.figure(figsize=(15, 10))
                        plt.suptitle(f"{suptitle}   [SWP={sigma_weighting_power}]")
                        colors = [
                            "#1f77b4",
                            "#ff7f0e",
                            "#2ca02c",
                            "#d62728",
                            "#9467bd",
                            "#8c564b",
                            "#e377c2",
                            "#7f7f7f",
                            "#bcbd22",
                            "#126290",
                        ]
                        for f0, f0_idx, color, subplot_idx in zip(
                            peak_freqs, peak_idxs, colors, [1, 2, 3, 4]
                        ):

                            # Fit peak
                            N_xi, N_xi_dict = pc.get_N_xi(
                                cgram,
                                f0,
                                decay_start_limit_xi_s=decay_start_limit_xi_s,
                                mse_thresh=mse_thresh,
                                stop_fit=stop_fit,
                                stop_fit_frac=stop_fit_frac,
                                start_fit_frac=start_fit_frac,
                                sigma_power=sigma_weighting_power,
                                A_max=A_max,
                                A_const=A_const,
                                noise_floor_bw_factor=noise_floor_bw_factor,
                                fit_func=fit_func,
                            )

                            # Unpack dictionary
                            f0_exact = N_xi_dict["f0_exact"]
                            N_xi = N_xi_dict["N_xi"]
                            N_xi_std = N_xi_dict["N_xi_std"]
                            T = N_xi_dict["T"]
                            T_std = N_xi_dict["T_std"]
                            A = N_xi_dict["A"]
                            A_std = N_xi_dict["A_std"]
                            mse = N_xi_dict["mse"]
                            decayed_idx = N_xi_dict["decayed_idx"]

                            # Plot the fit
                            plt.subplot(2, 2, subplot_idx)
                            pc.plot_N_xi_fit(N_xi_dict, color)

                            # Add params to a row dict
                            if good_peaks and output_spreadsheet:
                                row = {
                                    "Species": species,
                                    "WF Index": wf_idx,
                                    "Filename": wf_fn,
                                    "Frequency": f0_exact,
                                    "N_xi": N_xi,
                                    "N_xi_std": N_xi_std,
                                    "T": T,
                                    "T_std": T_std,
                                    "A": A,
                                    "A_std": A_std,
                                    "MSE": mse,
                                    "Decayed Xi": xis_s[decayed_idx],
                                    "Decayed Num Cycles": xis_s[decayed_idx] * f0_exact,
                                }
                                if N_bs > 0:
                                    row.update(
                                        {"Average CI Width": N_xi_dict["avg_delta_CI"]}
                                    )
                                if species not in ["Tokay", "V Sim Human"]:
                                    SNRfit, fwhm = get_params_from_df(df, f0)
                                    row["SNRfit"], row["FWHM"] = SNRfit, fwhm
                                rows.append(row)
                        # Book it!
                        plt.tight_layout()
                        os.makedirs(results_folder, exist_ok=True)
                        # fits_folder = f'{fig_folder}' if good_peaks else 'Additional Figures'
                        # fits_mod_str = f' (BRP={bs_resample_prop}, A_max={A_max})'
                        # fits_mod_str = f' (A_const={A_const})'
                        fits_mod_str = ""
                        fits_str = (
                            f"Fits{fits_mod_str}"
                            if good_peaks
                            else f"Bad Fits{fits_mod_str}"
                        )
                        for folder in [results_folder, all_results_folder]:
                            os.makedirs(rf"{folder}/{fits_str}", exist_ok=True)
                            plt.savefig(
                                rf"{folder}/{fits_str}/{fn_id} ({fits_str}).png",
                                dpi=300,
                            )
                        if show_plots:
                            plt.show()

        if output_spreadsheet and not only_calc_new_coherences:
            # Save parameter data as xlsx
            df_fitted_params = pd.DataFrame(rows)
            N_xi_fitted_parameters_fn = rf"{results_folder}/N_xi Fitted Parameters ({win_meth_str}, PW={pw}, tau={tau_s*1000:.2f}ms, {fits_mod_str})"
            df_fitted_params.to_excel(rf"{N_xi_fitted_parameters_fn}.xlsx", index=False)

    print("Done!")
