import numpy as np
from N_xi_fit_funcs import *
import os
from phaseco import *
import matplotlib.pyplot as plt
import pandas as pd
import phaseco as pc

for hop_s in [0.01]:
    for pw in [True]:
        print(f"Calculating hop={hop_s}s")
        for species in ["Human"]:
            # Initialize list for row dicts for xlsx file
            rows = []
            for wf_idx in [0]:
                "Get/preprocess waveform"
                # High pass filter cutoff freq, transition band width, and max allowed ripple (in dB)
                hpf = {"type": "kaiser", "cf": 300, "df": 50, "rip": 100} 
                # Will crop waveform to this length (in seconds)
                wf_len_s = 60  
                
                wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
                    species=species,
                    wf_idx=wf_idx,
                    scale=True,
                    hpf=hpf,
                    wf_len_s=wf_len_s,
                )

                "PARAMETERS"
                # WF pre-processing parameters
                

                # Coherence Parameters
                win_meth = {"method": "rho", "rho": 0.7, "snapping_rhortle": 0}
                tau_s = 2**13 / 44100  # Everyone uses the same tau_s
                tau = round(
                    tau_s * fs
                )  # This is just 2**13 for efficient FFT implementation, unless fs!=44100
                xi_min_s = 0.001
                delta_xi_s = 0.001
                hop = round(hop_s * fs)
                force_recalc_colossogram = 0
                const_N_pd = 1

                # Output options
                output_colossogram = 1
                output_peak_picks = 0
                output_fits = 0
                output_bad_fits = 0
                output_spreadsheet = 0
                show_plots = 1

                # Fitting Parameters
                trim_step = 1
                A_max = np.inf  # 1 or np.inf
                sigma_weighting_power = (
                    0  # > 0 means less weight on lower coherence bins in fit
                )
                ddx_thresh_in_num_cycles = True

                # Plotting parameters
                fits_noise_bin = None
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
                

                # Species specific params

                # Maximum xi value
                xi_max_ss = {"Anole": 0.1, "Owl": 0.1, "Human": 0.5, "Tokay": 0.1}

                # Maximum frequency to plot (in khz)
                max_khzs = {"Anole": 6, "Tokay": 6, "Human": 10, "Owl": 12}

                # This determines where to start the fit as the latest peak in the range defined by xi=[0, decay_start_max_xi]
                decay_start_limit_xi_ss = {
                    "Anole": 0.02,
                    "Tokay": 0.02,
                    "Owl": 0.02,
                    "Human": 0.2,
                }

                # Decay Star Method
                # This is how many standard deviations away from the mean to set the noise floor
                noise_floor_bw_factors = {
                    "Anole": 0.5,
                    "Tokay": 0.5,
                    "Owl": 0,
                    "Human": 2,
                }

                # The derivative is very negative at first, and then at some point it flattens out -- this is the point at which it's flattened out enough that we stop the fit
                ddx_threshes = {"Anole": -3, "Tokay": -3, "Owl": -3, "Human": -1}

                ddx_threshes_num_cycles = {
                    "Anole": -0.001,
                    "Tokay": -0.001,
                    "Owl": -0.0005,
                    "Human": -0.00005,
                }

                # Get species-specific params

                # decay_start_limit_xi_s = decay_start_limit_xi_ss[species]
                # TEST
                decay_start_limit_xi_s = None
                max_khz = max_khzs[species]
                xi_max_s = xi_max_ss[species]
                noise_floor_bw_factor = noise_floor_bw_factors[species]
                ddx_thresh = (
                    ddx_threshes_num_cycles[species]
                    if ddx_thresh_in_num_cycles
                    else ddx_threshes[species]
                )

                global_xi_max_s = max(xi_max_ss.values()) if const_N_pd else None

                # Raise warning if tau is not a power of two AND the samplerate is indeed 44100
                if np.log2(tau) != int(np.log2(tau)) and fs == 44100:
                    raise ValueError(
                        "tau is not a power of 2, but the samplerate is 44100!"
                    )

                "Calculate/load things"
                # This will either load it if it's there or calculate it (and pickle it) if not
                N_xi_folder = r"N_xi Fits/"
                colossogram_dict = load_calc_colossogram(
                    wf,
                    wf_idx,
                    wf_fn,
                    wf_len_s,
                    species,
                    fs,
                    hpf,
                    N_xi_folder,
                    pw,
                    tau,
                    tau_s,
                    xi_min_s,
                    xi_max_s,
                    global_xi_max_s,
                    hop,
                    win_meth,
                    force_recalc_colossogram,
                    const_N_pd,
                )

                # Load everything that wasn't explicitly "saved" in the filename
                colossogram = colossogram_dict["colossogram"]
                fn_id = colossogram_dict["fn_id"]
                win_meth_str = colossogram_dict["win_meth_str"]
                hpf_str = colossogram_dict["hpf_str"]
                f = colossogram_dict["f"]
                xis_s = colossogram_dict["xis_s"]
                N_pd_min = colossogram_dict["N_pd_min"]
                N_pd_max = colossogram_dict["N_pd_max"]
                hop = colossogram_dict["hop"]

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
                    N_xi_folder + rf"Results/Results ({win_meth_str}, PW={pw})/"
                )
                all_results_folder = N_xi_folder + rf"Results/Results (All)/"
                os.makedirs(results_folder, exist_ok=True)
                os.makedirs(all_results_folder, exist_ok=True)
                os.makedirs(N_xi_folder + r"Additional Figures/", exist_ok=True)

                "Plots"
                if const_N_pd:
                    if N_pd_min != N_pd_max:
                        raise Exception(
                            "If N_pd is constant, then N_pd_min and N_pd_max should be equal..."
                        )
                    N_pd_str = rf"$N_{{pd}}={N_pd_min}$"
                else:
                    N_pd_str = rf"$N_{{pd}} \in [{N_pd_min}, {N_pd_max}]$"
                suptitle = rf"[{species} {wf_idx}]   [{wf_fn}]   [{win_meth_str}]   [$\tau$={tau_s*1000:.2f}ms]   [$\xi_{{\text{{max}}}}={xi_max_s*1000:.0f}$ms]   [Hop = {(hop / fs)*1000:.0f}ms]   [{wf_len_s}s WF]   [{N_pd_str}]"

                if output_colossogram:
                    print("Plotting Colossogram")
                    plt.close("all")
                    plt.figure(figsize=(15, 5))
                    plot_colossogram(
                        xis_s,
                        f,
                        colossogram,
                        title=rf"Colossogram ($\tau={tau_s:.3f}$s)",
                        max_khz=max_khz,
                        cmap="magma",
                    )
                    for f0_idx in good_peak_idxs:
                        plt.scatter(
                            xi_min_s * 1000 + (xi_max_s * 1000) / 50,
                            f[f0_idx] / 1000,
                            c="w",
                            marker=">",
                            label="Peak at " + f"{f[f0_idx]:0f}Hz",
                            alpha=0.5,
                        )
                    plt.title(f"Colossogram", fontsize=18)
                    plt.suptitle(suptitle, fontsize=10)
                    for folder in [results_folder, all_results_folder]:
                        os.makedirs(rf"{folder}\Colossograms", exist_ok=True)
                        plt.savefig(
                            rf"{folder}\Colossograms\{fn_id} (Colossogram).png", dpi=300
                        )
                    if show_plots:
                        plt.show()

                if output_peak_picks:
                    print("Plotting Peak Picks")
                    target_xi = 0.01
                    xi_idx = np.argmin(np.abs(xis_s - target_xi))
                    coherence_slice = colossogram[:, xi_idx]
                    psd = get_welch(wf=wf, fs=fs, tau=tau)[1]
                    plt.close("all")
                    plt.figure(figsize=(11, 8))
                    plt.suptitle(suptitle)
                    # Coherence slice plot
                    plt.subplot(2, 1, 1)
                    plt.title(rf"Colossogram Slice at $\xi={xis_s[xi_idx]:.3f}$")
                    plt.plot(
                        f / 1000, coherence_slice, label=r"$C_{\xi}$, $\xi={target_xi}$"
                    )
                    for f0_idx in good_peak_idxs:
                        plt.scatter(f[f0_idx] / 1000, coherence_slice[f0_idx], c="r")
                    plt.xlabel("Frequency (kHz)")
                    plt.ylabel(r"$C_{\xi}$")
                    plt.xlim(0, max_khz)
                    # PSD plot
                    plt.subplot(2, 1, 2)
                    plt.title(rf"Power Spectral Density")
                    plt.plot(f / 1000, 10 * np.log10(psd), label="PSD")
                    for f0_idx in good_peak_idxs:
                        plt.scatter(f[f0_idx] / 1000, 10 * np.log10(psd[f0_idx]), c="r")
                    plt.xlabel("Frequency (kHz)")
                    plt.ylabel("PSD [dB]")
                    plt.legend()
                    plt.xlim(0, max_khz)
                    plt.tight_layout()
                    plt.savefig(
                        rf"{results_folder}\Peak Picks\{fn_id} (Peak Picks).png",
                        dpi=300,
                    )
                    if show_plots:
                        plt.show()

                "FITTING"
                if output_fits:
                    print(rf"Fitting {wf_fn}")
                    # Get becky's dataframe
                    if species != "Tokay":
                        df = get_spreadsheet_df(wf_fn, species)

                    p0 = [1, 0.5]
                    bounds = ([0, 0], [np.inf, A_max])  # [T, amp]
                    fit_func = exp_decay

                    for peak_freqs, peak_idxs, good_peaks in zip(
                        [good_peak_freqs, bad_peak_freqs],
                        [good_peak_idxs, bad_peak_idxs],
                        [True, False],
                    ):
                        # If there are no peaks, continue
                        if len(peak_idxs) == 0:
                            if good_peaks:
                                print("WARNING: No good peaks were picked!")
                            # Could also just be that there are no bad peaks!
                            continue

                        if not good_peaks and not output_bad_fits:
                            continue

                        if not good_peaks:
                            print("Now fitting bad peaks...")

                        plt.close("all")
                        plt.figure(figsize=(15, 10))
                        plt.suptitle(f"{suptitle}   [SWP={sigma_weighting_power}]")

                        for f0, f0_idx, color, subplot_idx in zip(
                            peak_freqs, peak_idxs, colors, [1, 2, 3, 4]
                        ):

                            # Fit peak
                            N_xi, N_xi_dict = get_N_xi(
                                xis_s,
                                f,
                                colossogram,
                                f0,
                                decay_start_limit_xi_s,
                                sigma_power=sigma_weighting_power,
                                A_max=A_max,
                            )

                            # Unpack dictionary
                            f0_exact               = N_xi_dict["f0_exact"]
                            N_xi                   = N_xi_dict["N_xi"]
                            N_xi_std               = N_xi_dict["N_xi_std"]
                            T                      = N_xi_dict["T"]
                            T_std                  = N_xi_dict["T_std"]
                            A                      = N_xi_dict["A"]
                            A_std                  = N_xi_dict["A_std"]
                            mse                    = N_xi_dict["mse"]
                            decayed_idx            = N_xi_dict["decayed_idx"]
                            
                            

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
                                    "Decayed Num Cycles":xis_s[decayed_idx] * f0_exact
                                }
                                if species != "Tokay":
                                    SNRfit, fwhm = get_params_from_df(df, f0)
                                    row["SNRfit"], row["FWHM"] = SNRfit, fwhm
                                rows.append(row)

                        # Book it!
                        plt.tight_layout()
                        os.makedirs(results_folder, exist_ok=True)
                        # fits_folder = f'{fig_folder}' if good_peaks else 'Additional Figures'
                        fits_str = f"Fits" if good_peaks else "Bad Fits"
                        for folder in [results_folder, all_results_folder]:
                            os.makedirs(rf"{folder}\{fits_str}", exist_ok=True)
                            plt.savefig(
                                rf"{folder}\{fits_str}\{fn_id} ({fits_str}).png",
                                dpi=300,
                            )
                        if show_plots:
                            plt.show()

        if output_spreadsheet:
            # Save parameter data as xlsx
            df_fitted_params = pd.DataFrame(rows)
            N_xi_fitted_parameters_fn = (
                rf"{results_folder}\N_xi Fitted Parameters ({win_meth_str}, PW={pw})"
            )
            df_fitted_params.to_excel(rf"{N_xi_fitted_parameters_fn}.xlsx", index=False)

        print("Done!")
