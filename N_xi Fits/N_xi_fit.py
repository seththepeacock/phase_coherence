import numpy as np
from N_xi_fit_funcs import *
import os
from phaseco import *
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
for hop_s in [0.004]:
    print(f"Calculating hop={hop_s}s")
    for species in ["Human"]:
        # Initialize list for row dicts for xlsx file
        rows = []
        for wf_idx in [0]:
            "Get/preprocess waveform"  
            # hpf = {'type':'spectral', 'cf':300}
            hpf = {'type':'kaiser', 'cf':300, 'df':50, 'rip':80}
            wf_len_s = 30 # Crop to this length (in seconds)
            wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
                species=species, wf_idx=wf_idx, scale=True, hpf=hpf, wf_len_s=wf_len_s
            )             
            

            "PARAMETERS"
            # Coherence Parameters
            win_meth = {'method':'rho', 'rho':0.7, 'snapping_rhortle':0}
            tau_s = 2**13 / 44100  # Everyone uses the same tau_s
            tau = round(
                tau_s * fs
            )  # This is just 2**13 for efficient FFT implementation, unless fs!=44100
            xi_min_s = 0.001
            delta_xi_s = 0.001
            # hop_s = xi_min_s
            # hop_s = tau // 2
            # hop_s = 0.004
            hop = round(hop_s * fs)
            pw = False
            force_recalc_coherences = 0
            const_N_pd = 1

            # Output options
            output_colossogram = 1
            output_peak_picks = 0
            output_fits = 1
            output_bad_fits = 0
            output_spreadsheet = 0
            show_plots = 0

            # Fitting Parameters
            trim_step = 1
            A_max = np.inf  # 1 or np.inf
            sigma_weighting_power = (
                0  # > 0 means less weight on lower coherence bins in fit
            )
            ddx_thresh_in_num_cycles = True
            

            # Plotting parameters
            plot_noise_on_fits = 1
            plot_single_noise_bin_on_fits = (
                0  # Set this to the frequency you want to plot
            )
            s_signal = 5
            s_noise = 5
            s_decayed = 100
            marker_signal = "o"
            marker_noise = "o"
            marker_decayed = "*"
            lw_fit = 1.5
            alpha_fit = 1
            pe_stroke_fit = [
                pe.Stroke(linewidth=2, foreground="black", alpha=1),
                pe.Normal(),
            ]
            edgecolor_signal = None
            edgecolor_noise = "yellow"
            edgecolor_decayed = "black"
            crop = False
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
                "#17becf",
            ]

            # Species specific params

            # Maximum xi value
            xi_max_ss = {"Anole": 0.1, "Owl": 0.1, "Human": 1.5, "Tokay": 0.1}

            # Maximum frequency to plot (in khz)
            max_khzs = {"Anole": 6, "Tokay": 6, "Human": 10, "Owl": 12}

            # This determines where to start the fit as the latest peak in the range defined by xi=[0, decay_start_max_xi]
            decay_start_xi_max_ss = {
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

            decay_start_xi_max_ss = decay_start_xi_max_ss[species]
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
            coherences_dict = load_calc_coherences(
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
                force_recalc_coherences,
                const_N_pd,
            )

            # Load everything that wasn't explicitly "saved" in the filename
            coherences = coherences_dict["coherences"]
            fn_id = coherences_dict['fn_id']
            win_meth_str = coherences_dict['win_meth_str']
            hpf_str = coherences_dict['hpf_str']
            f = coherences_dict["f"]
            xis_s = coherences_dict["xis_s"]
            N_pd_min = coherences_dict["N_pd_min"]
            N_pd_max = coherences_dict["N_pd_max"]
            hop = coherences_dict["hop"]

            # Handle transpose from old way
            if coherences.shape[0] != xis_s.shape[0]:
                coherences = coherences.T
            
            # Get peak bin indices now that we have f
            good_peak_idxs = np.argmin(
                np.abs(f[:, None] - good_peak_freqs[None, :]), axis=0
            )
            bad_peak_idxs = np.argmin(
                np.abs(f[:, None] - bad_peak_freqs[None, :]), axis=0
            )

            "Make more directories"
            results_folder = N_xi_folder + rf"Results/Results ({win_meth_str}, PW={pw})/"
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
            suptitle = rf"[{species} {wf_idx}]   [{wf_fn}]   [{win_meth_str}]   [$\tau$={tau_s*1000:.2f}ms]   [HPF=({hpf_str})]   [$\xi_{{\text{{max}}}}={xi_max_s*1000:.0f}$ms]   [Hop = {(hop / fs)*1000:.3f}ms]   [{wf_len_s}s WF]   [{N_pd_str}]"

            if output_colossogram:
                print("Plotting Colossogram")
                plt.close("all")
                plt.figure(figsize=(15, 5))
                plot_colossogram(
                    xis_s, f, coherences, title=rf'Colossogram ($\tau={tau_s:.3f}$s)', max_khz=max_khz, cmap="magma"
                )
                for peak_idx in good_peak_idxs:
                    plt.scatter(
                        xi_min_s * 1000 + (xi_max_s * 1000) / 50,
                        f[peak_idx] / 1000,
                        c="w",
                        marker=">",
                        label="Peak at " + f"{f[peak_idx]:0f}Hz",
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
                coherence_slice = coherences[:, xi_idx]
                psd = welch(wf=wf, fs=fs, tau=tau)[1]
                plt.close("all")
                plt.figure(figsize=(11, 8))
                plt.suptitle(suptitle)
                # Coherence slice plot
                plt.subplot(2, 1, 1)
                plt.title(rf"Colossogram Slice at $\xi={xis_s[xi_idx]:.3f}$")
                plt.plot(
                    f / 1000, coherence_slice, label=r"$C_{\xi}$, $\xi={target_xi}$"
                )
                for peak_idx in good_peak_idxs:
                    plt.scatter(
                        f[peak_idx] / 1000, coherence_slice[peak_idx], c="r"
                    )
                plt.xlabel("Frequency (kHz)")
                plt.ylabel(r"$C_{\xi}$")
                plt.xlim(0, max_khz)
                # PSD plot
                plt.subplot(2, 1, 2)
                plt.title(rf"Power Spectral Density")
                plt.plot(f / 1000, 10 * np.log10(psd), label="PSD")
                for peak_idx in good_peak_idxs:
                    plt.scatter(
                        f[peak_idx] / 1000, 10 * np.log10(psd[peak_idx]), c="r"
                    )
                plt.xlabel("Frequency (kHz)")
                plt.ylabel("PSD [dB]")
                plt.legend()
                plt.xlim(0, max_khz)
                plt.tight_layout()
                plt.savefig(
                    rf"{results_folder}\Peak Picks\{fn_id} (Peak Picks).png", dpi=300
                )
                if show_plots:
                    plt.show()

            "FITTING"
            if output_fits:
                print(rf"Fitting {wf_fn}")
                # Get becky's dataframe
                if species != "Tokay":
                    df = get_spreadsheet_df(wf_fn, species)

                p0 = [1, 1]
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
                    unit = "(1/#)" if ddx_thresh_in_num_cycles else "(1/s)"
                    plt.suptitle(f"{suptitle}   [ddx_thresh={ddx_thresh} {unit}]")

                    for peak_freq, peak_idx, color, subplot_idx in zip(
                        peak_freqs, peak_idxs, colors, [1, 2, 3, 4]
                    ):
                        # Pack all parameters for fit_peak together into a tuple for compactness
                        peak_fit_params = (
                            f,
                            peak_idx,
                            noise_floor_bw_factor,
                            decay_start_xi_max_ss,
                            trim_step,
                            sigma_weighting_power,
                            bounds,
                            p0,
                            coherences,
                            xis_s,
                            wf_fn,
                            win_meth_str,
                            ddx_thresh,
                            ddx_thresh_in_num_cycles,
                        )

                        # Fit peak
                        (
                            T,
                            T_std,
                            A,
                            A_std,
                            mse,
                            is_signal,
                            is_noise,
                            decay_start_idx,
                            decayed_idx,
                            target_coherence,
                            xis_s_fit_crop,
                            fitted_exp_decay,
                            noise_means,
                            noise_stds,
                        ) = fit_peak(*peak_fit_params)

                        # Note that all fitting was done on while xis was in seconds, but all subsequent plotting will use xis in num cycles
                        # Nondimensionalize everything
                        freq = f[peak_idx]
                        N_xi = T * freq
                        N_xi_std = T_std * freq
                        xis_num_cycles = xis_s * freq

                        xis_num_cycles_fit_crop = xis_s_fit_crop * freq

                        # Plot the fit
                        plt.subplot(2, 2, subplot_idx)
                        # Handle the case where the peak fit failed
                        if mse == -1:
                            plt.title(rf"{freq:.0f}Hz Peak (FIT FAILED)")
                        # Handle the case where the peak fit succeeded
                        else:
                            plt.title(rf"{freq:.0f}Hz Peak")

                            if T_std < np.inf and A_std < np.inf:
                                fit_label = rf"$N_{{\xi}}={N_xi:.3g}\pm{N_xi_std:.3g}$, $A={A:.3g}\pm{A_std:.3g}$, MSE={mse:.3g}"
                            else:
                                fit_label = ""
                                print("One or more params is infinite!")
                            plt.plot(
                                xis_num_cycles_fit_crop,
                                fitted_exp_decay,
                                color=color,
                                label=fit_label,
                                lw=lw_fit,
                                path_effects=pe_stroke_fit,
                                alpha=alpha_fit,
                                zorder=2,
                            )

                        # Plot the coherence
                        plt.scatter(
                            xis_num_cycles[is_signal],
                            target_coherence[is_signal],
                            s=s_signal,
                            edgecolors=edgecolor_signal,
                            marker=marker_signal,
                            color=color,
                            zorder=1,
                            label="Above Noise Floor",
                        )
                        plt.scatter(
                            xis_num_cycles[is_noise],
                            target_coherence[is_noise],
                            s=s_noise,
                            color=color,
                            edgecolors=edgecolor_noise,
                            zorder=1,
                            label="Below Noise Floor",
                        )
                        # Mark decayed point
                        plt.scatter(
                            xis_num_cycles[decayed_idx],
                            target_coherence[decayed_idx],
                            s=s_decayed,
                            marker=marker_decayed,
                            color=color,
                            edgecolors=edgecolor_decayed,
                            zorder=3,
                        )
                        if plot_noise_on_fits:
                            # plt.scatter(xis_num_cycles, noise_means, label='Noise Mean (Above 12kHz)', s=1, color=colors[4])
                            noise_floor_bw_factor_str = (
                                rf"(\sigma*{noise_floor_bw_factor})"
                                if noise_floor_bw_factor != 1
                                else r"\sigma"
                            )
                            plt.plot(
                                xis_num_cycles,
                                noise_means,
                                label=rf"All Bins $\mu \pm {noise_floor_bw_factor_str}$",
                                color=colors[4],
                            )
                            plt.fill_between(
                                xis_num_cycles,
                                noise_means - noise_stds * noise_floor_bw_factor,
                                noise_means + noise_stds * noise_floor_bw_factor,
                                color=colors[4],
                                alpha=0.3,
                            )
                        if plot_single_noise_bin_on_fits:
                            noise_freq = plot_single_noise_bin_on_fits
                            noise_target_idx = np.argmin(np.abs(f - noise_freq))
                            # plt.scatter(xis_num_cycles, coherences[noise_target_idx, :], label=f'Noise Bin ({noise_freq/1000:.0f}kHz)', s=1, color=colors[5])
                            plt.plot(
                                xis_num_cycles,
                                coherences[noise_target_idx, :],
                                label=f"Noise Bin ({noise_freq/1000:.0f}kHz)",
                                color=colors[5],
                            )

                        # Finish plot
                        plt.xlabel(r"# Cycles")
                        plt.ylabel(r"$C_{\xi}$")
                        plt.ylim(0, 1)
                        plt.legend()

                        # Add params to a row dict
                        if good_peaks and output_spreadsheet:
                            row = {
                                    "Species": species,
                                    "WF Index": wf_idx,
                                    "Filename": wf_fn,
                                    "Frequency": freq,
                                    "N_xi": N_xi,
                                    "N_xi_std": N_xi_std,
                                    "T": T,
                                    "T_std": T_std,
                                    "A": A,
                                    "A_std": A_std,
                                    "MSE": mse,
                                }
                            if species != "Tokay":
                                SNRfit, fwhm = get_params_from_df(df, peak_freq)
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

