import numpy as np
from N_xi_fit_funcs import *
import os
from phaseco import *
import matplotlib.pyplot as plt
import pandas as pd
import phaseco as pc
from scipy.fft import rfftfreq

all_species = ["Human", "Anole", "Owl", "Tokay"]
speciess = ["Human"]
wf_idxs = range(1)

for win_meth in [
    # {"method": "rho", "rho": 0.7},
    # {"method": "zeta", "zeta": 0.01, "win_type": "hann"},
    # {"method": "zeta", "zeta": 0.01, "win_type": "boxcar"},
    {"method": "static", "win_type": "hann"},
]:
    for tau_power in [13]:
        # Initialize list for row dicts for xlsx file
        rows = []
        for species in speciess:
            for wf_idx in wf_idxs:
                if wf_idx == 4 and species != "Owl":
                    continue

                "PARAMETERS"
                # WF pre-processing parameters
                # bpf_center_freq = 3710
                # bpf_bandwidth = 250
                filter = {
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
                hop_s = 0.01
                tau_s = 2**tau_power / 44100  # Everyone uses the same tau_s
                xi_min_s = 0.001
                delta_xi_s = 0.001
                const_N_pd = 1
                quantize_f0 = 1

                # Output options
                output_fits = 0
                output_bad_fits = 0
                output_spreadsheet = 0
                show_plots = 0

                # Fitting Parameters
                bootstrap = 0  # Bootstrap fits for a CI
                bs_resample_prop = 1.0  # Proportion of fit points to resample
                # mse_thresh = 0.0001 # Decay start is pushed forward xi by xi until MSE thresh falls below this value
                mse_thresh = np.inf
                trim_step = 1
                A_max = np.inf  # 1 or np.inf
                A_const = True  # Fixes the amplitude of the decay at 1
                stop_fit = "frac"  # stops fit when it reaches a fraction of the fit start value
                stop_fit_frac = 0.1  # aforementioned fraction
                sigma_weighting_power = (
                    0  # < 0 -> less weight on lower coherence part of fit
                )
                fit_func = "exp"  # 'exp' or 'gauss'

                # Plotting parameters
                fits_noise_bin = None

                # Species specific params

                # Maximum xi value
                xi_max_ss = {
                    "Anole": 0.1,
                    "Owl": 0.1,
                    "Human": 0.3,
                    # "Human": 1.0,
                    "V Sim Human": 0.2,
                    "Tokay": 0.1,
                }

                "Get and process waveform"
                wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
                    species=species,
                    wf_idx=wf_idx,
                )

                # Scale wf
                if species in ["Anole", "Human"]:  # Scale wf
                    wf = scale_wf(wf)

                # Crop wf
                wf = crop_wf(wf, fs, wf_len_s)

                # Apply filter
                wf = filter_wf(wf, fs, filter, species)

                "Bookkeeping on params"

                decay_start_limit_xi_s = None  # Defaults to 25% of the waveform
                xi_max_s = xi_max_ss[species]
                noise_floor_bw_factor = 1
                global_xi_max_s = max(xi_max_ss.values()) if const_N_pd else None

                # Convert to samples
                xi_min = round(xi_min_s * fs)
                xi_max = round(xi_max_s * fs)
                tau = round(
                    tau_s * fs
                )  # This is just 2**13 for (power of 2 = maximally efficient FFT), except for owls where fs!=44100
                hop = round(hop_s * fs)

                # Raise warning if tau is not a power of two AND the samplerate is indeed 44100
                if np.log2(tau) != int(np.log2(tau)) and fs == 44100:
                    raise ValueError(
                        "tau is not a power of 2, but the samplerate is 44100!"
                    )

                "Make more directories"
                filter_str = get_filter_str(filter)
                win_meth_str = get_win_meth_str(win_meth)
                # N_pd_str = get_N_pd_str(const_N_pd, N_pd_min, N_pd_max)
                method_id = rf"[{win_meth_str}]   [$\tau$={tau_s*1000:.2f}ms]   [$\xi_{{\text{{max}}}}={xi_max_s*1000:.0f}$ms]   [Hop={(hop / fs)*1000:.0f}ms]"

                paper_analysis_folder = r"paper_analysis/"
                results_super_folder = (
                    rf"{paper_analysis_folder}Results [SINGLE f0 TEST]/"
                )
                results_folder = (
                    rf"{results_super_folder}/Results (PW={pw}, {win_meth_str})"
                )
                all_results_folder = rf"{results_super_folder}/Results (All)"
                os.makedirs(results_folder, exist_ok=True)
                os.makedirs(all_results_folder, exist_ok=True)
                os.makedirs(
                    paper_analysis_folder + r"Additional Figures/", exist_ok=True
                )

                "Plots"
                suptitle = rf"[{species} {wf_idx}]   [{wf_fn}]   [{wf_len_s}s WF]   {method_id}"

                "FITTING"
                print(rf"Fitting {wf_fn}")
                # Get becky's dataframe
                if species not in ["Tokay", "V Sim Human"]:
                    df = get_spreadsheet_df(wf_fn, species)

                p0 = [1, 0.5]
                bounds = ([0, 0], [np.inf, A_max])  # [T, amp]

                for peak_freqs, good_peaks in zip(
                    [good_peak_freqs, bad_peak_freqs],
                    [True, False],
                ):
                    # If there are no peaks, continue
                    if len(peak_freqs) == 0:
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
                    colors = [
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",
                        "#9467bd",
                        "#8c564b",
                    ]
                    for f0, color, subplot_idx in zip(peak_freqs, colors, [1, 2, 3, 4]):
                        if quantize_f0:
                            f_dft = rfftfreq(tau, 1 / fs)
                            f0 = f_dft[np.argmin(np.abs(f_dft - f0))]
                        print(f0)
                        # Calculate cgram slice
                        cgram_slice_dict = pc.get_colossogram(
                            wf,
                            fs,
                            xis={
                                "xi_min": xi_min,
                                "xi_max": xi_max,
                                "delta_xi": xi_min,
                            },
                            hop=hop,
                            tau=tau,
                            f0=f0,
                            win_meth=win_meth,
                            pw=pw,
                            const_N_pd=const_N_pd,
                            global_xi_max_s=global_xi_max_s,
                            return_dict=True,
                        )

                        # Fit peak
                        N_xi, N_xi_dict = pc.get_N_xi(
                            cgram_slice_dict,
                            f0,
                            decay_start_limit_xi_s=decay_start_limit_xi_s,
                            mse_thresh=mse_thresh,
                            stop_fit=stop_fit,
                            stop_fit_frac=stop_fit_frac,
                            sigma_power=sigma_weighting_power,
                            A_max=A_max,
                            A_const=A_const,
                            noise_floor_bw_factor=noise_floor_bw_factor,
                            bootstrap=bootstrap,
                            bs_resample_prop=bs_resample_prop,
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
                        xis_s = N_xi_dict["xis_s"]

                        # Plot the fit
                        plt.subplot(2, 2, subplot_idx)
                        pc.plot_N_xi_fit(N_xi_dict, color, bootstrap=bootstrap)

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
                            if bootstrap:
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
                    # A_str_mod = f'BRP={bs_resample_prop}, A_max={A_max}'
                    fits_mod_str = f"fit_func={fit_func}, A_const={A_const}"
                    fits_str = (
                        f"Fits ({fits_mod_str})"
                        if good_peaks
                        else f"Bad Fits ({fits_mod_str})"
                    )
                    for folder in [results_folder, all_results_folder]:
                        os.makedirs(rf"{folder}/{fits_str}", exist_ok=True)
                        plt.savefig(
                            rf"{folder}/{fits_str}/{fn_id} ({fits_str}).png",
                            dpi=300,
                        )
                    if show_plots:
                        plt.show()

        if output_spreadsheet:
            # Save parameter data as xlsx
            fn_id = rf"{species} {wf_idx}, PW={pw}, {win_meth_str}, hop={(hop/fs)*1000:.0f}ms, tau={tau_s*1000:.0f}ms, {filter_str}, xi_max={xi_max_s*1000:.0f}ms, wf_len={wf_len_s}s, wf={wf_fn.split('.')[0]}"
            df_fitted_params = pd.DataFrame(rows)
            N_xi_fitted_parameters_fn = rf"{results_folder}/N_xi Fitted Parameters ({win_meth_str}, PW={pw}, tau={tau_s*1000:.2f}ms, {fits_mod_str})"
            df_fitted_params.to_excel(rf"{N_xi_fitted_parameters_fn}.xlsx", index=False)

    print("Done!")
