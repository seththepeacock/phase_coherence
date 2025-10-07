import numpy as np
from N_xi_fit_funcs import *
import os
from phaseco import *
import matplotlib.pyplot as plt
import pandas as pd
import phaseco as pc

os.chdir(r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence")

"PARAMETERS"
# Loop parameters
all_species = [
    "Anole",
    "Human",
    "Owl",
    "Tokay",
]

speciess = [
    # "Tokay",
    # "Human",
    "Owl",
    # "Anole"
]


wf_idxs = [1]

# speciess = all_species
# wf_idxs = range(4)


long = 0

# WF pre-processing parameters
filter_meths = [
    # {
    #     "type": "kaiser",
    #     "cf": 300,
    #     "df": 50,
    #     "rip": 100,
    # },
    None,
]
# cutoff freq (HPF if one value, BPF if two), transition band width, and max allowed ripple (in dB)

wf_len_s = 60  # Will crop waveform to this length (in seconds)
scale = True  # Scale the waveform for dB SPL (shouldn't have an effect outisde of vertical shift on PSD;
# only actually scales if we know the right scaling constant, which is only Anoles and Humans)

# Coherence Parameters
pws = [False]
bws = [200]
rhos = [None]
hop_props = [0.1]
wa = False
const_N_pd = 0
nfft = 2**13


# PSD Parameters
tau_psd = nfft


# Options for iterating through subjects
force_recalc_colossogram = 0
plot_what_we_got = 0
only_calc_new_coherences = 0


# Fitting Parameters
noise_freqs = [10000, 20000]
N_bs = 0  # Num of bootstraps to do to calculate fit func CI (0 = standard, no bootstrapping)
mse_thresh = np.inf
A_max = np.inf  # 1 or np.inf
A_const = False  # Fixes the amplitude of the decay at 1
start_fit = "frac"
start_fit_frac = 0.9
stop_fit = "frac"  # stops fit when it reaches a fraction of the fit start value
stop_fit_frac = 0.1  # aforementioned fraction
sigma_weighting_power = 0  # < 0 -> less weight on lower coherence part of fit
fit_func = "exp"  # 'exp' or 'gauss'

# Plotting/output parameters
output_colossogram = 0
output_peak_picks = 1
output_fits = 0
output_bad_fits = 0
output_spreadsheet = (
    (wf_idxs == range(4)) and (speciess == all_species)
) and not plot_what_we_got and not long and output_fits
output_spreadsheet = 0
show_plots = 1
force_all_freqs = True

# Species specific params

# Maximum xi value
xi_max_ss = {
    "Anole": 0.1,
    "Owl": 0.1,
    "Human":1.0,
    # "Human":1.5,
    "V Sim Human": 0.2,
    "Tokay": 0.1,
}

xi_min_ss = {
    "Anole": 0.001,
    "Owl": 0.001,
    "Human": 0.001,
    "Tokay": 0.001,
}

# Maximum frequency to plot (in khz)
max_khzs = {
    "Anole": 6,
    "Tokay": 6,
    "Human": 10,
    "V Sim Human": 10,
    "Owl": 12,
}



# Define a folder
pkl_folder = os.path.join("paper_analysis", "pickles")

"Loops"
for filter_meth in filter_meths:
    for pw in pws:
        for rho in rhos:
            if rho is not None:
                win_meth = {"method": "rho", "win_type": "flattop", "rho": rho}
            else:
                win_meth = {"method": "static", "win_type": "flattop"}
            for hop_prop in hop_props:
                for bw in bws:
                    # Initialize list for row dicts for xlsx file
                    rows = []
                    for species in speciess:
                        for wf_idx in wf_idxs:
                            wf_pp = None
                            xi_min_s = xi_min_ss[species]
                            if wf_idx == 4 and species != "Owl":
                                continue
                            if bw in [100, 150, 200] and species == "Human":
                                continue
                            # if species == "V Sim Human" and wf_idx != 0:
                            #     continue
                            # if not pw and wf_idx != 2:
                            #     continue

                            "Get waveform"
                            wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
                                species=species,
                                wf_idx=wf_idx,
                            )
                            all_sel_freqs = np.concat(
                                (good_peak_freqs, bad_peak_freqs, noise_freqs)
                            )
                            f0s_cgram = (
                                None
                                if output_colossogram or force_all_freqs
                                else all_sel_freqs
                            )
                            # Get precalculated tau for this bandwidth
                            tau = get_precalc_tau_from_bw(bw, fs, win_meth["win_type"])

                            # Check we haven't exceeded our max set by nfft
                            if tau > nfft:
                                raise ValueError(
                                    f"Can't have tau = {tau} > {nfft} = nfft!"
                                )

                            # Process species-specific params
                            decay_start_limit_xi_s = 0.05 if species == 'Human' else 0.02
                            max_khz = max_khzs[species]
                            xi_max_s = xi_max_ss[species]
                
                            # LONG FITS
                            if long:
                                xi_max_s = 10.0
                                xi_min_s = 0.1

                            # These ones didn't end by 1.0, so will take em out to 1.5 instead
                            if species=='Human' and wf_idx in [2, 3]:
                                if xi_max_s==1.0:
                                    xi_max_s = 1.5
                            
                            # # STATIC WINDOW PARAMS
                            # if rho is None:
                            #     xi_min_s = 0.001
                            #     xi_max_s = 0.05

                            "Calculate/load things"
                            # This will either load it if it's there or calculate it (and pickle it) if not
                            paper_analysis_folder = r"paper_analysis/"
                            cgram_dict = load_calc_colossogram(
                                **{
                                    "wf": wf,
                                    "wf_idx": wf_idx,
                                    "wf_fn": wf_fn,
                                    "wf_len_s": wf_len_s,
                                    "wf_pp": wf_pp,
                                    "species": species,
                                    "fs": fs,
                                    "filter_meth": filter_meth,
                                    "pkl_folder": pkl_folder,
                                    "pw": pw,
                                    "tau": tau,
                                    "nfft": nfft,
                                    "xi_min_s": xi_min_s,
                                    "xi_max_s": xi_max_s,
                                    "hop": hop_prop,
                                    "win_meth": win_meth,
                                    "force_recalc_colossogram": force_recalc_colossogram,
                                    "plot_what_we_got": plot_what_we_got,
                                    "only_calc_new_coherences": only_calc_new_coherences,
                                    "const_N_pd": const_N_pd,
                                    "scale": scale,
                                    "N_bs": N_bs,
                                    "f0s": f0s_cgram,
                                }
                            )
                            if "plot_what_we_got" in cgram_dict.keys():
                                print("NO PICKLE, SKIPPING")
                                continue
                            if "only_calc_new_coherences" in cgram_dict.keys():
                                print("ALREADY CALCULATED, SKIPPING")
                                continue
                                
                            # Load everything that wasn't explicitly "saved" in the filename
                            colossogram = cgram_dict["colossogram"]
                            wf_pp = cgram_dict['wf_pp'] # this is now no longer None and we can use it next time (unless it gets reset because we're not doing anything else with this subject)
                            win_meth_str = cgram_dict["win_meth_str"]
                            f = cgram_dict[
                                "f"
                            ]  # This will be just f0s if they were passed in
                            xis_s = cgram_dict["xis_s"]
                            N_pd_min = cgram_dict["N_pd_min"]
                            N_pd_max = cgram_dict["N_pd_max"]

                            # Handle transpose from old way
                            if colossogram.shape[0] != xis_s.shape[0]:
                                colossogram = colossogram.T

                            # Get peak bin indices now that we have f
                            # (this is either the full nfft freq axis or just the short list, works either way)
                            good_peak_idxs = np.argmin(
                                np.abs(f[:, None] - good_peak_freqs[None, :]), axis=0
                            )
                            bad_peak_idxs = np.argmin(
                                np.abs(f[:, None] - bad_peak_freqs[None, :]), axis=0
                            )

                            all_sel_freq_idxs = np.argmin(
                                np.abs(f[:, None] - all_sel_freqs[None, :]), axis=0
                            )  # Trivial in the case that f = all_sel_freqs

                            "Make more directories"
                            long_str = "LONG " if long else ""
                            results_folder = (
                                paper_analysis_folder
                                + rf"Results/{long_str}Results (PW={pw}, BW={bw}Hz, {win_meth_str})"
                            )
                            all_results_folder = (
                                paper_analysis_folder + rf"Results/{long_str}Results (All Static)"
                            )
                            os.makedirs(results_folder, exist_ok=True)
                            os.makedirs(all_results_folder, exist_ok=True)
                            os.makedirs(
                                paper_analysis_folder + r"Additional Figures/",
                                exist_ok=True,
                            )

                            "Plots"
                            # Build plot-related strings
                            N_bs_str = "" if N_bs == 0 else f"N_bs={N_bs}, "
                            pw_str = f"{pw}" if not wa or not pw else "WA"
                            const_N_pd_str = "" if const_N_pd else "N_pd=max, "
                            f0s_str = (
                                ""
                                if f0s_cgram is None
                                else f"f0s={np.array2string(f0s_cgram, formatter={'float' : lambda x: "%.0f" % x})}, "
                            )
                            nfft_str = "" if nfft is None else f"nfft={nfft}, "
                            delta_xi_str = (
                                ""
                                if xi_min_s == 0.001
                                else f"delta_xi={xi_min_s*1000:.0f}ms, "
                            )
                            bw_str = (
                                f"HPBW={bw}Hz"
                                if bw is not None
                                else f"tau={1000 * tau / fs:.0f}ms"
                            )
                            N_pd_str = pc.get_N_pd_str(const_N_pd, N_pd_min, N_pd_max)
                            filter_str = get_filter_str(filter_meth)
                            # Build IDs
                            plot_fn_id = rf"{species} {wf_idx}, {bw_str}, hop={hop_prop:.2g}, PW={pw_str}, {win_meth_str}, {filter_str}, xi_max={xi_max_s*1000:.0f}ms, {delta_xi_str}{nfft_str}{f0s_str}{const_N_pd_str}{N_bs_str}wf_len={wf_len_s}s, wf={wf_fn.split('.')[0]}"
                            method_id = rf"[$\tau$={(tau/fs)*1000:.2f}ms]   [PW={pw}]   [{win_meth_str}]   [Hop={hop_prop:.2g}$\tau$]   [{N_pd_str}]   [nfft={nfft}]"
                            suptitle = rf"[{species} {wf_idx}]   [{wf_fn}]   [HPBW={bw}Hz]   {method_id}   [{filter_str}]"
                            f_khz = f / 1000
                            good_colors = [
                                "#1f77b4",
                                "#ff7f0e",
                                "#e377c2",
                                "#9467bd",
                            ]
                            bad_colors = [
                                "#d62728",
                                "#8c564b",
                                "#7f7f7f",
                                "#bcbd22",
                            ]
                            if output_colossogram:
                                print("Plotting Colossogram")
                                plt.close("all")
                                plt.figure(figsize=(15, 5))
                                pc.plot_colossogram(
                                    xis_s,
                                    f,
                                    colossogram,
                                    pw=pw,
                                    cmap="magma",
                                )
                                plt.title(rf"Colossogram")
                                plt.ylim(0, max_khz)
                                for peak_idx in good_peak_idxs:
                                    plt.scatter(
                                        xi_min_s * 1000 + (xi_max_s * 1000) / 50,
                                        f_khz[peak_idx],
                                        c="w",
                                        marker=">",
                                        label="Peak at " + f"{f[peak_idx]:0f}Hz",
                                        alpha=0.5,
                                    )
                                plt.title(f"Colossogram", fontsize=18)
                                plt.suptitle(suptitle, fontsize=10)
                                for folder in [results_folder, all_results_folder]:
                                    os.makedirs(
                                        rf"{folder}/Colossograms", exist_ok=True
                                    )
                                    plt.savefig(
                                        rf"{folder}/Colossograms/{plot_fn_id} (Colossogram).jpg",
                                        dpi=300,
                                    )
                                if show_plots:
                                    plt.show()

                            if output_peak_picks:
                                print("Plotting Peak Picks")
                                # See if we need to preprocess waveform
                                if wf_pp is None:
                                    wf = crop_wf(wf, fs, wf_len_s)
                                    wf = filter_wf(wf, fs, filter_meth, species)
                                    if (
                                        species in ["Anole", "Human"] and scale
                                    ):  # Scale wf
                                        wf = scale_wf(wf)
                                else:
                                    wf = wf_pp

                                # Calculate arrays for plotting
                                target_xi_s = 0.01
                                xi_idx = np.argmin(np.abs(xis_s - target_xi_s))
                                coherence_slice = colossogram[xi_idx, :]
                                f_psd, psd = pc.get_welch(
                                    wf=wf, fs=fs, tau=tau_psd, win="hann"
                                )
                                psd_db = 10 * np.log10(psd)
                                f_psd_khz = f_psd / 1000
                                good_peak_idxs_psd = np.argmin(
                                    np.abs(good_peak_freqs[:, None] - f_psd[None, :]),
                                    axis=1,
                                )
                                bad_peak_idxs_psd = np.argmin(
                                    np.abs(bad_peak_freqs[:, None] - f_psd[None, :]),
                                    axis=1,
                                )

                                # Initialize plot
                                plt.close("all")
                                plt.figure(figsize=(12, 8))
                                plt.suptitle(suptitle)

                                # Coherence slice plot
                                if f0s_cgram is None:
                                    plt.subplot(2, 1, 1)
                                    plt.title(
                                        rf"Colossogram Slice at $\xi={xis_s[xi_idx]:.3f}$"
                                    )
                                    plt.plot(
                                        f_khz,
                                        coherence_slice,
                                        label=r"$C_{\xi}$, $\xi={target_xi}$",
                                        color="k",
                                    )
                                    for peak_idxs, colors in zip(
                                        [good_peak_idxs, bad_peak_idxs],
                                        [good_colors, bad_colors],
                                    ):
                                        for peak_idx, color in zip(peak_idxs, colors):
                                            plt.scatter(
                                                f_khz[peak_idx],
                                                coherence_slice[peak_idx],
                                                c=color,
                                            )
                                    plt.xlabel("Frequency (kHz)")
                                    plt.ylabel(r"$C_{\xi}$")
                                    plt.xlim(0, max_khz)
                                    plt.ylim(0, 1)
                                    plt.subplot(2, 1, 2)

                                # PSD plot
                                plt.title(rf"Power Spectral Density")
                                plt.plot(f_psd_khz, psd_db, label="PSD", color="k")
                                bw_khz = bw / 1000
                                for peak_idxs_psd, colors in zip(
                                    [good_peak_idxs_psd, bad_peak_idxs_psd],
                                    [good_colors, bad_colors],
                                ):
                                    for peak_idx_psd, color in zip(
                                        peak_idxs_psd, colors
                                    ):
                                        plt.scatter(
                                            f_psd_khz[peak_idx_psd],
                                            psd_db[peak_idx_psd],
                                            c=color,
                                        )
                                        f0_khz = f_psd_khz[peak_idx_psd]
                                        # Color according to bandwidth
                                        a, b = f0_khz - bw_khz / 2, f0_khz + bw_khz / 2

                                        # Interpolate so you get exact endpoints
                                        ya = np.interp(a, f_psd_khz, psd_db)
                                        yb = np.interp(b, f_psd_khz, psd_db)

                                        # Restrict x,y to the band plus the interpolated endpoints
                                        mask = (f_psd_khz >= a) & (f_psd_khz <= b)
                                        f_band = np.concatenate(
                                            ([a], f_psd_khz[mask], [b])
                                        )
                                        psd_band = np.concatenate(
                                            ([ya], psd_db[mask], [yb])
                                        )
                                        plt.plot(
                                            f_band,
                                            psd_band,
                                            c=color,
                                        )

                                plt.xlabel("Frequency (kHz)")
                                plt.ylabel("PSD [dB]")
                                plt.legend()
                                plt.xlim(0, max_khz)
                                plt.tight_layout()
                                for folder in [results_folder, all_results_folder]:
                                    os.makedirs(f"{folder}/Peak Picks/", exist_ok=True)
                                    plt.savefig(
                                        rf"{folder}/Peak Picks/{plot_fn_id} (Peak Picks).jpg",
                                        dpi=300,
                                    )
                                if show_plots:
                                    plt.show()

                            "FITTING"
                            if output_fits:
                                print(rf"Fitting {wf_fn}")
                                # Get becky's dataframe
                                if species not in ["Tokay", "V Sim Human"] and output_spreadsheet:
                                    df = get_spreadsheet_df(wf_fn, species)

                                p0 = [1, 0.5]
                                bounds = ([0, 0], [np.inf, A_max])  # [T, amp]
                                for zoom_to_fit in [True, False]:
                                    if long and zoom_to_fit:
                                        continue
                                    for peak_freqs, peak_idxs, colors, good_peaks in zip(
                                        [good_peak_freqs, bad_peak_freqs],
                                        [good_peak_idxs, bad_peak_idxs],
                                        [good_colors, bad_colors],
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
                                        swp_str = "" if sigma_weighting_power==0 else f"   [SWP={sigma_weighting_power}]"
                                        plt.suptitle(
                                            f"{suptitle}{swp_str}"
                                        )

                                        for f0, peak_idx, color, subplot_idx in zip(
                                            peak_freqs, peak_idxs, colors, [1, 2, 3, 4]
                                        ):

                                            # Fit peak
                                            N_xi, N_xi_dict = pc.get_N_xi(
                                                cgram_dict,
                                                f0,
                                                decay_start_limit_xi_s=decay_start_limit_xi_s,
                                                mse_thresh=mse_thresh,
                                                stop_fit=stop_fit,
                                                stop_fit_frac=stop_fit_frac,
                                                start_fit_frac=start_fit_frac,
                                                sigma_power=sigma_weighting_power,
                                                A_max=A_max,
                                                A_const=A_const,
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
                                            plot_noise_floor = (
                                                True if f0s_cgram is None else False
                                            )

                                            pc.plot_N_xi_fit(
                                                N_xi_dict,
                                                color,
                                                plot_noise_floor=plot_noise_floor,
                                                zoom_to_fit=zoom_to_fit
                                            )

                                            # Add params to a row dict
                                            if good_peaks and output_spreadsheet and zoom_to_fit:
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
                                                    "Decayed Num Cycles": xis_s[decayed_idx]
                                                    * f0_exact,
                                                }
                                                if N_bs > 0:
                                                    row.update(
                                                        {
                                                            "Average CI Width": N_xi_dict[
                                                                "avg_delta_CI"
                                                            ]
                                                        }
                                                    )
                                                if species not in ["Tokay", "V Sim Human"]:
                                                    SNRfit, fwhm = get_params_from_df(df, f0)
                                                    row["SNRfit"], row["FWHM"] = SNRfit, fwhm
                                                rows.append(row)
                                        # Book it!
                                        plt.tight_layout()
                                        os.makedirs(results_folder, exist_ok=True)
                                        zoom_str = "[ZOOMED]" if zoom_to_fit else ""
                                        fits_str = (
                                            f"Fits [Good]"
                                            if good_peaks
                                            else f"Fits [Bad]"
                                        )
                                        for folder in [results_folder, all_results_folder]:
                                            os.makedirs(
                                                rf"{folder}/{fits_str}", exist_ok=True
                                            )
                                            plt.savefig(
                                                rf"{folder}/{fits_str}/{plot_fn_id} ({fits_str}{zoom_str}).jpg",
                                                dpi=300,
                                            )
                                            # os.makedirs(rf"{folder}/Fits [All]{fits_mod_str}", exist_ok=True)
                                            # plt.savefig(
                                            #     rf"{folder}/Fits [All]{fits_mod_str}/{plot_fn_id} ({fits_str}).jpg",
                                            #     dpi=300,
                                            # )
                                        if show_plots:
                                            plt.show()

                    if output_spreadsheet and not only_calc_new_coherences:
                        # Save parameter data as xlsx
                        df_fitted_params = pd.DataFrame(rows)
                        N_xi_fitted_parameters_fn = rf"{results_folder}/N_xi Fitted Parameters ({win_meth_str}, PW={pw}, BW={bw}Hz{zoom_str})"
                        df_fitted_params.to_excel(
                            rf"{N_xi_fitted_parameters_fn}.xlsx", index=False
                        )

                print("Done!")
