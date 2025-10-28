from N_xi_fit_funcs import *
import phaseco as pc
import numpy as np
import os

os.chdir(r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence")
# ,"Human", "Owl", "Tokay"
# Choose waveform and center frequency


# GLobal params
plot_xtg = False
plot_comparison = True

# nfft = None # Setting this to 2**14 will make sure all frequencies are pulled from the same array
# nfft = 2**14
pw = True
hop = 0.2
const_N_pd = 0
# filter_meth = {
#     "type": "kaiser",
#     "cf": 300,
#     "df": 50,
#     "rip": 100,
#                 }
filter_meth = None

C_xi_thresh = 0.5
wf_idxs = [0, 1]




for nfft in [None]:
    for win_meth in [{"method": "static", "win_type": "hann"}, {"method": "static", "win_type": "flattop"}]:
        if plot_comparison:
            plt.close('all')
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Slopes")
            plt.subplot(1, 2, 2)
            plt.title("Y intercepts")
            colors = [
                "#1f77b4",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#7f7f7f",
                "#bcbd22",
            ]
        for wf_idx in wf_idxs:

            for species, color in zip(["Human", "Owl", "Anole", "Tokay"], colors):
                wf_pp = None
                if wf_idx >= 4 and species != "Owl":
                    continue

                # Load waveform
                wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
                    species=species, wf_idx=wf_idx
                )
                noise_freqs = np.array([10000, 20000])
                freqs = np.concat((good_peak_freqs, bad_peak_freqs, noise_freqs))

                # Set parameters
                match species:
                    case "Anole":
                        xi_min_s = 0.001
                        xi_max_s = 0.05
                    case "Owl":
                        xi_min_s = 0.001
                        xi_max_s = 0.05
                    case "Tokay":
                        xi_min_s = 0.001
                        xi_max_s = 0.05
                    case "Human":
                        xi_min_s = 0.002
                        xi_max_s = 0.1
                tau_max_s = 0.35
                delta_tau_s = 0.025

                taus_s = np.linspace(0, 0.35, round(tau_max_s / delta_tau_s) + 1)[1:]

                # Create axes
                taus = np.array(np.round(fs * taus_s), dtype=int)
                xis = pc.get_xis_array(
                    {
                        "xi_min_s": xi_min_s,
                        "xi_max_s": xi_max_s,
                        "delta_xi_s": xi_min_s,
                    },
                    fs,
                )

                xis_s = xis / fs
                taus_s = taus / fs
                N_xis = len(xis)
                N_taus = len(taus)
                thresh = 0.5
                for freq_idx, freq in enumerate(freqs):
                    print(f"Processing f0 {freq_idx+1}/{len(freqs)} ({freq:.2f})")
                    # Allocate xtg
                    xi_tau_gram_f0 = np.empty((N_xis, N_taus))
                    f0_exacts = np.empty(N_taus)
                    for tau_idx, tau in enumerate(taus):

                        print(
                            f"Processing {taus_s[tau_idx]*1000:.1f}ms {tau_idx+1}/{N_taus}"
                        )
                        lcc_kwargs = {
                            "wf": wf,
                            "wf_idx": wf_idx,
                            "wf_fn": wf_fn,
                            "wf_len_s": 60,
                            "wf_pp": wf_pp,
                            "species": species,
                            "fs": fs,
                            "filter": filter_meth,
                            "paper_analysis_folder": r"paper_analysis/",
                            "pw": pw,
                            "tau": tau,
                            "nfft": nfft,
                            "xi_min_s": xi_min_s,
                            "xi_max_s": xi_max_s,
                            "global_xi_max_s": None,
                            "hop": hop,
                            "win_meth": win_meth,
                            "force_recalc_colossogram": 0,
                            "plot_what_we_got": 0,
                            "only_calc_new_coherences": 0,
                            "const_N_pd": const_N_pd,
                            "scale": True,
                            "N_bs": 0,
                            "f0s": freqs,
                        }
                        # Add to xi_tau_gram
                        cgram_dict, wf_pp = load_calc_colossogram(**lcc_kwargs)
                        f = cgram_dict["f"]
                        cgram = cgram_dict["colossogram"]
                        freq_idx = np.argmin(np.abs(f - freq))
                        f0_exacts[tau_idx] = f[freq_idx]
                        cgram_f0 = cgram[:, freq_idx]
                        xi_tau_gram_f0[:, tau_idx] = cgram_f0

                    # Plotting loops
                    for y_axis in ["tau"]:
                        for x_axis in ["xi"]:
                            match y_axis:
                                case "bw":
                                    # bw_omega = 20*np.pi / tau
                                    # bw_f = bw_omega * fs / 2pi
                                    y = (20 * np.pi / taus) * (fs / (2 * np.pi))
                                    ymin = np.min(y)
                                    ymax = np.max(y)
                                    ylabel = r"BPF Width [Hz]"
                                case "tau":
                                    y = taus_s * 1000
                                    ymin = np.min(y)
                                    ymax = np.max(y)
                                    ylabel = r"$\tau$ [ms]"
                            match x_axis:
                                case "num_cycles":
                                    x = xis_s * freq
                                    xlabel = r"$\xi$ [# Cycles]"
                                    xmax = np.max(freqs[:-2]) * np.max(xis_s)
                                    xmin = np.min(freqs[:-2]) * np.min(xis_s)
                                case "xi":
                                    x = xis_s * 1000
                                    xlabel = r"$\xi$ [ms]"
                                    xmax = np.max(xis_s * 1000)
                                    xmin = np.min(xis_s * 1000)
                            # Build strings
                            win_meth_str = pc.get_win_meth_str(
                                win_meth, latex=True
                            )  # This will also check that our win_meth was passed correctly
                            filter_str = get_filter_str(filter_meth)
                            f0_exact_str = (
                                f"{f0_exacts.min():.0f}Hz, {f0_exacts.max():.0f}Hz"
                                if f0_exacts.min() != f0_exacts.max()
                                else f"{f0_exacts[0]:.0f}Hz"
                            )
                            suptitle = rf"[{species} {wf_idx}]   [f0={freq:.0f}Hz ({f0_exact_str})]   [{wf_fn}]   [{win_meth_str}]   [Hop={hop}]"
                            paper_analysis_folder = r"paper_analysis/"
                            xtg_folder = (
                                paper_analysis_folder
                                + rf"Results/Xi-Tau-Grams/y={y_axis}, x={x_axis}/PW={pw}"
                            )
                            full_comparison_id = rf"{species} {wf_idx} f0={freq}, PW={pw}, {win_meth_str}, hop={hop}, nfft={nfft}, xi_max={xi_max_s*1000:.0f}ms, {filter_str}, const_N_pd={const_N_pd}, wf={wf_fn.split('.')[0]}"
                            os.makedirs(xtg_folder, exist_ok=True)

                            if plot_comparison:
                                # for thresh, color in zip([0.25, 0.50, 0.75], ['blue', 'magenta', 'yellow']):
                                proportion = 0.5
                                xis_thresh_idxs = np.argmin(
                                    np.abs(xi_tau_gram_f0 - thresh), axis=0
                                )  # Index into the zeroth (xi) array
                                # Check if we hit the end of the xi array (this would mess up our line fit)
                                for tau_idx, xis_thresh_idx in enumerate(
                                    xis_thresh_idxs
                                ):
                                    taus_s_fit = taus_s
                                    if (
                                        xis_thresh_idx == len(x) - 1
                                    ):  # If we reached the end, crop to there
                                        xis_thresh_idxs = xis_thresh_idxs[:tau_idx]
                                        taus_s_fit = taus_s[:tau_idx]
                                        break
                                if len(xis_thresh_idxs > 0):
                                    xi_threshes = xis_s[xis_thresh_idxs]
                                    p = np.polyfit(
                                        xi_threshes, taus_s_fit, 1
                                    )  # p[0] = slope, p[1] = intercept
                                    if freq_idx == 0 and wf_idx == 0:
                                        label = f"{species}"
                                        # if species == 'Human':
                                        #     label += " [Good]" if good_peaks else " [Bad]"
                                    else:
                                        label = None
                                    if freq in good_peak_freqs:
                                        marker = '*'
                                    elif freq in bad_peak_freqs:
                                        marker = 'o'
                                    elif freq in noise_freqs:
                                        marker = 'x'
                                    plt.subplot(1, 2, 1)
                                    plt.scatter(freq, p[0], color=color, label=label, marker=marker)
                                    plt.subplot(1, 2, 2)
                                    plt.scatter(freq, p[1], color=color, label=label, marker=marker)

                            # Plot
                            if plot_xtg:
                                plt.close("all")
                                plt.figure(figsize=(12, 8))
                                plt.suptitle(suptitle)
                                plt.title(rf"$\xi$-$\tau$-gram")
                                # make meshgrid
                                xx, yy = np.meshgrid(x, y)  # Note we convert to ms

                                # plot the heatmap
                                heatmap = plt.pcolormesh(
                                    xx,
                                    yy,
                                    xi_tau_gram_f0.T,
                                    vmin=0,
                                    vmax=1,
                                    cmap="bone_r",
                                    shading="nearest",
                                )

                                # get and set label for cbar
                                cbar_label = r"$C_\xi$" if pw else r"$C_\xi^\phi$"
                                cbar = plt.colorbar(heatmap)
                                cbar.set_label(cbar_label, labelpad=30)
                                # Plot line of best fit

                                if y_axis == "tau":
                                    for thresh, color in zip(
                                        [0.25, 0.50, 0.75],
                                        ["blue", "magenta", "yellow"],
                                    ):
                                        taus_ms = taus_s * 1000
                                        xis_thresh_idxs = np.argmin(
                                            np.abs(xi_tau_gram_f0 - thresh), axis=0
                                        )  # Index into the zeroth (xi) array
                                        # Check if we hit the end of the xi array (this would mess up our line fit)
                                        for tau_idx, xis_thresh_idx in enumerate(
                                            xis_thresh_idxs
                                        ):
                                            if (
                                                xis_thresh_idx == len(x) - 1
                                            ):  # If we reached the end, crop to there
                                                xis_thresh_idxs = xis_thresh_idxs[
                                                    :tau_idx
                                                ]
                                                taus_ms = taus_ms[:tau_idx]
                                                break
                                        if len(xis_thresh_idxs > 0):
                                            x_thresh = x[xis_thresh_idxs]

                                            plt.scatter(x_thresh, taus_ms, color=color)
                                            p = np.polyfit(
                                                x_thresh, taus_ms, 1
                                            )  # p[0] = slope, p[1] = intercept
                                            plt.plot(
                                                x_thresh,
                                                np.polyval(p, x_thresh),
                                                color=color,
                                                label=rf" $C_\xi^*={thresh:.2f}$ $m={p[0]:.2g}x + {p[1]:.2g}$",
                                            )

                                # set axes labels and titles
                                plt.legend(loc="upper left")
                                plt.xlabel(xlabel)
                                plt.ylabel(ylabel)
                                plt.xlim(xmin, xmax)
                                plt.ylim(ymin, ymax)
                                plt.savefig(
                                    f"{xtg_folder}/{full_comparison_id} (Xi-Tau-Gram).png", dpi=300
                                )

                                # plt.show()

        if plot_comparison:
            plt.subplot(1, 2, 1)
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.legend()
            full_comparison_id = f"C_xi_thresh={C_xi_thresh}, PW={pw}, {win_meth_str}, hop={hop}, nfft={nfft}, xi_max={xi_max_s*1000:.0f}ms, {filter_str}, const_N_pd={const_N_pd}"
            plt.suptitle(full_comparison_id)
            plt.savefig(rf"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence\paper_analysis\Results\Xi-Tau-Grams\{full_comparison_id}.jpg", dpi=300)
            # plt.show()
