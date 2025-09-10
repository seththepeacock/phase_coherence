import phaseco as pc
from N_xi_fit_funcs import *

all_species = ["Anole", "Human", "Owl", "Tokay"]

for win_meth in [
    # {"method": "zeta", "zeta": 0.01, "win_type": "hann"},
    {"method": "rho", "rho": 0.7},
    # {"method": "zeta", "zeta": 0.01, "win_type": "boxcar"},
    # {"method": "static", "win_type": "hann"},
]:
    for pw in [True]:
        for species in all_species:
            for wf_idx in range(1):
                "Get waveform"
                wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
                    species=species,
                    wf_idx=wf_idx,
                )

                "PARAMETERS"
                # WF pre-processing parameters
                filter = {
                    "type": "kaiser",
                    # "cf": 300,
                    "df": 50,
                    "rip": 100,
                }  # cutoff freq (HPF if one value, BPF if two), transition band width, and max allowed ripple (in dB)
                wf_len_s = 60  # Will crop waveform to this length (in seconds)
                scale = True  # Scale the waveform for dB SPL (shouldn't have an effect outisde of vertical shift on PSD;
                # only actually scales if we know the right scaling constant, which is only Anoles and Humans)

                # Species parameters
                max_khzs = {
                    "Anole": 6,
                    "Tokay": 6,
                    "Human": 10,
                    "V Sim Human": 10,
                    "Owl": 12,
                }
                max_khz = max_khzs[species]

                # Coherence Parameters
                hop_s = 0.01
                xi_s = 0.01
                tau_s = 2**13 / 44100  # Everyone uses the same tau_s

                tau = round(
                    tau_s * fs
                )  # This is just 2**13 for (power of 2 = maximally efficient FFT), except for owls where fs!=44100
                hop = round(hop_s * fs)
                xi = round(xi_s * fs)

                pw = False

                # GET AC
                ac_dict_xi = pc.get_autocoherence(
                    wf,
                    fs,
                    xi,
                    pw,
                    tau,
                    hop=hop,
                    win_meth=win_meth,
                    ref_type="next_seg",
                    return_avg_abs_pd=True,
                    return_dict=True,
                )
                f_xi, avg_abs_pd_xi = ac_dict_xi["f"], ac_dict_xi["avg_abs_pd"]

                ac_dict_omega = pc.get_autocoherence(
                    wf,
                    fs,
                    xi,
                    pw,
                    tau,
                    hop=hop,
                    ref_type="next_freq",
                    return_avg_abs_pd=True,
                    return_dict=True,
                )
                f_omega, avg_abs_pd_omega = ac_dict_omega["f"], ac_dict_omega["avg_abs_pd"]

                # PLOT <|phi|>
                plt.figure(figsize=(8, 12))
                plt.suptitle(f"{species} {wf_idx}")
                plt.subplot(1, 2, 1)
                plt.plot(
                    f_xi / 1000,
                    avg_abs_pd_xi,
                    label=rf"$\left < | \Delta \phi_\xi | \right >$",
                )

                plt.xlim(0, max_khz)
                plt.xlabel("Frequency [kHz]")
                plt.ylim(0, np.pi)
                plt.ylabel(r"$\left < | \Delta \phi_\xi | \right >$")

                plt.subplot(1, 2, 2)
                plt.plot(
                    f_omega / 1000,
                    avg_abs_pd_omega,
                    label=rf"$\left < | \Delta \phi_\omega | \right >$",
                )
                plt.xlim(0, max_khz)
                plt.xlabel("Frequency [kHz]")
                plt.ylim(0, np.pi)
                plt.ylabel(r"$\left < | \Delta \phi_\omega | \right >$")
                plt.show()
