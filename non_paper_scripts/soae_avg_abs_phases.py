import phaseco as pc
from helper_funcs import *

all_species = ["Anole", "Human", "Owl", "Tokay"]
speciess = ["Anole"]
wf_idxs = [2]


for species in speciess:
    for wf_idx in wf_idxs:
        species = "Anole"
        wf_idx = 2
        "Get waveform"
        wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
            species=species,
            wf_idx=wf_idx,
        )

        "PARAMETERS"
        # WF pre-prcessing parameters
        filter = {
            "type": "kaiser",
            "cf": 300,
            "df": 50,
            "rip": 100,
        }  # cutoff freq (HPF if one value, BPF if two), transition band width, and max allowed ripple (in dB)
        wf_len_s = 60  # Will crop waveform to this length (in seconds)
        wf = crop_wf(wf, fs, wf_len_s)
        wf = filter_wf(wf, fs, filter, species)

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
        # win_meth = {"method":"static", "win_type":"boxcar"}
        win_meth = {"method":"rho", "rho":0.7}
        
        # hop_s = 0.01
        # xi_s = 0.01
        # tau_s = 2**13 / 44100  # Everyone uses the same tau_s
        tau_s = 0.02322
        xi_s = 0.00290
        hop_s = xi_s

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
            phase_corr=False,
            return_dict=True,
        )
        f_xi, avg_abs_pd_xi = ac_dict_xi["f"], ac_dict_xi["avg_abs_pd"]

        ac_dict_xi_phase_corr = pc.get_autocoherence(
            wf,
            fs,
            xi,
            pw,
            tau,
            hop=hop,
            win_meth=win_meth,
            ref_type="next_seg",
            return_avg_abs_pd=True,
            phase_corr=True,
            return_dict=True,
        )
        f_xi_phase_corr, avg_abs_pd_xi_phase_corr = ac_dict_xi_phase_corr["f"], ac_dict_xi_phase_corr["avg_abs_pd"]

        ac_dict_tau = pc.get_autocoherence(
            wf,
            fs,
            xi=tau,
            pw=pw,
            tau=tau,
            hop=hop,
            win_meth=win_meth,
            ref_type="next_seg",
            return_avg_abs_pd=True,
            phase_corr=False,
            return_dict=True,
        )
        f_tau, avg_abs_pd_tau = ac_dict_tau["f"], ac_dict_tau["avg_abs_pd"]

        ac_dict_tau_phase_corr = pc.get_autocoherence(
            wf,
            fs,
            xi=tau,
            pw=pw,
            tau=tau,
            hop=hop,
            win_meth=win_meth,
            ref_type="next_seg",
            return_avg_abs_pd=True,
            phase_corr=True,
            return_dict=True,
        )
        f_tau_phase_corr, avg_abs_pd_tau_phase_corr = ac_dict_tau_phase_corr["f"], ac_dict_tau_phase_corr["avg_abs_pd"]

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

        ac_dict_omega_phase_corr = pc.get_autocoherence(
            wf,
            fs,
            xi,
            pw,
            tau,
            hop=hop,
            ref_type="next_freq",
            return_avg_abs_pd=True,
            phase_corr=True,
            return_dict=True,
        )
        f_omega_phase_corr, avg_abs_pd_omega_phase_corr = ac_dict_omega_phase_corr["f"], ac_dict_omega_phase_corr["avg_abs_pd"]

        # PLOT <|phi|>
        plt.figure(figsize=(6, 8))
        plt.suptitle(rf"{species} {wf_idx}   [{wf_fn}]   [$\tau$={tau_s:.4g}]   [$\xi$={xi_s:.4g}]")
        plt.subplot(3, 1, 1)
        plt.title(r"$\xi$-Referenced Phases")
        plt.scatter(
            f_xi / 1000,
            avg_abs_pd_xi,
            label=rf"Uncorrected",
        )
        plt.scatter(
            f_xi / 1000,
            avg_abs_pd_xi_phase_corr,
            label=rf"Phase Corrected",
        )
        plt.legend()

        plt.xlim(0.5, 5)
        plt.xlabel("Frequency [kHz]")
        plt.ylim(0, np.pi)
        plt.ylabel(r"$\left < | \Delta \phi_\xi | \right >$")

        plt.subplot(3, 1, 2)
        plt.title(r"$\omega$-Referenced Phases")
        plt.scatter(
            f_omega / 1000,
            avg_abs_pd_omega,
            label=rf"Uncorrected",
        )
        plt.scatter(
            f_omega / 1000,
            avg_abs_pd_omega_phase_corr,
            label=rf"Phase Corrected",
        )
        plt.xlim(0.5, 5)
        plt.xlabel("Frequency [kHz]")
        plt.ylim(0, np.pi)
        plt.ylabel(r"$\left < | \Delta \phi_\omega | \right >$")

        plt.subplot(3, 1, 3)
        plt.title(r"$\tau$-Referenced Phases")
        plt.scatter(
            f_tau / 1000,
            avg_abs_pd_tau,
            label=rf"Uncorrected",
        )
        plt.scatter(
            f_tau_phase_corr / 1000,
            avg_abs_pd_tau_phase_corr,
            label=rf"Phase Corrected",
        )
        plt.legend()

        plt.xlim(0.5, 5)
        plt.xlabel("Frequency [kHz]")
        plt.ylim(0, np.pi)
        plt.ylabel(r"$\left < | \Delta \phi_\tau | \right >$")

plt.tight_layout()
plt.savefig("avg_abs_pd_plot.png")
plt.show()
