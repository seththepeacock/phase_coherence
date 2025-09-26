import phaseco as pc
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

win_meth = {"method": "rho", "win_type": "flattop", "rho": 1.0}
win_meth_str = pc.get_win_meth_str(win_meth)
bw = 50
for pw in [True, False]:
    results_folder = rf"paper_analysis/Results/Results (PW={pw}, {win_meth_str})"
    N_xi_fitted_parameters_fp = rf"{results_folder}/N_xi Fitted Parameters ({win_meth_str}, PW={pw}, BW={bw}Hz).xlsx"
    df = pd.read_excel(N_xi_fitted_parameters_fp)
    As, N_xis, T_xis, freqs, speciess = df["A"], df["N_xi"], df["T"], df["Frequency"], df["Species"]

    markers = {"Anole": ('limegreen', 'o'), "Owl": ('orange', 'd'), "Human": ('blue', 'x'), "Tokay":('green', '+')}
    for A_mult in [True, False]:
            plt.close('all')
            plt.figure(figsize=(10, 5))
            for loglog, subplot_idx in zip([True, False], [1, 2]):
                plt.subplot(1, 2, subplot_idx)
                plt.title(rf"$N_\xi$ Comparison [PW={pw}, A_mult={A_mult}]")
                firsts = {"Anole": 1, "Owl": 1, "Human": 1, "Tokay": 1}
                for k in range(len(df)):
                    species, N_xi, A, freq = speciess[k], N_xis[k], As[k], freqs[k] 
                    N_xi_mod = N_xi * A if A_mult else N_xi / np.pi
                    if firsts[species]:
                        label = f"{species}"
                        firsts[species] = 0
                    else:
                        label = None
                    plt.scatter(freq/1000, N_xi_mod, label=label, color = markers[species][0], marker=markers[species][1])
                if loglog:
                    plt.loglog()
                    plt.legend()
                plt.grid(which='both')
                plt.xlabel("Frequency [kHz]")
                plt.ylabel(rf"$A N_\xi$" if A_mult else rf"$N_\xi/\pi$")
            A_mult_str = r"A_mult/" if A_mult else ""
            folder = f"paper_analysis/Results/N_xi vs Freq/{A_mult_str}"
            os.makedirs(folder, exist_ok=True)
            fig_fp = rf"{folder}N_xi vs Freq (PW={pw}, {win_meth_str}).jpg"
            plt.savefig(fig_fp, dpi=300)
        
    # Repeat for time constants (not num cycles)
    for A_mult in [True, False]:
        plt.close('all')
        plt.figure(figsize=(10, 5))
        for loglog, subplot_idx in zip([True, False], [1, 2]):
            plt.subplot(1, 2, subplot_idx)
            plt.title(rf"$T_\xi$ Comparison [PW={pw}, A_mult={A_mult}]")
            firsts = {"Anole": 1, "Owl": 1, "Human": 1, "Tokay": 1}
            for k in range(len(df)):
                species, T_xi, A, freq = speciess[k], T_xis[k], As[k], freqs[k] 
                T_xi_mod = T_xi * A if A_mult else T_xi / np.pi
                if firsts[species]:
                    label = f"{species}"
                    firsts[species] = 0
                else:
                    label = None
                plt.scatter(freq/1000, T_xi_mod, label=label, color = markers[species][0], marker=markers[species][1])
            if loglog:
                plt.loglog()
                plt.legend()
            plt.grid(which='both')
            plt.xlabel("Frequency [kHz]")
            plt.ylabel(rf"$A T_\xi$" if A_mult else rf"$T_\xi/\pi$")
        A_mult_str = r"A_mult/" if A_mult else ""
        folder = f"paper_analysis/Results/N_xi vs Freq/{A_mult_str}"
        os.makedirs(folder, exist_ok=True)
        fig_fp = rf"{folder}T_xi vs Freq (PW={pw}, {win_meth_str}).jpg"
        plt.savefig(fig_fp, dpi=300)

