import phaseco as pc
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


bw = 50
rho=1.0
win_type = "flattop"
win_meth = {"method": "rho", "win_type": win_type, "rho": rho}
win_meth_str = pc.get_win_meth_str(win_meth)
for pw in [False]:
    results_folder = rf"paper_analysis/Results/Results (PW={pw}, BW={bw}Hz, {win_meth_str})"
    N_xi_fitted_parameters_fp = rf"{results_folder}/N_xi Fitted Parameters ({win_meth_str}, PW={pw}, BW={bw}Hz).xlsx"
    df = pd.read_excel(N_xi_fitted_parameters_fp)
    As, N_xis, T_xis, freqs, speciess = df["A"], df["N_xi"], df["T"], df["Frequency"], df["Species"]

    markers = {"Anole": ('limegreen', 'o'), "Owl": ('orange', 'd'), "Human": ('blue', 'x'), "Tokay":('green', '+')}
    for A_mult in [False]:
            plt.close('all')
            plt.figure(figsize=(10, 10))
            plt.suptitle(f"PW={pw}")
            for loglog, subplot_idx in zip([True, False], [1, 2]):
                plt.subplot(2, 2, subplot_idx)
                log_str = " (Log-Log)" if loglog else ""
                plt.title(rf"$N_\xi=T_\xi \cdot f$ Comparison{log_str}")
                firsts = {"Anole": 1, "Owl": 1, "Human": 1, "Tokay": 1}
                for k in range(len(df)):
                    species, N_xi, A, freq = speciess[k], N_xis[k], As[k], freqs[k] 
                    N_xi_mod = N_xi * A if A_mult else N_xi / (2*np.pi)
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
                plt.ylabel(rf"$A N_\xi$" if A_mult else rf"$N_\xi/2\pi$")
            # A_mult_str = r"A_mult/" if A_mult else ""
            # folder = f"paper_analysis/Results/N_xi vs Freq/{A_mult_str}"
            # os.makedirs(folder, exist_ok=True)
            # fig_fp = rf"{folder}N_xi vs Freq (PW={pw}, BW={bw}, {win_meth_str}).jpg"
            # plt.savefig(fig_fp, dpi=300)
        
    # Repeat for time constants (not num cycles)
    for A_mult in [False]:
        # plt.close('all')
        # plt.figure(figsize=(10, 5))
        for loglog, subplot_idx in zip([True, False], [1, 2]):
            plt.subplot(2, 2, subplot_idx + 2)
            log_str = " (Log-Log)" if loglog else ""
            plt.title(rf"$T_\xi$ Comparison{log_str}")
            
            firsts = {"Anole": 1, "Owl": 1, "Human": 1, "Tokay": 1}
            for k in range(len(df)):
                species, T_xi, A, freq = speciess[k], T_xis[k], As[k], freqs[k] 
                T_xi_mod = T_xi * A if A_mult else T_xi / (2*np.pi)
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
            plt.ylabel(rf"$A T_\xi$" if A_mult else rf"$T_\xi/2\pi$")
        A_mult_str = r"A_mult/" if A_mult else ""
        folder = os.path.join('paper_analysis', 'Results', f'Results (PW={pw}, BW={bw}Hz, {win_meth_str})', A_mult_str)
        os.makedirs(folder, exist_ok=True)
        fig_fp = rf"{folder}N_xi, T_xi vs Freq (PW={pw}, BW={bw}, {win_meth_str}).jpg"
        plt.savefig(fig_fp, dpi=300)

