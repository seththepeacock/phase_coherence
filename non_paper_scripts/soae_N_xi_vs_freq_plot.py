import phaseco as pc
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


bw = 200
rho=None
win_type = "flattop"
win_meth = {"method": "rho", "win_type": win_type, "rho": rho} if rho is not None else {"method": "static", "win_type": win_type}
win_meth_str = pc.get_win_meth_str(win_meth)
T_xi_label = rf"$T_\xi$"
N_xi_label = rf"$N_\xi$"
for pw in [False]:
    results_folder = rf"paper_analysis/Results/Results (PW={pw}, BW={bw}Hz, {win_meth_str})"
    N_xi_fitted_parameters_fp = rf"{results_folder}/N_xi Fitted Parameters ({win_meth_str}, PW={pw}, BW={bw}Hz).xlsx"
    df = pd.read_excel(N_xi_fitted_parameters_fp)
    As, N_xis, T_xis, freqs, speciess, fwhms, snrs = df["A"], df["N_xi"], df["T"], df["Frequency"], df["Species"], df["FWHM"], df["SNRfit"]

    markers = {"Anole": ('limegreen', 'o'), "Owl": ('orange', 'd'), "Human": ('blue', 'x'), "Tokay":('green', '+')}

        
    # Plot N_xi, T_xi against Frequency on loglog plot
    xaxis = 'freq'
    xaxis = 'FWHM'
    # xaxis = 'SNR'
    if xaxis == 'freq':
        xlabel = "Frequency [kHz]"
        xs = freqs / 1000
    elif xaxis == 'FWHM':
        xlabel = 'FWHM'
        xs = fwhms
    if 1:
        # plt.close('all')
        # plt.figure(figsize=(10, 5))
        plt.close('all')
        plt.figure(figsize=(10, 10))
        plt.suptitle(f"PW={pw}")
        for loglog, subplot_idx in zip([True, False], [1, 2]):
            plt.subplot(2, 2, subplot_idx)
            log_str = " (Log-Log)" if loglog else ""
            plt.title(rf"$N_\xi=T_\xi \cdot f$ Comparison{log_str}")
            firsts = {"Anole": 1, "Owl": 1, "Human": 1, "Tokay": 1}
            for k in range(len(df)):
                species, N_xi, A, x, = speciess[k], N_xis[k], As[k], xs[k] 
                N_xi_mod = N_xi
                if firsts[species]:
                    label = f"{species}"
                    firsts[species] = 0
                else:
                    label = None
                plt.scatter(x, N_xi_mod, label=label, color = markers[species][0], marker=markers[species][1])
            if loglog:
                plt.loglog()
                plt.legend()
            # else:
            #     plt.ylim(0, 0.2)
            
            plt.grid(which='both')
            plt.xlabel(xlabel)
            plt.ylabel(N_xi_label)
            # Now do T_xi
            plt.subplot(2, 2, subplot_idx + 2)
            log_str = " (Log-Log)" if loglog else ""
            plt.title(rf"$T_\xi$ Comparison{log_str}")
            
            firsts = {"Anole": 1, "Owl": 1, "Human": 1, "Tokay": 1}
            for k in range(len(df)):
                species, T_xi, A, x = speciess[k], T_xis[k], As[k], xs[k] 
                T_xi_mod = T_xi
                if firsts[species]:
                    label = f"{species}"
                    firsts[species] = 0
                else:
                    label = None
                plt.scatter(x, T_xi_mod, label=label, color = markers[species][0], marker=markers[species][1])
            if loglog:
                plt.loglog()
                plt.legend()
            else:
                plt.ylim(0, 0.02)
                plt.legend()
            
            plt.grid(which='both')
            plt.xlabel(xlabel)
            plt.ylabel(T_xi_label)
        folder = os.path.join('paper_analysis', f'N_xi, T_xi vs X')
        os.makedirs(folder, exist_ok=True)
        fig_fp = rf"{folder}N_xi, T_xi vs {xaxis.upper()} (PW={pw}, BW={bw}, {win_meth_str}).jpg"
        plt.tight_layout()
        plt.savefig(fig_fp, dpi=300)
