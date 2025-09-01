import phaseco as pc
from generate_NDDHO import generate_nddho
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import numpy as np



# Fixed params
rho = 0.7
win_meths = [{"method": "rho", "rho": rho}]
stop_fit = 'frac'
stop_fit_frac = 0.1



# Loop Params
A_consts = [True, False]
num_iters = 10
qs = [5, 10, 15, 20, 25, 50, 75, 100]
pws = [True]
f0s = [10, 100, 1000]

# num_iters = 2
# f0s = [100, 1000]
# pws = [True, False]
# qs = [5, 10, 25, 50, 75, 100]



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
"Plotting Parameters"
gen_plots = 0
show_plots = 0
fontsize = 8
output_spreadsheet = 1

"Start loop"
for win_meth in win_meths:
    win_meth_str = pc.get_win_meth_str(win_meth)
    for pw in pws:
        for A_const in A_consts:
            # Spreadsheet starts now
            rows = []
            relevant_comp_str = f"PW={pw}, {win_meth_str}, A_const={A_const}"
            for f0_i, f0 in enumerate(f0s):
                plt.figure(figsize=(19.2 * 0.8, 12 * 0.8))
                for q_i, q in enumerate(qs):
                    plt.subplot(2, int(len(qs) / 2), q_i + 1)
                    for iter, color in zip(range(num_iters), colors[0:num_iters]):
                        print(
                            f"Q={q} ({q_i+1}/{len(qs)}), f0={f0} ({f0_i+1}/{len(f0s)}), Iter {iter+1}/{num_iters}"
                        )

                        "NDDHO Parameters"

                        fs = 44100
                        wf_len = 60
                        sigma = 1
                        # Set in for loops above
                        # f0 = 1000
                        # q=1

                        "Coherence Parameters"
                        tau = 2**13
                        hop = round(0.01 * fs)

                        # These ones are set above
                        # pw = True
                        # win_meth = {'method':'rho', 'rho':0.7}

                        # xis
                        if q <= 10:
                            xi_max_s = 0.1
                        elif q <= 50:
                            xi_max_s = 0.5
                        elif q <= 100:
                            xi_max_s = 1
                        elif q <= 500:
                            xi_max_s = 5
                        elif q <= 1000:
                            xi_max_s = 10
                        else:
                            raise ValueError("Haven't picked a xi_max for q>100!")

                        # Adjust for f0=10 and f0=1000
                        match f0:
                            case 10:
                                xi_max_s = xi_max_s * 10
                            case 100:
                                xi_max_s = xi_max_s
                            case 1000:
                                xi_max_s = xi_max_s / 10

                        delta_xi_s = xi_max_s / 100
                        xi_min_s = delta_xi_s

                        xis = {
                            "xi_min_s": xi_min_s,
                            "delta_xi_s": delta_xi_s,
                            "xi_max_s": xi_max_s,
                        }

                        "Filepaths"
                        # Pickle folder
                        pkl_folder = "NDDHO/Pickles/"
                        os.makedirs(pkl_folder, exist_ok=True)
                        # NDDHO WF FP
                        wf_id_sans_q = f"f0={f0}, sigma={sigma}, len={wf_len}, fs={fs}, iter={iter}"
                        wf_id = f"q={q}, {wf_id_sans_q}"
                        wf_fn = f"{wf_id} [NDDHO WF]"
                        wf_fp = pkl_folder + wf_fn + ".pkl"

                        # Colossogram FP
                        cgram_id = f"PW={pw}, {win_meth_str}, xi_max={xi_max_s*1000:.0f}ms, hop={(hop/fs)*1000:.0f}ms, tau={(tau/fs)*1000:.0f}ms"
                        cgram_fn = f"{cgram_id}, {wf_id} [COLOSSOGRAM]"
                        cgram_fp = pkl_folder + cgram_fn + ".pkl"

                        # Load/calc waveform
                        if os.path.exists(wf_fp):
                            print("Already got this wf, loading!")
                            with open(wf_fp, "rb") as file:
                                wf_x = pickle.load(file)
                        else:
                            print(f"Generating NDDHO {wf_fn}")
                            wf_x, wf_y = generate_nddho(q, f0, sigma, fs, wf_len)
                            with open(wf_fp, "wb") as file:
                                pickle.dump(wf_x, file)

                        # Load/calc colossogram
                        if os.path.exists(cgram_fp):
                            print("Already got this colossogram, loading!")
                            with open(cgram_fp, "rb") as file:
                                cgram_dict = pickle.load(file)
                        else:
                            print(f"Calculating Colossogram ({cgram_id})")
                            cgram_dict = pc.get_colossogram(
                                wf_x,
                                fs,
                                xis,
                                pw=pw,
                                win_meth=win_meth,
                                tau=tau,
                                hop=hop,
                                return_dict=True,
                            )
                            with open(cgram_fp, "wb") as file:
                                pickle.dump(cgram_dict, file)

                        xis_s = cgram_dict["xis_s"]
                        f = cgram_dict["f"]
                        colossogram = cgram_dict["colossogram"]
                        # Calculate f0_d
                        omega_0 = f0 * (2 * np.pi)
                        gamma = omega_0 / q
                        omega_d = np.sqrt(omega_0**2 - gamma**2)
                        f0_d = omega_d / (2 * np.pi)

                        N_xi, N_xi_dict = pc.get_N_xi(
                            xis_s, f, colossogram, f0_d, A_const=A_const, stop_fit=stop_fit, stop_fit_frac=stop_fit_frac
                        )

                        # Unpack dictionary
                        f0_bin_center = N_xi_dict["f0_exact"]
                        N_xi = N_xi_dict["N_xi"]
                        N_xi_std = N_xi_dict["N_xi_std"]
                        T = N_xi_dict["T"]
                        T_std = N_xi_dict["T_std"]
                        A = N_xi_dict["A"]
                        A_std = N_xi_dict["A_std"]
                        mse = N_xi_dict["mse"]
                        decayed_idx = N_xi_dict["decayed_idx"]

                        # cgram_dicts[cgram_id] = cgram_dict
                        # N_xi_dicts[cgram_id] = N_xi_dict

                        if output_spreadsheet:
                            rows.append(
                                {
                                    "Q": q,
                                    "Undamped CF": f0,
                                    "N_xi": N_xi,
                                    "N_xi_std": N_xi_std,
                                    "T": T,
                                    "T_std": T_std,
                                    "A": A,
                                    "A_std": A_std,
                                    "MSE": mse,
                                    "Iter": iter,
                                    "Sigma": sigma,
                                    "DFT Freq Bin": f0_bin_center,
                                    "Damped CF": f0_d,
                                    "NDDHO Params": wf_fn,
                                }
                            )

                        if gen_plots:
                            pc.plot_N_xi_fit(
                                N_xi_dict, color=color, plot_noise_floor=False
                            )
                            plt.legend(fontsize=fontsize)
                            plt.title(f"Q={q}", fontsize=fontsize)

                # Outside of q/iter loop now (but inside f0 loop)
                if gen_plots:
                    plt.suptitle(
                        f"f0={f0}Hz, PW={pw}, A_const={A_const}, noise_sigma={sigma}, {win_meth_str}, hop={(hop/fs)*1000:.0f}ms, tau={(tau/fs)*1000:.0f}ms, wf_len={wf_len}s, fs={fs}Hz",
                        fontsize=fontsize,
                    )
                    plt.tight_layout()
                    plots_folder = f"NDDHO/Results [PW={pw}]/Fits Varying Q [PW={pw}]/"
                    os.makedirs(plots_folder, exist_ok=True)
                    plot_fp = f"{plots_folder}/f0={f0}Hz, {relevant_comp_str}, noise_sigma={sigma} [FITS VARYING Q].jpg"
                    plt.savefig(plot_fp, dpi=300)
                if show_plots:
                    plt.show()
            # Outside of f0 loop now
            if output_spreadsheet:
                # Save parameter data as xlsx
                df_fitted_params = pd.DataFrame(rows)
                spreadsheet_fn = rf"NDDHO/Results [PW={pw}]/NDDHO N_xi Data ({relevant_comp_str})"
                df_fitted_params.to_excel(rf"{spreadsheet_fn}.xlsx", index=False)


# gamma_inv_ish = q / f0
# if gamma_inv_ish <= 0.01:         # q <= 10
#     xi_max_s = 0.01
# elif gamma_inv_ish <= 0.1:        # 10 < q <= 100
#     xi_max_s = 0.1
# elif gamma_inv_ish <= 1.0:        # 100 < q <= 1000
#     xi_max_s = 1.0
# elif gamma_inv_ish <= 10.0:       # 1000 < q <= 10000
#     xi_max_s = 10.0
