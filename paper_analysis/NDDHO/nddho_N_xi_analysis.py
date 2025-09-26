import phaseco as pc
from generate_NDDHO import generate_nddho
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import numpy as np

os.chdir(r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence")

# Fixed params
win_meths = [{"method": "rho", "rho": 1.0, "win_type": "flattop"}]
stop_fit = "frac"
stop_fit_frac = 0.1


# Loop Params
A_consts = [True, False]
# NEW RUN
num_iters = 10
# qs = [5, 10, 15, 20, 25, 50, 75, 100]
# f_ds = [100, 500, 1000, 5000] 
qs = [5, 10, 15, 20, 25, 50, 75, 100]
f_ds = [100, 500, 1000]
# Have num_iter=1 for 5000 but doesn't work well
pws = [True, False]

# OG MAIN PAPER RUN
# num_iters = 10
# qs = [5, 10, 15, 20, 25, 50, 75, 100]
# pws = [True, False]
# f0s = [10, 100, 1000]



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
gen_plots = 1
show_plots = 0
fontsize = 8
output_spreadsheet = 0

"Start loop"
for win_meth in win_meths:
    win_meth_str = pc.get_win_meth_str(win_meth)
    for pw in pws:
        for A_const in A_consts:
            # Spreadsheet starts now
            rows = []
            relevant_comp_str = f"PW={pw}, {win_meth_str}, A_const={A_const}"
            for f_d_idx, f_d in enumerate(f_ds):
                plt.figure(figsize=(19.2 * 0.8, 12 * 0.8))
                for q_i, q in enumerate(qs):
                    N_cols = int(round((len(qs) / 2))) if len(qs) > 1 else 1
                    plt.subplot(2, N_cols, q_i + 1)
                    for iter, color in zip(range(num_iters), colors[0:num_iters]):
                        print(
                            f"Q={q} ({q_i+1}/{len(qs)}), f_d={f_d} ({f_d_idx+1}/{len(f_ds)}), Iter {iter+1}/{num_iters}"
                        )

                        "NDDHO Parameters"

                        fs = 44100
                        wf_len_s = 60
                        # Set in for loops above
                        # f0 = 1000
                        # q=1

                        "Coherence Parameters"
                        hop = 0.1
                        nfft = 8192
                        const_N_pd = 0
                        tau = 3285

                        # f0s_cgram = None
                        f0s_cgram = np.array([f_d])


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
                        match f_d:
                            case 10:
                                xi_max_s = xi_max_s * 10
                            case 100:
                                xi_max_s = xi_max_s
                            case 500: 
                                xi_max_s = xi_max_s / 5
                            case 1000:
                                xi_max_s = xi_max_s / 10
                            case 5000:
                                xi_max_s = xi_max_s / 50
                            case 10000:
                                xi_max_s = xi_max_s / 100

                        delta_xi_s = max(xi_max_s / 100, 1/fs)
                        xi_min_s = delta_xi_s

                        xis = {
                            "xi_min_s": xi_min_s,
                            "delta_xi_s": delta_xi_s,
                            "xi_max_s": xi_max_s,
                        }

                        "Filepaths"
                        # Pickle folder
                        old_pkl_folder = "paper_analysis/NDDHO/Pickles (Old)/"
                        new_pkl_folder = "paper_analysis/NDDHO/Pickles/"
                        os.makedirs(old_pkl_folder, exist_ok=True)
                        os.makedirs(new_pkl_folder, exist_ok=True)
                        
                        # NDDHO WF FP
                        wf_id = f"q={q}, f_d={f_d}, len={wf_len_s}, fs={fs}, iter={iter}"
                        wf_fn = f"{wf_id} [NDDHO WF]"
                        old_wf_fp = old_pkl_folder + wf_fn + ".pkl"
                        new_wf_fp = new_pkl_folder + wf_fn + ".pkl"

                        # Colossogram FP
                        const_N_pd_str = "N_pd=const" if const_N_pd else "N_pd=max"
                        f0s_str = (
                            ""
                            if f0s_cgram is None
                            else f"f0s={np.array2string(f0s_cgram, formatter={'float' : lambda x: "%.0f" % x})}, "
                        )
                        nfft_str = "" if nfft is None else f"nfft={nfft}, "
                        delta_xi_str = "" if delta_xi_s == 0.001 else f"delta_xi={delta_xi_s*1000:.2g}ms, "
                        cgram_id = rf"PW={pw}, {win_meth_str}, hop={hop}, tau={tau}, xi_max={xi_max_s*1000:.0f}ms, {delta_xi_str}{nfft_str}{f0s_str}{const_N_pd_str}"
                        cgram_fn = f"{cgram_id}, {wf_id} [COLOSSOGRAM]"
                        old_cgram_fp = old_pkl_folder + cgram_fn + ".pkl"
                        new_cgram_fp = new_pkl_folder + cgram_fn + ".pkl"

                        # Load/calc waveform
                        if os.path.exists(old_wf_fp):
                            print("Already got this wf, loading!")
                            with open(old_wf_fp, "rb") as file:
                                wf_x = pickle.load(file)
                            # Dump to new folder (if we haven't already)
                            if not os.path.exists(new_wf_fp):
                                print("Dumping into new folder!")
                                with open(new_wf_fp, "wb") as file:
                                    pickle.dump(wf_x, file)
                        else:
                            print(f"Generating NDDHO {wf_fn}")
                            wf_x, wf_y = generate_nddho(q, f_d, fs, wf_len_s)
                            with open(old_wf_fp, "wb") as file:
                                pickle.dump(wf_x, file)

                        # Load/calc colossogram
                       

                        if os.path.exists(old_cgram_fp):
                            print("Already got this colossogram, loading!")
                            with open(old_cgram_fp, "rb") as file:
                                cgram_dict = pickle.load(file)
                            # Dump to new folder (if we haven't already)
                            if not os.path.exists(new_cgram_fp):
                                print("Dumping into new folder")
                                with open(new_cgram_fp, "wb") as file:
                                    pickle.dump(cgram_dict, file)
                        else:
                            print(f"Calculating Colossogram ({cgram_id})")
                            cgram_dict = pc.get_colossogram(
                                wf_x,
                                fs,
                                xis=xis,
                                hop=hop,
                                tau=tau,
                                win_meth=win_meth,
                                pw=pw,
                                nfft=nfft,
                                const_N_pd=const_N_pd,
                                f0s=f0s_cgram,
                                return_dict=True,
                            )
                            with open(old_cgram_fp, "wb") as file:
                                pickle.dump(cgram_dict, file)
                        
                        xis_s = cgram_dict["xis_s"]
                        f = cgram_dict["f"]
                        colossogram = cgram_dict["colossogram"]
                        # plt.close('all')
                        # pc.plot_colossogram(xis_s, f, colossogram, pw=pw)
                        # plt.show()
                        N_xi, N_xi_dict = pc.get_N_xi(
                            cgram_dict,
                            f_d,
                            A_const=A_const,
                            stop_fit=stop_fit,
                            stop_fit_frac=stop_fit_frac,
                            pw=pw,
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

                        if output_spreadsheet:
                            rows.append(
                                {
                                    "Q": q,
                                    "Undamped CF": f_d,
                                    "N_xi": N_xi,
                                    "N_xi_std": N_xi_std,
                                    "T": T,
                                    "T_std": T_std,
                                    "A": A,
                                    "A_std": A_std,
                                    "MSE": mse,
                                    "Iter": iter,
                                    "DFT Freq Bin": f0_bin_center,
                                    "Damped CF": f_d,
                                    "NDDHO Params": wf_fn,
                                }
                            )

                        if gen_plots:
                            pc.plot_N_xi_fit(
                                N_xi_dict, color=color, plot_noise_floor=False
                            )
                            plt.legend(fontsize=fontsize / 2)
                            plt.title(f"Q={q}", fontsize=fontsize)

                # Outside of q/iter loop now (but inside f0 loop)
                if gen_plots:
                    plt.suptitle(
                        rf"f0={f_d}Hz, PW={pw}, A_const={A_const}, {win_meth_str}, hop={hop}$\tau$, tau={(tau/fs)*1000:.0f}ms, wf_len={wf_len_s}s, fs={fs}Hz",
                        fontsize=fontsize,
                    )
                    plt.tight_layout()
                    plots_folder = f"paper_analysis/NDDHO/Results [{win_meth_str}]/Fits Varying Q [A_const={A_const}]/"
                    os.makedirs(plots_folder, exist_ok=True)
                    plot_fp = f"{plots_folder}/f0={f_d}Hz, {relevant_comp_str}, [FITS VARYING Q].jpg"
                    plt.savefig(plot_fp, dpi=300)
                if show_plots:
                    plt.show()
            # Outside of f0 loop now
            if output_spreadsheet:
                # Save parameter data as xlsx
                df_fitted_params = pd.DataFrame(rows)
                spreadsheet_fn = rf"NDDHO/Results [PW={pw}, {win_meth_str}]/NDDHO N_xi Data ({relevant_comp_str})"
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
