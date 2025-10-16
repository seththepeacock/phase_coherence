import phaseco as pc
import pandas as pd
import matplotlib.pyplot as plt
import os

# NDDHO params
f_ds = [1000, 2000, 3000]


# Coherence params
pws = [False]
rho_bw_hops = [(None, 'gamma', ('int', 1)), (1.0, 50, ('s', 0.01))]
win_type = 'flattop'

# Plotting Params
s=25
fontsize = 10
marker='*'
plot_mse = False
nrows = 2 if plot_mse else 1
ncols = 3
fig_size = (ncols*5, nrows*5)
show_plot = 1

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

A_consts=[False]

for A_const in A_consts:
    for rho, bw_type, hop_thing in rho_bw_hops:
        for pw in pws:
            plt.close('all')
            plt.figure(figsize=fig_size)
            for (f_d_i, f_d), color in zip(enumerate(f_ds), colors[0 : len(f_ds)]):
                # Deal with windowing method
                if rho is None:
                    win_meth = {'method':'static', 'win_type':win_type}
                else:
                    win_meth = {'method':'rho', 'rho':rho, 'win_type':win_type}
                
                win_meth_str = pc.get_win_meth_str(win_meth)
                bw_str = "BW=0.5gamma" if bw_type == 'gamma' else f"BW={bw_type}Hz"
                relevant_comp_str = f"PW={pw}, {win_meth_str}, {bw_str}, A_const={A_const}"

                results_folder = os.path.join('paper_analysis', 'results', 'nddho', f'NDDHO Results [{relevant_comp_str}]')
                os.makedirs(results_folder, exist_ok=True)
                spreadsheet_fp = os.path.join(results_folder, f"NDDHO Q N_xi Data [{relevant_comp_str}].xlsx")
                # print("ANALYSIS FROM ", spreadsheet_fp)
                df = pd.read_excel(spreadsheet_fp)
                num_iters = df["Iter"].max() + 1
                qs = df["Q"].unique()

                for iter in range(num_iters):
                    for q_k, q in enumerate(qs):
                        # Deal with bandwidth

                        row = df.loc[
                            (df["CF"] == f_d) & (df["Q"] == q) & (df["Iter"] == iter)
                        ]

                        N_xi = row["N_xi"].iloc[0]
                        T_xi = row["T_xi"].iloc[0]
                        mse = row["MSE"].iloc[0]
                        # Only add one label per f0
                        label = f"UCF={f_d}Hz" if q_k == 0 and iter == 0 else None

                        # N_xi vs gamma Plot
                        # plt.subplot(nrows, ncols, f0_i + 1)
                        plt.scatter(q, N_xi, label=label, c=color, marker=marker, s=s)
                        plt.legend()
                        plt.xlabel(r"Q", fontsize=fontsize)
                        plt.ylabel(r"$N_\xi$", fontsize=fontsize)
                        # plt.ylim(0, 35)
                        if plot_mse:
                            # MSE vs gamma Plot
                            plt.subplot(nrows, ncols, f_d_i + 1 + 2)
                            plt.scatter(q, mse, label=label, c=color, marker=marker, s=s)
                            plt.legend()
            
            
            if plot_mse:
                plt.subplot(nrows, ncols, 3)
                plt.title(r"MSE vs $Q$", fontsize=fontsize)
                plt.xlabel(r"$Q$", fontsize=fontsize)
                plt.ylabel(r"MSE", fontsize=fontsize)
                plt.legend()
                plt.subplot(nrows, ncols, 4)
                plt.title(r"MSE vs $Q$", fontsize=fontsize)
                plt.xlabel(r"$Q$", fontsize=fontsize)
                plt.ylabel(r"MSE", fontsize=fontsize)
                plt.legend()

            # Adjust subplots
            sp_space = 0.25
            plt.subplots_adjust(wspace=sp_space, hspace=sp_space)

            # Save Figure
            scatter_fp = os.path.join(results_folder, f'N_xi vs Q [{relevant_comp_str}].jpg')
            # plt.tight_layout()
            plt.suptitle(rf'$N_\xi$ vs $Q$ [{relevant_comp_str}]')
            if show_plot:
                plt.show()
            plt.savefig(scatter_fp, dpi=300)
                