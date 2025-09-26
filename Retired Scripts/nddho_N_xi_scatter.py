import phaseco as pc
import pandas as pd
import matplotlib.pyplot as plt
import os

# Coherence / Fitting Params
win_meths = [{"method": "rho", "rho": 0.7}]
pws = [True]
A_const = True
f0s = [10, 100, 1000]

# Plotting Params
s=25
fontsize = 10
marker='*'
plot_mse = False
nrows = 2 if plot_mse else 1
ncols = 3
fig_size = (ncols*5, nrows*5)

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


for win_meth in win_meths:
    for pw in pws:
        win_meth_str = pc.get_win_meth_str(win_meth)

        plt.figure(figsize=fig_size)
        for (f0_i, f0), color in zip(enumerate(f0s), colors[0 : len(f0s)]):
            relevant_comp_str = f'PW={pw}, {win_meth_str}, A_const={A_const}'
            spreadsheet_fn = rf"NDDHO/Results [PW={pw}]/NDDHO N_xi Data ({relevant_comp_str}).xlsx"
            df = pd.read_excel(spreadsheet_fn)
            num_iters = df["Iter"].max() + 1
            # f0s = df["Undamped CF"].unique()
            Qs = df["Q"].unique()
            sigmas = df["Sigma"].unique()
            if len(sigmas) > 1:
                raise ValueError(f"Should only have one sigma in this comparison! Instead, have {sigmas}")
            sigma = sigmas[0]
            for iter in range(num_iters):
                for q_i, q in enumerate(Qs):
                    row = df.loc[
                        (df["Undamped CF"] == f0) & (df["Q"] == q) & (df["Iter"] == iter)
                    ]

                    N_xi = row["N_xi"].iloc[0]
                    mse = row["MSE"].iloc[0]
                    # Only add one label per f0
                    label = f"UCF={f0}Hz" if q_i == 0 and iter == 0 else None

                    # N_xi vs Q Plot
                    plt.subplot(nrows, ncols, f0_i + 1)
                    plt.scatter(q, N_xi, label=label, c=color, marker=marker, s=s)
                    plt.legend()
                    plt.xlabel("Q", fontsize=fontsize)
                    plt.ylabel(r"$N_\xi$", fontsize=fontsize)
                    plt.ylim(0, 35)
                    if plot_mse:
                        # MSE vs Q Plot
                        plt.subplot(nrows, ncols, f0_i + 1 + 2)
                        plt.scatter(q, mse, label=label, c=color, marker=marker, s=s)
                        plt.legend()
        
        
        if plot_mse:
            plt.subplot(nrows, ncols, 3)
            plt.title("MSE vs Q", fontsize=fontsize)
            plt.xlabel("Q", fontsize=fontsize)
            plt.ylabel(r"MSE", fontsize=fontsize)
            plt.legend()
            plt.subplot(nrows, ncols, 4)
            plt.title("MSE vs Q", fontsize=fontsize)
            plt.xlabel("Q", fontsize=fontsize)
            plt.ylabel(r"MSE", fontsize=fontsize)
            plt.legend()

        # Adjust subplots
        sp_space = 0.25
        plt.subplots_adjust(wspace=sp_space, hspace=sp_space)

        # Save Figure
        os.makedirs(f'NDDHO/Results [PW={pw}]', exist_ok=True)
        scatter_fp = f'NDDHO/Results [PW={pw}]/N_xi vs Q [f0 Separated] [PW={pw}, {win_meth_str}].jpg'
        # plt.tight_layout()
        plt.suptitle(rf'$N_\xi$ vs Q [PW={pw}, {win_meth_str}, A=1]')
        plt.savefig(scatter_fp, dpi=300)
            