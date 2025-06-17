import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rhos = [0.65, 0.7, 0.75]
log_mse = 0
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

ymaxes = {
    'Anole' : 125,
    'Owl' : 200,
    'Human' : 1250,
}

# So far, we're gonna keep doing things with both methods
# for end_decay_at in ['Next Min', 'Noise Floor']:
for end_decay_at in ['Next Min', 'Noise Floor']:
    # Load all the spreadsheets as dataframes
    dfs = {}
    max_MSE = -np.inf
    min_MSE = np.inf
    for rho in rhos:
        filepath = rf'N_xi Fits/Results (rho={rho})/N_xi Fitted Parameters (rho={rho}, End Decay at {end_decay_at}).xlsx'
        df = pd.read_excel(filepath)
        dfs[rho] = df
        mse_column = df['MSE']
        if log_mse:
            mse_column = np.log(mse_column)
        max_mse_rho = mse_column.max()
        min_mse_rho = mse_column.min()
        if max_mse_rho > max_MSE:
            max_MSE = max_mse_rho
        if min_mse_rho < min_MSE:
            min_MSE = min_mse_rho
    
    # Make the plot
    plt.close('all')
    plt.figure(figsize=(15, 9), num=1)
    plt.suptitle(f"Rho Comparison (End Decay at {end_decay_at})")
    subplot_idx = 0
    for species in ['Anole', 'Owl', 'Human']:
        for wf_idx in range(4):
            subplot_idx+=1
            plt.subplot(3, 4, subplot_idx)
            plt.title(f"{species} {wf_idx}")
            nontrivial_peak = True
            for peak_num in range(4):
                color = colors[peak_num]
                N_xis = []
                MSEs = []
                for rho in rhos:
                    if nontrivial_peak:
                        df = dfs[rho]
                        df = df[df['Species']==species]
                        df = df[df['WF Index']==wf_idx]
                    if len(df) < peak_num+1:
                        nontrivial_peak = False
                        continue
                    df_row = df.iloc[peak_num]
                    freq = df_row['Frequency']
                    N_xi = df_row['N_xi']
                    MSE = df_row['MSE']
                    if log_mse:
                        MSE = np.log(MSE)
                    N_xis.append(N_xi)
                    MSEs.append(MSE)
    
                if nontrivial_peak:
                    MSE_scaled = (np.array(MSEs) - min_MSE)/(max_MSE - min_MSE)
                    plt.plot(rhos, N_xis, linestyle='-', label=f'{freq:.0f} Hz', color=color)
                    plt.scatter(rhos, N_xis, color=color, s=100)
                    for idx in range(len(rhos)):
                        cmap = plt.get_cmap(name='gray_r')
                        mse_color = cmap(MSE_scaled[idx])
                        plt.scatter(rhos[idx], N_xis[idx], color=mse_color, s=10, zorder=3)
                        
    
                    
                    
            plt.legend(fontsize=7)
            plt.ylim(0, ymaxes[species])
            plt.xlabel(r'$\rho$')
            plt.ylabel(r'$N_{\xi}$')
    # Close er out
    fig_folder = r'N_xi Fits/Results/Rho Comparison Plots/'
    plot_name = rf'Rho Comparison (End Decay at {end_decay_at})'
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_folder + plot_name + '.png')
    