import importlib
from SOAEpeaks import load_df
import phaseco as pc
from phaseco import *
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal.windows import gaussian
from scipy.optimize import curve_fit
importlib.reload(pc)

def get_is_signal(coherences, f, xis, f_target_idx, f_noise=12000, sample_hw=5, z_alpha=0.05, crop=False):
    f_noise_idx = (np.abs(f - f_noise)).argmin()  # find frequency bin index closest to our 12kHz cutoff
    # Get mean and std dev of coherence (over frequency axis, axis=0) for each xi value (using only "noise" frequencies above our cutoff)
    noise_means = np.mean(coherences[f_noise_idx:, :], axis=0) 
    noise_stds = np.std(coherences[f_noise_idx:, :], axis=0, ddof=1) # ddof=1 since we're using sample mean (not true mean) in sample std estimate
    # Now for each xi value, run a z-test to see if it's noise or not
    is_signal = np.full(len(xis), True, dtype=bool)
    for xi_idx in range(len(xis)):
        # Skip xi values that are too close to the edges to get a full sample
        if xi_idx < sample_hw:
            is_signal[xi_idx] = True
            continue
        elif xi_idx >= len(xis) - sample_hw:
            is_signal[xi_idx] = False
        coherence_sample = coherences[f_target_idx, xi_idx - sample_hw: xi_idx + sample_hw]
        # Calculate z test statistic
        sample_mean = np.mean(coherence_sample)
        z = (sample_mean - noise_means[xi_idx]) / (noise_stds[xi_idx]/np.sqrt(len(coherence_sample)))
        # Calculate p-value for a one-tailed test (sf = survival function = 1 - cdf)
        p = sp.stats.norm.sf(z)
        is_signal[xi_idx] = p < z_alpha
    target_coherences=coherences[f_target_idx, :]
    if crop:
        is_signal = is_signal[sample_hw:-sample_hw]
        xis = xis[sample_hw:-sample_hw]
        target_coherences = target_coherences[f_target_idx, sample_hw:-sample_hw]
    
    decayed_idx = -1
    # Find decayed point
    for i in range(len(is_signal)):
        if not crop and i < sample_hw: # These are automatically set to signal anyway but what the heck
            continue
        if not is_signal[i]:
            decayed_idx = i
            break
    if decayed_idx == -1:
        print(f"Signal at {f[f_target_idx]:.0f}Hz never decays!")
    xi_decayed = xis[decayed_idx]
    
    return is_signal, xis, target_coherences, xi_decayed, decayed_idx
def exp_decay(x, a, timeconst):
    return a * np.exp(-x/timeconst)



# Get different species
df = load_df(laptop=True, dfs_to_load=["Curated Data"])
wf_list = []
for species in ['Anolis', 'Owl', 'Human']:
    df_species = df[df['species'] == species]
    for i in range (3):
        row = df_species.iloc[i]
        wf_fn = row['filepath'].split('\\')[-1]
        wf_list.append((row['wf'], row['sr'], wf_fn, species))
del df

wf_idx = 7 # Started with 0, 2 (Lizard) - 4, 5 (Owl) - 6, 7 (Human)
for wf_idx in tqdm([0, 2, 4, 5, 6, 7]):
    wf, fs, wf_fn, species = wf_list[wf_idx]

    # Set parameters (same for human and lizard)
    tau = 2**12 / 44100 # Everyone uses the same tau
    tauS = int(tau*fs)
    # Raise warning if tauS is not a power of two AND the samplerate is indeed 44100
    if np.log2(tauS) != int(np.log2(tauS)):
        if fs == 44100:
            raise ValueError("tauS is not a power of 2, but the samplerate is 44100!")
        else:
            print(f"WARNING: tauS is not a power of 2, but the samplerate is {fs} (not 44100), so we'll assume you're just ensuring tau aligns with other waveforms!")

    rho = 0.7

    if species == 'Human':
        # Human parameters
        min_xi = 0.001
        max_xi = 1
        delta_xi = 0.001
        if wf_fn == 'human_TH14RearwaveformSOAE.mat':
            max_xi = 2.5
        max_khz = 6
    elif species in ['Lizard', 'Anolis']:
        # Lizard parameters
        min_xi = 0.001
        max_xi = 0.1
        delta_xi = 0.0005
        max_khz = 6
    elif species == 'Owl':
        # Owl parameters
        min_xi = 0.001
        max_xi = 0.1
        delta_xi = 0.0005
        max_khz = 12
        
    if 0:
        operation = 'generate+save'
    else:
        operation = 'open'
        
        
    # Save/open coherences
    suptitle=rf"[{wf_fn}]   [$\rho$={rho}]   [$\tau$={tau*1000:.2f}ms]"
    
    fn_id = rf"tau={tau*1000:.0f}, rho={rho}, {species}, {wf_fn.split('.')[0]}"
    pkl_fn = f'C_xi Decay Coherences - {fn_id}'
    pkl_folder = r'Pickles/'
    fig_folder = r'Colossogram Figures/C_xi Decay Figures/'

    if operation == 'generate+save':
        with open(pkl_folder + pkl_fn + '.pkl', 'wb') as file:
            pickle.dump((coherences, f, xis, tau, rho, wf_fn, species), file)
    elif operation == 'open':
        with open(pkl_folder + pkl_fn + '.pkl', 'rb') as file:
            coherences, f, xis, tau, rho, wf_fn, species = pickle.load(file)
    
    # Peak pick the target bins
    if wf_fn == 'anole_AC6rearSOAEwfB1.mat': # 0
        peak_freqs = [1225, 2150, 4300]
        noise_freqs = [400, 12000]
    elif wf_fn == 'anole_ACsb18learSOAEwfG4.mat': # 2
        peak_freqs = [990, 2000, 3670]
        noise_freqs = [400, 12000]
    elif wf_fn == 'owl_Owl6L1.mat': # 4
        peak_freqs = [4867, 6384, 7235]
        noise_freqs = [400, 12000]
    elif wf_fn == 'owl_TAG4learSOAEwf1.mat': # 5
        peak_freqs = [6280, 7820, 10487]
        noise_freqs = [400, 12000]
    elif wf_fn == 'human_TH14RearwaveformSOAE.mat': # 6
        peak_freqs = [603, 2250, 4370]
        noise_freqs = [400, 12000]
    elif wf_fn == 'human_TH21RearwaveformSOAE.mat': # 7
        peak_freqs = [2000, 2605, 4135]
        noise_freqs = [400, 12000]
    else:
        raise(Exception("Haven't peak picked this waveform yet!"))
        

    bin_idxs = []
    bin_names = []

    for peak_freq in peak_freqs:
        bin_idxs.append(np.argmin(np.abs(f - peak_freq)))
        bin_names.append(f"{peak_freq:.0f}Hz Peak")

    for noise_freq in noise_freqs:
        bin_idxs.append(np.argmin(np.abs(f - noise_freq)))
        bin_names.append(f"{noise_freq:.0f}Hz Noise")
        
    freq_list = peak_freqs + noise_freqs
    
    
    # Compare xi scale vs # cycles scale

    # Z-Test Parameters
    sample_hw = 10
    z_alpha = 0.05 # Minimum p-value for z-test; we assume noise unless p < z_alpha (so higher z_alpha means more signal bins)

    # Fitting Parameters
    fit = True
    min_fit_xi_idx = 1
    trim_step = 5
    sigma_weighting_power = 1 # > 0 means less weight on lower coherence bins
    A_restrict = False
         

    # Plotting parameters
    fig_subfolder = r'/Xi vs # Cycles (Fitted)/'
    s_signal=1
    s_noise=1
    s_decayed = 100
    plot_noise = False
    marker_signal='o'
    marker_noise='o'
    marker_decayed='*'
    edgecolor_signal=None
    edgecolor_noise=None
    edgecolor_decayed='black'
    crop=False
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


    # Take care of A restriction
    if A_restrict:
        A_max = 1
        suptitle = suptitle + r"   [$A \in [0, 1]$]]"
        fn_plot_type = "A in [0, 1]"
    else:
        A_max = np.inf
        fn_plot_type = "A in [0, infty]"
        suptitle = suptitle + r"   [$A \in [0, \infty]$]]"
    
    plt.close('all')
    plt.figure(figsize=(12, 10))
    plt.suptitle(suptitle)

    for f_target_idx, bin_name, color in zip(bin_idxs, bin_names, colors):    
        # (Possibly) skip noise bins
        noise_bin = 'Noise' in bin_name
        if not plot_noise and noise_bin: 
            continue   
        
        # Calculate signal vs noise and point of decay
        is_signal, xis_plot, target_coherences_plot, xi_decayed, decayed_idx = get_is_signal(coherences, f, xis, f_target_idx, f_noise=12000, sample_hw=sample_hw, z_alpha=z_alpha, crop=crop)
        is_noise = ~is_signal
        
                
            
        # Curve Fit
        
        if fit and not noise_bin:
            fit_start_idx = np.argmax(target_coherences_plot[min_fit_xi_idx:]) + min_fit_xi_idx
            x_fit = xis_plot[fit_start_idx:decayed_idx]
            y_fit = target_coherences_plot[fit_start_idx:decayed_idx]
            sigma = 1 / (y_fit**sigma_weighting_power+ 1e-9)  # Inverse variance weighting
            attempts = 0
            popt = None
            while len(x_fit) > trim_step and popt is None:
                try:
                    popt, pcov = curve_fit(exp_decay, x_fit, y_fit, p0=[1, 1], sigma=sigma, bounds=([0, 0], [A_max, np.inf]), absolute_sigma=True)
                    break  # Fit succeeded!
                except (RuntimeError, ValueError) as e:
                    attempts += 1
                    x_fit = x_fit[:-trim_step]
                    y_fit = y_fit[:-trim_step]
                    sigma = sigma[:-trim_step]
                    print(f"Fit failed (attempt {attempts}): {e} — trimmed to {len(x_fit)} points")
            if popt is None:
                raise RuntimeError("Curve fit failed after all attempts.")
            y_fitted = exp_decay(x_fit, *popt)
            timeconst = popt[1]
        
        # Plot this peak
        f_target = f[f_target_idx]
        fit_labels = [bin_name + rf" (TC={timeconst*1000:.1f}ms)", bin_name + rf" (TC={timeconst*f_target:.0f} Cycles)"]
        for xdim, subplot_idx, fit_label in zip(['Xi', '# Cycles'], [1, 2], fit_labels):
            plt.subplot(2, 1, subplot_idx)
            
            alpha=0.3 if noise_bin else 1 
            scatter_label = None if fit else bin_name
            x = xis_plot  * f_target if xdim == '# Cycles' else xis_plot * 1000
            x_fitted = x_fit * f_target if xdim == '# Cycles' else x_fit * 1000
            
            
            plt.scatter(x[is_signal], target_coherences_plot[is_signal], s=s_signal, edgecolors=edgecolor_signal, marker=marker_signal, color=color, alpha=alpha, zorder=1)
            plt.scatter(x[is_noise], target_coherences_plot[is_noise], s=s_noise, color=color, label=scatter_label, edgecolors=edgecolor_noise, alpha=alpha, zorder=1)
            if not noise_bin: 
                # Mark decayed point
                plt.scatter(x[decayed_idx], target_coherences_plot[decayed_idx], s=s_decayed, marker=marker_decayed, color=color, edgecolors=edgecolor_decayed, zorder=3)
                if fit:
                    plt.plot(x_fitted, y_fitted, color=color, label=fit_label, zorder=2)
            

    # Finish up plot
    plt.subplot(2, 1, 1)
    plt.xlabel(r'$\xi$ [ms]')
    plt.ylabel(r'$C_{\xi}$')           
    plt.title(r"$C_{\xi}$ Decays")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title(r"$C_{\xi}$ Decays (# Cycles)")
    plt.xlabel(r'# of Cycles')
    plt.ylabel(r'$C_{\xi}$')           
    plt.legend()
    # Prevent us from getting hella noise for the 12kHz one
    max_target_f_idx = np.max(bin_idxs[0:-2]) # Get the largest peak frequency
    max_num_cycles = xis[-1] * f[max_target_f_idx] # Get corresponding max number of cycles
    # plt.xlim(0, max_num_cycles) 

    # Book it!
    plt.tight_layout()
    plt.savefig(f'{fig_folder}{fig_subfolder}C_xi Decays Xi vs # Cycles (Fitted, {fn_plot_type}) - {fn_id}.png', dpi=300)
    # plt.show()
        
    