import numpy as np
from N_xi_fit_funcs import *
import pickle
import os
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
import phaseco as pc
from phaseco import *
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from tqdm import tqdm

for wf_idx in [0, 1, 2, 3]:
    for species in ['Anole', 'Human', 'Owl']:
        for rho in [0.7]:
            for dense_stft, const_Npd in [(0, 0)]:
                    # if rho != 0.6 and (wf_idx, species) != (1, 'Anole'):
                    #     continue
                    "Get waveform"
                    wf, wf_fn, fs, peak_freqs, bad_fit_freqs = get_wf(species=species, wf_idx=wf_idx)
                    print(f"Processing {wf_fn}")
                    
                    # Apply a high pass filter
                    hpf_cutoff_freq = 300
                    wf = spectral_filter(wf, fs, hpf_cutoff_freq, type='hp')
                    
                    # Crop to desired length
                    if not dense_stft:
                        wf_length = 60
                    else:
                        wf_length = 30
                    wf = wf[:int(wf_length*fs)]
                    
                    "PARAMETERS"
                    # Coherence Parameters
                    # rho = None
                    tau = 2**13 / 44100 # Everyone uses the same tau
                    tauS = int(tau*fs)
                    delta_xi = 0.001
                    min_xi = 0.001
                    force_recalc_coherences = 0
                    
                    # Plotting options
                    plotting_colossogram = 1
                    plotting_peak_picks = 0
                    plotting_fits = 0
                    show_plots = 0
                 
                    # Z-Test Parameters
                    sample_hw = 10
                    z_alpha = 0.05 # Minimum p-value for z-test; we assume noise unless p < z_alpha (so higher z_alpha means more signal bins)

                    # Fitting Parameters
                    min_fit_xi_idx = 1
                    trim_step = 10
                    A_max = np.inf # 1 or np.inf
                    sigma_weighting_power = 0 # > 0 means less weight on lower coherence bins in fit
                    
                    # Plotting parameters
                    s_signal=1
                    s_noise=1
                    s_decayed = 100
                    plot_noise = False
                    marker_signal='o'
                    marker_noise='o'
                    marker_decayed='*'
                    lw_fit = 1.5
                    alpha_fit = 1
                    pe_stroke_fit = [pe.Stroke(linewidth=2, foreground='black', alpha=1), pe.Normal()]  
                    edgecolor_signal=None
                    edgecolor_noise=None
                    edgecolor_decayed='black'
                    crop=False
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    
                    # Species specific params
                    max_xis = {
                        'Anole': 0.2,
                        'Human': 1.0,
                        'Owl': 0.2
                    }
                    
                    max_khzs = {
                        'Anole': 6,
                        'Human': 6,
                        'Owl': 12
                    }
                    
                    max_xi = max_xis[species]
                    # global_max_xi = max(max_xis.values())
                    global_max_xi = max_xi
                    max_khz = max_khzs[species]
                    
                            
                    
                    "Set filepaths"
                    fn_id = rf"{species} {wf_idx}, const_Npd={const_Npd}, dense_stft={dense_stft}, rho={rho}, tau={tau*1000:.0f}ms, max_xi={max_xi}, wf_length={wf_length}s, HPF={hpf_cutoff_freq}Hz, wf={wf_fn.split('.')[0]}"
                    # Calclulate Npd if we're going to hold it constant
                    pkl_fn = f'{fn_id} (Coherences)'
                    N_xi_folder = r'N_xi Fits/'
                    pkl_folder = N_xi_folder + r'Coherences Pickles/'
                    fig_folder = N_xi_folder + rf'Figures (rho={rho})/'
                    if const_Npd:
                        global_max_xiS = global_max_xi*fs
                        if dense_stft:
                            seg_spacingS = min_xi*fs
                            # There are int((len(wf)-tauS)/seg_spacingS)+1 full tau-segments. But the last xiS/seg_spacingS (=1 in non-dense_stft-case) ones won't have a reference.
                            const_Npd = int((len(wf) - tauS) / seg_spacingS) + 1 - int(global_max_xiS/seg_spacingS) 
                        else:
                            # The minimum number (at the maximum xiS) of full tau-segments is int((len(wf)-tauS)/global_max_xiS)+1
                            # ...but the last one won't have a reference, so take off the + 1
                            const_Npd = int((len(wf) - tauS) / (global_max_xiS))
                    suptitle = rf"[{wf_fn}]   [$\rho$={rho}]   [$\tau$={tau*1000:.2f}ms]   [HPF at {hpf_cutoff_freq}Hz]   [$\xi_{{\text{{max}}}}={max_xi}$]   [{wf_length}s WF]   [const_Npd={const_Npd}]   [dense_stft={dense_stft}]"
                    

                    "Calculate things"
                    # Raise warning if tauS is not a power of two AND the samplerate is indeed 44100
                    if np.log2(tauS) != int(np.log2(tauS)) and fs == 44100:
                        raise ValueError("tauS is not a power of 2, but the samplerate is 44100!")
                    # Get coherences
                    os.makedirs(pkl_folder, exist_ok=True)
                    if os.path.exists(pkl_folder + pkl_fn + '.pkl') and not force_recalc_coherences:
                        with open(pkl_folder + pkl_fn + '.pkl', 'rb') as file:
                            coherences, f, xis, tau, rho, wf_fn, species = pickle.load(file)
                    else:
                        print(f"Calculating coherences for {fn_id}")
                        f, xis, coherences = get_coherences(wf, fs, tauS, min_xi, max_xi, delta_xi, rho, const_Npd=const_Npd, dense_stft=dense_stft, global_max_xi=global_max_xi)
                        with open(pkl_folder + pkl_fn + '.pkl', 'wb') as file:
                            pickle.dump((coherences, f, xis, tau, rho, wf_fn, species), file)
                    
                    # Get peak bin indices
                    peak_idxs = np.argmin(np.abs(f[:, None] - peak_freqs[None, :]), axis=0) 
                    
                    "Preliminary Plots"
                    if plotting_colossogram:
                        print("Plotting Colossogram")
                        plt.close('all')
                        plt.figure(figsize=(15, 5))
                        plot_colossogram(coherences, f, xis, tau, max_khz=max_khz, cmap='magma')
                        for peak_idx in peak_idxs:
                            plt.scatter(min_xi*1000 + (max_xi*1000)/50, f[peak_idx] / 1000, c='w', marker='>', label="Peak at " + f"{f[peak_idx]:0f}Hz", alpha=0.5)
                        plt.title(f"{species} Colossogram", fontsize=18)
                        plt.suptitle(suptitle, fontsize=10)
                        os.makedirs(f'{fig_folder}\Colossograms', exist_ok=True)
                        plt.savefig(f'{fig_folder}\Colossograms\{fn_id} (Colossogram).png', dpi=300)
                        if show_plots:
                            plt.show()
                            
                    if plotting_peak_picks:
                        print("Plotting Peak Picks")
                        target_xi = 0.01
                        xi_idx = np.argmin(np.abs(xis - target_xi))
                        coherence_slice = coherences[:, xi_idx]
                        psd = get_welch(wf=wf, fs=fs, tauS=tauS)[1]
                        plt.close('all')
                        plt.figure(figsize=(11, 8))
                        plt.suptitle(suptitle)
                        # Coherence slice plot
                        plt.subplot(2, 1, 1)
                        plt.title(rf"{species} Colossogram Slice at $\xi={xis[xi_idx]:.3f}$")
                        plt.plot(f / 1000, coherence_slice, label=r'$C_{\xi}$, $\xi={target_xi}$')
                        for peak_idx in peak_idxs:
                            plt.scatter(f[peak_idx] / 1000, coherence_slice[peak_idx], c='r')
                        plt.xlabel("Frequency (kHz)")
                        plt.ylabel(r'$C_{\xi}$')
                        plt.xlim(0, max_khz)
                        # PSD plot
                        plt.subplot(2, 1, 2)
                        plt.title(rf"{species} PSD")
                        plt.plot(f / 1000, 10*np.log10(psd), label='PSD')
                        for peak_idx in peak_idxs:
                            plt.scatter(f[peak_idx] / 1000, 10*np.log10(psd[peak_idx]), c='r')
                        plt.xlabel("Frequency (kHz)")
                        plt.ylabel("PSD [dB]")  
                        plt.legend()
                        plt.xlim(0, max_khz)
                        plt.tight_layout()
                        os.makedirs(f'{fig_folder}\Peak Picks', exist_ok=True)
                        plt.savefig(f'{fig_folder}\Peak Picks\{fn_id} (Peak Picks).png', dpi=300)
                        if show_plots:
                            plt.show()
                        
                    "FITTING"
                    row = {} # Initialize row dict for xlsx file
                    if plotting_fits:
                        print(f"Fitting {wf_fn}")
                        p0 = [1, 1]
                        bounds = ([0, 0], [np.inf, A_max]) # [tc, amp]
                        fit_func = exp_decay

                        plt.close('all')
                        plt.figure(figsize=(12, 10))
                        plt.suptitle(suptitle)

                        for peak_idx, color, subplot_idx in zip(peak_idxs, colors, [1, 2, 3, 4]):
                            # Fit peak
                            fit_peak_params = f, peak_idx, sample_hw, z_alpha, min_fit_xi_idx, trim_step, sigma_weighting_power, bounds, p0, coherences, xis, wf_fn, rho
                            tc, A, freq, is_signal, is_noise, decayed_idx, xis_plot, target_coherences_plot, x, x_fitted, y_fitted = fit_peak(*fit_peak_params)
                            
                            # Get chi square
                            
                            # Add params to row dict
                            
                            # Plot this peak
                            tc_label = rf"{tc*freq:.0f} \text{{ Cycles}}"
                            plt.subplot(2, 2, subplot_idx)
                            fit_label = rf"{freq:.0f}Hz ($T={tc_label}$, $A={A:.2f}$)"

                            
                            plt.scatter(x[is_signal], target_coherences_plot[is_signal], s=s_signal, edgecolors=edgecolor_signal, marker=marker_signal, color=color, zorder=1)
                            plt.scatter(x[is_noise], target_coherences_plot[is_noise], s=s_noise, color=color, edgecolors=edgecolor_noise, zorder=1)

                            # Mark decayed point
                            plt.scatter(x[decayed_idx], target_coherences_plot[decayed_idx], s=s_decayed, marker=marker_decayed, color=color, edgecolors=edgecolor_decayed, zorder=3)
                            plt.plot(x_fitted, y_fitted, color=color, label=fit_label, lw=lw_fit, path_effects=pe_stroke_fit, alpha=alpha_fit, zorder=2)
                            
                            plt.xlabel(r'# Cycles')
                            plt.ylabel(r'$C_{\xi}$')           
                            plt.ylim(0, 1)
                            plt.legend()
                            
                        for peak_freq in bad_fit_freqs:
                            fit_peak_params = f, peak_idx, sample_hw, z_alpha, min_fit_xi_idx, trim_step, sigma_weighting_power, bounds, p0, coherences, xis, wf_fn, rho
                            tc, A, freq, is_signal, is_noise, decayed_idx, xis_plot, target_coherences_plot, x, x_fitted, y_fitted = fit_peak(*fit_peak_params)
                            
                            # Get chi square
                            
                            # Add params to row dict
                            

                        # Book it!
                        plt.tight_layout()
                        os.makedirs(fig_folder, exist_ok=True)   
                        os.makedirs(f'{fig_folder}\Fits', exist_ok=True) 
                        plt.savefig(f'{fig_folder}\Fits\{fn_id} (Fits).png', dpi=300)
                        if show_plots:
                            plt.show()
                            
                            
                