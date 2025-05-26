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


# for species in ['Anole', 'Human', 'Owl']:
#     for wf_idx in [0, 1, 2, 3]:
species = 'Anole'
wf_idx = 0
for species in ['Anole']:
    for seths_way in [True, False]:
        for N_phase_diffs_held_const in [True, False]:
            print("Seths way:", seths_way, "N held const:", N_phase_diffs_held_const)
            wf, wf_fn, fs, peak_freqs, bad_fit_freqs = get_wf(species=species, wf_idx=wf_idx)
            print(f"Processing {wf_fn}")
            
            "PARAMETERS"
            # Coherence Parameters
            rho = 0.7
            tau = 2**12 / 44100 # Everyone uses the same tau
            tauS = int(tau*fs)
            delta_xi = 0.001
            min_xi = 0.001
            force_recalc_coherences = 0
            
            # Z-Test Parameters
            sample_hw = 10
            z_alpha = 0.05 # Minimum p-value for z-test; we assume noise unless p < z_alpha (so higher z_alpha means more signal bins)

            # Fitting Parameters
            min_fit_xi_idx = 1
            trim_step = 10
            A_max = np.inf # 1 or np.inf
            sigma_weighting_power = 0 # > 0 means less weight on lower coherence bins 

            # Plotting parameters
            plotting_colossogram = 1
            plotting_peak_picks = 0
            plotting_fits = 1
            show_plots = 0
            s_signal=1
            s_noise=1
            s_decayed = 100
            plot_noise = False
            marker_signal='o'
            marker_noise='o'
            marker_decayed='*'
            lw_fit = 1.5
            alpha_fit = 1
            alpha_bad_fit = 0.2
            pe_stroke_fit = [pe.Stroke(linewidth=2, foreground='black', alpha=1), pe.Normal()]  
            edgecolor_signal=None
            edgecolor_noise=None
            edgecolor_decayed='black'
            crop=False
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Species specific params
            # Lizard
            if wf_fn in ['AC6rearSOAEwfB1.mat', 'ACsb4rearSOAEwf1.mat', 'ACsb24rearSOAEwfA1.mat', 'ACsb30learSOAEwfA2.mat']:
                max_xi = 0.06
                max_khz = 6
            # Human
            elif wf_fn in ['ALrearSOAEwf1.mat', 'JIrearSOAEwf2.mat', 'LSrearSOAEwf1.mat', 'TH13RearwaveformSOAE.mat']:
                max_xi = 1
                max_khz = 6
            # Owl
            elif wf_fn in ['Owl2R1.mat', 'Owl7L1.mat', 'TAG6rearSOAEwf1.mat', 'TAG9rearSOAEwf2.mat']:
                max_xi = 0.1
                max_khz = 12
            
            "Set filepaths"
            fn_id = rf"tau={tau*1000:.0f}ms, rho={rho}, {species}, {wf_fn.split('.')[0]}"
            suptitle = rf"[{wf_fn}]   [$\rho$={rho}]   [$\tau$={tau*1000:.2f}ms]"
            if seths_way:
                suptitle+= "   [seth new way]"
                fn_id += ", seth new way"
            if N_phase_diffs_held_const:
                suptitle+= "   [N_phase_diffs held constant]"
                fn_id += ", N_phase_diffs held const"
            else:
                suptitle+= "   [Use max amount of wf]"
                fn_id += ", max amount of wf"
            pkl_fn = f'C_xi Decay Coherences - {fn_id}'
            N_xi_folder = r'N_xi Fits/N_phase_diffs Comparison/'
            pkl_folder = N_xi_folder + r'Coherences Pickles/'
            fig_folder = N_xi_folder + r'Figures/'
            
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
                print(f"Calculating coherences for {wf_fn} with rho={rho}")
                if seths_way:
                    f, xis, coherences = get_coherences(wf, fs, tauS, min_xi, max_xi, delta_xi, rho, N_phase_diffs_held_const)
                else:
                    num_xis = int((max_xi - min_xi) / delta_xi) + 1
                    xis = np.linspace(min_xi, max_xi, num_xis)

                    max_xiS = max(xis) * fs
                    f = np.array(rfftfreq(tauS, 1/fs))
                    # Make sure we have a consistent number of segments to take vector strength over since this will change with xi
                    if N_phase_diffs_held_const:
                        N_segs = int((len(wf) - tauS) / max_xiS) + 1
                    else:
                        N_segs = None

                    coherences = np.zeros((len(f), len(xis)))
                    for i, xi in enumerate(tqdm(xis)):
                        coherences[:, i] = get_coherence(wf=wf, fs=fs, tauS=tauS, xi=xi, ref_type="next_seg", N_segs=N_segs, rho=rho)[1]
                    
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
                target_xi = 0.02
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
                    xdim = '# Cycles'
                    tc_label = rf"{tc*freq:.0f} \text{{ Cycles}}"
                    plt.subplot(2, 2, subplot_idx)
                    fit_label = rf"{freq:.0f}Hz ($T={tc_label}$, $A={A:.2f}$)"

                    
                    plt.scatter(x[is_signal], target_coherences_plot[is_signal], s=s_signal, edgecolors=edgecolor_signal, marker=marker_signal, color=color, zorder=1)
                    plt.scatter(x[is_noise], target_coherences_plot[is_noise], s=s_noise, color=color, edgecolors=edgecolor_noise, zorder=1)

                    # Mark decayed point
                    plt.scatter(x[decayed_idx], target_coherences_plot[decayed_idx], s=s_decayed, marker=marker_decayed, color=color, edgecolors=edgecolor_decayed, zorder=3)
                    plt.plot(x_fitted, y_fitted, color=color, label=fit_label, lw=lw_fit, path_effects=pe_stroke_fit, alpha=alpha_fit, zorder=2)
                    
                    plt.xlabel(r'$\xi$ [ms]')
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
                    
                    
        