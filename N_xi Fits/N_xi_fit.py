import numpy as np
from N_xi_fit_funcs import *
import pickle
import os
import phaseco
from phaseco import *
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# Initialize list for row dicts for xlsx file
rows = []

# for rho in [0.7, 0.75, 0.65]:
for rho in [0.7]:
    for species in ['Anole', 'Owl', 'Human']:
        for wf_idx in [0, 1, 2, 3]:
            for dense_stft, const_N_pd in [(1, 1)]:
                print(f"Processing {species} {wf_idx}")
                if species == 'Human' and wf_idx ==1:
                    # ALSO REMOVE THE ONE BELOW CROPPING MAX XI for 2
                    continue
                
                "Get waveform"
                wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(species=species, wf_idx=wf_idx)
                
                # Apply a high pass filter
                hpf_cutoff_freq = 300
                wf = spectral_filter(wf, fs, hpf_cutoff_freq, type='hp')
                
                # Crop to desired length
                wf_length = 30 if dense_stft else 60                  
                wf = wf[:int(wf_length*fs)]
                
                "PARAMETERS"
                # Coherence Parameters
                # rho = None
                snapping_rhortle = 0
                tau = 2**13 / 44100 # Everyone uses the same tau
                tauS = int(tau*fs)
                min_xi = 0.001
                delta_xi = min_xi
                # skip_min_xi = True if (dense_stft, const_N_pd) == (0, 0) else False
                skip_min_xi = False
                force_recalc_coherences = 0
                
                
                # Plotting options
                plotting_colossogram = 0
                plotting_peak_picks = 1
                plotting_fits = 1
                show_plots = 0
                
                
                # Decay Star Method
                noise_floor_bw_factor = 0.7 # This is how many standard deviations away from the mean to set the noise floor
                
                # Fitting Parameters
                trim_step = 1
                A_max = np.inf # 1 or np.inf
                sigma_weighting_power = 0 # > 0 means less weight on lower coherence bins in fit
                
                # Plotting parameters
                plot_noise_on_fits = 1
                plot_single_noise_bin_on_fits = 0 # Set this to the frequency you want to plot
                s_signal=5
                s_noise=5
                s_decayed = 100
                marker_signal='o'
                marker_noise='o'
                marker_decayed='*'
                lw_fit = 1.5
                alpha_fit = 1
                pe_stroke_fit = [pe.Stroke(linewidth=2, foreground='black', alpha=1), pe.Normal()]  
                edgecolor_signal=None
                edgecolor_noise='yellow'
                edgecolor_decayed='black'
                crop=False
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
                # Species specific params
                
                # Maximum xi value
                max_xis = {
                    'Anole': 0.1,
                    'Owl': 0.1,
                    'Human': 1.5
                }
                
                # Maximum frequency to plot (in khz)
                max_khzs = {
                    'Anole': 6,
                    'Human': 10,
                    'Owl': 12
                }
                
                # This determines where to start the fit as the latest peak in the range defined by xi=[0, decay_start_max_xi] 
                decay_start_max_xis = {
                    'Anole' : 0.03,
                    'Owl' : 0.03,
                    'Human' : 0.2
                }

                decay_start_max_xi = decay_start_max_xis[species]
                max_khz = max_khzs[species]
                max_xi = max_xis[species]
                # CHANGE
                if species == 'Human' and wf_idx == 2:
                    max_xi = 1.0
                global_max_xi = max(max_xis.values()) if const_N_pd else None

                "Set filepaths"
                fn_id = rf"{species} {wf_idx}, const_Npd={const_N_pd}, dense_stft={dense_stft}, rho={rho}, snapping_rhortle={snapping_rhortle}, tau={tau*1000:.0f}ms, max_xi={max_xi}, wf_length={wf_length}s, HPF={hpf_cutoff_freq}Hz, wf={wf_fn.split('.')[0]}"
                # Calclulate Npd if we're going to hold it constant
                pkl_fn = f'{fn_id} (Coherences)'
                N_xi_folder = r'N_xi Fits/'
                pkl_folder = N_xi_folder + r'Pickles/'
                fig_folder = N_xi_folder + rf'Figures/'

                "Calculate things"
                # Raise warning if tauS is not a power of two AND the samplerate is indeed 44100
                if np.log2(tauS) != int(np.log2(tauS)) and fs == 44100:
                    raise ValueError("tauS is not a power of 2, but the samplerate is 44100!")
                # Get coherences
                os.makedirs(pkl_folder, exist_ok=True)
                if os.path.exists(pkl_folder + pkl_fn + '.pkl') and not force_recalc_coherences:
                    with open(pkl_folder + pkl_fn + '.pkl', 'rb') as file:
                        coherences, f, xis, tau, rho, N_pd_min, N_pd_max, seg_spacing, snapping_rhortle, wf_fn, species = pickle.load(file)
                else:
                    print(f"Calculating coherences for {fn_id}")
                    coherences_dict = get_colossogram_coherences(wf, fs, min_xi, max_xi, delta_xi, tauS=tauS, rho=rho, const_N_pd=const_N_pd, snapping_rhortle=snapping_rhortle, dense_stft=dense_stft, global_max_xi=global_max_xi, skip_min_xi=skip_min_xi, return_dict=True)
                    coherences = coherences_dict['coherences']
                    f = coherences_dict['f']
                    xis = coherences_dict['xis']    
                    N_pd_min = coherences_dict['N_pd_min']
                    N_pd_max = coherences_dict['N_pd_max']
                    seg_spacing = coherences_dict['seg_spacing']
                    snapping_rhortle = coherences_dict['snapping_rhortle']
                    
                    with open(pkl_folder + pkl_fn + '.pkl', 'wb') as file:
                        # pickle.dump((coherences, f, xis, tau, rho, wf_fn, species), file)
                        pickle.dump((coherences, f, xis, tau, rho, N_pd_min, N_pd_max, seg_spacing, snapping_rhortle, wf_fn, species), file)
                
                # Get peak bin indices
                good_peak_idxs = np.argmin(np.abs(f[:, None] - good_peak_freqs[None, :]), axis=0) 
                bad_peak_idxs = np.argmin(np.abs(f[:, None] - bad_peak_freqs[None, :]), axis=0)
                
                # Temporary hack
                if skip_min_xi and coherences.shape[1] == xis.shape[0] + 1:
                    coherences = coherences[:, 1:]
                
                "Plots"
                if const_N_pd:
                    if N_pd_min != N_pd_max:
                        raise Exception("If N_pd is constant, then N_pd_min and N_pd_max should be equal...")
                    N_pd_str = f"$N_{{pd}}={N_pd_min}$"
                else:
                    N_pd_str = f"$N_{{pd}} \in [{N_pd_min}, {N_pd_max}]$"
                if snapping_rhortle:
                    rho_str = rf"$\rho={rho}$ - Snapping Rhortle"
                else:
                    rho_str = rf"$\rho={rho}$"
                suptitle = rf"[{species} {wf_idx}]   [{wf_fn}]   [{rho_str}]   [$\tau$={tau*1000:.2f}ms]   [HPF at {hpf_cutoff_freq}Hz]   [$\xi_{{\text{{max}}}}={max_xi*1000:.0f}$ms]   [{wf_length}s WF]   [{N_pd_str}]"
                if dense_stft:
                    suptitle += f'   [Dense STFT ({seg_spacing*1000}ms)]'
                
                
                if plotting_colossogram:
                    print("Plotting Colossogram")
                    plt.close('all')
                    plt.figure(figsize=(15, 5))
                    plot_colossogram(coherences, f, xis, tau, max_khz=max_khz, cmap='magma')
                    for peak_idx in good_peak_idxs:
                        plt.scatter(min_xi*1000 + (max_xi*1000)/50, f[peak_idx] / 1000, c='w', marker='>', label="Peak at " + f"{f[peak_idx]:0f}Hz", alpha=0.5)
                    plt.title(f"Colossogram", fontsize=18)
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
                    plt.title(rf"Colossogram Slice at $\xi={xis[xi_idx]:.3f}$")
                    plt.plot(f / 1000, coherence_slice, label=r'$C_{\xi}$, $\xi={target_xi}$')
                    for peak_idx in good_peak_idxs:
                        plt.scatter(f[peak_idx] / 1000, coherence_slice[peak_idx], c='r')
                    plt.xlabel("Frequency (kHz)")
                    plt.ylabel(r'$C_{\xi}$')
                    plt.xlim(0, max_khz)
                    # PSD plot
                    plt.subplot(2, 1, 2)
                    plt.title(rf"Power Spectral Density")
                    plt.plot(f / 1000, 10*np.log10(psd), label='PSD')
                    for peak_idx in good_peak_idxs:
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
                if plotting_fits:
                    print(f"Fitting {wf_fn}")
                    p0 = [1, 1]
                    bounds = ([0, 0], [np.inf, A_max]) # [T, amp]
                    fit_func = exp_decay
                    for good_peak_idxs, good_peaks in zip([good_peak_idxs, bad_peak_idxs], [True, False]):
                        # If there are no bad peaks, skip
                        if len(good_peak_idxs) == 0:
                            if good_peaks:
                                print("No peaks were picked!")
                            else:
                                continue
                        
                        plt.close('all')
                        plt.figure(figsize=(15, 10))

                        for peak_idx, color, subplot_idx in zip(good_peak_idxs, colors, [1, 2, 3, 4]):
                            # Pack all parameters for fit_peak together into a tuple for compactness
                            peak_fit_params = f, peak_idx, noise_floor_bw_factor, decay_start_max_xi, trim_step, sigma_weighting_power, bounds, p0, coherences, xis, wf_fn, rho
                            # Fit peak
                            fit_peak_output = fit_peak(*peak_fit_params)
                            
                            # Handle the case where the peak fit failed
                            if len(fit_peak_output) == 9:
                                freq, is_signal, is_noise, xi_decayed, decayed_idx, xis_num_cycles, target_coherence, noise_means, noise_stds = fit_peak_output
                                T, T_std, A, A_std, mse = np.nan, np.nan, np.nan, np.nan, np.nan
                                # Begin plot
                                plt.subplot(2, 2, subplot_idx)
                                plt.title(rf"{freq:.0f}Hz Peak (FIT FAILED)")
                            # Handle the case where the peak fit succeeded
                            else:
                                T, T_std, A, A_std, mse, freq, is_signal, is_noise, decayed_idx, target_coherence, target_coherence_cropped, xis_num_cycles, x_fitted, y_fitted, noise_means, noise_stds = fit_peak_output
                                # Plot the fit
                                plt.subplot(2, 2, subplot_idx)
                                plt.title(rf"{freq:.0f}Hz Peak")
                                T_label = rf"{T*freq:.0f} \text{{ Cycles}}"
                                if T_std < np.inf and A_std < np.inf:
                                    fit_label = rf"$T={T_label}\pm{T_std:.2g}$, $A={A:.2f}\pm{A_std:.2g}$, MSE={mse:.2g}"
                                else:
                                    print("FIX FIT LABEL")
                                plt.plot(x_fitted, y_fitted, color=color, label=fit_label, lw=lw_fit, path_effects=pe_stroke_fit, alpha=alpha_fit, zorder=2)
                    
                            # Plot the coherence
                            plt.scatter(xis_num_cycles[is_signal], target_coherence[is_signal], s=s_signal, edgecolors=edgecolor_signal, marker=marker_signal, color=color, zorder=1, label='Above Noise Floor')
                            plt.scatter(xis_num_cycles[is_noise], target_coherence[is_noise], s=s_noise, color=color, edgecolors=edgecolor_noise, zorder=1, label='Below Noise Floor')
                            # Mark decayed point
                            plt.scatter(xis_num_cycles[decayed_idx], target_coherence[decayed_idx], s=s_decayed, marker=marker_decayed, color=color, edgecolors=edgecolor_decayed, zorder=3)
                            if plot_noise_on_fits:
                                # plt.scatter(xis_num_cycles, noise_means, label='Noise Mean (Above 12kHz)', s=1, color=colors[4])
                                plt.plot(xis_num_cycles, noise_means, label=f'All Bins $\mu \pm (\sigma*{noise_floor_bw_factor})$', color=colors[4])
                                plt.fill_between(xis_num_cycles,
                                noise_means - noise_stds*noise_floor_bw_factor,
                                noise_means + noise_stds*noise_floor_bw_factor,
                                color=colors[4],
                                alpha=0.3)
                            if plot_single_noise_bin_on_fits:
                                noise_freq = plot_single_noise_bin_on_fits
                                noise_target_idx = np.argmin(np.abs(f-noise_freq))
                                # plt.scatter(xis_num_cycles, coherences[noise_target_idx, :], label=f'Noise Bin ({noise_freq/1000:.0f}kHz)', s=1, color=colors[5])
                                plt.plot(xis_num_cycles, coherences[noise_target_idx, :], label=f'Noise Bin ({noise_freq/1000:.0f}kHz)', color=colors[5])
                            
                            # Finish plot
                            plt.xlabel(r'# Cycles')
                            plt.ylabel(r'$C_{\xi}$')           
                            plt.ylim(0, 1)
                            plt.legend()
                            
                            # Add params to a row dict
                            if good_peaks:
                                rows.append({
                                    'Species':species,
                                    'WF Index':wf_idx,
                                    'Filename':wf_fn,
                                    'Frequency':freq,
                                    'T':T,
                                    'T_std':T_std,
                                    'A':A,
                                    'A_std':A_std,
                                    'MSE':mse,
                                })
                            
                        # Book it!
                        plt.tight_layout()
                        os.makedirs(fig_folder, exist_ok=True)
                        fits_folder = f'{fig_folder}' if good_peaks else 'Additional Figures'
                        fits_str = f'Fits' if good_peaks else 'Bad Fits'   
                        os.makedirs(f'{fits_folder}\{fits_str}', exist_ok=True) 
                        plt.savefig(f'{fits_folder}\{fits_str}\{fn_id} ({fits_str}).png', dpi=300)
                        if show_plots:
                            plt.show()
                        
                    
                    
                    
                    
                        
    # Save parameter data as xlsx
    df = pd.DataFrame(rows)
    N_xi_fitted_parameters_fn = rf'{fig_folder}\N_xi Fitted Parameters (rho={rho})'
    df.to_excel(rf'{N_xi_fitted_parameters_fn}.xlsx', index=False)
                        
                        
            