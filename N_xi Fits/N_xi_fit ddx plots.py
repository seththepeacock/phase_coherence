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
for end_decay_at in ['Next Min', 'Noise Floor']:
    for rho in [0.7]:
        # Initialize list for row dicts for xlsx file
        rows = []
        for wf_idx in range(4):
            for species in ['Tokay', 'Anole', 'Owl', 'Human']:
            

                print(f"Processing {species} {wf_idx} (rho={rho})")
                
                
                "Get/preprocess waveform"
                wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(species=species, wf_idx=wf_idx)
                hpf_cutoff_freq = 300
                # Apply a high pass filter
                wf = spectral_filter(wf, fs, hpf_cutoff_freq, type='hp')
                # Crop to desired length
                wf_length = 30
                wf = crop_wf(wf, fs, wf_length, species)
                
                
                "PARAMETERS"
                # Coherence Parameters
                # rho = None    
                snapping_rhortle = 0
                tau = 2**13 / 44100 # Everyone uses the same tau
                tauS = int(tau*fs) # This is just 2**13 for efficient FFT implementation, unless fs!=44100
                min_xi = 0.001
                delta_xi = min_xi
                # skip_min_xi = True if (dense_stft, const_N_pd) == (0, 0) else False
                skip_min_xi = False
                force_recalc_coherences = 0
                dense_stft = 1
                const_N_pd = 1
                
                
                # Output options
                output_colossogram = 0
                output_peak_picks = 0
                output_fits = 1
                output_spreadsheet = 0
                show_plots = 0
            
                
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
                    'Human': 1.5,
                    'Tokay': 0.1
                }
                
                # Maximum frequency to plot (in khz)
                max_khzs = {
                    'Anole': 6,
                    'Tokay' : 6,
                    'Human': 10,
                    'Owl': 12
                }
                
                # This determines where to start the fit as the latest peak in the range defined by xi=[0, decay_start_max_xi] 
                decay_start_max_xis = {
                    'Anole' : 0.02,
                    'Tokay' : 0.02,
                    'Owl' : 0.02,
                    'Human' : 0.2
                }
                
                # Decay Star Method
                # This is how many standard deviations away from the mean to set the noise floor
                noise_floor_bw_factors = {
                    'Anole' : 0.5,
                    'Tokay' : 0.5,
                    'Owl' : 0,
                    'Human' : 2
                }
                
                
                # The derivative is very negative at first, and then at some point it flattens out -- this is the point at which it's flattened out enough that we stop the fit
                ddx_threshes = {
                    'Anole' : -2,
                    'Tokay' : -2,
                    'Owl' : -2,
                    'Human' : -2
                }
                
                    
                # Get species-specific params

                decay_start_max_xi = decay_start_max_xis[species]
                max_khz = max_khzs[species]
                max_xi = max_xis[species]
                noise_floor_bw_factor = noise_floor_bw_factors[species]
                ddx_thresh = ddx_threshes[species]
                

                global_max_xi = max(max_xis.values()) if const_N_pd else None
                
                
                
            
                

                "Set filepaths"
                fn_id = rf"{species} {wf_idx}, const_Npd={const_N_pd}, dense_stft={dense_stft}, rho={rho}, snapping_rhortle={snapping_rhortle}, tau={tau*1000:.0f}ms, max_xi={max_xi}, wf_length={wf_length}s, HPF={hpf_cutoff_freq}Hz, wf={wf_fn.split('.')[0]}"
                # Calclulate Npd if we're going to hold it constant
                pkl_fn = f'{fn_id} (Coherences)'
                N_xi_folder = r'N_xi Fits/'
                pkl_folder = N_xi_folder + r'Pickles/'
                results_folder = N_xi_folder + rf'Additional Figures/ddxs (rho={rho}) -- Num Cycles/' 
                all_results_folder = N_xi_folder + rf'Results (All rho)/'
                
                "Calculate things"
                # Raise warning if tauS is not a power of two AND the samplerate is indeed 44100
                if np.log2(tauS) != int(np.log2(tauS)) and fs == 44100:
                    raise ValueError("tauS is not a power of 2, but the samplerate is 44100!")
                # Get coherences
                os.makedirs(pkl_folder, exist_ok=True)
                if os.path.exists(pkl_folder + pkl_fn + '.pkl') and not force_recalc_coherences:
                    with open(pkl_folder + pkl_fn + '.pkl', 'rb') as file:
                        coherences, f, xis, tau, rho, N_pd_min, N_pd_max, hop, snapping_rhortle, wf_fn, species = pickle.load(file)
                        
                else:
                    # continue
                    print(f"Calculating coherences for {fn_id}")
                    coherences_dict = get_colossogram_coherences(wf, fs, min_xi, max_xi, delta_xi, tauS=tauS, rho=rho, const_N_pd=const_N_pd, snapping_rhortle=snapping_rhortle, dense_stft=dense_stft, global_max_xi=global_max_xi, skip_min_xi=skip_min_xi, return_dict=True)
                    coherences = coherences_dict['coherences']
                    f = coherences_dict['f']
                    xis = coherences_dict['xis']    
                    N_pd_min = coherences_dict['N_pd_min']
                    N_pd_max = coherences_dict['N_pd_max']
                    hop = coherences_dict['hop']
                    snapping_rhortle = coherences_dict['snapping_rhortle']
                    
                    with open(pkl_folder + pkl_fn + '.pkl', 'wb') as file:
                        # pickle.dump((coherences, f, xis, tau, rho, wf_fn, species), file)
                        pickle.dump((coherences, f, xis, tau, rho, N_pd_min, N_pd_max, hop, snapping_rhortle, wf_fn, species), file)
                
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
                    suptitle += f'   [Dense STFT ({hop*1000}ms)]'
                
                
                if output_colossogram:
                    print("Plotting Colossogram")
                    plt.close('all')
                    plt.figure(figsize=(15, 5))
                    plot_colossogram(coherences, f, xis, tau, max_khz=max_khz, cmap='magma')
                    for peak_idx in good_peak_idxs:
                        plt.scatter(min_xi*1000 + (max_xi*1000)/50, f[peak_idx] / 1000, c='w', marker='>', label="Peak at " + f"{f[peak_idx]:0f}Hz", alpha=0.5)
                    plt.title(f"Colossogram", fontsize=18)
                    plt.suptitle(suptitle, fontsize=10)
                    for folder in [results_folder, all_results_folder]:
                        os.makedirs(f'{folder}\Colossograms', exist_ok=True)
                        plt.savefig(f'{folder}\Colossograms\{fn_id} (Colossogram).png', dpi=300)
                    if show_plots:
                        plt.show()
                        
                if output_peak_picks:
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
                    plt.savefig(f'{results_folder}\Peak Picks\{fn_id} (Peak Picks).png', dpi=300)
                    if show_plots:
                        plt.show()
                    
                "FITTING"
                if output_fits:
                    print(f"Fitting {wf_fn}")
                    # Get becky's dataframe
                    if species != 'Tokay':
                        df = get_spreadsheet_df(wf_fn, species)
                    
                    p0 = [1, 1]
                    bounds = ([0, 0], [np.inf, A_max]) # [T, amp]
                    fit_func = exp_decay
                    for peak_freqs, peak_idxs, good_peaks in zip([good_peak_freqs, bad_peak_freqs], [good_peak_idxs, bad_peak_idxs], [True, False]):
                        # If there are no bad peaks, skip
                        if len(peak_idxs) == 0:
                            if good_peaks:
                                print("No peaks were picked!")
                            else:
                                continue
                        
                        if not good_peaks:
                            print("Now fitting bad peaks...")
                        
                        plt.close('all')
                        plt.figure(figsize=(15, 10))
                        plt.suptitle(suptitle + f"   [ddx_thresh={ddx_thresh}]")

                        for peak_freq, peak_idx, color, subplot_idx in zip(peak_freqs, peak_idxs, colors, [1, 2, 3, 4]):
                            target_coherences = coherences[peak_idx, :]
                            ddxs = np.diff(target_coherences) / np.diff(xis*peak_freq)
                            plt.subplot(2, 2, subplot_idx)
                            ax = plt.gca()
                            ax2 = ax.twinx()
                            ax2.plot(xis*peak_freq, target_coherences, label=r'$C_{\xi}$', color=color)
                            ax.plot((xis*peak_freq)[:-1], ddxs, 'o', color='k', label=r'$\frac{\partial C_{\xi}}{\partial \xi}$')
                            # ax.set_ylim(-0.001, 0.001)
                            ax.set_ylim(-10, 10)
                            
                        # Book it!
                        plt.tight_layout()
                        os.makedirs(results_folder, exist_ok=True)
                        # fits_folder = f'{fig_folder}' if good_peaks else 'Additional Figures'
                        fits_str = f'Fits' if good_peaks else 'Bad Fits'
                        for folder in [results_folder, all_results_folder]:   
                            os.makedirs(f'{folder}\{fits_str}', exist_ok=True) 
                            plt.savefig(rf'{folder}\{fits_str}\{fn_id} ({fits_str}).png', dpi=300)
                        if show_plots:
                            plt.show()
                        
                        
                        
                        
                        
        if output_spreadsheet:
            # Save parameter data as xlsx
            df_fitted_params = pd.DataFrame(rows)
            N_xi_fitted_parameters_fn = rf'{results_folder}\N_xi Fitted Parameters (rho={rho})'
            df_fitted_params.to_excel(rf'{N_xi_fitted_parameters_fn}.xlsx', index=False)
                            
                            
        print("Done!")