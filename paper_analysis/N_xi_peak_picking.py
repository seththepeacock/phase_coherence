import numpy as np
from N_xi_fit_funcs import *
import os
import phaseco as pc
from phaseco import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
 
for species in ['Anole','Tokay', 'Owl', 'Human']:
    for wf_idx in range(5):
        if wf_idx == 4 and species != 'Owl':
            continue
        
        
        "Get waveform"
        wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(species=species, wf_idx=wf_idx)
        print(f"Processing {species} {wf_idx} ({fs}Hz)")
        wf_len_s = 60
        wf = crop_wf(wf, fs, wf_len_s)

        "PARAMETERS"
        plot = 0
        tau = 2**13
        win_type = 'hann'
        # Everyone can use the same tau; slightly less finegrained for owl but it's already way oversampled
        
        max_khzs = {
                        'Anole': 6,
                        'Tokay': 6,
                        'Human': 10,
                        'V Sim Human': 10,
                        'Owl': 12
                    }
                    
        max_khz = max_khzs[species]
        # Get peak bin indices
        fig_folder = r'N_xi Fits/Auto Peak Picks'
        fn_id = rf"{species} {wf_idx}, $\tau={tau / fs *1000:.0f}$ms, wf_length={wf_len_s:.3f}s"
        f, psd = pc.get_welch(wf=wf, fs=fs, tau=tau, win=win_type)
        
        # Guesses
        peak_guesses = np.concatenate((good_peak_freqs, bad_peak_freqs))
        # match wf_fn:
        #     # V Sim Human
        #     case 'longMCsoaeL1_20dBdiff100dB_InpN1InpYN0gain85R1rs43.mat':
        #         peak_guesses = [1156, 1246, 1519, 1975]
        #     # Anoles
        #     case 'AC6rearSOAEwfB1.mat': #0 
        #         peak_guesses = [1232, 2153, 3710, 4501]
        #     case 'ACsb4rearSOAEwf1.mat': #1
        #         peak_guesses = [964, 3028, 3160, 3960]
        #     case 'ACsb24rearSOAEwfA1.mat': #2    
        #         peak_guesses = [2169, 2503, 3112, 3478, 1728, 1809,]
        #     case 'ACsb30learSOAEwfA2.mat': #3
        #         peak_guesses = [1800, 2139, 2401, 2774, 3052]
        #     # Tokays
        #     case 'tokay_GG1rearSOAEwf.mat': # 0
        #         peak_guesses = [1184, 1717, 1572, 3214, 3714]
        #     case 'tokay_GG2rearSOAEwf.mat': # 1
        #         peak_guesses = [1324, 1565, 2901, 3176, 3450, 3876]
        #     case 'tokay_GG3rearSOAEwf.mat': # 2
        #         peak_guesses = [1109, 1322, 2813, 3133, 1620, 2266]
        #     case 'tokay_GG4rearSOAEwf.mat': # 3
        #         peak_guesses = [1100, 2288, 2840, 3160]
        #     # Owls
        #     case 'Owl2R1.mat': #0
        #         peak_guesses = [8016, 8458, 4342, 5578, 5953, 7090, 7451, 9031, 9574]
        #     case 'Owl7L1.mat': #1
        #         peak_guesses = [7941, 7535, 8861,6164, 8426, 9252, 9779]
        #     case 'TAG6rearSOAEwf1.mat': #2
        #         peak_guesses = [6029, 8096, 8484, 9862, 5626, ]
        #     case 'TAG9rearSOAEwf2.mat': #3
        #         peak_guesses = [6993, 3461, 4613, 4931, 6164, 7450, 9878, 10270]
        #     case "owl_TAG4learSOAEwf1.mat": #4
        #         peak_guesses = [5771, 7176, 9631, 4958, 8463, 8839]
        #     # Humans
        #     case 'ALrearSOAEwf1.mat': #0
        #         peak_guesses = [2805, 2945, 3865, 904, 980, 2665, 3219,]
        #     case 'JIrearSOAEwf2.mat': #1
        #         peak_guesses = [2342, 4048, 5841, 3402, 8312, 8678]
        #     case 'LSrearSOAEwf1.mat': #2
        #         peak_guesses = [732, 985, 1637, 2229, 985, 1637, 3122]
        #     case 'TH13RearwaveformSOAE.mat': #3
        #         peak_guesses = [904, 1518, 2040, 2697]
        
        

        psd_db = 10*np.log10(psd)
        peak_indices, _ = find_peaks(psd_db, prominence=1, distance=5)
        # # Inspect prominence values of detected peaks
        # prominences = peak_prominences(10*np.log10(psd), peak_indices)[0]

        # # Print them out
        # for i, prom in zip(peak_indices, prominences):
        #     print(f"Peak at f={f[i]:.1f} Hz has prominence {prom:.2f} dB")

        
        f = np.array(f)
        if plot:
            plt.close('all')
            plt.figure(figsize=(10, 5))
            plt.suptitle(fn_id)
            plt.title(rf"Power Spectral Density")
            plt.plot(f, psd_db, label='PSD', c='k')
            
        for kind, c, peak_guesses in zip(['good', 'bad'], ['g', 'r'], [good_peak_freqs, bad_peak_freqs]):
            picked_peaks = []
            if kind == 'good':
                print("Good peak freqs:")
            elif kind == 'bad':
                print('Bad peak freqs:')
            for peak_guess in peak_guesses:
                # Find index of nearest true peak
                nearest_idx = np.argmin(np.abs(f[peak_indices] - peak_guess))
                
                f_picked_peak_idx = peak_indices[nearest_idx]
                peak = f[f_picked_peak_idx]
                picked_peaks.append(peak)
                if plot:
                    plt.scatter(f[f_picked_peak_idx], psd_db[f_picked_peak_idx], marker='x', c=c, s=30)
            picked_peaks = np.array(picked_peaks)
            peak_str = ""
            for peak_freq in picked_peaks:
                peak_str += f"{peak_freq:.0f}, "
            print(peak_str)
            
        if plot:
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD [dB]")  
            plt.legend()
            plt.xlim(0, max_khz*1000)
            plt.tight_layout()
            plt.show()
        