import numpy as np
from N_xi_fit_funcs import *
import os
import phaseco as pc
from phaseco import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences

for species in ['V Sim Human']:
    for wf_idx in [0]:
        print(f"Processing {species} {wf_idx}")
        
        "Get waveform"
        wf, wf_fn, fs, peak_guesses, bad_fit_freqs = get_wf(species=species, wf_idx=wf_idx)
        wf_length = len(wf) / fs

        "PARAMETERS"
        plot = 1
        tau_s = 2**13 / 44100 # Everyone uses the same tau_s
        tau = round(tau_s*fs)
        
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
        fn_id = rf"{species} {wf_idx}, $\tau={tau_s*1000:.0f}$ms, wf_length={wf_length:.3f}s"
        f, psd = pc.get_welch(wf=wf, fs=fs, tau=tau)
        
        # Guesses
        match wf_fn:
            # V Sim Human
            case 'longMCsoaeL1_20dBdiff100dB_InpN1InpYN0gain85R1rs43.mat':
                peak_guesses = [1156, 1246, 1519, 1975]
            # Anoles
            case 'AC6rearSOAEwfB1.mat': #0
                peak_guesses = [1232, 2153, 3710, 4501]
            case 'ACsb4rearSOAEwf1.mat': #1
                peak_guesses = [964, 3028, 3160, 3960]
            case 'ACsb24rearSOAEwfA1.mat': #2    
                peak_guesses = [1811, 2177, 3109, 3486]
            case 'ACsb30learSOAEwfA2.mat': #3
                peak_guesses = [1800, 2139, 2401, 2774]
            # Tokays
            case 'tokay_GG1rearSOAEwf.mat':
                peak_guesses = [1187, 1570, 3218, 3720]
            case 'tokay_GG2rearSOAEwf.mat':
                peak_guesses = [1200, 1560, 3180, 3870]
            case 'tokay_GG3rearSOAEwf.mat':
                peak_guesses = [1115, 1620, 2270, 3149]
            case 'tokay_GG4rearSOAEwf.mat':
                peak_guesses = [1100, 2288, 2840, 3160]
            # Humans
            case 'ALrearSOAEwf1.mat': #0
                peak_guesses = [2660, 2940, 3220, 3870]
            case 'JIrearSOAEwf2.mat': #1
                peak_guesses = [2343, 3401, 8312, 8678]
            case 'LSrearSOAEwf1.mat': #2
                peak_guesses = [736, 983, 1638, 2225]
            case 'TH13RearwaveformSOAE.mat': #3
                peak_guesses = [905, 1522, 2049, 2692]
            # Owls
            case 'Owl2R1.mat': #0
                peak_guesses = [4341, 7456, 8458, 9031]
            case 'Owl7L1.mat': #1
                peak_guesses = [6897, 7940, 8854, 9263]
            case 'TAG6rearSOAEwf1.mat': #2
                peak_guesses = [5609, 8090, 8492, 9862]
            case 'TAG9rearSOAEwf2.mat': #3
                peak_guesses = [4928, 6993, 7450, 9869]
        
        
        peak_indices, _ = find_peaks(10*np.log10(psd), prominence=2)
        

        # # Inspect prominence values of detected peaks
        # prominences = peak_prominences(10*np.log10(psd), peak_indices)[0]

        # # Print them out
        # for i, prom in zip(peak_indices, prominences):
        #     print(f"Peak at f={f[i]:.1f} Hz has prominence {prom:.2f} dB")

        
        picked_peaks = []
        for peak_guess in peak_guesses:
            # Start a list with length > 2 to enter a while loop
            selected_index = [0, 0]
            bw = 50
            while len(selected_index) > 1:
                band = [peak_guess - bw, peak_guess + bw]
                # Select peaks within the frequency band
                band_mask = (f[peak_indices] >= band[0]) & (f[peak_indices] <= band[1])
                # Get all peaks within the frequency band
                selected_index = peak_indices[band_mask]
                # Narrow down the bandwidth until there is only one peak
                bw = bw - 1
                if bw == 0:
                    raise ValueError("No nearby peaks -- bad guess?!")
            peak = f[selected_index[0]]
            picked_peaks.append(peak)
    
        peak_str = ""
        for peak_freq in picked_peaks:
            peak_str += f"{peak_freq:.0f}, "
        peak_str = peak_str[:-2]
        print(peak_str)
        if plot:
            f = np.array(f)
            picked_peaks = np.array(picked_peaks)
            peak_idxs = np.argmin(np.abs(f[:, None] - picked_peaks[None, :]), axis=0) 
            plt.close('all')
            plt.figure(figsize=(10, 5))
            plt.suptitle(fn_id)
            plt.title(rf"Power Spectral Density")
            plt.plot(f, 10*np.log10(psd), label='PSD')
            for peak_idx in peak_idxs:
                plt.scatter(f[peak_idx], 10*np.log10(psd[peak_idx]), marker='x', c='r', s=10)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD [dB]")  
            plt.legend()
            plt.xlim(0, max_khz*1000)
            plt.tight_layout()
            plt.show()
        