import numpy as np
import scipy.io as sio
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

def get_wf(wf_fn=None, species=None, wf_idx=None):
    if wf_fn is None:
        if species is None or wf_idx is None:
            raise ValueError("You must input either fn or species and idx!")
        else:
            wf_fn = get_fn(species, wf_idx)
    # print("WF file name:", wf_fn)
    # Load matlab file
    N_xi_folder = r'N_xi Fits/'
    data_folder = N_xi_folder + r'Data/'
    wf = sio.loadmat(data_folder + wf_fn)['wf'][:, 0]
    # Get fs
    if wf_fn in ['Owl2R1.mat', 'Owl7L1.mat']:
        fs = 48000
    else:
        fs = 44100
    # Get peak list
    match wf_fn:
        # Anoles
        case 'AC6rearSOAEwfB1.mat': #0
            seth_good_peak_freqs = [1233, 2164, 3714, 4500]
            becky_good_peak_freqs = [1233, 2164, 3709, 4506]
            bad_peak_freqs = []
        case 'ACsb4rearSOAEwf1.mat': #1
            seth_good_peak_freqs = [964, 3031, 3160, 3957]
            becky_good_peak_freqs = [964, 3025, 3155, 3951]
            bad_peak_freqs = []
        case 'ACsb24rearSOAEwfA1.mat': #2    
            seth_good_peak_freqs = [1809, 2169, 3112, 3478]
            becky_good_peak_freqs = [2175, 2503, 3112, 3478]
            bad_peak_freqs = [1728, 1814]
        case 'ACsb30learSOAEwfA2.mat': #3
            seth_good_peak_freqs = [1803, 2137, 2406, 2778]
            becky_good_peak_freqs = [1798, 2143, 2406, 2778]
            bad_peak_freqs = []
            
        # Tokays
        case 'tokay_GG1rearSOAEwf.mat': #0
            seth_good_peak_freqs = []
            bad_peak_freqs = []
        case 'tokay_GG2rearSOAEwf.mat': #1
            seth_good_peak_freqs = []
            bad_peak_freqs = []
        case 'tokay_GG3rearSOAEwf.mat': #2
            seth_good_peak_freqs = []
            bad_peak_freqs = []
        case 'tokay_GG4rearSOAEwf.mat': #3
            seth_good_peak_freqs = []
            bad_peak_freqs = []
            
        # Owls
        case 'Owl2R1.mat': #0
            seth_good_peak_freqs = [4355, 7451, 8458, 9039]
            becky_good_peak_freqs = [5953, 7090, 7453, 8016]
            bad_peak_freqs = [4342, 5578, 8450, 9035, 9574]
        case 'Owl7L1.mat': #1
            seth_good_peak_freqs = [6896, 7941, 8861, 9271]
            becky_good_peak_freqs = [7535, 7922, 8426, 9779]
            bad_peak_freqs = [6164, 6896, 8854, 9252]
        case 'TAG6rearSOAEwf1.mat': #2
            seth_good_peak_freqs = [5626, 8096, 8484, 9862]
            becky_good_peak_freqs = [5626, 6029, 8102, 9857]
            bad_peak_freqs = [8489]
        case 'TAG9rearSOAEwf2.mat': #3
            seth_good_peak_freqs = [4931, 6993, 7450, 9878]
            becky_good_peak_freqs = [3461, 6977, 9846, 10270]
            bad_peak_freqs = [4613, 4920, 6164, 7445]    
        
        # Humans
        case 'ALrearSOAEwf1.mat': #0
            seth_good_peak_freqs = [2665, 2945, 3219, 3865]
            becky_good_peak_freqs = [2805, 2945, 3865]
            bad_peak_freqs = [904, 980, 2659, 3219]
        case 'JIrearSOAEwf2.mat': #1
            seth_good_peak_freqs = [2342, 3402, 8312, 8678]
            becky_good_peak_freqs = [2342, 3402, 4048, 5841]
            bad_peak_freqs = [8312, 8678]
        case 'LSrearSOAEwf1.mat': #2
            seth_good_peak_freqs = [732, 985, 1637, 2229]
            becky_good_peak_freqs = [732, 985, 2230]
            bad_peak_freqs = [1637, 3122]
        case 'TH13RearwaveformSOAE.mat': #3
            seth_good_peak_freqs = [904, 1518, 2040, 2697]
            becky_good_peak_freqs = [904, 1518, 2040, 2697]
            bad_peak_freqs = []
        
    return wf, wf_fn, fs, np.array(becky_good_peak_freqs), np.array(bad_peak_freqs)
        
def get_fn(species, idx):
    match species:
        case 'Anole':
            match idx:
                case 0:
                    return 'AC6rearSOAEwfB1.mat'
                case 1:
                    return 'ACsb4rearSOAEwf1.mat'
                case 2: 
                    return 'ACsb24rearSOAEwfA1.mat'
                case 3:
                    return 'ACsb30learSOAEwfA2.mat'
        case 'Tokay':
            return f'tokay_GG{idx+1}rearSOAEwf.mat'

        case 'Human':
            match idx:
                case 0:
                    return 'ALrearSOAEwf1.mat'
                case 1:
                    return 'JIrearSOAEwf2.mat'
                case 2: 
                    return 'LSrearSOAEwf1.mat'
                case 3:
                    return 'TH13RearwaveformSOAE.mat'
        case 'Owl':
            match idx:
                case 0:
                    return 'Owl2R1.mat'
                case 1:
                    return 'Owl7L1.mat'
                case 2:
                    return 'TAG6rearSOAEwf1.mat'
                case 3:
                    return 'TAG9rearSOAEwf2.mat'
        case _:
            raise ValueError("Species must be 'Anole', 'Human', 'Tokay', or 'Owl'!")

def get_is_signal(coherences, f, xis, f_peak_idx, f_noise=0, noise_floor_bw_factor=None):
    if noise_floor_bw_factor is None:
        raise ValueError("You must input noise_floor_bw_factor!")
    # Get the coherence slice we care about
    target_coherence=coherences[f_peak_idx, :]
    # find frequency bin index closest to our cutoff (NOW JUST 0)
    f_noise_idx = (np.abs(f - f_noise)).argmin()  
    # Get mean and std dev of coherence (over frequency axis, axis=0) for each xi value (using ALL frequencies)
    noise_means = np.mean(coherences[f_noise_idx:, :], axis=0) 
    noise_stds = np.std(coherences[f_noise_idx:, :], axis=0, ddof=1) # ddof=1 since we're using sample mean (not true mean) in sample std estimate
    # Now for each xi value, see if it's noise by determining if it's noise_floor_bw_factor*sigma away from the mean
    is_signal = np.full(len(xis), True, dtype=bool)
    
    for xi_idx in range(len(xis)):
        if xi_idx < 5:
            is_signal[xi_idx] = True
            continue
        coherence_value = coherences[f_peak_idx, xi_idx]
        noise_floor_upper_limit = noise_means[xi_idx] + noise_floor_bw_factor * noise_stds[xi_idx]
        # Calculate whether we're above noise floor for each xi value
        is_signal[xi_idx] = coherence_value > noise_floor_upper_limit

    
    return is_signal, target_coherence, noise_means, noise_stds

def exp_decay(x, T, amp):
    return amp * np.exp(-x/T)

def fit_peak(f, f_peak_idx, noise_floor_bw_factor, decay_start_max_xi, trim_step, sigma_weighting_power, bounds, p0, coherences, xis, wf_fn, rho, end_decay_at):
    
    if sigma_weighting_power == 0:
        get_fit_sigma = lambda y, sigma_weighting_power: np.ones(len(y))
    else: 
        get_fit_sigma = lambda y, sigma_weighting_power: 1 / (y**sigma_weighting_power+ 1e-9)  
    # Calculate signal vs noise and point of decay
    is_signal, target_coherence, noise_means, noise_stds = get_is_signal(coherences, f, xis, f_peak_idx, noise_floor_bw_factor=noise_floor_bw_factor)
    is_noise = ~is_signal
    # Get target frequency
    freq = f[f_peak_idx]

    
    # Find where to start the fit as the latest peak in the range defined by xi=[0, decay_start_max_xi] 
    decay_start_max_xi_idx = np.argmin(np.abs(xis-decay_start_max_xi))
    maxima = find_peaks(target_coherence[:decay_start_max_xi_idx], prominence=0.01)[0]
    num_maxima = len(maxima)
    match num_maxima:
        case 1:
            print(f"One peak found in first {decay_start_max_xi*1000:.0f}ms of xi, starting fit here")
            decay_start_idx = maxima[0]
        case 2:
            print(f"Two peaks found in first {decay_start_max_xi*1000:.0f}ms of xi, starting fit at second one!")
            decay_start_idx = maxima[1]
        case 0:
            print(f"No peaks found in first {decay_start_max_xi*1000:.0f}ms of xi, starting fit at zero!")
            decay_start_idx = 0
        case _:
            print(f"Three or more peaks found in first {decay_start_max_xi*1000:.0f}ms of xi, starting fit at third one!")
            decay_start_idx = maxima[2]
    
    # Find decayed point
    dip_below_noise_floor_idx = -1
    for i in range(decay_start_idx, len(is_signal)):
        if not is_signal[i]:
            dip_below_noise_floor_idx = i
            break
    if dip_below_noise_floor_idx == -1:
        print(f"Signal at {freq:.0f}Hz never decays!")

    # Find all minima after the dip below the noise floor
    if end_decay_at == 'Next Min':
        minima = find_peaks(-target_coherence[dip_below_noise_floor_idx:])[0]
        if len(minima) == 0:
            # If no minima, just set decayed_idx to the dip below noise floor
            decayed_idx = dip_below_noise_floor_idx
        else:
            # If there are minima, take the first one after the dip below noise floor
            decayed_idx = dip_below_noise_floor_idx + minima[0]
    else: 
        decayed_idx = dip_below_noise_floor_idx
        
    
    # Curve Fit
    print(f"Fitting exp decay to {freq:.0f}Hz peak on {wf_fn} with rho={rho}")
    # Crop arrays to the fit range
    xis_fit_crop = xis[decay_start_idx:decayed_idx]
    target_coherence_fit_crop = target_coherence[decay_start_idx:decayed_idx]
    sigma = get_fit_sigma(target_coherence_fit_crop, sigma_weighting_power) 
    failures = 0
    popt = None
    
    
    
    while len(xis_fit_crop) > trim_step and popt is None:
        try:
            popt, pcov = curve_fit(exp_decay, xis_fit_crop, target_coherence_fit_crop, p0=p0, sigma=sigma, bounds=bounds)
            break  # Fit succeeded!
        except (RuntimeError, ValueError) as e:
            # Trim the x, y, 
            failures += 1
            xis_fit_crop = xis_fit_crop[trim_step:-trim_step]
            target_coherence_fit_crop = target_coherence_fit_crop[trim_step:-trim_step]
            sigma = sigma[trim_step:-trim_step]
            
            print(f"Fit failed (attempt {failures}): — trimmed to {len(xis_fit_crop)} points")
    
 
    # HAndle case where curve fit fails
    if popt is None:
        print(f"Curve fit failed after all attempts ({freq:.0f}Hz from {wf_fn})")
        T, T_std, A, A_std, mse, xis_fit_crop, fitted_exp_decay = (-1, -1, -1, -1, -1, -1, -1)
        # raise RuntimeError(f"Curve fit failed after all attempts ({freq:.0f}Hz from {wf_fn})")
    else:
        # If successful, get the paramters and standard deviation
        perr = np.sqrt(np.diag(pcov))
        T = popt[0]
        T_std = perr[0]
        A = popt[1]
        A_std = perr[1]
        # Get the fitted exponential decay
        fitted_exp_decay = exp_decay(xis_fit_crop, *popt)
        
        
        
        # Calculate MSE
        mse = np.mean((fitted_exp_decay - target_coherence_fit_crop)**2)
        
   
    
    return T, T_std, A, A_std, mse, is_signal, is_noise, decay_start_idx, decayed_idx, target_coherence, xis_fit_crop, fitted_exp_decay, noise_means, noise_stds

def get_spreadsheet_df(wf_fn, species):
    df = pd.read_excel(r'N_xi Fits/Data/2024.07analysisSpreadsheetV8_RW.xlsx', sheet_name=species if species != 'Anole' else 'Anolis')
    df.iloc[0]
    if wf_fn == 'TAG9rearSOAEwf2.mat': # This one has trailing whitespace in Becky's excel sheet
        wf_fn += ' '
    return df[df['rootWF'].str.split(r'/').str[-1] == wf_fn].copy()
    
def get_params_from_df(df, peak_freq):
    df = df[df['CF'] == peak_freq]
    if len(df) == 0:
        raise ValueError("Dataframe is empty...")
    row = df.iloc[0]
    SNRfit = row['SNRfit']
    fwhm = row['FWHM']
    
    return SNRfit, fwhm
    