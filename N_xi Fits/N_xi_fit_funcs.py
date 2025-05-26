import numpy as np
import scipy.io as sio
import scipy as sp
from scipy.optimize import curve_fit

def get_wf(wf_fn=None, species=None, wf_idx=None):
    wf_length = 60
    if wf_fn is None:
        if species is None or wf_idx is None:
            raise ValueError("You must input either fn or species and idx!")
        else:
            wf_fn = get_fn(species, wf_idx)
    # Load matlab file
    N_xi_folder = r'N_xi Fits/'
    wf_folder = N_xi_folder + r'Waveforms/'
    wf = sio.loadmat(wf_folder + wf_fn)['wf'][:, 0]
    # Get fs
    if wf_fn in ['Owl2R1.mat', 'Owl7L1.mat']:
        fs = 48000
    else:
        fs = 44100
    # Crop to 60s for consistency
    wf = wf[:int(wf_length*fs)]
    # Get peak list
    match wf_fn:
        # Anoles
        case 'AC6rearSOAEwfB1.mat': #0
            peak_freqs = [1232, 2153, 3710, 4501]
            bad_fit_freqs = []
        case 'ACsb4rearSOAEwf1.mat': #1
            peak_freqs = []
            bad_fit_freqs = []
        case 'ACsb24rearSOAEwfA1.mat': #2    
            peak_freqs = []
            bad_fit_freqs = []
        case 'ACsb30learSOAEwfA2.mat': #3
            peak_freqs = []
            bad_fit_freqs = []
        # Humans
        case 'ALrearSOAEwf1.mat': #0
            peak_freqs = [2660, 2940, 3220, 3870]
            bad_fit_freqs = []
        case 'JIrearSOAEwf2.mat': #1
            peak_freqs = []
            bad_fit_freqs = []
        case 'LSrearSOAEwf1.mat': #2
            peak_freqs = []
            bad_fit_freqs = []
        case 'TH13RearwaveformSOAE.mat': #3
            peak_freqs = []
            bad_fit_freqs = []
        # Owls
        case 'Owl2R1.mat': #0
            peak_freqs = []
            bad_fit_freqs = []
        case 'Owl7L1.mat': #1
            peak_freqs = []
            bad_fit_freqs = []
        case 'TAG6rearSOAEwf1.mat': #2
            peak_freqs = []
            bad_fit_freqs = []
        case 'TAG9rearSOAEwf2.mat': #3
            peak_freqs = []
            bad_fit_freqs = []
    return wf, wf_fn, fs, np.array(peak_freqs), np.array(bad_fit_freqs)
        
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

def get_is_signal(coherences, f, xis, f_target_idx, f_noise=12000, sample_hw=10, z_alpha=0.05):
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
    
    decayed_idx = -1
    # Find decayed point
    for i in range(len(is_signal)):
        if i < sample_hw: # These are automatically set to signal anyway but what the heck
            continue
        if not is_signal[i]:
            decayed_idx = i
            break
    if decayed_idx == -1:
        print(f"Signal at {f[f_target_idx]:.0f}Hz never decays!")
    xi_decayed = xis[decayed_idx]
    
    return is_signal, xis, target_coherences, xi_decayed, decayed_idx

def exp_decay(x, T, amp):
    return amp * np.exp(-x/T)

def fit_peak(f, peak_idx, sample_hw, z_alpha, min_fit_xi_idx, trim_step, sigma_weighting_power, bounds, p0, coherences, xis, wf_fn, rho):
    if sigma_weighting_power == 0:
        get_fit_sigma = lambda y, sigma_weighting_power: np.ones(len(y))
    else: 
        get_fit_sigma = lambda y, sigma_weighting_power: 1 / (y**sigma_weighting_power+ 1e-9)
    freq = f[peak_idx]  
    # Calculate signal vs noise and point of decay
    is_signal, xis_plot, target_coherences_plot, xi_decayed, decayed_idx = get_is_signal(coherences, f, xis, peak_idx, f_noise=12000, sample_hw=sample_hw, z_alpha=z_alpha)
    is_noise = ~is_signal
    
    # Curve Fit
    print(f"Fitting exp decay to {freq}Hz peak on {wf_fn} with rho={rho}")
    fit_start_idx = np.argmax(target_coherences_plot[min_fit_xi_idx:]) + min_fit_xi_idx
    x_to_fit = xis_plot[fit_start_idx:decayed_idx]
    y_to_fit = target_coherences_plot[fit_start_idx:decayed_idx]
    sigma = get_fit_sigma(y_to_fit, sigma_weighting_power) 
    attempts = 0
    popt = None
    
    while len(x_to_fit) > trim_step and popt is None:
        try:
            popt, pcov = curve_fit(exp_decay, x_to_fit, y_to_fit, p0=p0, sigma=sigma, bounds=bounds)
            break  # Fit succeeded!
        except (RuntimeError, ValueError) as e:
            attempts += 1
            x_to_fit = x_to_fit[:-trim_step]
            y_to_fit = y_to_fit[:-trim_step]
            sigma = sigma[:-trim_step]
            print(f"Fit failed (attempt {attempts}): — trimmed to {len(x_to_fit)} points")
    if popt is None:
        raise RuntimeError(f"Curve fit failed after all attempts ({freq:.0f}Hz from {wf_fn})")
    else:
        print("Fitting successful!")
    y_fitted = exp_decay(x_to_fit, *popt)
    tc = popt[0]
    A = popt[1]
    x = xis_plot  * freq
    x_fitted = x_to_fit * freq
    
    return tc, A, freq, is_signal, is_noise, decayed_idx, xis_plot, target_coherences_plot, x, x_fitted, y_fitted