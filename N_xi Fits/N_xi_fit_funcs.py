import numpy as np
import scipy.io as sio
import scipy as sp
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def get_wf(wf_fn=None, species=None, wf_idx=None):
    if wf_fn is None:
        if species is None or wf_idx is None:
            raise ValueError("You must input either fn or species and idx!")
        else:
            wf_fn = get_fn(species, wf_idx)
    print("WF file name:", wf_fn)
    # Load matlab file
    N_xi_folder = r'N_xi Fits/'
    wf_folder = N_xi_folder + r'Waveforms/'
    wf = sio.loadmat(wf_folder + wf_fn)['wf'][:, 0]
    # Get fs
    if wf_fn in ['Owl2R1.mat', 'Owl7L1.mat']:
        fs = 48000
    else:
        fs = 44100
    # Get peak list
    match wf_fn:
        # Anoles
        case 'AC6rearSOAEwfB1.mat': #0
            good_peak_freqs = [1233, 2164, 3714, 4500]
            bad_peak_freqs = []
        case 'ACsb4rearSOAEwf1.mat': #1
            good_peak_freqs = [964, 3031, 3160, 3957]
            bad_peak_freqs = []
        case 'ACsb24rearSOAEwfA1.mat': #2    
            good_peak_freqs = [1809, 2169, 3112, 3478]
            bad_peak_freqs = []
        case 'ACsb30learSOAEwfA2.mat': #3
            good_peak_freqs = [1803, 2137, 2406, 2778]
            bad_peak_freqs = []
        # Humans
        case 'ALrearSOAEwf1.mat': #0
            good_peak_freqs = [2665, 2945, 3219, 3865]
            bad_peak_freqs = []
        case 'JIrearSOAEwf2.mat': #1
            good_peak_freqs = [2342, 3402, 8312, 8678]
            bad_peak_freqs = []
        case 'LSrearSOAEwf1.mat': #2
            good_peak_freqs = [732, 985, 1637, 2229]
            bad_peak_freqs = []
        case 'TH13RearwaveformSOAE.mat': #3
            good_peak_freqs = [904, 1518, 2040, 2697]
            bad_peak_freqs = []
        # Owls
        case 'Owl2R1.mat': #0
            good_peak_freqs = [4355, 7451, 8458, 9039]
            bad_peak_freqs = []
        case 'Owl7L1.mat': #1
            good_peak_freqs = [6896, 7941, 8861, 9271]
            bad_peak_freqs = []
        case 'TAG6rearSOAEwf1.mat': #2
            good_peak_freqs = [5626, 8096, 8484, 9862]
            bad_peak_freqs = []
        case 'TAG9rearSOAEwf2.mat': #3
            good_peak_freqs = [4931, 6993, 7450, 9878]
            bad_peak_freqs = []
    return wf, wf_fn, fs, np.array(good_peak_freqs), np.array(bad_peak_freqs)
        
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
        case _:
            raise ValueError("Species must be 'Anole', 'Human', or 'Owl'!")

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
        # As n (sample size) gets smaller, sqrt(n) gets smaller → the standard error increases -> more likely to fail to reject null -> more noise bins.
        # Think of the standard error as the "blur" or "noise" around your estimate. 
        # Smaller samples = blurrier estimates = harder to be confident your sample mean is really different from the null mean.
        # Calculate p-value for a one-tailed test (sf = survival function = 1 - cdf for a right tailed z test)
        # p = Pr(Z > z), i.e., sample mean > null mean
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
    
    return is_signal, target_coherences, xi_decayed, decayed_idx, noise_means, noise_stds

def exp_decay(x, T, amp):
    return amp * np.exp(-x/T)

def fit_peak(f, peak_idx, sample_hw, z_alpha, decay_start_max_xi, trim_step, sigma_weighting_power, bounds, p0, coherences, xis, wf_fn, rho):
    # Get freq
    freq = f[peak_idx]
    # Convert the xis array to number of cycles
    xis_num_cycles = xis  * freq
    
    if sigma_weighting_power == 0:
        get_fit_sigma = lambda y, sigma_weighting_power: np.ones(len(y))
    else: 
        get_fit_sigma = lambda y, sigma_weighting_power: 1 / (y**sigma_weighting_power+ 1e-9)  
    # Calculate signal vs noise and point of decay
    is_signal, target_coherence, xi_decayed, decayed_idx, noise_means, noise_stds = get_is_signal(coherences, f, xis, peak_idx, f_noise=12000, sample_hw=sample_hw, z_alpha=z_alpha)
    is_noise = ~is_signal
    
    # Curve Fit
    print(f"Fitting exp decay to {freq:.0f}Hz peak on {wf_fn} with rho={rho}")
    # Find where to start the fit as the latest peak in the range defined by xi=[0, decay_start_max_xi] 
    decay_start_max_xi_idx = np.argmin(np.abs(xis-decay_start_max_xi))
    # But if the signal has decayed by this point, then the latest the fit should end is, of course, the decay point!
    decay_start_max_xi_idx = min(decay_start_max_xi_idx, decayed_idx)
    maxima = find_peaks(target_coherence[:decay_start_max_xi_idx])[0]
    num_maxima = len(maxima)
    match num_maxima:
        case 1:
            fit_start_idx = maxima[0]
        case 2:
            fit_start_idx = maxima[1]
        case 0:
            print(f"No peaks found in first {decay_start_max_xi*1000:.0f}ms of xi!")
            fit_start_idx = 0
        case _:
            print(f"Three or more peaks found in first {decay_start_max_xi*1000:.0f}ms of xi!")
            fit_start_idx = 0
    x_to_fit = xis[fit_start_idx:decayed_idx]
    y_to_fit = target_coherence[fit_start_idx:decayed_idx]
    sigma = get_fit_sigma(y_to_fit, sigma_weighting_power) 
    failures = 0
    popt = None
    
    while len(x_to_fit) > trim_step and popt is None:
        try:
            popt, pcov = curve_fit(exp_decay, x_to_fit, y_to_fit, p0=p0, sigma=sigma, bounds=bounds)
            break  # Fit succeeded!
        except (RuntimeError, ValueError) as e:
            # Trim the x, y, 
            failures += 1
            x_to_fit = x_to_fit[trim_step:-trim_step]
            y_to_fit = y_to_fit[trim_step:-trim_step]
            sigma = sigma[trim_step:-trim_step]
            
            print(f"Fit failed (attempt {failures}): — trimmed to {len(x_to_fit)} points")
            
    if popt is None:
        print("Curve fit failed after all attempts ({freq:.0f}Hz from {wf_fn})")
        # raise RuntimeError(f"Curve fit failed after all attempts ({freq:.0f}Hz from {wf_fn})")
        return freq, is_signal, is_noise, xi_decayed, decayed_idx, xis_num_cycles, target_coherence, noise_means
    else:
        print("Fitting successful!")
        # Get the paramters and stndard devition
        perr = np.sqrt(np.diag(pcov))
        T = popt[0]
        T_std = perr[0]
        A = popt[1]
        A_std = perr[1]
        # Get the fitted exponential decay
        y_fitted = exp_decay(x_to_fit, *popt)
        # Convert the x array for the fit into number of cycles
        x_fitted = x_to_fit * freq
        # Get coherence array that corresponds to the final fit
        target_coherence_cropped = target_coherence[fit_start_idx:decayed_idx]
        if failures > 0:
            target_coherence_cropped = target_coherence[failures*trim_step:-failures*trim_step]

        # Calculate MSE
        mse = np.mean((y_fitted - target_coherence_cropped)**2)
    
        return T, T_std, A, A_std, mse, freq, is_signal, is_noise, decayed_idx, target_coherence, target_coherence_cropped, xis_num_cycles, x_fitted, y_fitted, noise_means, noise_stds