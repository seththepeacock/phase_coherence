@njit
def fft_convolve_numba(x, h):
    """Numba-compatible FFT-based convolution ('valid' mode)."""
    n = len(x)
    m = len(h)
    n_fft = next_pow2(n + m - 1)  # next power of 2
    X = np.fft.rfft(x, n_fft)
    H = np.fft.rfft(h, n_fft)
    y = np.fft.irfft(X * H, n_fft)
    y = y[:n + m - 1]
    return y[m-1:n]  # mimic 'valid' mode

@njit
def fft_autocorr_numba(x):
    """Numba-compatible FFT-based autocorrelation."""
    n = len(x)
    n_fft = next_pow2(2*n - 1)
    X = np.fft.rfft(x, n_fft)
    Sxx = X * np.conjugate(X)
    acf = np.fft.irfft(Sxx, n_fft)
    acf = np.real(acf[:2*n - 1])
    return acf

@njit
def next_pow2(n):
    return 1 << int(math.ceil(math.log2(n)))

@njit
def get_nbacf_cgram(wf, fs, xis, f, win, pw):
    colossogram = np.empty((len(xis), len(f)))
    
    for f_idx in range(len(f)):
        f0_exact = f[f_idx]
        omega_0_norm = f0_exact * 2 * np.pi / fs
        n = np.arange(len(win))
        kernel = win * np.exp(1j * omega_0_norm * n)
        
        # Filtering via convolution
        wf_filtered = fft_convolve_numba(wf, kernel)
        
        # Normalize amplitude (if not power-weighted)
        if not pw:
            wf_filtered = wf_filtered / np.abs(wf_filtered)
        
        # Autocorrelation
        acf = fft_autocorr_numba(wf_filtered)
        N = len(wf_filtered)
        
        # Build lags
        lags = np.arange(-N + 1, N)
        zero_lag_idx = N - 1
        
        # Crop to requested xi lags
        xi_idxs = xis + zero_lag_idx
        acf = acf[xi_idxs]
        lags = lags[xi_idxs]
        
        # Normalization factors
        lags_abs = np.abs(lags)
        num_terms = N - lags_abs
        
        if pw:
            var_xi = np.empty(len(lags))
            for k in range(len(lags_abs)):
                lag_abs = lags_abs[k]
                if lag_abs == 0:
                    var_xi[k] = np.var(wf_filtered)
                else:
                    v1 = np.var(wf_filtered[lag_abs:])
                    v2 = np.var(wf_filtered[:-lag_abs])
                    var_xi[k] = np.sqrt(v1 * v2)
            colossogram[:, f_idx] = np.abs(acf) / (num_terms * var_xi)
        else:
            colossogram[:, f_idx] = np.abs(acf) / num_terms

    return colossogram