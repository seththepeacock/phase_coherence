def fit_peak(
    f,
    f_peak_idx,
    noise_floor_bw_factor,
    decay_start_max_xi,
    trim_step,
    sigma_weighting_power,
    bounds,
    p0,
    colossogram,
    xis_s,
    wf_fn,
    win_meth_str,
    ddx_thresh,
    ddx_thresh_in_num_cycles,
):
    # Get the coherence slice we care about
    target_coherence = colossogram[:, f_peak_idx]

    if sigma_weighting_power == 0:
        get_fit_sigma = lambda y, sigma_weighting_power: np.ones(len(y))
    else:
        get_fit_sigma = lambda y, sigma_weighting_power: 1 / (
            y**sigma_weighting_power + 1e-9
        )
    # Calculate signal vs noise and point of decay
    is_signal, noise_means, noise_stds = get_is_signal(
        colossogram,
        f,
        xis_s,
        target_coherence,
        noise_floor_bw_factor=noise_floor_bw_factor,
    )
    is_noise = ~is_signal
    # Get target frequency
    freq = f[f_peak_idx]

    # Find where to start the fit as the latest peak in the range defined by xi=[0, decay_start_max_xi]
    decay_start_max_xi_idx = np.argmin(np.abs(xis_s - decay_start_max_xi))
    maxima = find_peaks(target_coherence[:decay_start_max_xi_idx], prominence=0.01)[0]
    num_maxima = len(maxima)
    match num_maxima:
        case 1:
            print(
                f"One peak found in first {decay_start_max_xi*1000:.0f}ms of xi, starting fit here"
            )
            decay_start_idx = maxima[0]
        case 2:
            print(
                f"Two peaks found in first {decay_start_max_xi*1000:.0f}ms of xi, starting fit at second one!"
            )
            decay_start_idx = maxima[1]
        case 0:
            print(
                f"No peaks found in first {decay_start_max_xi*1000:.0f}ms of xi, starting fit at first xi!"
            )
            decay_start_idx = 0
        case _:
            print(
                f"Three or more peaks found in first {decay_start_max_xi*1000:.0f}ms of xi, starting fit at last one!"
            )
            decay_start_idx = maxima[-1]


    # Find first time there is a "minimum" OR a dip below the noise floor
    decayed_idx = -1
    if ddx_thresh_in_num_cycles:
        ddx_thresh = ddx_thresh * freq


    # Since we use a derivative criteria and it starts at a local max, we should give it a few ms for the derivative to get nice and negative
    ddx_search_buffer_sec = 0.005  # Corresponds to ~5 points since xi=0.001
    decay_start_s = xis_s[decay_start_idx]
    ddx_search_start_s = decay_start_s + ddx_search_buffer_sec
    ddx_search_start_idx = np.argmin(np.abs(xis_s - ddx_search_start_s))

    for i in range(ddx_search_start_idx, len(target_coherence) - 1):
        if not is_signal[i]:
            decayed_idx = i
            break
        else:
            ddx = (target_coherence[i + 1] - target_coherence[i]) / (
                xis_s[i + 1] - xis_s[i]
            )
            if ddx > ddx_thresh:
                decayed_idx = i
                break


    if decayed_idx == -1:
        print(f"Signal at {freq:.0f}Hz never decays!")
    # TEST
    # decayed_idx = -1

    # # Find all minima after the dip below the noise floor
    # if end_decay_at == 'Next Min':
    #     minima = find_peaks(-target_coherence[dip_below_noise_floor_idx:])[0]
    #     if len(minima) == 0:
    #         # If no minima, just set decayed_idx to the dip below noise floor
    #         decayed_idx = dip_below_noise_floor_idx
    #     else:
    #         # If there are minima, take the first one after the dip below noise floor
    #         decayed_idx = dip_below_noise_floor_idx + minima[0]
    # else:
    #     decayed_idx = dip_below_noise_floor_idx


    # Curve Fit
    print(f"Fitting exp decay to {freq:.0f}Hz peak on {wf_fn} with win_meth={win_meth_str}")
    # Crop arrays to the fit range
    xis_s_fit_crop = xis_s[decay_start_idx:decayed_idx]
    target_coherence_fit_crop = target_coherence[decay_start_idx:decayed_idx]
    sigma = get_fit_sigma(target_coherence_fit_crop, sigma_weighting_power)
    failures = 0
    popt = None

    while len(xis_s_fit_crop) > trim_step and popt is None:
        try:
            popt, pcov = curve_fit(
                exp_decay,
                xis_s_fit_crop,
                target_coherence_fit_crop,
                p0=p0,
                sigma=sigma,
                bounds=bounds,
            )
            break  # Fit succeeded!
        except (RuntimeError, ValueError) as e:
            # Trim the x, y,
            failures += 1
            xis_s_fit_crop = xis_s_fit_crop[trim_step:-trim_step]
            target_coherence_fit_crop = target_coherence_fit_crop[trim_step:-trim_step]
            sigma = sigma[trim_step:-trim_step]

            print(
                f"Fit failed (attempt {failures}): — trimmed to {len(xis_s_fit_crop)} points"
            )

    # HAndle case where curve fit fails
    if popt is None:
        print(f"Curve fit failed after all attempts ({freq:.0f}Hz from {wf_fn})")
        T, T_std, A, A_std, mse, xis_s_fit_crop, fitted_exp_decay = (
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        )
        # raise RuntimeError(f"Curve fit failed after all attempts ({freq:.0f}Hz from {wf_fn})")
    else:
        # If successful, get the paramters and standard deviation
        perr = np.sqrt(np.diag(pcov))
        T = popt[0]
        T_std = perr[0]
        A = popt[1]
        A_std = perr[1]
        # Get the fitted exponential decay
        fitted_exp_decay = exp_decay(xis_s_fit_crop, *popt)

        # Calculate MSE
        mse = np.mean((fitted_exp_decay - target_coherence_fit_crop) ** 2)

    return (
        T,
        T_std,
        A,
        A_std,
        mse,
        is_signal,
        is_noise,
        decay_start_idx,
        decayed_idx,
        target_coherence,
        xis_s_fit_crop,
        fitted_exp_decay,
        noise_means,
        noise_stds,
    )