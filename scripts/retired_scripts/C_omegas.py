    # or we can reference it against the phase of both the lower and higher frequencies (at the same point in time)
    case "freqs":
        
        match mode:

            # C_omegas^phi
            case "phi":
                # Get phases
                _, f, stft = get_stft(
                    wf=wf,
                    fs=fs,
                    tau=tau_updated,
                    hop=hop,
                    win=win,
                    nfft=nfft,
                    N_segs=N_pd,
                    f0s=f0s,
                    demod=demod,
                )
                # Get angles
                phases = np.angle(stft)
                # Calculate N_segs and N_bins
                N_segs = stft.shape[0]
                N_bins = len(f)
                # initialize arrays
                # even though we only lose ONE freq point with lower and one with higher, we want to get all the points we can get from BOTH so we do - 2
                pds_low = np.zeros((N_segs, N_bins - 2))
                pds_high = np.zeros((N_segs, N_bins - 2))
                # take the first and last bin off the freq ax
                f = f[1:-1]

                # calc phase diffs
                for seg in range(N_segs):
                    for freq_bin in range(1, N_bins - 1):
                        # the - 1 is so that we start our pd_low and pd_high arrays at 0 and put in N_bins-2 points.
                        # These will correspond to our new frequency axis.
                        pds_low[seg, freq_bin - 1] = (
                            phases[seg, freq_bin] - phases[seg, freq_bin - 1]
                        )
                        pds_high[seg, freq_bin - 1] = (
                            phases[seg, freq_bin + 1] - phases[seg, freq_bin]
                        )
                # set the phase diffs to one of these so we can return (could've also been pd_high)
                pds = pds_low
                # Calculate avg vector for low and high
                avg_vector_low = np.mean(np.exp(1j * pds_low), axis=0, dtype=complex)
                autocoherence_low = np.abs(avg_vector_low)
                avg_vector_high = np.mean(np.exp(1j * pds_high), axis=0, dtype=complex)
                autocoherence_high = np.abs(avg_vector_high)
                # average the colossogram you would get from either of these
                autocoherence = (autocoherence_low + autocoherence_high) / 2

            # C_omegas^P or C_omegas^M
            case "P" | "M":
                raise RuntimeError(
                    "Neither C_omegas^M or C_omegas^P has been implemented (only C_omegas^phi)."
                )