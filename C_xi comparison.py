from N_xi_fit_funcs import *
import phaseco as pc
import numpy as np
import os
os.chdir(r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence")
# ,"Human", "Owl", "Tokay"
    # Choose waveform and center frequency

plt.figure()

colors = [
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",

    "#8c564b",
    "#7f7f7f",
    "#bcbd22",

]

for species, color in zip(["Human", "Owl", "Anole", "Tokay"], colors):
    for wf_idx in range(5):
        print(f"{species} {wf_idx}")
        wf_pp = None
        if wf_idx >= 4 and species != "Owl":
            continue


        # Load waveform
        wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(species=species, wf_idx=wf_idx)
        # f0s = np.concat((good_peak_freqs, bad_peak_freqs, [20000]))

        # Set parameters
        xi_s = 0.01
        tau_s = 0.025

        hop = 0.5
        win_meth = {"method": "static", "win_type": "flattop"}
        filter_meth = None

        # Convert
        tau = round(tau_s * fs)
        xi = round(xi_s * fs)
        f, ac = pc.get_autocoherence(wf, fs, tau=tau, xi=xi, win_meth=win_meth, hop=hop, pw=True)

        for peak_freqs, good_peaks, marker in zip(
                [good_peak_freqs, bad_peak_freqs],
                [True, False], ['o', 'x']
            ):
                peak_idxs = np.argmin(np.abs(f[:, None]-peak_freqs[None, :]), axis=0)
                # If there are no peaks, continue
                if len(peak_idxs) == 0:
                    if good_peaks:
                        print("WARNING: No good peaks were picked!")
                    # Could also just be that there are no bad peaks! either way...
                    continue
                
                for k, (f0, peak_idx) in enumerate(zip(
                    peak_freqs, peak_idxs,
                )):
                    C_xi = ac[peak_idx]
                    if k == 0 and wf_idx == 1:
                        label = f"{species}"
                        if species == 'Human':
                            label += " [Good]" if good_peaks else " [Bad]"
                    else:
                        label = None
                    plt.scatter(f0, C_xi, color=color, marker=marker, label=label)
plt.legend()
plt.show()
                         
        





        