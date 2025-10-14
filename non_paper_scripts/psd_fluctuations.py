import phaseco as pc
from N_xi_fit_funcs import *

all_species = [
    "Anole",
    "Human",
    "Owl",
    "Tokay",
]
speciess = all_species
wf_idxs = [2]

# PSD Params
tau = 8192
win_type = 'hann'
hop = 0.1




for species in speciess:
    for wf_idx in wf_idxs:
        wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
            species=species,
            wf_idx=wf_idx,
        )
        print(f"Processing {species} {wf_idx}")
        t, f, stft = pc.get_stft(wf, fs, tau, hop=hop, win=win_type)
        powers = pc.magsq(stft)
        variance = np.var(powers, axis=0)
        psd = np.mean(powers, axis=0) 
        plt.close('all')
        plt.title(f"Variance of {species} {wf_idx}")
        plt.figure()
        plt.plot(f/1000, 10*np.log10(variance), c='b')
        plt.xlim(0, 12)
        plt.ylabel("Variance [dB]", c='b')
        plt.xlabel("Frequency [kHz]")
        plt.twinx()
        plt.plot(f/1000, 10*np.log10(psd), c='r')
        plt.ylabel("PSD [dB]", c='r')
        folder = "paper_analysis/PSD Fluctuations"
        os.makedirs(folder, exist_ok=True)
        plt.savefig(rf"{folder}/{species} {wf_idx}.jpg", dpi=200)
        # plt.show()
        




