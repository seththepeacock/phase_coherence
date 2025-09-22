from N_xi_fit_funcs import *
import phaseco as pc
import numpy as np
import scipy
from scipy.fft import fft, fftfreq
import os
os.chdir(r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence")

# Choose waveform and center frequency
species = "Human"
wf_idx = 0

# Load waveform
wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(species=species, wf_idx=wf_idx)

# Set filterbank parameters
hop = 1
tau = 2**13
tau_s = tau / fs
nfft = tau
win_type = 'flattop'
wf_len_s = 10
f0 = good_peak_freqs[0]
predemod=False

# Set phi parameters
tau_phi = 2**12
hop_phi = round(tau_phi / 2)
win_type_phi = 'boxcar'

# Plotting parameters
log_y = True


filter_meth = {
    "type": "kaiser",
    "cf": 300,
    "df": 50,
    "rip": 100,
}

peak_freqs = good_peak_freqs
pkl_folder = r"paper_analysis/Pickles/Filterbanks/"
filter_str = get_filter_str(filter_meth)
fn_id = rf"{species} {wf_idx}, {win_type}, hop={hop}, tau={tau_s*1000:.0f}ms, {filter_str}, wf_len={wf_len_s}s, predemod={predemod}, wf={wf_fn.split('.')[0]}"
pkl_fn = f"{fn_id} (Filterbank)"

paper_analysis_folder = r"paper_analysis/"
phase_psd_folder = (
    paper_analysis_folder + rf"Results/Phase PSDs"
)
os.makedirs(phase_psd_folder, exist_ok=True)
if os.path.exists(pkl_folder + pkl_fn + ".pkl"):
    with open(pkl_folder + pkl_fn + ".pkl", "rb") as file:
        filterbank_dict = pickle.load(file)
else:
    win = scipy.signal.get_window(win_type, tau)
    wf = filter_wf(wf, fs, filter_meth, species)
    wf = crop_wf(wf, fs, wf_len_s)
    t, f, stft = pc.get_stft(
        wf,
        fs=fs,
        tau=tau,
        nfft=nfft,
        hop=hop,
        win=win,
        demod=predemod,
        verbose=True,
    )
    peak_freq_idxs = np.argmin(np.abs(f[:, None] - peak_freqs[None, :]), 0)
    f = f[peak_freq_idxs]
    stft = stft[:, peak_freq_idxs]

    print("Packing into dictionary")
    filterbank_dict = {
        'stft':stft,
        't':t,
        'f':f,
        'fs':fs,
        'tau':tau,
        'fn_id':fn_id,
        'hop':'hop',
        'wf_fn':wf_fn,
    }
    print("Dumping")
    os.makedirs(pkl_folder, exist_ok=True)
    with open(pkl_folder + pkl_fn + ".pkl", "wb") as file:   
        pickle.dump(filterbank_dict, file)

f = filterbank_dict['f']
t = filterbank_dict['t']

f0_idx = np.argmin(np.abs(f-f0))
wf_f0 = filterbank_dict['stft'][:, f0_idx]

print("Calculating PSD of phi")
if not predemod:
    wf_f0 = wf_f0 * np.exp(-1j * f0 * t)
# Unwrap
wf_phi_f0 = np.unwrap(np.angle(wf_f0))

# Detrend
# Fit a straight line: slope = frequency, intercept = initial phase
p = np.polyfit(t, wf_phi_f0, 1)   # p[0] = slope, p[1] = intercept

# Subtract the fitted line
wf_phi_dtrnd = wf_phi_f0 - np.polyval(p, t)


plt.subplot(2, 1, 1)
plt.scatter(t, wf_phi_f0, s=5)
plt.plot(t, np.polyval(p, t))
plt.subplot(2, 1, 2)
plt.plot(t, wf_phi_dtrnd)
plt.show()


f_phi, psd_phi_f0 = pc.get_welch(wf_phi_dtrnd, fs, f0, tau_phi, hop=hop_phi, win=win_type_phi, realfft=False)

if log_y:
    psd_phi_f0 = 10*np.log10(psd_phi_f0)
    ylabel = f'PSD [dB]'

plt.figure()
plt.scatter(f_phi, psd_phi_f0, label=f"{f0}Hz", s=3)
plt.title("Phase Noise PSD")
plt.ylabel(ylabel)
plt.xlim(-1000, 1000)
plt.xlabel
plt.savefig(f"{phase_psd_folder}/{fn_id}.jpg", dpi=300)
plt.show()





    
