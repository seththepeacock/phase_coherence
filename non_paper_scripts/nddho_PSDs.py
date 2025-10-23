import matplotlib.pyplot as plt
import os
import phaseco as pc
from nddho_generator import nddho_generator
import pickle
import numpy as np

plot_folder = os.path.join("paper_analysis", "results", "nddho", "psd")
pkl_folder = os.path.join('paper_analysis', 'pickles', 'nddho')
os.makedirs(plot_folder, exist_ok=True)
os.makedirs(pkl_folder, exist_ok=True)

wf_len_s = 60
fs = 44100
tau_psd = 2**14
win_psd = 'hann'
iter = 1

# The HWHM of an NDDHO seems to be about (1/8)*gamma, so to get most of the full peak we'll take 1/4 gamma on either side
gamma_mult = 1/4 
# So to approximate the SOAEs, which have FWHM in [50, 300], then we want our HWHMs in [25, 150], so gammas in 8*[25, 150] = [200, 1200]
# Then we want our Qs to stay in this range as well... note Q ~ 6 * f_d / gamma so gamma ~ 6 * f_d / Q
# So we can't have gamma exceed 1200 which holds if Q is no lower than 6 * 5000 / 1200 = 30
# And we can't have gamma below 200 which holds if Q is no higher than 6 * 2000 / 200 = 60

do_gamma = 0
do_q = 1

for f_d in [1000, 2000, 3000]:
    if do_q:
        for q in [25, 30, 35, 40, 45, 50, 55, 60]:
            wf_id = f"q={q}, f_d={f_d}, len={wf_len_s}, fs={fs}, iter={iter}"
            wf_fn = f"{wf_id} [NDDHO WF].pkl"
            wf_fp = os.path.join(pkl_folder, wf_fn)
            # Load/calc waveform
            print(wf_id)
            if os.path.exists(wf_fp):
                print("Already got this wf, loading!")
                with open(wf_fp, "rb") as file:
                    wf = pickle.load(file)
            else:
                print(f"Generating NDDHO {wf_fn}")
                wf, wf_y = nddho_generator(f_d, q=q, fs=fs, t_max=wf_len_s)
                with open(wf_fp, "wb") as file:
                    pickle.dump(wf, file)
            f, psd = pc.get_welch(wf, fs, tau_psd, win=win_psd, hop = tau_psd //2)
            # Find gamma
            gamma = (f_d*2*np.pi) / np.sqrt(q**2-1/4)
            hwhm_ish = gamma * gamma_mult
            plt.close('all')
            plt.figure()
            plt.title(wf_id + rf" [$\gamma=${gamma:.0f}]")
            plt.plot(f, psd)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel("PSD")
            plt.vlines([f_d+hwhm_ish, f_d-hwhm_ish], ymin=0, ymax=np.max(psd), label=rf"$\pm \gamma\cdot{gamma_mult}$", color='g')
            plt.vlines([f_d + 50, f_d - 50], ymin=0, ymax=np.max(psd), label=r"$\pm 50$Hz", color='r')
            plt.xlim(f_d-2*hwhm_ish, f_d+2*hwhm_ish)
            plt.legend()
            plot_fp = os.path.join(plot_folder, wf_id + " [PSD Peak].jpg")
            plt.savefig(plot_fp, dpi=100)
    if do_gamma:
        for gamma in [200, 400, 800, 1200]:
            hwhm_ish = gamma * gamma_mult
            wf_id = f"gamma={gamma}, f_d={f_d}, len={wf_len_s}, fs={fs}, iter={iter}"
            wf_fn = f"{wf_id} [NDDHO WF].pkl"
            wf_fp = os.path.join(pkl_folder, wf_fn)
            print(wf_id)
            # Load/calc waveform
            if os.path.exists(wf_fp):
                print("Already got this wf, loading!")
                with open(wf_fp, "rb") as file:
                    wf = pickle.load(file)
            else:
                print(f"Generating NDDHO {wf_fn}")
                wf, wf_y = nddho_generator(f_d, gamma=gamma, fs=fs, t_max=wf_len_s)
                with open(wf_fp, "wb") as file:
                    pickle.dump(wf, file)
            f, psd = pc.get_welch(wf, fs, tau_psd, win=win_psd, hop=tau_psd//2)
            plt.close('all')
            plt.figure()
            plt.title(wf_id)
            plt.plot(f, psd)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel("PSD")
            plt.vlines([f_d+hwhm_ish, f_d-hwhm_ish], ymin=0, ymax=np.max(psd), label=rf"$\pm \gamma\cdot{gamma_mult}$", color='g')
            plt.vlines([f_d+50, f_d-50], ymin=0, ymax=np.max(psd), label=r"$\pm 50$Hz", color='r')
            plt.xlim(f_d-2*hwhm_ish, f_d+2*hwhm_ish)
            plt.legend()
            plot_fp = os.path.join(plot_folder, wf_id + " [PSD Peak].jpg")
            plt.savefig(plot_fp, dpi=100)
            # plt.show()
            

            
