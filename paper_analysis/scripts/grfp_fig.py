from helper_funcs import *
import os
import matplotlib.pyplot as plt
import phaseco as pc

# Folders
os.chdir(r"C:\Users\setht\Dropbox\Citadel\GitHub\phase-coherence")
paper_figures_folder = os.path.join('non_paper_plots', 'grfp_figs')
pkl_folder = os.path.join('paper_analysis', 'pickles', 'soae')
tau_pkl_folder = os.path.join('paper_analysis', 'pickles')
os.makedirs(paper_figures_folder, exist_ok=True)


"Static vs Dynamic Colossogram (TIGHT)"

# Choose subject
species = "Anole"
wf_idx = 0

# Get waveform
wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
    species=species,
    wf_idx=wf_idx,
)


# Get/set coherence params
filter_meth=None
wf_len_s = 60  # Will crop waveform to this length (in seconds)
scale = True  # Scale the waveform (only actually scales if we know the right scaling constant, which is only Anoles and Humans)
hop_s = 0.01
win_type = "flattop"
rho = 0.7
win_meth = {"method": "rho", "rho": rho, "win_type": win_type}
pw = False
bw = 50
xi_min_s = 0.001
delta_xi_s = 0.0005
xi_max_s = 0.1
nfft = 2**14
const_N_pd = 0


tau = get_precalc_tau_from_bw(bw, fs, win_type, tau_pkl_folder)

# Start building LCC kwargs dict with constants
lcc_kwargs_tight = {
    "wf_len_s": wf_len_s,
    "filter_meth": filter_meth,
    "pw": pw,
    "xi_min_s": xi_min_s,
    "win_meth": win_meth,
    "nfft": nfft,
    "pkl_folder": pkl_folder,
    "xi_max_s": xi_max_s,
    "species": species,
    "fs": fs,
    "tau": tau,
    "hop": int(round(hop_s * fs)),
    "wf": wf,
    "wf_idx": wf_idx,
    "wf_fn": wf_fn,
}





# INITIALIZE PLOT
plt.figure(figsize=(8.5 - 2, 4))

# Set labels
xlabel = "Timescale [ms]"
ylabel = "Frequency [kHz]"
cmap = "magma"
cbar_label = "Coherence"
fontsize_tight = 10
labelpad_tight = 5
ymax = 4
ymin = 1
xmax = 50


# Load Colossogram
cgram_dict = load_calc_colossogram(**lcc_kwargs_tight)

# Load everything that wasn't explicitly "saved" in the filename
colossogram = cgram_dict["colossogram"]
xis_s = cgram_dict["xis_s"]
f = cgram_dict["f"]

xmin = xis_s[0] * 10000

# MAKE PLOT
plt.subplot(1, 3, 3)
cbar = pc.plot_colossogram(xis_s, f, colossogram, cmap=cmap, return_cbar=True)
# Set Chris' requested fontsizes
ax = plt.gca()


# ax.set_xlabel(xlabel, labelpad=labelpad_tight, fontsize=fontsize_tight)
ax.set_ylabel(None)

ax.tick_params("both", labelsize=fontsize_tight)
cbar.ax.set_ylabel(cbar_label, labelpad=labelpad_tight)
cbar.ax.yaxis.label.set_fontsize(fontsize_tight)
cbar.ax.tick_params(labelsize=fontsize_tight)


plt.ylim(1, 3.5)
plt.xlim(xmin, xmax)


"Repeat for static"
lcc_kwargs_tight["win_meth"] = {"method": "static", "win_type": win_type}
bw_static = 150
tau_static_150 = get_precalc_tau_from_bw(bw_static, fs, win_type, tau_pkl_folder)
lcc_kwargs_tight['tau'] = tau_static_150

# Load Colossogram
cgram_dict = load_calc_colossogram(**(lcc_kwargs_tight))

# Load everything that wasn't explicitly "saved" in the filename
colossogram = cgram_dict["colossogram"]
xis_s = cgram_dict["xis_s"]
f = cgram_dict["f"]

# MAKE PLOT
plt.subplot(1, 3, 2)
cbar = pc.plot_colossogram(xis_s, f, colossogram, cmap=cmap, return_cbar=True)
# Set Chris' requested fontsizes
ax = plt.gca()


ax.set_xlabel(xlabel, labelpad=labelpad_tight, fontsize=fontsize_tight)
ax.set_ylabel(None)

ax.tick_params("both", labelsize=fontsize_tight)
cbar.remove()

plt.ylim(1, 3.5)
plt.xlim(xmin, xmax)


"Repeat for static (50Hz)"
bw_static = 50
tau_static_50 = get_precalc_tau_from_bw(bw_static, fs, win_type, tau_pkl_folder)
lcc_kwargs_tight["tau"] = tau_static_50

# Load Colossogram
cgram_dict = load_calc_colossogram(**(lcc_kwargs_tight))

# Load everything that wasn't explicitly "saved" in the filename
colossogram = cgram_dict["colossogram"]
xis_s = cgram_dict["xis_s"]
f = cgram_dict["f"]

# MAKE PLOT
plt.subplot(1, 3, 1)
cbar = pc.plot_colossogram(xis_s, f, colossogram, cmap=cmap, return_cbar=True)
# Set Chris' requested fontsizes
ax = plt.gca()


# ax.set_xlabel(xlabel, labelpad=labelpad_tight, fontsize=fontsize_tight)
ax.set_ylabel(ylabel, labelpad=labelpad_tight, fontsize=fontsize_tight)

ax.tick_params("both", labelsize=fontsize_tight)

cbar.remove()

plt.ylim(1, 3.5)
plt.xlim(xmin, xmax)


plt.tight_layout()

plt.savefig(os.path.join(paper_figures_folder, f"dyn win cgram fig.jpg"), dpi=500)