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


"Static vs Dynamic Colossogram (GRFP Proposal)"

# Choose subject
species = "Tokay"
wf_idx = 2

# Get waveform
wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
    species=species,
    wf_idx=wf_idx,
)

# Plot params
xlabel = "Timescale [ms]"
ylabel = "Frequency [kHz]"
cmap = "magma"
cbar_label = "Phase Self-Coherence"
fontsize = 8
fontsize_ticks = 5
labelpad = 5
ymax = 3.0
ymin = 1.25
xmin = 5
xmax = 30
vmax_1 = 1.0
vmax_2 = 0.3

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
xi_min_s = 0.0005
xi_max_s = 0.1
nfft = 2**14
const_N_pd = 0
show_plots = 1




tau = get_precalc_tau_from_bw(bw, fs, win_type, tau_pkl_folder)

# Start building LCC kwargs dict with constants
lcc_kwargs_alt = {
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
plt.figure(figsize=(8.5 - 2, 2))




# Load Colossogram
cgram_dict = load_calc_colossogram(**lcc_kwargs_alt)

# Load everything that wasn't explicitly "saved" in the filename
colossogram = cgram_dict["colossogram"]
xis_s = cgram_dict["xis_s"]
f = cgram_dict["f"]

# MAKE PLOT
plt.subplot(1, 3, 3)
cbar = pc.plot_colossogram(xis_s, f, colossogram, cmap=cmap, return_cbar=True, vmax=vmax_2)
plt.title("Dynamic", fontsize=fontsize)
# Set Chris' requested fontsizes
ax = plt.gca()

ax.set_ylabel(None)
ax.set_xlabel(None)

ax.tick_params("both", labelsize=fontsize_ticks)
cbar.ax.set_ylabel(cbar_label, labelpad=labelpad)
cbar.ax.yaxis.label.set_fontsize(fontsize)
cbar.ax.tick_params(labelsize=fontsize_ticks)



plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)

"Repeat for static (50Hz)"
# Switch to static
lcc_kwargs_alt["win_meth"] = {"method": "static", "win_type": win_type}

# Change bandwidth
bw_static = 50
tau_static_50 = get_precalc_tau_from_bw(bw_static, fs, win_type, tau_pkl_folder)
lcc_kwargs_alt["tau"] = tau_static_50

# Load Colossogram
cgram_dict = load_calc_colossogram(**(lcc_kwargs_alt))

# Load everything that wasn't explicitly "saved" in the filename
colossogram = cgram_dict["colossogram"]
xis_s = cgram_dict["xis_s"]
f = cgram_dict["f"]

# MAKE PLOT
plt.subplot(1, 3, 1)
plt.title("50Hz Bandwidth", fontsize=fontsize)
cbar = pc.plot_colossogram(xis_s, f, colossogram, cmap=cmap, return_cbar=True, vmax=vmax_1)
# Set Chris' requested fontsizes
ax = plt.gca()


# ax.set_xlabel(xlabel, labelpad=labelpad_tight, fontsize=fontsize_tight)
ax.set_ylabel(ylabel, labelpad=labelpad, fontsize=fontsize)
ax.set_xlabel(None)

ax.tick_params("both", labelsize=fontsize_ticks)


# cbar.remove()
cbar.ax.set_ylabel(None)
cbar.ax.tick_params(labelsize=fontsize_ticks)

plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)


"Repeat for static (150Hz)"
bw_static = 150
tau_static_150 = get_precalc_tau_from_bw(bw_static, fs, win_type, tau_pkl_folder)
lcc_kwargs_alt['tau'] = tau_static_150

# Load Colossogram
cgram_dict = load_calc_colossogram(**(lcc_kwargs_alt))

# Load everything that wasn't explicitly "saved" in the filename
colossogram = cgram_dict["colossogram"]
xis_s = cgram_dict["xis_s"]
f = cgram_dict["f"]


# MAKE PLOT
plt.subplot(1, 3, 2)
plt.title("150Hz Bandwidth", fontsize=fontsize)
cbar = pc.plot_colossogram(xis_s, f, colossogram, cmap=cmap, return_cbar=True, vmax=vmax_2)
# Set Chris' requested fontsizes
ax = plt.gca()



ax.set_xlabel(xlabel, labelpad=labelpad, fontsize=fontsize)
ax.set_ylabel(None)

ax.tick_params("both", labelsize=fontsize_ticks)
# cbar.remove()
cbar.ax.set_ylabel(None)
cbar.ax.tick_params(labelsize=fontsize_ticks)

plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)


plt.tight_layout()

plt.savefig(os.path.join(paper_figures_folder, f"dyn win cgram fig.jpg"), dpi=500)

if show_plots:
    plt.show()