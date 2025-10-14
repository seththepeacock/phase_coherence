import phaseco as pc
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# A script to compare N_xi vs freq plots between two windowing methods


# Switches
show_plot = 1
save_fig = 1


# Globals
nfft = 2**14
win_type = "flattop"
species_bws = {"Anole": 150, "Human": 50, "Owl": 300, "Tokay": 150}
markers = {
    "Anole": ("limegreen", "o"),
    "Owl": ("orange", "d"),
    "Human": ("blue", "x"),
    "Tokay": ("green", "+"),
}
results_folder = os.path.join("paper_analysis", "Results")
mse_thresh = 0.001
xmin = 0.5
xmax = 12
ymin = 1
ymax = 2e3

# Define our methods
static_meth = {
    "win_meth": {"method": "static", "win_type": win_type},
    "hop_thing": ("int", 1),
    "bw_type": "species",
    "pw": False,
}

dynamic_meth = {
    "win_meth": {"method": "rho", "win_type": win_type, "rho": 1.0},
    "hop_thing": ("s", 0.01),
    "bw_type": 50,
    "pw": False,
}

# Define helper func to handle legend labels
def get_label_from_firsts(firsts, species):
    if firsts[species]:
        label = f"{species}"
        firsts[species] = 0
    else:
        label = None
    return label, firsts


# First, we gather everything from the two dataframes
N_xis = {}
T_xis = {}
freqs = {}
speciess = {}
titles = {}
mse_statics = {}
hwhms = {}
for method, key in zip([static_meth, dynamic_meth], ['static', 'dynamic']):
    # Define strings
    bw_str = (
        f"BW=Species" if method["bw_type"] == "species" else f"BW={method['bw_type']}Hz"
    )
    win_meth_str = pc.get_win_meth_str(method["win_meth"])
    relevant_comp_str = rf"PW={method['pw']}, {bw_str}, {win_meth_str}"
    # Define paths
    specific_results_folder = os.path.join(
        results_folder, rf"Results ({relevant_comp_str})"
    )
    N_xi_fitted_parameters_fp = os.path.join(
        specific_results_folder, rf"N_xi Fitted Parameters ({relevant_comp_str}).xlsx"
    )
    # Read df
    df = pd.read_excel(N_xi_fitted_parameters_fp)
    # Gather in lists
    N_xis[key]=df["N_xi"]
    T_xis[key]=df["T_xi"]
    freqs[key]=df["Frequency"]
    speciess[key]=df["Species"]
    titles[key]=f"PW={method['pw']}, {bw_str}, {win_meth_str}"
    if method['win_meth']['method'] == 'static':
        hwhms[key]=df['L_gamma']
        mse_statics[key]=df["MSE"]

# Now we plot!
plt.close("all")
plt.figure(figsize=(20, 20))
plt.suptitle("Windowing Comparison")

"1: Plot static method"
plt.subplot(2, 2, 1)
plt.title(titles['static'])
firsts = {"Anole": 1, "Owl": 1, "Human": 1, "Tokay": 1}
for k in range(len(N_xis['static'])):
    (
        species,
        N_xi,
        freq,
        mse_static,
    ) = (
        speciess['static'][k],
        N_xis['static'][k],
        freqs['static'][k],
        mse_statics['static'][k],
    )
    if mse_static > mse_thresh:
        continue
    label, firsts = get_label_from_firsts(firsts, species)
    plt.scatter(
        freq / 1000,
        N_xi,
        label=label,
        color=markers[species][0],
        marker=markers[species][1],
    )
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.loglog()
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("Frequency [kHz]")
    plt.ylabel(r"$N_\xi$")

"2: Plot dynamic method"
plt.subplot(2, 2, 2)
plt.title(titles['dynamic'])
firsts = {"Anole": 1, "Owl": 1, "Human": 1, "Tokay": 1}
for k in range(len(N_xis['static'])):
    species, N_xi, freq, mse_static = speciess['dynamic'][k], N_xis['dynamic'][k], freqs['dynamic'][k], mse_statics['static'][k]
    if mse_static > mse_thresh:
        continue
    label, firsts = get_label_from_firsts(firsts, species)
    plt.scatter(
        freq / 1000,
        N_xi,
        label=label,
        color=markers[species][0],
        marker=markers[species][1],
    )
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.loglog()
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("Frequency [kHz]")
    plt.ylabel(r"$N_\xi$")
plt.tight_layout()

"3: Plot the deltas"
plt.subplot(2, 2, 3)
plt.title(f"Static Windowing (Faint) vs Dynamic Windowing")
firsts = {"Anole": 1, "Owl": 1, "Human": 1, "Tokay": 1}
for k in range(len(N_xis['static'])):
    species, freq, mse_static = speciess['static'][k], freqs['static'][k], mse_statics['static'][k]
    # Skip egregious static fits
    if mse_static > mse_thresh:
        print("SKIPPING")
        continue
        

    # Get N_xis
    N_xi_1 = N_xis['dynamic'][k]
    N_xi_0 = N_xis['static'][k]

    label, firsts = get_label_from_firsts(firsts, species)

    # Plot line between them
    plt.vlines(freq / 1000, ymin=min(N_xi_0, N_xi_1), ymax=max(N_xi_0, N_xi_1), lw=1, color='k', zorder=0)

    # Plot each method
    plt.scatter(
        freq / 1000,
        N_xi_0,
        alpha=0.5,
        color=markers[species][0],
        marker=markers[species][1],
        zorder=1
    )
    plt.scatter(
        freq / 1000,
        N_xi_1,
        label=label,
        color=markers[species][0],
        marker=markers[species][1],
        zorder=1
    )
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.loglog()
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("Frequency [kHz]")
    plt.ylabel(r"$N_\xi$")


"4: Plot HWHM vs T_xi (static)"
plt.title(rf"Static $T_\xi$ vs PSD Peak HWHM ($\gamma$)")
firsts = {"Anole": 1, "Owl": 1, "Human": 1, "Tokay": 1}
for k in range(len(N_xis['static'])):
    species, freq, mse_static, T_xi, hwhm  = speciess['static'][k], freqs['static'][k], mse_statics['static'][k], T_xis['static'][k], hwhm['static'][k]
    # Skip egregious static fits
    if mse_static > mse_thresh:
        print("SKIPPING")
        continue
    # Plot
    label = get_label_from_firsts(firsts, species)
    plt.scatter(
        hwhm,
        T_xi,
        color=markers[species][0],
        marker=markers[species][1],
        zorder=1
    )
    plt.legend()
    plt.xlabel(r"PSD HWHM [Hz]")
    plt.ylabel(r"$T_\xi$")



plt.tight_layout()


if save_fig:
    fig_folder = os.path.join(results_folder, "Windowing Comparisons")
    os.makedirs(fig_folder, exist_ok=True)
    fig_fp = os.path.join(fig_folder, rf"{titles['static']} vs {titles['dynamic']}.jpg")
    plt.savefig(fig_fp, dpi=300)

if show_plot:
    plt.show()
