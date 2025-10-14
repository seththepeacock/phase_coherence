from phaseco_funcs import * # For get_coherences()
import pandas as pd # For loading SOAE dataframe
import matplotlib.pyplot as plt


"""
----------------------------------------------
Load waveform
----------------------------------------------
"""

# Load a waveform from the SOAE dataframe

if 1: # Change to 0 to bypass and load your own waveform (call it wf) and sample rate (fs)
    print("Loading dataframe...")
    df = pd.read_parquet('Curated Data.parquet')
    print("Dataframe loaded!")
    print(df.value_counts('species')) # Prints all species options with number of samples
    
    # Set which species you want
    species = 'Anolis'
    
    # Set which number you want from this species (they're not in any particular order)
    sample_idx = 2 
    
    # Crop dataframe to just the desired species and the row you picked above
    df_species = df[df['species'] == species] 
    row = df_species.iloc[sample_idx]
    
    # Get wf, fs, and filename
    wf = row['wf']
    fs = row['sr']
    filename = row['filepath'].split('\\')[-1] # Note the entire filepath is stored in the dataframe to maintain folder information, since some of the other dataframes have data from different labs and years and such in different folders 


"""
----------------------------------------------
Calculate Colossogram
----------------------------------------------
"""


# Set parameters
tauS = 2**11 # ~0.046s for fs=44.1kHz 
tau = tauS / fs # Convert to seconds
# Generate xi array
min_xi = 0.0025 
max_xi = tau 
num_xis = 50
xis = np.linspace(min_xi, max_xi, num_xis)
# Generate frequency array
f = rfftfreq(tauS, 1/fs)

# Set windowing parameters
rho = 1 # Set rho for a dynamically changing Gaussian with FWHM = rho*xi 

"""
Other windowing options:
- Set rho=None for no windowing
- Set rho=None and use a different window via the win_type parameter in get_coherence()
    - win_type can be a string like win_type='hanning' (use one of the strings in the SciPy get_window() docs)
    - win_type can also be an array of window coefficients like win_type=[0, 0.1, 0.2, ... , 0.1, 0]
"""

# Make sure we have a consistent number of segments to take vector strength over since larger xi => less segments in waveform
max_xiS = max(xis) * fs # Calculate max_xi in samples
N_segs = int((len(wf) - tauS) / max_xiS) # Calculate the number of segments which we can get out of the waveform for even the biggest xi value  

# Initialize coherences array
coherences = np.zeros((len(f), len(xis)))

# Calculate coherences
for i, xi in enumerate(xis):
    coherences[:, i] = get_coherence(wf=wf, fs=fs, tauS=tauS, xi=xi, N_segs=N_segs, rho=rho)[1] 
    # This returns (frequency_axis, coherences) so we grab just the coherences with [1] 
    print(f"Coherence for xi {i+1}/{num_xis} complete!")
    

"""
----------------------------------------------
Plot Colossogram
----------------------------------------------
"""

print("Plotting!")
plt.figure(figsize=(12, 6))

# Plotting parameters
cmap = 'magma' # colormap
max_khz = 10 # maximum frequency to plot
title = rf"Colossogram with $\tau={tau:.3f}$" # Plot title

# make meshgrid
xx, yy = np.meshgrid(xis * 1000, f / 1000) # Note we convert xis to ms and f to kHz

# Plot the heatmap
(vmin, vmax) = (0, 1) # min and max values for colorbar
heatmap = plt.pcolormesh(xx, yy, coherences, vmin=vmin, vmax=vmax, cmap=cmap, shading='nearest')

# get and set label for cbar
cbar = plt.colorbar(heatmap)
cbar.set_label("Vector Strength")
# set axes limits/labels and titles
plt.ylim(0, max_khz)
plt.xlabel(rf"$\xi$ [ms]")
plt.ylabel("Frequency [kHz]")
plt.title(title)
# plt.suptitle(filename) # Add filename to plot if we have/want it
plt.tight_layout()
plt.show()





