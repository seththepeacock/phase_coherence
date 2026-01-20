

"""
CompareNxiAcrossSpecies2.py


Notes
o modified from Seth's opening_N_xi_xlsx.py and intended to make a better
figure than compareAllFourSpecies.m

Created on Mon Jun 23 14:01:12 2025
@author: pumpkin CB
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# ======================================================
fName= 'N_xi Fitted Parameters (rho=0.7, PW=True)E.xlsx'
#fName= 'N_xi Fitted Parameters (zeta=0.01, Hann, PW=True).xlsx'
# ----

fact= 1/(np.pi);  # scaling factor for all Nxi {1/pi}

# --- owl ANF tuning curve data (Excel files provided by Christine)
ANFfileOwl= './Owl Qerb_analysis_CK2.xlsx';

ANFfileTokay= './manley99.txt'
# ======================================================


# --- read in data
df = pd.read_excel(fName)
owlANF= pd.read_excel(ANFfileOwl)
toakyANFq10 = np.loadtxt(ANFfileTokay)

# ---------
if (1==0):
    # Here's all the columns in the df
    print("Columns in dataframe:")
    for col in df.columns:
        print(col)
# ---------
# Get peak frequency and corresponding val to plot
anole_freqs = df[df['Species'] == 'Anole']['Frequency'].values
anole_Nxi = df[df['Species'] == 'Anole']['N_xi'].values
owl_freqs = df[df['Species'] == 'Owl']['Frequency'].values
owl_Nxi = df[df['Species'] == 'Owl']['N_xi'].values
human_freqs = df[df['Species'] == 'Human']['Frequency'].values
human_Nxi = df[df['Species'] == 'Human']['N_xi'].values

tokay_freqs = df[df['Species'] == 'Tokay']['Frequency'].values
tokay_Nxi = df[df['Species'] == 'Tokay']['N_xi'].values


# =========================
# deal w/ owl ANF vals
owlANF= owlANF[1:]  # discard first line
freqANFowl= owlANF['CF'].values
QerbANFowl= owlANF['Qerb'].values

# =========================
# deal w/ tokay ANF vals
freqANFtokay= toakyANFq10[:,0]
QerbANFtokay= 6*toakyANFq10[:,1]*1000/np.pi;  # convert to Qerb(?) via Bergevin & Shera 2010



# =======================================================================
# ==== visualize
plt.close("all")
# ------------------------------------------------
fig1, ax1 = plt.subplots()


# --- human
fig1= plt.plot(human_freqs/1000,human_Nxi*fact,'x',color='blue',
               alpha=1,ms=5,markeredgewidth=2,markerfacecolor='none',label=r'Human $N_{\xi}$')

# --- owl
fig1= plt.plot(owl_freqs/1000,owl_Nxi*fact,'d',color='orange',
               alpha=1.0,ms=6,markeredgewidth=2,markerfacecolor='none',label=r'Owl $N_{\xi}$')

fig1= plt.plot(freqANFowl,QerbANFowl,'d',color='orange',
               alpha=0.2,ms=3,lw=1,markerfacecolor='orange',label=r'Owl $Q_{erb}$')

# --- tokay
fig1= plt.plot(tokay_freqs/1000,tokay_Nxi*fact,'+',color='green',
               alpha=1,ms=6,markeredgewidth=2,markerfacecolor='none',label=r'Tokay $N_{\xi}$')
fig1= plt.plot(freqANFtokay,QerbANFtokay,'o',color='green',
               alpha=0.3,ms=2,lw=1,markerfacecolor='green',label=r'Tokay $Q_{erb}$')

# --- anole
fig1= plt.plot(anole_freqs/1000,anole_Nxi*fact,'o',color='k',markeredgewidth=0.5,
               alpha=0.8,ms=6,markerfacecolor='lime',label=r'Anole $N_{\xi}$')
# ---
ax1.set_xscale('log')
ax1.set_yscale('log')
fig1= plt.xlim([0.1,12])
fig1= plt.ylim([1,300])
fig1= plt.xlabel('Frequency [kHz]')
fig1= plt.ylabel(r'SOAE-based $N_{\xi}/\pi$ & ANF-based $Q_{erb}$') 
fig1= plt.grid(True, which="both", ls="-", color='0.9')
fig1= plt.legend()
if (1==0):
    fig1= plt.title('Comparison of Colossogram Time Constants and ANF-derived Tuning') 




