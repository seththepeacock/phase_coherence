"Figure 4CDE"
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os



# -----------------------------
# function to extract Nsoae and geomean freq from SOAE peak array
def computeNsoae(arr):
    tempARR= arr
    cnt= 0
    M= len(tempARR)  # numb. of peaks
    Mint= M-1   # only consider adjacent peak pairs
    nsoae= np.empty([Mint])
    geofreq= np.empty([Mint])
    # --- only adjacent neighboring pairs
    for nn in range(0,M-1):
        fL= tempARR[nn]*1000 # pick off lowest freq. yet to analyze  [Hz]
        fH= tempARR[nn+1]*1000  # higher freq. of the pair [Hz]
        freqGM= np.sqrt(fH*fL)  # geometric mean freq.
        freqDiff= fH-fL
        geofreq[cnt]= freqGM  # stored geometric mean freq. [Hz]
        nsoae[cnt]= freqGM/freqDiff # stored Nsoae
        cnt= cnt+1
        
    return geofreq, nsoae


"Filepaths"
method_id_N_xi = 'rho=1.0, Flattop, BW=50Hz, Mode=phi'
fp_N_xi= os.path.join('results', 'soae', f'SOAE Results ({method_id_N_xi})', f'SOAE N_xi Fitted Parameters ({method_id_N_xi}).xlsx')
ssheet_fp = os.path.join('results', 'human_peak_picking', 'Mags vs C_xi_M Picked Peaks.xlsx')

"Build arrays from spreadsheet"
# Load Excel file into two DataFrames
df_mags = pd.read_excel(ssheet_fp, sheet_name='Mags')
df_C = pd.read_excel(ssheet_fp, sheet_name='C_xi_M')
# Get all the filenames
wf_fns = np.array(df_mags.columns.to_list())
N_wfs = len(wf_fns)
# Initialize final arrays to build
all_geofreqs_mags = []
all_geofreqs_C = []
all_Nsoae_mags = []
all_Nsoae_C = []
all_freq_ratios_C = []
each_geofreqs_C = np.empty(shape=(N_wfs), dtype=list)
each_Nsoae_C = np.empty(shape=(N_wfs), dtype=list)
for k, wf_fn in enumerate(wf_fns):
    # Get freqs
    peak_freqs_mags = df_mags[wf_fn].dropna().to_numpy()
    peak_freqs_C = df_C[wf_fn].dropna().to_numpy()
    # Sort by increasing frequency
    peak_freqs_mags = np.sort(peak_freqs_mags)
    peak_freqs_C = np.sort(peak_freqs_C)
    # Compute Nsoae and geometric mean freqs
    geofreqs_mags, Nsoae_mags = computeNsoae(peak_freqs_mags)
    geofreqs_C, Nsoae_C = computeNsoae(peak_freqs_C)
    # Compute freq ratios (higher/lower)
    ratios_C = peak_freqs_C[1:] / peak_freqs_C[:-1]
    # Add to overall lists
    all_geofreqs_mags.extend(geofreqs_mags)
    all_geofreqs_C.extend(geofreqs_C)
    all_Nsoae_mags.extend(Nsoae_mags)
    all_Nsoae_C.extend(Nsoae_C)
    all_freq_ratios_C.extend(ratios_C)
    each_geofreqs_C[k] = geofreqs_C
    each_Nsoae_C[k] = Nsoae_C
# Convert to ndarray
all_geofreqs_mags = np.array(all_geofreqs_mags)
all_geofreqs_C = np.array(all_geofreqs_C)
all_Nsoae_mags = np.array(all_Nsoae_mags)
all_Nsoae_C = np.array(all_Nsoae_C)
all_freq_ratios_C = np.array(all_freq_ratios_C)
# Compute freq diffs
freq_diffs_C= all_geofreqs_C/all_Nsoae_C



"Chris' original plotting code"
# ======================================================
binsN= 37    # number of bins for Nsoae histogram
N = 1000   # number of times to bootstrap re xi-adjusted power law fit
ratioBinCNT= 200   # numb. of bins for freq. ratio histogram
fact= 1/(8*np.pi);  # scaling factor for all Nxi {1/4pi}
show_plots = 1
save_plots = 1
# ======================================================


# --- send some vals to screen for ref
print(f'# of mag-avgd peaks found = {str(len(all_geofreqs_mags))} ')
print(f'# of C_xi^M peaks found = {str(len(all_geofreqs_C))} ')
percent_inc= 100*(len(all_geofreqs_C)-len(all_geofreqs_mags) )/len(all_geofreqs_mags)
print(f'% increase in # of peaks found = {percent_inc:.2f}% ')




# =======================================================================
# ==== Shera 2003 Nsoae vals
# (previously computed for CB 2012 HR paper; vals. extracted from older analysis
# code: ...//Users/pumpkin/Dropbox/Collaborations B/Oldenburg/Analysis/
#    /2014 MoH owl/Analysis/2012 HR human analysis codesModelPredictN2.m

Shera03n=np.array([1.1686,1.5560,1.9393,2.2290,2.5734,2.9629,3.4175,3.9169,
    4.5260,5.2217,6.0284,6.9145,7.9564,9.1602,10.4685,12.1349,13.7273,
   16.1044,18.6965,21.4918,24.6847,28.2800,32.4947,37.3661,43.2892])
Shera03cnt=np.array([4.2852,6.3118,8.2251,5.3621,11.4314,7.3887,18.3734,10.4826,
   15.4564,12.4530,19.4773,16.6705,21.4195,19.3430,33.3050,47.6042,
   47.6622,69.4051,67.4692,58.4829,24.4698,16.6070,7.3961,2.3141,1.3894])




# ------------------------------------------------
# ------------------------------------------------
# --- grab a subset of Nsoae vals above a certain min. val.
# (a la Shera 2003 analysis to "the peak of the distribution [...]
# To reduce bias in the fit"
# NOTE: unless there is a good reason to utilize this set of #s, seems 
# wiser to avoid
NsoaeMIN= 0.75*np.mean(all_Nsoae_C)
gmTthresh= all_geofreqs_C[all_Nsoae_C>NsoaeMIN]
nTthresh= all_Nsoae_C[all_Nsoae_C>NsoaeMIN]


# =======================================================================
# ==== (single) Power Law fit (a la Shera 2003 approach)
# --
def powlawFunc(f,A,c):
    return A*(f**c)

# === do the fits via the  scipy blackbox
# --- standard spectral averaging peak freqs
paramsS, covarS = curve_fit(powlawFunc,all_geofreqs_mags,all_Nsoae_mags)
AfitS, cfitS = paramsS
# --- xi-adjusted temporal averaging peak freqs
paramsT, covarT = curve_fit(powlawFunc,all_geofreqs_C,all_Nsoae_C)
AfitT, cfitT = paramsT
# --- Shera03-style "thresholded" vals.
paramsTthresh, covarTthresh = curve_fit(powlawFunc,gmTthresh,nTthresh)
AfitTthresh, cfitTthresh = paramsTthresh

# ==== compute fit curves to plot
fitF= np.logspace(np.log10(300),np.log10(max(all_geofreqs_C)), 50)
fitNS= powlawFunc(fitF,AfitS, cfitS)
fitNT= powlawFunc(fitF,AfitT, cfitT)
fitNTthresh= powlawFunc(fitF,AfitTthresh, cfitTthresh)
# --- Shera's fit vals (via Table I in 2003 paper)
fitFshera= np.logspace(np.log10(550),np.log10(7000), 50)
fitShera= powlawFunc(fitFshera/1000,13.7,0.31)


# =======================================================================
# ==== (bootstrapped I) Power Law fit --> amongst pooled datas
# [adapting bits from my EXstatBootstrap2.py code]

nbs= len(all_Nsoae_C)
#nbs= int(np.round(0.9*len(nTall)))
indx= np.arange(nbs)  # create array index 

for n in range(0,N):
    # === grab a resampled array 
    indxBS= np.random.choice(indx,replace=1,size=nbs)
    xT= all_geofreqs_C[indxBS] # 
    yT= all_Nsoae_C[indxBS] # 
    pTtemp, covarTtemp = curve_fit(powlawFunc,xT,yT,maxfev=2000)
    AfitTtemp, cfitTtemp = pTtemp
    fitTtemp= powlawFunc(fitF,AfitTtemp, cfitTtemp)
    # -- store away fits
    if n==0:
        yDf= fitTtemp  # kludgy
    else:
        yDf= np.vstack((yDf,fitTtemp)) 

# === determine mean and SD (& SE) from bootstrapped loess      
yDfitM= np.mean(yDf,0)  # mean loess fit
yDfitSD= np.std(yDf,axis=0)  # standard deviation
SE= yDfitSD/np.sqrt(N)  # " standard error
CI= 2*SE # confidence intervals (CIs) as +/-95% (i.e., 2*SE)


# =======================================================================
# ==== (bootstrapped II) Power Law fit --> amongst subjects
# Bootstrap resampling number (equal to number of subjects, 12)
nbs2= len(wf_fns)
# NOTE: can make this smaller if you want fewer resmapled subjects
indx2= np.arange(nbs2)  # create array index 
# --
for mm in range(0,nbs2-1):
    # Get a reproducible rng
    rng = np.random.default_rng(seed=24) 
    # === grab a resampled list of subject
    indxBS2 = rng.choice(indx2, size=nbs2, replace=True)
    xT2= []
    yT2= []
    # --- compile #s together via for loop (better way to do??)
    for nn in range(0,nbs2-1):
        # Replaced Chris' code 
        # xT2= np.concatenate((xT2,eval('gmT'+str(indxBS2[nn]+1))))
        # yT2= np.concatenate((yT2,eval('nT'+str(indxBS2[nn]+1))))
        # with the following equivalent code with my variables
        xT2.extend(each_geofreqs_C[indxBS2[nn]])
        yT2.extend(each_Nsoae_C[indxBS2[nn]])
    xT2 = np.array(xT2)
    yT2 = np.array(yT2)
    
    pTtemp2, covarTtemp2 = curve_fit(powlawFunc,xT2,yT2)
    AfitTtemp2, cfitTtemp2 = pTtemp2
    fitTtemp2= powlawFunc(fitF,AfitTtemp2, cfitTtemp2)
    # -- store away fits
    if mm==0:
        yDf2= fitTtemp2  # kludgy
    else:
        yDf2= np.vstack((yDf2,fitTtemp2))

# === determine mean and SD (& SE) from bootstrapped loess      
yDfitM2= np.mean(yDf2,0)  # mean loess fit
yDfitSD2= np.std(yDf2,axis=0)  # standard deviation
SE2= yDfitSD2/np.sqrt(N)  # " standard error
CI2= 2*SE2 # confidence intervals (CIs) as +/-95% (i.e., 2*SE)


# =======================================================================
# ==== create histogram of Nsoae vals (akin to Fig.2 of Shera 2003)
log_bins = np.logspace(np.log10(all_Nsoae_C.min()), np.log10(all_Nsoae_C.max()),binsN) 
log_binsShera = np.logspace(np.log10(Shera03n.min()), np.log10(Shera03n.max()),len(Shera03n)) 
bin_centers = (log_bins[:-1] + log_bins[1:]) / 2
# --
countsT, binsREP = np.histogram(all_Nsoae_C,log_bins)
countsS, binsREP = np.histogram(all_Nsoae_mags,log_bins)



# =======================================================================
# -- determine average Nsoae vals. in oct.-wide bins (starting at 0.3 kHz)

freqOct= [300,600,1200,2400,4800,9600]  # oct. range bounds

avgN= []
stdN= []
serrN= []
avgGM= []
for pp in range(0,len(freqOct)-1):
    tmp0= len(all_Nsoae_C[np.where(np.logical_and(all_geofreqs_C>=freqOct[pp],all_geofreqs_C<freqOct[pp+1]))])
    tmp1= np.mean(all_Nsoae_C[np.where(np.logical_and(all_geofreqs_C>=freqOct[pp],all_geofreqs_C<freqOct[pp+1]))])
    tmp2= np.std(all_Nsoae_C[np.where(np.logical_and(all_geofreqs_C>=freqOct[pp],all_geofreqs_C<freqOct[pp+1]))])
    tmp3= np.mean(all_geofreqs_C[np.where(np.logical_and(all_geofreqs_C>=freqOct[pp],all_geofreqs_C<freqOct[pp+1]))])
    avgN.append(tmp1)
    stdN.append(tmp2)
    serrN.append(tmp2/np.sqrt(tmp0))
    avgGM.append(tmp3)
    
    
avgN= np.array(avgN)  # kludge
stdN= np.array(stdN)
serrN= np.array(serrN)
avgGM= np.array(avgGM)

#val1= np.mean(nTall[np.where(np.logical_and(gmTall>=300,gmTall<600))])
#std1= np.std(nTall[np.where(np.logical_and(gmTall>=300,gmTall<600))])

# --- also create the assoc. vers. of the freq. diff. from Shera's 2003
# power law fit


fDiffSheraPLoct= np.log2(fitFshera/fitShera)


# =======================================================================
# read in human Nxi vals.


# --- read in data
df = pd.read_excel(fp_N_xi)
human_freqs = df[df['Species'] == 'Human']['Frequency'].values
human_Nxi = df[df['Species'] == 'Human']['N_xi'].values

# =======================================================================
# ==== visualize
plt.close("all")
# ------------------------------------------------
# Fig.1 - Nsoae vs freq. (along with various fits/comps)
fig1, ax1 = plt.subplots()

# --- plot all compiled points
#fig1= plt.plot(gmSall/1000,nSall,'x',color='r',alpha=0.3,ms=5,markerfacecolor='none',label='Spectral Avg.')
fig1= plt.scatter(all_geofreqs_mags/1000,all_Nsoae_mags,marker='x', color='darkorange',s=24,alpha=0.5,linewidths=2,label='Spectral Avg.')
fig1= plt.plot(all_geofreqs_C/1000,all_Nsoae_C,'s',color='royalblue',alpha=0.4,ms=4,markerfacecolor='royalblue',
               markeredgecolor='none',label='Temporal Avg.')
# --- plot power law fits
#fig1= plt.plot(fitF/1000,fitNS,'r--',lw=1,label='Spectral Avg.')
#fig1= plt.plot(fitF/1000,fitNT,'k-',lw=2,alpha=0.3,label='Power fit (all)')
#fig1= plt.plot(fitF/1000,fitNTthresh,'c-',lw=3,label='Thresholded')
fig1= plt.plot(fitFshera/1000,fitShera,'-.',lw=2,color='black',label='Shera (2003)')
# --- plot bootstrapped power law fits (all data pooled for bootstrap)
fig1= plt.plot(fitF/1000,yDfitM,'-',color='royalblue',lw=2,label='Bootstrapped power law fit')
fig1= plt.fill_between(fitF/1000, (yDfitM-yDfitSD), (yDfitM+yDfitSD), 
                       color='royalblue',alpha=0.1)   
# - Subj.-pooled bootstrapped fit?
if (1==0):
    fig1= plt.plot(fitF/1000,yDfitM2,'-',color='magenta',lw=2,label='Subj.-pooled bootstrapped fit')
    fig1= plt.fill_between(fitF/1000, (yDfitM2-yDfitSD2), (yDfitM2+yDfitSD2), 
                           color='magenta',alpha=0.1)
# --- plot mean oct-wide bin vals?
if (1==1):
    #fig1= plt.plot(avgGM/1000,avgN,'^',color='magenta')
    # --- plotting w/ standard error
    fig1= plt.errorbar(avgGM/1000,avgN, yerr=serrN, fmt='d',capsize=4,lw=1,
                       color='blue',alpha=0.7,label='octave-wide averages')
    

# --- bookeeping
ax1.set_xscale('log')
ax1.set_yscale('log')
fig1= plt.xlim([0.27,15])
fig1= plt.ylim([1,300])
fig1= plt.xlabel('Frequency [kHz]',fontsize=12)
fig1= plt.ylabel(r"$N_{SOAE}$",fontsize=12) 
#fig1= plt.title('Human Nsoae: Spec-avg. (triangle) vs xi-adjust. Temp.-avg. (dot)') 
fig1= plt.grid(True, which="both", ls="-", color='0.9')
ax1.set_axisbelow(True)
fig1= plt.legend()
if show_plots:
    plt.show()

# ------------------------------------------------
# Fig.2 - Histogram of Nsoae
if (1==1):
    fig2, ax2 = plt.subplots()
    # ---
    plt.bar(bin_centers,countsT/np.sum(countsT),alpha=0.6,width=np.diff(log_bins),ec="k", align="edge",label='Temporal avg.')
    plt.bar(bin_centers,countsS/np.sum(countsS),alpha=0.5,width=np.diff(log_bins),ec="k",align="edge",label='Spectral avg.')
    # ----
    plt.plot(Shera03n,Shera03cnt/np.sum(Shera03cnt),'k--',lw=2.5,alpha=0.5,label='Shera (2003)')
    plt.xscale('log') # Still useful to ensure proper display of log ticks
    plt.xlabel(r"$N_{SOAE}$",fontsize=12)
    plt.ylabel("Probability",fontsize=12)
    #plt.title(r'Comparison of $N_{SOAE}$')
    plt.legend()
    if show_plots:
        plt.show()

# ------------------------------------------------
# Fig.3 - Freq. spacings vs GM freq.
# Purpose: Provide visualization to compare to SSOAE spacings as shown in
# Fig.2 of Bell and Jedrzejczak (2017)
if (1==1):
    fig3, ax3 = plt.subplots(2,1)
    ax3[0].plot(all_geofreqs_C/1000,np.log2(all_freq_ratios_C),'bo',ms=3,alpha=0.4)
    ax3[0].set_ylabel("Freq. Pair Spacing [oct]",fontsize=12)
    ax3[0].set_xlabel("Geometric Mean Frequency [kHz]",fontsize=12)
    ax3[0].grid()
    # --
    n, bins, patches = plt.hist(all_freq_ratios_C,bins=ratioBinCNT,label='SOAE spacing',
                                color='orange',edgecolor='black',alpha=0.5)
    maxCNT = n.max()
    ax3[1].plot([1.06,1.06],[0,maxCNT+2],'r--',lw=3,label='SSOAE spacing (B&J2017)')
    ax3[1].set_xlim([1,1.5])
    ax3[1].set_xlabel("Freq. Ratio",fontsize=12)
    ax3[1].set_ylabel("Counts",fontsize=12)
    ax3[1].grid(True, which="both", ls="-", color='0.7')
    ax3[1].set_axisbelow(True)
    ax3[1].legend(loc="upper right")
    
    plt.tight_layout()
    if show_plots:
        plt.show()
    
    
# ------------------------------------------------
# Fig.4 - Compare human Nsoae versus Nxi
# Purpose: 
if (1==1):
    fig4, ax4 = plt.subplots()
    # ---
    fig4= plt.plot(all_geofreqs_C/1000,all_Nsoae_C,'s',color='royalblue',alpha=0.4,ms=4,markerfacecolor='royalblue',
                   markeredgecolor='none',label=r"$N_{SOAE}$")
    fig4= plt.plot(human_freqs/1000,human_Nxi*fact,'x',color='red',
                   alpha=1,ms=5,markeredgewidth=2,markerfacecolor='none',label=r'$N_{\xi}$')

    ax4.set_xscale('log')
    ax4.set_yscale('log')
    fig4= plt.xlim([0.27,15])
    fig4= plt.ylim([1,300])
    fig4= plt.xlabel('Frequency [kHz]',fontsize=12)
    fig4= plt.ylabel(r"Human $N$ values",fontsize=12) 
    #fig1= plt.title('Human Nsoae: Spec-avg. (triangle) vs xi-adjust. Temp.-avg. (dot)') 
    fig4= plt.grid(True, which="both", ls="-", color='0.9')
    ax4.set_axisbelow(True)
    fig4= plt.legend()
    if show_plots:
        plt.show()