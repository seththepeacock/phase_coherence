#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CompareNDDHO_Nxi_Q.py

Created on Mon Jul 28 13:14:29 2025
@author: pumpkin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# ======================================================
fName= 'NDDHO N_xi Data (PW=True, rho=0.7, A_const=True).xlsx'
jitterA= 0.5
# ======================================================

# use a function to do the averaging from the Excel file
def compAVGs(arrA):
    aMean= np.mean(arrA[CF10_Q==5])
    aSTD= np.std(arrA[CF10_Q==5])
    bMean= np.mean(arrA[CF10_Q==10])
    bSTD= np.std(arrA[CF10_Q==10])
    cMean= np.mean(arrA[CF10_Q==25])
    cSTD= np.std(arrA[CF10_Q==25])
    dMean= np.mean(arrA[CF10_Q==50])
    dSTD= np.std(arrA[CF10_Q==50])
    eMean= np.mean(arrA[CF10_Q==75])
    eSTD= np.std(arrA[CF10_Q==75])
    fMean= np.mean(arrA[CF10_Q==100])
    fSTD= np.std(arrA[CF10_Q==100])
    avgs= [aMean,bMean,cMean,dMean,eMean,fMean]
    stds= [aSTD,bSTD,cSTD,dSTD,eSTD,fSTD]
    return np.array([avgs,stds],float)

# --- read in data
df = pd.read_excel(fName)
Qs= [5,10,25,50,75,100]

# ----------------
# -- CF=10 case
CF10_Q = df[df['Undamped CF'] == 10]['Q'].values
CF10_Nxi = df[df['Undamped CF'] == 10]['N_xi'].values
CF10_NxiSTD = df[df['Undamped CF'] == 10]['N_xi_std'].values
tmp= compAVGs(CF10_Nxi)
CF10_NxiMean= tmp[0,:]
CF10_NxiSTD= tmp[1,:]
# -- CF=100 case
CF100_Q = df[df['Undamped CF'] == 100]['Q'].values
CF100_Nxi = df[df['Undamped CF'] == 100]['N_xi'].values
CF100_NxiSTD = df[df['Undamped CF'] == 100]['N_xi_std'].values
tmp= compAVGs(CF100_Nxi)
CF100_NxiMean= tmp[0,:]
CF100_NxiSTD= tmp[1,:]
# -- CF=1000 case
CF1000_Q = df[df['Undamped CF'] == 1000]['Q'].values
CF1000_Nxi = df[df['Undamped CF'] == 1000]['N_xi'].values
CF1000_NxiSTD = df[df['Undamped CF'] == 1000]['N_xi_std'].values
tmp= compAVGs(CF1000_Nxi)
CF1000_NxiMean= tmp[0,:]
CF1000_NxiSTD= tmp[1,:]


# ----------------
# linear regression fits
tmpFit= np.polyfit(Qs,CF10_NxiMean,1)
tmpFit2= np.poly1d(tmpFit)
fit10P= tmpFit2(Qs)

tmpFit= np.polyfit(Qs,CF100_NxiMean,1)
tmpFit2= np.poly1d(tmpFit)
fit100P= tmpFit2(Qs)

tmpFit= np.polyfit(Qs,CF1000_NxiMean,1)
tmpFit2= np.poly1d(tmpFit)
fit1000P= tmpFit2(Qs)


# =======================================================================
# ==== visualize
plt.close("all")
# ------------------------------------------------
fig1, ax1 = plt.subplots()

# --- CF=1000 case
fig1= plt.plot((CF1000_Q+jitterA*np.random.randn(1,len(CF1000_Q))).flatten(),CF1000_Nxi,
               '.',color='green',alpha=0.2,ms=4,markeredgewidth=2,markerfacecolor='none')
fig1= plt.plot(Qs,CF1000_NxiMean,'*',color='green',ms=6,
               label=r'$f_o={1000}\, Hz$',markeredgewidth=2)
fig1= plt.errorbar(Qs,CF1000_NxiMean,yerr=CF1000_NxiSTD,
                   fmt=' ',color='green')
fig1= plt.plot(Qs,fit1000P,'-',color='green')
# --- CF=100 case
fig1= plt.plot((CF100_Q+jitterA*np.random.randn(1,len(CF100_Q))).flatten(),CF100_Nxi,
               '.',color='orange',alpha=0.2,ms=2,markeredgewidth=4,markerfacecolor='none')
fig1= plt.plot(Qs,CF100_NxiMean,'o',color='orange',ms=4,alpha=0.5,
               label=r'$f_o={100}\, Hz$',markeredgewidth=2)
fig1= plt.errorbar(Qs,CF100_NxiMean,yerr=CF100_NxiSTD,
                   fmt=' ',color='orange')
fig1= plt.plot(Qs,fit100P,'-',color='orange')
# --- CF=10 case
#fig1= plt.plot(CF10_Q,CF10_Nxi,'.',color='blue',
#               alpha=0.2,ms=2,markeredgewidth=4,markerfacecolor='none')
fig1= plt.plot((CF10_Q+jitterA*np.random.randn(1,len(CF10_Q))).flatten(),CF10_Nxi,
               '.',color='blue',alpha=0.2,ms=2,markeredgewidth=4,markerfacecolor='none')
fig1= plt.plot(Qs,CF10_NxiMean,'s',color='blue',ms=4,alpha=0.5,
               label=r'$f_o={10}\, Hz$',markeredgewidth=2)
fig1= plt.errorbar(Qs,CF10_NxiMean,yerr=CF10_NxiSTD,
                   fmt=' ',color='blue')
fig1= plt.plot(Qs,fit10P,'-',color='blue')
# ---
fig1= plt.xlabel(r'NDDHO Quality Factor $Q$',fontsize=12)
fig1= plt.ylabel(r'NDDHO $N_{\xi}$',fontsize=12) 
fig1= plt.grid(True, which="both", ls="-", color='0.9')
fig1= plt.legend()