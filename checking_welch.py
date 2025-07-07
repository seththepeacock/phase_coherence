from scipy.signal import welch, get_window
import numpy as np
from phaseco import *
fs = 1000
N = 2**13
nperseg = 2**11
window = 'blackman'
scaling='spectrum'
x = np.random.normal(0, 1, N)
f, Pxx = welch(x, fs=fs, window=window, scaling=scaling, nperseg=nperseg, noverlap=0, detrend=False)
f, Pxx_phaseco = welch(x, fs=fs, win_type=window, scaling=scaling, tauS=nperseg)

plt.scatter(f, Pxx_phaseco, label='phaseco', s=20)
plt.scatter(f, Pxx , label='scipy', s=10)
plt.legend()
plt.show()
