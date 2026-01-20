import phaseco as pc
import numpy as np
from helper_funcs import *


A = 10
wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(species='Human', wf_idx=0)

rng = np.random.default_rng()

wf_noisy = wf + A*rng.normal(loc=0, scale=1)
xis={'xi_min_s':}
pc.get_colossogram(wf, fs, )