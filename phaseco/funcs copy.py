import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq, fftshift
import pywt
import warnings
from tqdm import tqdm


"""
HELPER FUNCTIONS
"""



def get_avg_vector(phases):
  """ Returns magnitude, phase of vector made by averaging over unit vectors with angles given by input phases
  
  Parameters
  ------------
      phase_diffs: array
        array of phase differences
  """
  # get the sin and cos of the phase diffs, and average over the window pairs
  xx= np.mean(np.sin(phases),axis=0)
  yy= np.mean(np.cos(phases),axis=0)
  
  # finally, output the averaged vector's vector strength and angle with x axis (each a 1D array along the frequency axis) 
  return np.sqrt(xx**2 + yy**2), np.arctan2(yy, xx)

def spectral_filter(wf, fs, cutoff_freq, type='hp'):
  """ Filters waveform by zeroing out frequencies above/below cutoff frequency
  
  Parameters
  ------------
      wf: array
        waveform input array
      fs: int
        sample rate of waveform
      cutoff_freq: float
        cutoff frequency for filtering
      type: str, Optional
        Either 'hp' for high-pass or 'lp' for low-pass
  """
  fft_coefficients = np.fft.rfft(wf)
  frequencies = np.fft.rfftfreq(len(wf), d=1/fs)

  if type == 'hp':
    # Zero out coefficients from 0 Hz to cutoff_frequency Hz
    fft_coefficients[frequencies <= cutoff_freq] = 0
  elif type == 'lp':
    # Zero out coefficients from cutoff_frequency Hz to Nyquist frequency
    fft_coefficients[frequencies >= cutoff_freq] = 0

  # Compute the inverse real-valued FFT (irfft)
  filtered_wf = np.fft.irfft(fft_coefficients, n=len(wf))  # Ensure output length matches input
  
  return filtered_wf

def get_sigmaS(fwhm, fs):
  """ Gets sigmaS for (SciPy get_window) as a function of what you want the Gaussian FWHM to be (in seconds)
  
  Parameters
  ------------
      fwhm: float
        Desired FWHM of the Gaussian window (in seconds)
      fs: int
        sample rate
  """
  sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
  sigmaS = sigma * fs
  return sigmaS

def tau_or_tauS(fs, tau, tauS):
  if tauS is None:
    if tau is None:
      raise ValueError("You must input either tau or tauS!")
    else: 
      tauS = tau*fs
  else: 
    if tau is not None: # Here tauS is not None, and neither is tau...
      raise ValueError("You gave both tau and tauS... which do we use?")
    else:
      tau = tauS/fs
  return tau, int(tauS)



"""
PRIMARY FUNCTIONS
"""



def get_stft(wf, fs, tau=None, tauS=None, seg_spacing=None, rho=None, win_type='boxcar', N_segs=None, cutoff_freq=None, fftshift_segs=False, return_dict=False):
  """ Returns the segmented fft and associated freq ax of the given waveform

  Parameters
  ------------
      wf: array
        waveform input array
      fs: int
        sample rate of waveform
      tau: float
        length (in time) of each window
      seg_spacing: float
        length (in time) between the start of successive segments
      rho: double, Optional
        Applies a Gaussian window with whose FWHM is rho*xi
      win_type: String, Optional
        Window to apply before the FFT
      N_segs: int, Optional
        If this isn't passed, then just get the maximum number of segments of the given size
      cutoff_freq: double, Optional
        HPFs the total waveform by zeroing out frequencies above this frequency
      fftshift_segs: bool, Optional
        Shifts each time-domain window of the fft with fftshift() to center your window in time and make it zero-phase (no effect on coherence)
      return_dict: bool, Optional
        Returns a dict with:
        d["freq_ax"] = freq_ax
        d["stft"] = stft
        d["segmented_wf"] = segmented_wf
        d["seg_start_indices"] = seg_start_indices
  """
  # Make sure we got tau or tauS
  tau, tauS = tau_or_tauS(fs, tau, tauS)
  
  # HPF the waveform
  if cutoff_freq is not None:
    wf = spectral_filter(wf=wf, fs=fs, cutoff_freq=cutoff_freq, type='hp')
    
  # if you didn't pass in xi we'll assume you want no overlap - each new window starts at the end of the last!
  if seg_spacing is None:
    seg_spacing=tau

  # and the number of samples to shift
  seg_spacingS = int(seg_spacing*fs)

  # get sample_spacing
  delta_t = 1/fs

  # Girst, get the last index of the waveform
  final_wf_index = len(wf) - 1
    
  # next, we get what we would be the largest potential seg_start_index
  latest_potential_seg_start_index = final_wf_index - (tauS - 1) # start at the final_wf_index. we need to collect nperseg points. this final index is our first one, and then we need tauS - 1 more. 
  seg_start_indices = np.arange(0, latest_potential_seg_start_index + 1, seg_spacingS)
    # the + 1 here is because np.arange won't ever include the "stop" argument in the output array... but it could include (stop - 1) which is just our final_seg_start_index!

  # if number of segments is passed in, we make sure it's less than the length of seg_start_indices
  if N_segs is not None:
    max_N_segs = len(seg_start_indices)
    if N_segs > max_N_segs:
      raise Exception(f"That's more segments than we can manage - you want {N_segs}, but we can only do {max_N_segs}!")
  else:
    # if no N_segs is passed in, we'll just use the max number of segments
    N_segs = len(seg_start_indices)
    if N_segs != int((len(wf) - tauS) / seg_spacingS) + 1:
      print(f'Hm THATS WEIRD - N_SEGS METHOD 1 gives {N_segs} while other method gives {int((len(wf) - tauS) / seg_spacingS) + 1}')
    # this is equivalent to int((len(wf) - tauS) / xiS) + 1
  
  # Initialize segmented waveform matrix
  segmented_wf = np.zeros((N_segs, tauS))
  
  # Check how win_type has been passed in and get window accordingly
  if isinstance(win_type, str) or (isinstance(win_type, tuple) and isinstance(win_type[0], str)):
    # Get window function (boxcar is no window)
    win = get_window(win_type, tauS)
    if rho is not None:
      sigmaS = get_sigmaS(fwhm=rho*seg_spacing, fs=fs) # rho is the proportion of xi which we want the FWHM to be
      gaussian = get_window(('gauss', sigmaS), tauS)
      win = gaussian * win # In normal use, win will just be a boxcar, but... 
      if win_type != 'boxcar': # Throw a warning if you're going to double window
        warnings.warn(f"You're double-windowing with both a Gaussian and a {win_type}!")
        
  else: # Allow for passing in a custom window
    win = win_type
    if len(win) != tauS:
      raise Exception(f"Your custom window must be the same length as tauS ({tauS})!")
  
  # Check if we should be windowing or if unnecessary because they're all ones
  do_windowing = (np.any(win != 1))
  
  for k in range(N_segs):
    seg_start = seg_start_indices[k]
    seg_end = seg_start + tauS
    # grab the waveform in this segment
    seg = wf[seg_start:seg_end]
    if do_windowing:
      seg = seg * win
    if fftshift_segs: # optionally swap the halves of the waveform to effectively center it in time
      seg = fftshift(seg)      

    segmented_wf[k, :] = seg

  # Now we do the ffts!

  # Get frequency axis 
  freq_ax = rfftfreq(tauS, delta_t)
  N_bins = len(freq_ax)
  
  # initialize segmented fft array
  stft = np.zeros((N_segs, N_bins), dtype=complex)
  
  # get ffts
  for k in range(N_segs):
    stft[k, :] = rfft(segmented_wf[k, :])

  if return_dict:
    
    return {
      "freq_ax" : freq_ax,  
      "stft" : stft,
      "seg_start_indices" : seg_start_indices,
      "segmented_wf" : segmented_wf,
      "seg_spacing" : seg_spacing,
      "fs" : fs
      }
    
  else: 
    return freq_ax, stft
  
  
def get_coherence(wf, fs, tau=None, tauS=None, seg_spacing=None, xi=None, rho=None, win_type='boxcar', N_segs=None, reuse_stft=None, ref_type="next_seg", cutoff_freq=None, freq_bin_hop=1, return_dict=False):
  """ Gets the phase coherence of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      fs:
        sample rate of waveform
      tau: float
        length (in time) of each segment
      seg_spacing: float
        length (in time) between the start of successive segments
      xi: float
        length (in time) between phase references
      sigma: double, Optional
        Applies a Gaussian window with this standard deviation (in time) to each segment (win_type must be 'boxcar')
      win_type: String, Optional
        Window to apply before the FFT
      N_segs: int, Optional
        If this isn't passed, then just get the maximum number of segments of the given size
      reuse_stft: tuple, Optional
        If you want to avoid recalculating the STFT, pass it in here as a dictionary
      ref_type: str, Optional
        Either "next_seg" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
      cutoff_freq: double, Optional
        HPFs the total waveform by zeroing out frequencies above this frequency
      freq_bin_hop: int, Optional
        How many bins over to reference phase against for next_freq
      return_dict: bool, Optional
        Defaults to only returning the coherence; if this is enabled, then a dictionary is returned with keys:
        d["freq_ax"] = freq_ax
        d["coherence"] = coherence
        d["phases"] = phases
        d["phase_diffs"] = phase_diffs
        d["<|phase_diffs|>"] = avg_abs_phase_diffs
        d["avg_vector_angle"] = avg_vector_angle
        d["N_segs"] = N_segs
        d["stft"] = stft
  """

  
  # if nothing was passed into reuse_stft then we need to recalculate it
  if reuse_stft is None:
    stft_dict = get_stft(wf=wf, fs=fs, tau=tau, tauS=tauS, seg_spacing=seg_spacing, N_segs=N_segs, rho=rho, win_type=win_type, cutoff_freq=cutoff_freq, return_dict=True)
  else:
    stft_dict = reuse_stft

  # Retrieve necessary items from dictionary
  freq_ax = stft_dict["freq_ax"]
  stft = stft_dict["stft"]
  
  # Make sure these are valid
  if stft.shape[1] != freq_ax.shape[0]:
    raise Exception("STFT and frequency axis don't match!")
  # Handle errors from incorrect passing of tau/tauS in case they weren't caught in OG stft calculation
  tau, tauS = tau_or_tauS(fs, tau, tauS)
    
  
  # calculate necessary params from the stft
  N_segs, N_bins = np.shape(stft)

  # get phases
  phases=np.angle(stft)
  
  # we can reference each phase against the phase of the same frequency in the next window:
  if ref_type == "next_seg":
    # Make sure we can reference this xi in this STFT; xi should be an integer number of segs away
    xi_in_num_segs = xi/seg_spacing
    if int(xi_in_num_segs) - (xi_in_num_segs) > 1e-12:
          raise Exception(f"This xi corresponds to going {xi_in_num_segs} segs away, needs to be an integer! Change xi={xi} or seg_spacing={seg_spacing}.")
    
    # unwrap phases along time window axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=0)
    
    # initialize array for phase diffs; we won't be able to get it for the final window
    phase_diffs = np.zeros((N_segs - 1, N_bins))
    
    # calc phase diffs
    for seg in range(N_segs - 1):
      # take the difference between the phases in this current segment and the one xi seconds away
      phase_diffs[seg] = phases[seg + xi_in_num_segs] - phases[seg]
    
    coherence, avg_vector_angle = get_avg_vector(phase_diffs)
    
  # or we can reference it against the phase of the next frequency in the same window:
  elif ref_type == "next_freq":
    # unwrap phases along the frequency bin axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=1)
      
    # initialize array for phase diffs; -freq_bin_hop is because we won't be able to get it for the #(freq_bin_hop) freqs
    phase_diffs = np.zeros((N_segs, N_bins - freq_bin_hop))
    # we'll also need to take the last #(freq_bin_hop) bins off the freq_ax
    freq_ax = freq_ax[0:-freq_bin_hop]
    
    # calc phase diffs
    for seg in range(N_segs):
      for freq_bin in range(N_bins - freq_bin_hop):
        phase_diffs[seg, freq_bin] = phases[seg, freq_bin + freq_bin_hop] - phases[seg, freq_bin]
    
    # get final coherence
    coherence, avg_phase_diff = get_avg_vector(phase_diffs)
    
    # Since this references each frequency bin to its adjacent neighbor, we'll plot them w.r.t. the average frequency 
        # this corresponds to shifting everything over half a bin width (bin width is 1/tau)
    bin_width = freq_ax[1] - freq_ax[0]
    freq_ax = freq_ax + (bin_width / 2)
    
  
  # or we can reference it against the phase of both the lower and higher frequencies in the same window
  elif ref_type == "both_freqs":
    # unwrap phases along the frequency bin axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=1)
    
    # initialize arrays
      # even though we only lose ONE freq point with lower and one with higher, we want to get all the points we can get from BOTH so we do - 2
    pd_low = np.zeros((N_segs, N_bins - 2))
    pd_high = np.zeros((N_segs, N_bins - 2))
    # take the first and last bin off the freq ax
    freq_ax = freq_ax[1:-1]
    
    # calc phase diffs
    for seg in range(N_segs):
      for freq_bin in range(1, N_bins - 1):
        # the - 1 is so that we start our phase_diffs arrays at 0 and put in N_bins-2 points. 
        # These will correspond to our new frequency axis.
        pd_low[seg, freq_bin - 1] = phases[seg, freq_bin] - phases[seg, freq_bin - 1]
        pd_high[seg, freq_bin - 1] = phases[seg, freq_bin + 1] - phases[seg, freq_bin]
    coherence_low, _ = get_avg_vector(pd_low)
    coherence_high, _ = get_avg_vector(pd_high)
    # average the coherences you would get from either of these
    coherence = (coherence_low + coherence_high)/2
    # set the phase diffs to one of these (could've also been pd_high)
    phase_diffs = pd_low
    
  else:
    raise Exception("You didn't input a valid ref_type!")
  
  
  
  # get <|phase diffs|> (note we're unwrapping w.r.t. the frequency axis)
  avg_abs_phase_diffs = np.mean(np.abs(phase_diffs), 0)
  
  if not return_dict:
    return freq_ax, coherence
  
  else: # Return full dictionary
    return {
      "coherence": coherence,
      "phases" : phases,
      "phase_diffs" : phase_diffs,
      "<|phase_diffs|>" : avg_abs_phase_diffs,
      "avg_vector_angle" : avg_vector_angle,
      "N_segs" : N_segs,
      "freq_ax" : freq_ax,
      "stft" : stft,
      "stft_dict" : stft_dict
    }

def get_coherences(wf, fs, tauS, min_xi, max_xi, delta_xi, rho, const_Npd=False, dense_stft=False, global_max_xi=None):
  # Calculate xi array and N_bins
  num_xis = int((max_xi - min_xi) / delta_xi) + 1
  xis = np.linspace(min_xi, max_xi, num_xis)
  N_bins = len(np.array(rfftfreq(tauS, 1/fs)))
  # Initialize coherences array
  coherences = np.zeros((N_bins, len(xis)))
  
  # If we're going to hold the number of PDs constant, set the max xi that will determine this number 
  # (either max xi in htis colossogram or a global one so it's constant across all colossograms in comparison)
  if const_Npd:
    if global_max_xi is None:
      global_max_xi = max_xi 
    global_max_xiS = global_max_xi * fs
    
  

  if dense_stft:
    # Set the STFT seg_spacing according to the minimum xi in this colossogram
    seg_spacing = min_xi
    seg_spacingS = seg_spacing * fs
    # Make sure all xis correspond to an integer*seg_spacing; since seg_spacing=min_xi, this is equivalent to making sure seg_spacing=delta_xi
    if seg_spacing != delta_xi:
        raise Exception("seg_spacing(=min_xi) must be equal to delta_xi or else our xi values won't correspond to referencing to an integer number of segs away!")
    
    # Loop through xis
    for i, xi in enumerate(tqdm(xis)):
      xiS = xi * fs # Calculate current xi in samples
      # Get window
      win_type = 'boxcar' if rho is None else ("gaussian", get_sigmaS(fwhm=rho*xi, fs=fs))
      # Get N_pd
      if const_Npd:
        # There are int((len(wf)-tauS)/seg_spacingS)+1 full tau-segments. But the last xiS/seg_spacingS (=1 in non-dense_stft-case) ones won't have a reference.
        N_pd = int((len(wf) - tauS) / seg_spacingS) + 1 - int(global_max_xiS/seg_spacingS) 
      else:
        # Same as above, except just use the current xiS rather than the max xiS
        N_pd = int((len(wf) - tauS) / seg_spacingS) + 1 - int(xiS/seg_spacingS)
      # Get dense STFT (seg spacing according to minimum xi in this Colossogram)
      f, stft = get_stft(wf=wf, fs=fs, tauS=tauS, seg_spacing=seg_spacing, win_type=win_type)
      
      
      # initialize array for phase diffs
      phase_diffs = np.zeros((N_pd, N_bins))
      
      # get phases
      phases=np.angle(stft)
      
      # calc phase diffs
      for seg in range(N_pd):
        # take the difference between the phases in this current window and the one xi away
        # First, make sure this all makes sense
        if int(xiS/seg_spacingS) - (xiS/seg_spacingS) > 1e-12:
          raise Exception("xiS/seg_spacingS = " + str(xiS/seg_spacingS) + " -- this should be an integer!")
        phase_diffs[seg] = phases[seg + int(xiS/seg_spacingS)] - phases[seg]
      
      coherences[:, i] = get_avg_vector(phase_diffs)[0]
    
  else: # This is the classic way, where seg_spacing = xi
    # (note the number of PDs changes drastically from xi to xi, and if we make it constant, we have to clip very low if the max xi is large)
    if const_Npd:
      # There are int((len(wf)-tauS)/seg_spacingS)+1 full tau-segments. But the last xiS/seg_spacingS (=1 here since xiS=seg_spacingS) ones won't have a reference.
      # ... so there are int((len(wf)-tauS)/seg_spacingS) + 1 - 1 full tau segments (THAT HAVE A REFERENCE!)
      # And since we're holding N_pd constant by finding the minimum it will ever be, we set seg_SpacingS = global_max_xiS
      N_pd = int((len(wf) - tauS) / global_max_xiS) # Total number of PDs = number of full tau segments with a reference
      N_segs = N_pd + 1 # but there's xiS/seg_spacingS = 1 extra segment since the last one needs a reference too
    else: 
      # If we're not holding it constant, we just fill er up with as many segs as can fit!
      N_segs = None
    for i, xi in enumerate(tqdm(xis)):
      # Get coherence
      f, coherences[:, i] = get_coherence(wf=wf, fs=fs, tauS=tauS, xi=xi, ref_type="next_seg", N_segs=N_segs, rho=rho)
      
  return f, xis, coherences, const_Npd
  

def get_asym_coherence(wf, fs, tauS, xi, fwhm=None, rho=None, N_segs=None):
  """ Gets the coherence using an asymmetric window (likely will eventually be assimilated as an option in get_coherence())
  Parameters
  ------------
      wf: array
        waveform input array
      fs: int
        sampling rate of waveform
      tauS: int
        length (in samples) of each window
      xi: float
        amount (in time) between the start points of adjacent segments
      fwhm: double, Optional
        FWHM of the Gaussian window (in seconds), need either this or rho
      rho: double, Optional
        Applies a Gaussian window whose FWHM is rho*xi, need either this or fwhm
  """
  
  # Get sigmaS for the gaussian part of the window either via rho (dynamic FWHM = rho*xi) or fixed FWHM
  if fwhm is None and rho is None:
    raise ValueError("You must input either FWHM or rho!")
  elif fwhm is not None:
    sigmaS = get_sigmaS(fwhm=fwhm, fs=fs)
  elif rho is not None:
    sigmaS = get_sigmaS(fwhm=rho*xi, fs=fs)
    
    
  # Generate gaussian
  gaussian = get_window(('gauss', sigmaS), tauS)
  left_window = np.ones(int(tauS))
  right_window = np.ones(int(tauS))
  left_window[int(tauS/2):] = gaussian[int(tauS/2):] # Left window starts with ones and ends with gaussian (gaussian on overlapping side)
  right_window[0:int(tauS/2)] = gaussian[0:int(tauS/2)] # Vice versa

  # Get STFTs for each side
  f, left_stft = get_stft(wf=wf, fs=fs, tauS=tauS, seg_spacing=xi, N_segs=N_segs, win_type=left_window)
  f, right_stft = get_stft(wf=wf, fs=fs, tauS=tauS, seg_spacing=xi, N_segs=N_segs, win_type=right_window)
  # Extract angles
  left_phases = np.angle(left_stft)
  right_phases = np.angle(right_stft)
  # Calc phase diffs
  N_bins = len(f) # Number of frequency bins
  phase_diffs = np.zeros((N_segs - 1, N_bins))
  for win in range(N_segs - 1):
    # Take the difference between the phases, with the windows aligned as to minimize shared samples
    phase_diffs[win] = right_phases[win + 1] - left_phases[win]

  coherence, avg_vector_angle = get_avg_vector(phase_diffs) # Get vector strength

  return f, coherence

def get_welch(wf, fs, tau=None, tauS=None, seg_spacing=None, N_segs=None, win_type='boxcar', scaling='density', reuse_stft=None, return_dict=False):
  """ Gets the Welch averaged power of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      fs: int
        sample rate of waveform
      tau: float
        length (in time) of each window; used in get_stft and to calculate normalizing factor
      win_type: String, Optional
        Window to apply before the FFT
      N_segs: int, Optional
        Used in get_stft;
          if this isn't passed, then just gets the maximum number of segments of the given size
      scaling: String, Optional
        "mags" (magnitudes) or "density" (PSD) or "spectrum" (power spectrum)
      reuse_stft: tuple, Optional
        If you want to avoid recalculating the segmented fft, pass it in here along with the frequency axis as (freq_ax, stft)
      return_dict: bool, Optional
        Returns a dict with:
        d["freq_ax"] = freq_ax
        d["spectrum"] = spectrum
        d["segmented_spectrum"] = segmented_spectrum
  """
  # if nothing was passed into reuse_stft then we need to recalculate it
  if reuse_stft is None:
    freq_ax, stft = get_stft(wf=wf, fs=fs, tau=tau, tauS=tauS, seg_spacing=seg_spacing, N_segs=N_segs, win_type=win_type)
  else:
    freq_ax, stft = reuse_stft
    # Make sure these are valid
    if stft.shape[1] != freq_ax.shape[0]:
      raise Exception("STFT and frequency axis don't match!")

  # calculate necessary params from the stft
  N_segs, N_bins = np.shape(stft)
  
  # Handle possibilities of tau and tauS
  tau, tauS = tau_or_tauS(fs, tau, tauS)
  
  # initialize array
  segmented_spectrum = np.zeros((N_segs, N_bins))
  
  # get spectrum for each window
  for win in range(N_segs):
    segmented_spectrum[win, :] = ((np.abs(stft[win, :]))**2)
    
  # average over all segments (in power)
  spectrum = np.mean(segmented_spectrum, 0)
  
  window = get_window(win_type, tauS)
  S1 = np.sum(window)
  S2 = np.sum(window**2)
  ENBW = tauS * S2 / S1**2
  
  if scaling == 'mags':
    spectrum = np.sqrt(spectrum)
    normalizing_factor = 1 / S1
    
  elif scaling == 'spectrum':
    normalizing_factor = 1 / S1**2
    
  elif scaling == 'density':
    bin_width = 1/tau
    normalizing_factor = 1 / (S1**2 * ENBW * bin_width) # Start with standard periodogram/spectrum scaling, then divide by bin width (times ENBW in samples)

  else:
    raise Exception("Scaling must be 'mags', 'density', or 'spectrum'!")
  
  # Normalize; since this is an rfft, we should multiply by 2 
  spectrum = spectrum * 2 * normalizing_factor
  # Except DC bin should NOT be scaled by 2
  spectrum[0] = spectrum[0] / 2
  # Nyquist bin shouldn't either (note this bin only exists if nperseg is even)
  if tauS % 2 == 0:
    spectrum[-1] = spectrum[-1] / 2
  
  if not return_dict:
    return freq_ax, spectrum
  else:
    return {  
      "freq_ax" : freq_ax,
      "spectrum" : spectrum,
      "segmented_spectrum" : segmented_spectrum
      }    

def get_cwt(wf, fs, fb, f):
  """ Returns the CWT coefficients of the given waveform with a complex morelet wavelet
  Parameters
  ------------
      wf: array
          waveform input array
      fs: int
          sampling rate of waveform
      fb: float
          bandwidth of wavelet in time; bigger bandwidth = better frequency resolution but less time resolution (similar to tau)
      f: array
          frequency axis
  """
  # Perform a Continuous Wavelet Transform (CWT)
  dt = 1/fs
  # Define wavelet and get its center frequency (for scale-frequency conversion)
  wavelet_string = f'cmor{fb}-1.0'
  wavelet = pywt.ContinuousWavelet(wavelet_string)
  fc = pywt.central_frequency(wavelet) # This will always be 1.0 because we set it that way
  # Convert frequencies to scales
  scales = fc / (f * dt)
  coefficients, f_cwt = pywt.cwt(wf, scales, wavelet, method='fft', sampling_period=dt)
  return coefficients.T

def get_wavelet_coherence(wf, fs, f, fb=1.0, cwt=None, xi=0.0025):
  """ Returns the wavelet coherence of the given waveform with a complex morelet wavelet
  Parameters
  ------------
      wf: array
          waveform input array
      fs: int
          sampling rate of waveform
      f: array
          frequency axis
      fb: float, Optional
          bandwidth of wavelet in time; bigger bandwidth = better frequency resolution but less time resolution (similar to tau)
      cwt: array, Optional
          CWT coefficients of waveform if already calculated
      xi: float, Optional
          length (in time) between the start of successive segments
  """
  if cwt is None:
    cwt = get_cwt(wf=wf, fs=fs, fb=fb, f=f)
  # get phases
  phases=np.angle(cwt)
  xiS = int(xi*fs)

  N_segs = int(len(wf)/xiS) - 1

  # initialize array for phase diffs
  phase_diffs = np.zeros((N_segs, len(f)))

  # calc phase diffs
  for seg in range(N_segs):
      phase_diffs[seg, :] = phases[int(seg*xiS)] - phases[int((seg+1)*xiS)]

  wav_coherence, _ = get_avg_vector(phase_diffs)
  
  return wav_coherence