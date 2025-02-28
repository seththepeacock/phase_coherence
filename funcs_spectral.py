import numpy as np
from scipy.signal import butter, sosfilt, get_window
from scipy.fft import rfft, rfftfreq, fftshift


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
  
  # finally, output the averaged vector's vector strength and angle with x axis (for each frequency) 
  return np.sqrt(xx**2 + yy**2), np.arctan2(yy, xx)

def spectral_filter(wf, sr, cutoff_freq):
  fft_coefficients = np.fft.rfft(wf)
  frequencies = np.fft.rfftfreq(len(wf), d=1/sr)

  # Zero out coefficients from 0 Hz to cutoff_frequency Hz
  fft_coefficients[frequencies <= cutoff_freq] = 0

  # Compute the inverse real-valued FFT (irfft)
  filtered_wf = np.fft.irfft(fft_coefficients, n=len(wf))  # Ensure output length matches input
  
  return filtered_wf

# I THINK THE ISSUE HERE IS THAT THE ANGLE OF THIS AVERAGED VECTOR IS NOT THE SAME AS THE AVERAGE OF THE ANGLES OF THE INDIVIDUAL VECTORS
def get_stft(wf, sr, tau, xi=None, num_segs=None, win_type='boxcar', filter_seg=False, fftshift_segs=False, return_dict=False):
  """ Returns a dict with the segmented fft and associated freq ax of the given waveform

  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
        sample rate of waveform
      tau: float
        length (in time) of each window
      xi: float
        length (in time) between the start of successive segments
      num_segs: int, Optional
        If this isn't passed, then just get the maximum number of segments of the given size
      win_type: String, Optional
        Window to apply before the FFT
      filter_seg: bool/String, Optional
        Filters each individual segment via a 'butter' for Butterworth filter or 'brute' for "brute" spectral filtering
      fftshift_segs: bool, Optional
        Shifts each time-domain window of the fft with fftshift()
      return_dict: bool, Optional
        Returns a dict with extra variables (otherwise just returns freq_ax, stft)
  """
  
  # if you didn't pass in xi we'll assume you want no overlap - each new window starts at the end of the last!
  if xi is None:
    xi=tau
  
  # calculate the number of samples in the window
  nperseg = int(tau*sr)

  # and the number of samples to shift
  n_shift = int(xi*sr)

  # get sample_spacing
  sample_spacing = 1/sr

  # first, get the last index of the waveform
  final_wf_index = len(wf) - 1
    # - 1 is because of zero-indexing!
  # next, we get what we would be the largest potential seg_start_index
  final_seg_start_index = final_wf_index - (nperseg-1)
    # start at the final_wf_index. we need to collect nperseg points. this final index is our first one, and then we need nperseg - 1 more. 
    # So we march back nperseg-1 points, and then THAT is the last_potential_seg_start_index!
  seg_start_indices = np.arange(0, final_seg_start_index + 1, n_shift)
    # the + 1 here is because np.arange won't ever include the "stop" argument in the output array... but it could include (stop - 1) which is just our final_seg_start_index!

  # if number of segments is passed in, we make sure it's less than the length of seg_start_indices
  if num_segs is not None:
    if num_segs > len(seg_start_indices):
      raise Exception("That's more segments than we can manage! Decrease num_segs!")
  else:
    # if no num_segs is passed in, we'll just use the max number of segments
    num_segs = len(seg_start_indices)

  # Initialize segmented waveform matrix
  segmented_wf = np.zeros((num_segs, nperseg))
  
  # Get window function (boxcar is no window)
  win = get_window(win_type, nperseg)
  
  for k in range(num_segs):
    seg_start = seg_start_indices[k]
    seg_end = seg_start + nperseg
    # grab the waveform in this segment
    seg = wf[seg_start:seg_end]
    if win_type != "boxcar":
      seg = seg * win
    if fftshift_segs: # optionally swap the halves of the waveform to effectively center it in time
      seg = fftshift(seg)
    cutoff_freq = 25
    if filter_seg == 'brute':      
      seg = spectral_filter(seg, sr, cutoff_freq)
    elif filter_seg == 'butterworth':
      order = 100
      sos = butter(order, cutoff_freq, 'hp',fs=sr,output='sos')
      seg = sosfilt(sos, seg, padtype='constant', padlen=0)  # Ensure no padding
    segmented_wf[k, :] = seg
    # note this grabs the wf at indices seg_start, seg_start+1, ..., seg_end-1
      # if there are 4 samples and tau=xi=1 and SR=1, then nperseg=2, n_shift=1 and
      # Thus the first window will be samples 0 and 1, the next 1 and 2...

  # Now we do the ffts!

  # get frequency axis 
  freq_ax = rfftfreq(nperseg, sample_spacing)
  num_freq_pts = len(freq_ax)
  
  # initialize segmented fft array
  stft = np.zeros((num_segs, num_freq_pts), dtype=complex)
  
  # get ffts
  for k in range(num_segs):
    stft[k, :] = rfft(segmented_wf[k, :])

  if return_dict:
    return {  
      "stft" : stft,
      "freq_ax" : freq_ax,
      "seg_start_indices" : seg_start_indices
      }
  else: 
    return freq_ax, stft


def get_welch(wf, sr, tau, num_segs=None, win_type='boxcar', scaling='density', stft=None, freq_ax=None, return_dict=False):
  """ Gets the spectrum of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
        sample rate of waveform
      tau: float
        length (in time) of each window; used in get_stft and to calculate normalizing factor
      num_segs: int, Optional
        Used in get_stft;
          if this isn't passed, then just gets the maximum number of segments of the given size
      win_type: String, Optional
        Window to apply before the FFT
      scaling: String, Optional
        "mags" (magnitudes) or "density" (PSD) or "spectrum" (power spectrum)
      stft: any, Optional
        If you want to avoid recalculating the segmented fft, pass it in here!
      freq_ax: any, Optional
        We have to also pass in the freq_ax for the stft (either pass in both or neither!)
      return_dict: bool, Optional
        Defaults to only returning the spectrum averaged over all segments; if this is enabled, then a dictionary is returned with keys:
        "spectrum", "freq_ax", "win_spectrum"
  """
  # make sure we either have both or neither
  if (stft is None and freq_ax is not None) or (stft is not None and freq_ax is None):
    raise Exception("We need both stft and freq_ax (or neither)!")
  
  # if you passed the stft and freq_ax in then we'll skip over this
  if stft is None:
    freq_ax, stft = get_stft(wf=wf, sr=sr, tau=tau, num_segs=num_segs, win_type=win_type)

  # calculate necessary params from the stft
  stft_size = np.shape(stft)
  num_segs = stft_size[0]
  num_freq_pts = stft_size[1]

  # calculate the number of samples in the window for normalizing factor purposes
  nperseg = int(tau*sr)
  
  # initialize array
  win_spectrum = np.zeros((num_segs, num_freq_pts))
  
  # get spectrum for each window
  for win in range(num_segs):
    win_spectrum[win, :] = ((np.abs(stft[win, :]))**2)
    
  # average over all segments (in power)
  spectrum = np.mean(win_spectrum, 0)
  
  window = get_window(win_type, nperseg)
  S1 = np.sum(window)
  S2 = np.sum(window**2)
  ENBW = nperseg * S2 / S1**2
  
  if scaling == 'mags':
    spectrum = np.sqrt(spectrum)
    normalizing_factor = 1 / S1
    
  elif scaling == 'spectrum':
    normalizing_factor = 1 / S1**2
    
  elif scaling == 'density':
    bin_width = 1/tau
    normalizing_factor = 1 / (S1**2 * ENBW * bin_width) # Start with spectrum scaling, then divide by bin width (times ENBW in samples)

  else:
    raise Exception("scaling must be 'mags', 'density', or 'spectrum'!")
  
  # Normalize; since this is an rfft, we should multiply by 2 
  spectrum = spectrum * 2 * normalizing_factor
  # Except DC bin should NOT be scaled by 2
  spectrum[0] = spectrum[0] / 2
  # Nyquist bin shouldn't either (this bin only exists if nperseg is even)
  if nperseg % 2 == 0:
    spectrum[-1] = spectrum[-1] / 2
  
  if not return_dict:
    return freq_ax, spectrum
  else:
    return {  
      "spectrum" : spectrum,
      "freq_ax" : freq_ax,
      "win_spectrum" : win_spectrum
      }
    

def get_coherence(wf, sr, tau, xi, bin_shift=1, num_segs=None, win_type='boxcar', fftshift_segs=False, stft=None, freq_ax=None, ref_type="next_win", filter_seg=False, return_dict=False):
  """ Gets the phase coherence of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      sr:
        sample rate of waveform
      tau: float
        length (in time) of each segment
      xi: float
        length (in time) between the start of successive segments
      num_segs: int, Optional
        If this isn't passed, then just get the maximum number of segments of the given size
      win_type: String, Optional
        Window to apply before the FFT
      fftshift_segs: bool, Optional
        Shifts each time-domain segment of the fft
      stft: any, Optional
        If you want to avoid recalculating the segmented fft, pass it in here!
      freq_ax: any, Optional
        We have to also pass in the freq_ax for the stft (either pass in both or neither!)
      ref_type: str, Optional
        Either "next_win" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
      bin_shift: int, Optional
        How many bins over to reference phase against for next_freq
      return_dict: bool, Optional
        Defaults to only returning the coherence; if this is enabled, then a dictionary is returned with keys:
        d["coherence"] = coherence
        d["phases"] = phases
        d["phase_diffs"] = phase_diffs
        d["means"] = means
        d["avg_phase_diff"] = avg_phase_diff
        d["num_segs"] = num_segs
        d["freq_ax"] = freq_ax
        d["stft"] = stft
  """
  
  # make sure we either have both stft and freq_ax or neither
  if (stft is None and freq_ax is not None) or (stft is not None and freq_ax is None):
    raise Exception("We need both stft and freq_ax (or neither)!")
  
  # if you passed the stft and freq_ax in then we'll skip over this
  if stft is None:
    freq_ax, stft = get_stft(wf=wf, sr=sr, tau=tau, xi=xi, num_segs=num_segs, win_type=win_type, fftshift_segs=fftshift_segs, filter_seg=filter_seg)
  
  # calculate necessary params from the stft
  stft_size = np.shape(stft)
  num_segs = stft_size[0]
  num_freq_pts = stft_size[1]

  # get phases
  phases=np.angle(stft)
  
  # we can reference each phase against the phase of the same frequency in the next window:
  if ref_type == "next_win":
    # unwrap phases along time window axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=0)
    
    # initialize array for phase diffs; we won't be able to get it for the final window
    phase_diffs = np.zeros((num_segs - 1, num_freq_pts))
    
    # calc phase diffs
    for win in range(num_segs - 1):
      # take the difference between the phases in this current window and the next
      phase_diffs[win] = phases[win + 1] - phases[win]
    
    coherence, avg_phase_diff = get_avg_vector(phase_diffs)
    
  # or we can reference it against the phase of the next frequency in the same window:
  elif ref_type == "next_freq":
    # unwrap phases along the frequency bin axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=1)
      
    # initialize array for phase diffs; -bin_shift is because we won't be able to get it for the #(bin_shift) freqs
    phase_diffs = np.zeros((num_segs, num_freq_pts - bin_shift))
    # we'll also need to take the last #(bin_shift) bins off the freq_ax
    freq_ax = freq_ax[0:-bin_shift]
    
    # calc phase diffs
    for win in range(num_segs):
      for freq_bin in range(num_freq_pts - bin_shift):
        phase_diffs[win, freq_bin] = phases[win, freq_bin + bin_shift] - phases[win, freq_bin]
    
    # get final coherence
    coherence, avg_phase_diff = get_avg_vector(phase_diffs)
    
    # Since this references each frequency bin to its adjacent neighbor, we'll plot them w.r.t. the average frequency 
        # this corresponds to shifting everything over half a bin width (bin width is 1/tau)
    freq_ax = freq_ax + (1/2)*(1/tau)
    
  
  # or we can reference it against the phase of both the lower and higher frequencies in the same window
  elif ref_type == "both_freqs":
    # unwrap phases along the frequency bin axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=1)
    
    # initialize arrays
      # even though we only lose ONE freq point with lower and one with higher, we want to get all the points we can get from BOTH so we do - 2
    pd_low = np.zeros((num_segs, num_freq_pts - 2))
    pd_high = np.zeros((num_segs, num_freq_pts - 2))
    # take the first and last bin off the freq ax
    freq_ax = freq_ax[1:-1]
    
    # calc phase diffs
    for win in range(num_segs):
      for freq_bin in range(1, num_freq_pts - 1):
        # the - 1 is so that we start our phase_diffs arrays at 0 and put in num_freq_pts-2 points. 
        # These will correspond to our new frequency axis.
        pd_low[win, freq_bin - 1] = phases[win, freq_bin] - phases[win, freq_bin - 1]
        pd_high[win, freq_bin - 1] = phases[win, freq_bin + 1] - phases[win, freq_bin]
    coherence_low, _ = get_avg_vector(pd_low)
    coherence_high, _ = get_avg_vector(pd_high)
    # average the coherences you would get from either of these
    coherence = (coherence_low + coherence_high)/2
    # set the phase diffs to one of these (could've also been pd_high)
    phase_diffs = pd_low
    
  else:
    raise Exception("You didn't input a valid ref_type!")
  
  
  
  # get <|phase diffs|>
    # note we're unwrapping w.r.t. the frequency axis
  means = np.mean(np.abs(phase_diffs), 0)
  
  if not return_dict:
    return freq_ax, coherence
  
  else:
    # define output dictionary to be returned
    d = {}
    d["coherence"] = coherence
    d["phases"] = phases
    d["phase_diffs"] = phase_diffs
    d["means"] = means
    d["avg_phase_diff"] = avg_phase_diff
    d["num_segs"] = num_segs
    d["freq_ax"] = freq_ax
    d["stft"] = stft
    return d