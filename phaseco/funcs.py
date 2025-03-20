import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq, fftshift
import pywt
import warnings


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

def get_stft(wf, fs, tau, xi=None, sigma=None, beta=None, win_type='boxcar', N_segs=None, filter_seg='none', fftshift_segs=False, return_dict=False):
  """ Returns a dict with the segmented fft and associated freq ax of the given waveform

  Parameters
  ------------
      wf: array
        waveform input array
      fs: int
        sample rate of waveform
      tau: float
        length (in time) of each window
      xi: float
        length (in time) between the start of successive segments
      sigma: double, Optional
        Applies a Gaussian window with this standard deviation (in time) to each segment (win_type must be 'boxcar')
      win_type: String, Optional
        Window to apply before the FFT
      N_segs: int, Optional
        If this isn't passed, then just get the maximum number of segments of the given size
      filter_seg: String, Optional
        HPFs each individual segment by zeroing out frequencies above 100Hz frequency
      fftshift_segs: bool, Optional
        Shifts each time-domain window of the fft with fftshift()
      return_dict: bool, Optional
        Returns a dict with:
        d["freq_ax"] = freq_ax
        d["stft"] = stft
        d["segmented_wf"] = segmented_wf
        d["seg_start_indices"] = seg_start_indices
  """
  
  # if you didn't pass in xi we'll assume you want no overlap - each new window starts at the end of the last!
  if xi is None:
    xi=tau
  
  # calculate the number of samples in the window
  nperseg = int(tau*fs)

  # and the number of samples to shift
  n_shift = int(xi*fs)

  # get sample_spacing
  delta_t = 1/fs

  # Girst, get the last index of the waveform
  final_wf_index = len(wf) - 1
    
  # next, we get what we would be the largest potential seg_start_index
  latest_potential_seg_start_index = final_wf_index - (nperseg - 1) # start at the final_wf_index. we need to collect nperseg points. this final index is our first one, and then we need nperseg - 1 more. 
  seg_start_indices = np.arange(0, latest_potential_seg_start_index + 1, n_shift)
    # the + 1 here is because np.arange won't ever include the "stop" argument in the output array... but it could include (stop - 1) which is just our final_seg_start_index!

  # if number of segments is passed in, we make sure it's less than the length of seg_start_indices
  if N_segs is not None:
    if N_segs > len(seg_start_indices):
      raise Exception("That's more segments than we can manage! Decrease N_segs!")
  else:
    # if no N_segs is passed in, we'll just use the max number of segments
    N_segs = len(seg_start_indices)

  # Initialize segmented waveform matrix
  segmented_wf = np.zeros((N_segs, nperseg))
  
  # Get window function (boxcar is no window)
  win = get_window(win_type, nperseg)
  
  if beta is not None and sigma is not None:
    raise Exception("You're windowing with a Gaussian and a Kaiser! Pick one!")
  
  if beta is not None:
    kaiser = get_window(('kaiser', beta), nperseg)
    print(kaiser)
    win = kaiser * win # In normal use, win will just be a boxcar, but...
    print(win) 
    if win_type != 'boxcar': # Throw a warning if you're going to double window
      warnings.warn(f"You're double-windowing with both a Kaiser (beta = {beta}) and a {win_type}!")
  
  if sigma is not None:
    sigmaS = sigma*fs # Convert from time to samples
    gaussian = get_window(('gauss', sigmaS), nperseg)
    win = gaussian * win # In normal use, win will just be a boxcar, but... 
    if win_type != 'boxcar': # Throw a warning if you're going to double window
      warnings.warn(f"You're double-windowing with both a Gaussian (sigma = {sigma}) and a {win_type}!")
  
  
  
  for k in range(N_segs):
    seg_start = seg_start_indices[k]
    seg_end = seg_start + nperseg
    # grab the waveform in this segment
    seg = wf[seg_start:seg_end]
    if win_type != "boxcar" or sigma is not None or beta is not None:
      seg = seg * win
    if fftshift_segs: # optionally swap the halves of the waveform to effectively center it in time
      seg = fftshift(seg)
    if filter_seg:      
      seg = spectral_filter(seg, fs, cutoff_freq=100)

    segmented_wf[k, :] = seg

  # Now we do the ffts!

  # Get frequency axis 
  freq_ax = rfftfreq(nperseg, delta_t)
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
      "segmented_wf" : segmented_wf
      }
    
  else: 
    return freq_ax, stft
  
def get_coherence(wf, fs, tau, xi, sigma=None, beta=None, win_type='boxcar', N_segs=None, reuse_stft=None, ref_type="next_seg", filter_seg=False, bin_shift=1, return_dict=False):
  """ Gets the phase coherence of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      fs:
        sample rate of waveform
      tau: float
        length (in time) of each segment
      xi: float
        length (in time) between the start of successive segments
      sigma: double, Optional
        Applies a Gaussian window with this standard deviation (in time) to each segment (win_type must be 'boxcar')
      win_type: String, Optional
        Window to apply before the FFT
      N_segs: int, Optional
        If this isn't passed, then just get the maximum number of segments of the given size
      reuse_stft: tuple, Optional
        If you want to avoid recalculating the segmented fft, pass it in here along with the frequency axis as (freq_ax, stft)
      ref_type: str, Optional
        Either "next_seg" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
      filter_seg: bool, Optional
        HPFs each individual segment by zeroing out frequencies above 100Hz frequency
      bin_shift: int, Optional
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
    freq_ax, stft = get_stft(wf=wf, fs=fs, tau=tau, xi=xi, N_segs=N_segs, sigma=sigma, beta=beta, win_type=win_type, filter_seg=filter_seg)
  else:
    freq_ax, stft = reuse_stft
    # Make sure these are valid
    if stft.shape[1] != freq_ax.shape[0]:
      raise Exception("STFT and frequency axis don't match!")
  
  # calculate necessary params from the stft
  N_segs, N_bins = np.shape(stft)

  # get phases
  phases=np.angle(stft)
  
  # we can reference each phase against the phase of the same frequency in the next window:
  if ref_type == "next_seg":
    # unwrap phases along time window axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=0)
    
    # initialize array for phase diffs; we won't be able to get it for the final window
    phase_diffs = np.zeros((N_segs - 1, N_bins))
    
    # calc phase diffs
    for win in range(N_segs - 1):
      # take the difference between the phases in this current window and the next
      phase_diffs[win] = phases[win + 1] - phases[win]
    
    coherence, avg_vector_angle = get_avg_vector(phase_diffs)
    
  # or we can reference it against the phase of the next frequency in the same window:
  elif ref_type == "next_freq":
    # unwrap phases along the frequency bin axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=1)
      
    # initialize array for phase diffs; -bin_shift is because we won't be able to get it for the #(bin_shift) freqs
    phase_diffs = np.zeros((N_segs, N_bins - bin_shift))
    # we'll also need to take the last #(bin_shift) bins off the freq_ax
    freq_ax = freq_ax[0:-bin_shift]
    
    # calc phase diffs
    for win in range(N_segs):
      for freq_bin in range(N_bins - bin_shift):
        phase_diffs[win, freq_bin] = phases[win, freq_bin + bin_shift] - phases[win, freq_bin]
    
    # get final coherence
    coherence, avg_phase_diff = get_avg_vector(phase_diffs)
    
    # Since this references each frequency bin to its adjacent neighbor, we'll plot them w.r.t. the average frequency 
        # this corresponds to shifting everything over half a bin width (bin width is 1/tau)
    bin_width = 1/tau
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
    for win in range(N_segs):
      for freq_bin in range(1, N_bins - 1):
        # the - 1 is so that we start our phase_diffs arrays at 0 and put in N_bins-2 points. 
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
  avg_abs_phase_diffs = np.mean(np.abs(phase_diffs), 0)
  
  if not return_dict:
    return freq_ax, coherence
  
  else:
    # define output dictionary to be returned
    d = {}
    d["coherence"] = coherence
    d["phases"] = phases
    d["phase_diffs"] = phase_diffs
    d["<|phase_diffs|>"] = avg_abs_phase_diffs
    d["avg_vector_angle"] = avg_vector_angle
    d["N_segs"] = N_segs
    d["freq_ax"] = freq_ax
    d["stft"] = stft
    return d


def get_welch(wf, fs, tau, xi=None, beta=None, N_segs=None, win_type='boxcar', scaling='density', reuse_stft=None, return_dict=False):
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
    freq_ax, stft = get_stft(wf=wf, fs=fs, tau=tau, xi=xi, N_segs=N_segs, win_type=win_type, beta=beta)
  else:
    freq_ax, stft = reuse_stft
    # Make sure these are valid
    if stft.shape[1] != freq_ax.shape[0]:
      raise Exception("STFT and frequency axis don't match!")

  # calculate necessary params from the stft
  N_segs, N_bins = np.shape(stft)

  # calculate the number of samples in the window for normalizing factor purposes
  nperseg = int(tau*fs)
  
  # initialize array
  segmented_spectrum = np.zeros((N_segs, N_bins))
  
  # get spectrum for each window
  for win in range(N_segs):
    segmented_spectrum[win, :] = ((np.abs(stft[win, :]))**2)
    
  # average over all segments (in power)
  spectrum = np.mean(segmented_spectrum, 0)
  
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
    normalizing_factor = 1 / (S1**2 * ENBW * bin_width) # Start with standard periodogram/spectrum scaling, then divide by bin width (times ENBW in samples)

  else:
    raise Exception("Scaling must be 'mags', 'density', or 'spectrum'!")
  
  # Normalize; since this is an rfft, we should multiply by 2 
  spectrum = spectrum * 2 * normalizing_factor
  # Except DC bin should NOT be scaled by 2
  spectrum[0] = spectrum[0] / 2
  # Nyquist bin shouldn't either (note this bin only exists if nperseg is even)
  if nperseg % 2 == 0:
    spectrum[-1] = spectrum[-1] / 2
  
  if not return_dict:
    return freq_ax, spectrum
  else:
    return {  
      "freq_ax" : freq_ax,
      "spectrum" : spectrum,
      "segmented_spectrum" : segmented_spectrum
      }
    

