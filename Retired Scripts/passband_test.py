import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

def plot_kaiser_hpf(cutoff_hz, transition_bw_hz, attenuation_db, fs):
    """
    Design and plot a Kaiser-window high-pass FIR filter.

    Parameters:
        cutoff_hz (float): Cutoff frequency (start of transition band) in Hz
        transition_bw_hz (float): Desired transition bandwidth in Hz
        attenuation_db (float): Desired stopband attenuation in dB
        fs (float): Sampling frequency in Hz
    """
    # Calculate required filter order
    delta_f_norm = transition_bw_hz / fs

    N_taps = int(np.ceil((attenuation_db - 8) / (2.285 * delta_f_norm)))
    if N_taps % 2 == 0:
        N_taps += 1  # Make odd for symmetric filter

    beta = (
        0 if attenuation_db <= 21 else
        0.5842 * (attenuation_db - 21)**0.4 + 0.07886 * (attenuation_db - 21)
    )

    print(f"Kaiser HPF Design:")
    print(f"  Cutoff freq: {cutoff_hz} Hz")
    print(f"  Transition bandwidth: {transition_bw_hz} Hz")
    print(f"  Stopband attenuation: {attenuation_db} dB")
    print(f"  Estimated filter length: {N_taps} taps")
    print(f"  Kaiser beta: {beta:.2f}")

    # Design filter
    b = firwin(N_taps, cutoff=cutoff_hz, fs=fs, window=('kaiser', beta), pass_zero=False)

    # Frequency response
    w, h = freqz(b, worN=2048, fs=fs)
    # h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    h_db = 20 * np.log10(np.abs(h))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(w, h_db, label='Magnitude Response')
    plt.axvline(cutoff_hz, color='r', linestyle='--', label='Cutoff')
    plt.axhline(-3, color='gray', linestyle='--', label='-3 dB')
    plt.title('Kaiser High-Pass FIR Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    # plt.ylim(-100, 5)
    plt.xlim(0, 600)
    plt.tight_layout()
    plt.show()

# Example usage
plot_kaiser_hpf(cutoff_hz=300, transition_bw_hz=10, attenuation_db=50, fs=100000)
