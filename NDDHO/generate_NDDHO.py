import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
import matplotlib.pyplot as plt

def generate_nddho(q, f0, sigma=1, fs=44100, t_max=60):
    """
    Generate a noisy damped driven harmonic oscillator (NDDHO) waveform 
    using the exact update method.
    
    Parameters
    ----------
    q : float
        Quality factor.
    f0 : float
        Undamped natural frequency in Hz.
    sigma : float
        Standard deviation of the normally distributed white noise.
    fs : float
        Sampling frequency in Hz.
    t_max: float
        Length of waveform to generate
        
    Returns
    -------
    x : ndarray
        Position time series (waveform).
    v : ndarray
        Velocity time series.
    """

    dt = 1.0 / fs
    omega0 = 2 * np.pi * f0
    gamma = omega0 / q
    n_samples = round(t_max * fs)
    
    # Drift matrix A
    A = np.array([[0, 1],
                  [-omega0**2, - gamma]])
    
    # Diffusion matrix G (only in dv equation)
    G = np.array([[0],
                  [sigma]]) # sigma = sqrt(2D); that is, white noise forcing 𝜉(𝑡) satisfies ⟨𝜉(𝑡)𝜉(𝑡′)⟩ = 𝜎^2 𝛿(𝑡−𝑡′)
    
    # Exact update: Matrix exponential
    Phi = expm(A * dt)
    
    # Solve for covariance of the Brownian kick
    # Lyapunov equation: A Q + Q A^T = -G G^T
    # Continuous-time stationary covariance:
    Q_c = solve_continuous_lyapunov(A, -G @ G.T)
    
    # Discrete-time covariance of noise:
    Q_kick = Q_c - Phi @ Q_c @ Phi.T
    
    # Preallocate
    x = np.zeros(n_samples)
    v = np.zeros(n_samples)
    state = np.random.multivariate_normal(mean=[0, 0], cov=Q_c)
    
    # Iterate
    for i in range(1, n_samples):
        eta = np.random.multivariate_normal(mean=[0, 0], cov=Q_kick)
        state = Phi @ state + eta
        x[i], v[i] = state
    
    return x, v
