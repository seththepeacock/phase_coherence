import numpy as np
from scipy.linalg import expm
from tqdm import tqdm




def generate_nddho(q, f_d, fs=44100, t_max=60):
    """
    Generate a damped noise-driven harmonic oscillator (NDDHO) waveform
    using the exact discrete-time update scheme of Nørrelykke & Flyvbjerg.

    The system obeys
        x'' + gamma x' + k x = η(t),
    where η(t) is normalized unit white noise with
        ⟨η(t) η(t')⟩ = δ(t - t').

    The damping gamma and stiffness k are chosen to realize the requested
    quality factor Q and damped natural frequency f_d.



    Parameters
    ----------
    q : float
        Quality factor.
    f_d : float
        Damped natural frequency (in Hz).
    fs : float
        Sampling frequency (in Hz).
    t_max: float
        Length of waveform to generate (in sec).

    Returns
    -------
    x : ndarray
        Position time series (waveform).
    v : ndarray
        Velocity time series.
    """

    # Define variables
    dt = 1.0 / fs
    omega_d = 2 * np.pi * f_d  # Damped critical frequency
    n_samples = round(t_max * fs)

    # We're assuming m = 1 for simplicity (note k = omega_0**2 / m = omega_0**2, so we'll just use omega_0**2)
    # omega_d^2 = omega_0^2 - gamma^2 / 4 and gamma = omega_0 / q ==>
    omega_0 = omega_d / np.sqrt(1 - 1 / (4 * q**2))
    gamma = omega_0 / q
    # We also have gamma = omega_d / np.sqrt(q**2-1/4)
    
    if q <= 1/2:
        raise ValueError("Can't handle the underdamped case Q={q} <= 1/2!")
    
    # Since their driving force is F_therm(t) = sqrt(2 * k_B * T * gamma) eta(t)
    # and we just want eta(t), we want kBT = 1/(2*gamma)
    kBT = 1 / (2 * gamma)
    # Then we define D (Einstein's relation)
    D = kBT / gamma
    # Note that now sqrt(2D) / tau = (1/gamma) * gamma = 1
    omega = np.sqrt(omega_0**2 - (gamma / 2) ** 2)  # damped critical frequency

    # Define variables for equivalence with paper
    tau = 1 / gamma

    # Drift matrix M (Eq. 5)
    M = np.array([[0, -1], [omega_0**2, 1 / tau]])

    # Exact update: Matrix exponential
    Phi = expm(-M * dt)

    # Variances/covariances of increments (Eqs. 15–17)
    expfac = np.exp(-dt / tau)
    sig_xx2 = (D / (4 * omega**2 * omega_0**2 * tau**3)) * (
        4 * omega**2 * tau**2
        + expfac
        * (
            np.cos(2 * omega * dt)
            - 2 * omega * tau * np.sin(2 * omega * dt)
            - 4 * omega_0**2 * tau**2
        )
    )
    sig_vv2 = (D / (4 * omega**2 * tau**3)) * (
        4 * omega**2 * tau**2
        + expfac
        * (
            np.cos(2 * omega * dt)
            + 2 * omega * tau * np.sin(2 * omega * dt)
            - 4 * omega_0**2 * tau**2
        )
    )
    sig_xv2 = (D / (omega**2 * tau**2)) * expfac * np.sin(omega * dt) ** 2

    # Preallocate
    x = np.empty(n_samples)
    v = np.empty(n_samples)

    # make rng
    rng = np.random.default_rng()

    # sample initial state from stationary Gaussian (zero-mean, cov from Eq.23)
    cov_stationary = np.array([[kBT / omega_0**2, 0.0], [0.0, kBT]])
    ic = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_stationary)

    # Set initial conditions
    x[0] = ic[0]
    v[0] = ic[1]

    # And set as state
    state = np.array([x[0], v[0]])

    # Correlated Gaussian increments (Eqs. 13–14)
    xis = rng.normal(size=(n_samples-1))
    zetas = rng.normal(size=(n_samples-1))

    sig_xx = np.sqrt(sig_xx2)
    dxs = sig_xx * xis
    dvs = (sig_xv2 / sig_xx) * xis + np.sqrt(sig_vv2 - (sig_xv2**2) / sig_xx2) * zetas
    ds = np.stack((dxs, dvs), axis=1)

    # Iterate
    for j in tqdm(range(1, n_samples), mininterval=1.0):
        # Correlated Gaussian increments (Eqs. 13–14)
        state = Phi @ state + ds[j-1]  # (Eq. 7)
        x[j], v[j] = state

    return x, v


# def generate_nddho_f0nat(q, f0, fs=44100, t_max=60):
#     """
#     Generate a damped noise-driven (normalized white noise) harmonic oscillator (NDDHO) waveform
#     using an exact update method.

#     x'' + gamma x' + kx = eta(t)


#     Parameters
#     ----------
#     q : float
#         Quality factor.
#     f0 : float
#         Undamped natural frequency in Hz.
#     fs : float
#         Sampling frequency in Hz.
#     t_max: float
#         Length of waveform to generate

#     Returns
#     -------
#     x : ndarray
#         Position time series (waveform).
#     v : ndarray
#         Velocity time series.
#     """

#     # Define variables
#     dt = 1.0 / fs
#     omega_0 = 2 * np.pi * f0  # Undamped critical frequency
#     gamma = omega_0 / q  # by definition of Q
#     n_samples = round(t_max * fs)

#     # Define variables for equivalence with paper
#     # m = 1
#     tau = 1 / gamma  # m / gamma
#     # Since their driving force is F_therm(t) = sqrt(2 * k_B * T * gamma) eta(t)
#     # and we just want eta(t), we want::
#     kBT = 1 / (2 * gamma)
#     # Then we define D (Einstein's relation)
#     D = kBT / gamma
#     # Note that now sqrt(2D) / tau = (1/gamma) * (gamma / m) = 1/m = 1
#     omega = np.sqrt(omega_0**2 - 1 / (4 * tau**2))  # damped critical frequency

#     # Drift matrix M (Eq. 5)
#     M = np.array([[0, -1], [omega_0**2, 1 / tau]])

#     # Exact update: Matrix exponential
#     Phi = expm(-M * dt)

#     # Variances/covariances of increments (Eqs. 15–17)
#     expfac = np.exp(-dt / tau)
#     sig_xx2 = (D / (4 * omega**2 * omega_0**2 * tau**3)) * (
#         4 * omega**2 * tau**2
#         + expfac
#         * (
#             np.cos(2 * omega * dt)
#             - 2 * omega * tau * np.sin(2 * omega * dt)
#             - 4 * omega_0**2 * tau**2
#         )
#     )
#     sig_vv2 = (D / (4 * omega**2 * tau**3)) * (
#         4 * omega**2 * tau**2
#         + expfac
#         * (
#             np.cos(2 * omega * dt)
#             + 2 * omega * tau * np.sin(2 * omega * dt)
#             - 4 * omega_0**2 * tau**2
#         )
#     )
#     sig_xv2 = (D / (omega**2 * tau**2)) * expfac * np.sin(omega * dt) ** 2

#     # Preallocate
#     x = np.empty(n_samples)
#     v = np.empty(n_samples)

#     # make rng
#     rng = np.random.default_rng()

#     # sample initial state from stationary Gaussian (zero-mean, cov from Eq.23)
#     cov_stationary = np.array([[kBT / k, 0.0], [0.0, kBT / m]])
#     ic = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_stationary)

#     # Set initial conditions
#     x[0] = ic[0]
#     v[0] = ic[1]

#     # And set as state
#     state = np.array([x[0], v[0]])

#     # Correlated Gaussian increments (Eqs. 13–14)
#     xis = rng.normal(size=(n_samples))
#     zetas = rng.normal(size=(n_samples))

#     sig_xx = np.sqrt(sig_xx2)
#     dxs = sig_xx * xis
#     dvs = (sig_xv2 / sig_xx) * xis + np.sqrt(sig_vv2 - (sig_xv2**2) / sig_xx2) * zetas
#     ds = np.stack((dxs, dvs), axis=1)

#     # Iterate
#     for j in tqdm(range(1, n_samples), mininterval=1.0):
#         # Correlated Gaussian increments (Eqs. 13–14)
#         state = Phi @ state + ds[j]  # (Eq. 7)
#         x[j], v[j] = state

#     return x, v


# OG Implementation
# from scipy.linalg import expm, solve_continuous_lyapunov
# # Drift matrix M
# M = np.array([[0, -1],
#               [omega0**2, 1/tau]])

# # Diffusion matrix G (only in dv equation)
# G = np.array([[0],
#               [sigma]]) # sigma = sqrt(2D); that is, white noise forcing eta(𝑡) satisfies ⟨eta(𝑡)eta(𝑡′)⟩ = 𝜎^2 𝛿(𝑡−𝑡′)

# # Exact update: Matrix exponential
# Phi = expm(-M * dt)

# # Solve for covariance of the Brownian kick
# # Lyapunov equation: A Q + Q A^T = -G G^T
# # Continuous-time stationary covariance:
# Q_c = solve_continuous_lyapunov(-M, -G @ G.T)

# # Discrete-time covariance of noise:
# Q_kick = Q_c - Phi @ Q_c @ Phi.T

# # Preallocate
# x = np.zeros(n_samples)
# v = np.zeros(n_samples)
# state = np.random.multivariate_normal(mean=[0, 0], cov=Q_c)


# # Iterate
# for i in tqdm(range(1, n_samples), mininterval=1.0):
#     eta = np.random.multivariate_normal(mean=[0, 0], cov=Q_kick)
#     state = Phi @ state + eta
#     x[i], v[i] = state

# return x, v
