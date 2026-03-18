import numpy as np
from src.utils import C0, KB


class Target:
    def __init__(self, distance, velocity, rcs):
        self.distance = distance
        self.velocity = velocity
        self.rcs = rcs

def target2params(target, fc):
    rcs = 10 ** (target.rcs / 10)

    tau = 2.0 * target.distance / C0
    fd = 2.0 * target.velocity * fc / C0
    b = np.sqrt((C0 * rcs) / ((4*np.pi)**3 * target.distance**4 * fc**2))

    return tau, fd, b

def apply_target_echo(target, signal, cp_len, fs, fc):
    tau, fd, b = target2params(target, fc)

    n = np.arange(len(signal))

    delayed = np.zeros_like(signal, dtype=complex)
    lag = int(np.round(tau * fs)) # assuming integer delay; what about fractional delay?
    if lag > cp_len:
        raise ValueError(
            f"Target delay exceeds CP: lag={lag} samples, CP={cp_len}"
        )

    if lag == 0:
        delayed[:] = signal
    elif lag < len(signal):
        delayed[lag:] = signal[:-lag]

    doppler = np.exp(1j * 2 * np.pi * (fd/fs) * n)

    phi = np.random.uniform(0, 2 * np.pi)
    rand_phase = np.exp(1j * phi)

    return b * delayed * doppler * rand_phase

def apply_awgn(signal, snr_db):

    """
    Change function: generate noise power as in equation 3.35
    """
    sig_power = np.mean(np.abs(signal)**2)

    snr_lin = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_lin

    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))

    return signal + noise


