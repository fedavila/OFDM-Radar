import numpy as np
from src.utils import C0


class Target:
    def __init__(self, distance, velocity, rcs):
        self.distance = distance
        self.velocity = velocity
        self.rcs = rcs

def target2params(target, fc):
    rcs = 10 * (target.rcs / 10)

    tau = 2.0 * target.distance / C0
    fd = 2.0 * target.velocity * fc / C0
    b = np.sqrt((C0 * rcs) / ((4*np.pi)**3 * target.distance**4 * fc**2))

    return tau, fd, b

def apply_target_echo(target, signal, cp_len, fs, fc):
    tau, fd, b = target2params(target, fc)

    n = np.arange(len(signal))

    delayed = np.zeros_like(signal, type=complex)
    lag = int(np.round(tau * fs)) # assuming integer delay; what about fractional delay?
    if lag < len(cp_len):
        delayed[lag:] = signal[:-lag]

    return b * delayed * np.exp(1j * 2 * np.pi * (fd/fs) * n)




