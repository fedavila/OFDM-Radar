import numpy as np

def generate_bits(n_bits):
    return np.random.randint(0, 2, size=n_bits)

def bits2bpsk(bits):
    symbols = 2 * bits - 1
    return symbols

def bits2qpsk(bits):
    bits = bits.reshape(-1, 2)

    I = 2*bits[:, 0] - 1
    Q = 2*bits[:, 1] - 1

    symbols = (I + 1j*Q) / np.sqrt(2)
    return symbols

