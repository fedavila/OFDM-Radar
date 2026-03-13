import numpy as np

def generate_bits(n_bits):
    return np.random.randint(0, 2, size=n_bits)

def data_modulator(bits, BITS_PER_SYMBOL):
    if BITS_PER_SYMBOL == 1:
        symbols = 2 * bits - 1
    else:
        bits = bits.reshape(-1, 2)

def bits2bpsk(bits):
    symbols = 2 * bits - 1
    return symbols

def bits2qpsk(bits):
    bits = bits.reshape(-1, 2)

    I = 2*bits[:, 0] - 1
    Q = 2*bits[:, 1] - 1

    symbols = (I + 1j*Q) / np.sqrt(2)
    return symbols

def ofdm_modulation(F_tx, CP_LEN, N_fft):

    ofdm_frame = np.fft.ifft(F_tx, n=N_fft, axis=1)

    CP = ofdm_frame[:, -CP_LEN:]
    ofdm_frame_cp = np.concatenate((CP, ofdm_frame), axis=1)

    tx_signal = ofdm_frame_cp.reshape(-1)

    return tx_signal


