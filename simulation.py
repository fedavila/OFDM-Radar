import numpy as np
from src.utils import load_config, plot_periodogram, C0, KB
from src import environment as env
from src import transmitter as tx
from src import receiver as rx

config = load_config()

FC = config['radar']['fc']
SNR_DB = config['radar']['snr_db']
H = config['radar']['H']

MODULATION = config['ofdm']['modulation']
N_FFT = config['ofdm']['N_fft']
DELTA_F = config['ofdm']["delta_f"]
CP_LEN = config['ofdm']['CP']
N = config['ofdm']["N"]
M = config['ofdm']['M']

N_PER = config["periodogram"]['N_per']
M_PER = config["periodogram"]['M_per']

FS = N_FFT * DELTA_F
T_SYM = (1 / DELTA_F) + (CP_LEN / FS)

if MODULATION == "BPSK": 
    BITS_PER_SYMBOL = 1
else: 
    BITS_PER_SYMBOL = 2

N_BITS = BITS_PER_SYMBOL * N * M

# Transmitter 
bits = tx.generate_bits(N_BITS)
symbols = tx.data_modulator(bits, BITS_PER_SYMBOL)

F_tx = symbols.reshape(M, N)

tx_signal = tx.ofdm_modulation(F_tx, CP_LEN, N_FFT)

# Environment
targets = [env.Target(**t) for t in config['targets']]

echos = np.zeros_like(tx_signal, dtype=complex)

for target in targets:
    echos += env.apply_target_echo(target, tx_signal, CP_LEN, FS, FC)

rx_signal = env.apply_awgn(echos, SNR_DB)

# Receiver
F = rx.ofdm_demodulation(rx_signal, CP_LEN, N_FFT, F_tx)

per, n_ax, m_ax = rx.periodogram(F, N_PER, M_PER)

v_lim = [-30, 30]
d_lim = [0, 40.0]
plot_periodogram(per, n_ax, m_ax, DELTA_F, T_SYM, FC, v_lim, d_lim)

"""
Perform target detection (CFAR detection)
Target params estimation
Compute error
Plot
"""
