import numpy as np
from src.utils import load_config, plot_periodogram, plot_binary_map_with_detections, C0
from src import environment as env
from src import transmitter as tx
from src import receiver as rx
from src import post_processing as post

config = load_config()

FC = config['radar']['fc']
TEMP = config['radar']['temperature']
NF_DB = config['radar']['NF']
H = config['radar']['H']
FAR = config['radar']['FAR']

MODULATION = config['ofdm']['modulation']
N_FFT = config['ofdm']['N_fft']
DELTA_F = config['ofdm']["delta_f"]
T_CP = config['ofdm']['T_cp']
N = config['ofdm']["N"]
M = config['ofdm']['M']

N_PER = config["periodogram"]['N_per']
M_PER = config["periodogram"]['M_per']

BANDWIDTH = N * DELTA_F
FS = N_FFT * DELTA_F              # Hz
T_SYM = (1.0 / DELTA_F) + T_CP    # seconds
CP_LEN = int(np.round(T_CP * FS)) # samples

dmax = T_CP * C0 / 2
vmax = DELTA_F * C0 / (10 * FC)

N_MAX = int(np.ceil(2 * dmax * N_PER * DELTA_F) / C0)
M_MAX = int(np.ceil(2 * vmax * FC * T_SYM * M_PER) / C0)

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

rx_signal = env.apply_awgn_nf(echos, TEMP, NF_DB, FS)


# Receiver
F = rx.ofdm_demodulation(rx_signal, CP_LEN, N_FFT, F_tx)

per, n_idx, m_idx, noise_power_hat = rx.crop_periodogram(F, N_PER, M_PER, N_MAX, M_MAX, window="hamming")

# Plotting
d_ax = n_idx * C0 / (2 * N_PER * DELTA_F)
v_ax = m_idx * C0 / (2 * FC * T_SYM * M_PER)
vlim=[-100.0, 100.0] 
dlim=[0.0, 40.0]


plot_periodogram(per, d_ax, v_ax,
                 v_lim=vlim, 
                 d_lim=dlim,
                 title="Range-Doppler Map")

detections, eta, B = post.cfar_detector(per, noise_power_hat, FAR, N_win=4, M_win=64)

plot_binary_map_with_detections(B, detections, n_idx, m_idx, DELTA_F, T_SYM, FC, N_PER, M_PER,
                                v_lim=vlim,
                                d_lim=dlim,
                                title="Binary Map with Detections")



"""
Perform target detection (CFAR detection)
Target params estimation
Compute error
Plot
"""
