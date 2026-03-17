import numpy as np
from src.utils import load_config, C0, KB
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

FS = N_FFT * DELTA_F

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


"""
Implement superposition of all target effects on the tx_signal
Implement random phase rotation on the final echo signal
Add WGN respecting the SNR to be simulated
"""

# Receiver

"""
Perform OFDM demodulation
Perform perdiodogram over F matrix
Perform target detection (detection and interpolation algorithms)
Target params estimation (estimation algorithms)
Plot
"""
