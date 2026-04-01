import numpy as np
from src.utils import load_config, plot_distance_error, C0
from src import environment as env
from src import transmitter as tx
from src import receiver as rx
from src import post_processing as post

config = load_config(path='configs/evaluation_parameters.yaml')

FC = config['radar']['fc']
P_TX_DBM = config['radar']['Ptx'] 
G_DBI = config['radar']['G']
TEMP = config['radar']['temperature']
NF_DB = config['radar']['NF']
H = config['radar']['H']
FAR = config['radar']['FAR']

MODULATION = config['ofdm']['modulation']
DELTA_F = config['ofdm']["delta_f"]
T_CP = config['ofdm']['T_cp']
M = config['ofdm']['M']

RUNS = config['evaluation']['runs']
B_START = config['evaluation']['B_start']
B_END = config['evaluation']['B_end']
B_STEP = config['evaluation']['B_step']

N_PER = config["periodogram"]['N_per']
M_PER = config["periodogram"]['M_per']

P_tx = 10 ** (P_TX_DBM / 10) * 1e-3
G = 10 ** (G_DBI / 10)

T_SYM = (1.0 / DELTA_F) + T_CP    # seconds

dmax = T_CP * C0 / 2
vmax = DELTA_F * C0 / (10 * FC)

N_MAX = int(np.ceil(2 * dmax * N_PER * DELTA_F) / C0)
M_MAX = int(np.ceil(2 * vmax * FC * T_SYM * M_PER) / C0)

if MODULATION == "BPSK": 
    BITS_PER_SYMBOL = 1
else: 
    BITS_PER_SYMBOL = 2


bandwidths = np.arange(B_START, B_END + 1, B_STEP)
abs_errors = np.zeros((RUNS, len(bandwidths))) 

for j, B in enumerate(bandwidths):
    
    N = int(B * 1e6 / DELTA_F)
    N_FFT = int(2 ** np.ceil(np.log2(N)))
    FS = N_FFT * DELTA_F
    CP_LEN = int(np.round(T_CP * FS)) # samples
    N_BITS = BITS_PER_SYMBOL * N * M
    print(f"Running: B = {B} MHz; N = {N}")

    for k in range(RUNS):
        # Transmitter ============================================================================================================
        bits = tx.generate_bits(N_BITS)
        symbols = tx.data_modulator(bits, BITS_PER_SYMBOL)

        F_tx = symbols.reshape(M, N)

        tx_signal = np.sqrt(P_tx) * tx.ofdm_modulation(F_tx, CP_LEN, N_FFT)
        tx_signal *= np.sqrt(G)


        # Environment =============================================================================================================

        target = env.Target(**config['target'][0])

        echos = np.zeros_like(tx_signal, dtype=complex)

        echos += env.apply_target_echo(target, tx_signal, CP_LEN, FS, FC)

        rx_signal = env.apply_awgn_nf(echos, TEMP, NF_DB, FS)


        # Receiver ==================================================================================================================

        rx_signal *= np.sqrt(G)
        F = rx.ofdm_demodulation(rx_signal, CP_LEN, N_FFT, F_tx)

        per, n_idx, m_idx, noise_power_hat, c_norm = rx.crop_periodogram(F, N_PER, M_PER, N_MAX, M_MAX, window="hamming")


        # Post processing =======================================================================================
        detections, eta, B = post.cfar_detector(per, noise_power_hat, FAR, N_win=12, M_win=128)

        det_targets = []
        for t, det in enumerate(detections):
            n_hat = det["n_bin"]
            m_hat = det["m_bin"]
            peak_hat = det["peak_power"]

            d_hat = n_hat * C0 / (2 * DELTA_F * N_PER)
            v_hat = m_hat * C0 / (2 * FC * T_SYM * M_PER)
            rcs_hat = c_norm * peak_hat * (4*np.pi)**3 * FC**2 * d_hat**4 / (P_tx * N * M * G**2 * C0**2)

            det_targets.append(
                {
                    "d_hat": d_hat,
                    "v_hat": v_hat,
                    "rcs_hat": rcs_hat,
                }
            )
        
        abs_errors[k, j] = np.abs(det_targets[0]["d_hat"] - target.distance)

plot_distance_error(abs_errors, bandwidths)        

        


