# %% Imports

import numpy as np
from src.utils import load_config, plot_periodogram_and_detections, C0
from src import environment as env
from src import transmitter as tx
from src import receiver as rx
from src import post_processing as post


# %% Run simulation
config = load_config(path='configs/evaluation_parameters.yaml')

FC = config['radar']['fc']
P_TX_DBM = config['radar']['Ptx'] 
G_DBI = config['radar']['G']
TEMP = config['radar']['temperature']
NF_DB = config['radar']['NF']
PFA = config['radar']['PFA']

MODULATION = config['ofdm']['modulation']
DELTA_F = config['ofdm']["delta_f"]
T_CP = config['ofdm']['T_cp']
M = config['ofdm']['M']

RUNS = config['evaluation']['runs']
B_START = config['evaluation']['B_start']
B_END = config['evaluation']['B_end']
B_STEP = config['evaluation']['B_step']

WINDOW = config['periodogram']['window']

P_tx = 10 ** (P_TX_DBM / 10) * 1e-3
G = 10 ** (G_DBI / 10)

T_SYM = (1.0 / DELTA_F) + T_CP    # seconds

dmax = T_CP * C0 / 2
vmax = DELTA_F * C0 / (10 * FC)


if MODULATION == "BPSK": 
    BITS_PER_SYMBOL = 1
else: 
    BITS_PER_SYMBOL = 2


bandwidths = np.arange(B_START, B_END + 1, B_STEP)
abs_errors = np.zeros((RUNS, len(bandwidths))) 

for j, bw in enumerate(bandwidths):
    
    N = int(bw * 1e6 / DELTA_F)
    N_FFT = int(2 ** np.ceil(np.log2(N)))
    FS = N_FFT * DELTA_F
    CP_LEN = int(np.round(T_CP * FS)) # samples
    N_BITS = BITS_PER_SYMBOL * N * M

    if config["periodogram"]["configure"]:
        N_PER = config["periodogram"]['N_per']
        M_PER = config["periodogram"]['M_per']
    else:
        N_PER = 4 * N
        M_PER = 4 * M
        
    N_MAX = int(np.ceil(2 * dmax * N_PER * DELTA_F) / C0)
    M_MAX = int(np.ceil(2 * vmax * FC * T_SYM * M_PER) / C0)

    print(f"Running: B = {bw} MHz; N = {N}")
    for k in range(RUNS):
        # Transmitter ============================================================================================================
        bits = tx.generate_bits(N_BITS)
        symbols = tx.data_modulator(bits, BITS_PER_SYMBOL)

        F_tx = symbols.reshape(M, N)

        tx_signal = np.sqrt(P_tx) * tx.ofdm_modulation(F_tx, CP_LEN, N_FFT)
        tx_signal *= np.sqrt(G)


        # Environment =============================================================================================================
        real_distance = np.random.uniform(2.0, dmax - 1.0) 
        real_speed = np.random.uniform(-vmax + 1.0, vmax - 1.0)
        real_rcs = config['target'][0]['rcs']
        
        target = env.Target(distance=real_distance, velocity=real_speed, rcs=real_rcs)

        echos = np.zeros_like(tx_signal, dtype=complex)

        echos += env.apply_target_echo(target, tx_signal, CP_LEN, FS, FC)

        rx_signal = env.apply_awgn_nf(echos, TEMP, NF_DB, N * DELTA_F)


        # Receiver ==================================================================================================================

        rx_signal *= np.sqrt(G)
        F = rx.ofdm_demodulation(rx_signal, CP_LEN, N_FFT, F_tx)

        per, n_idx, m_idx, noise_power_hat, c_norm = rx.crop_periodogram(F, N_PER, M_PER, N_MAX, M_MAX, window=WINDOW)


        
        # Post processing =======================================================================================
        N_win = 7 * int(N_PER // N)
        M_win = 7 * int(M_PER // M)
        detections, eta, B = post.cfar_detector(per, noise_power_hat, PFA, N_win=N_win, M_win=M_win)

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
        d_ax = n_idx * C0 / (2 * N_PER * DELTA_F)
        v_ax = m_idx * C0 / (2 * FC * T_SYM * M_PER)
        vlim=[-100.0, 100.0] 
        dlim=[0.0, 40.0]

        # plot_periodogram_and_detections(per, B, det_targets, eta, d_ax, v_ax, v_lim=None, d_lim=None)
        abs_errors[k, j] = np.abs(det_targets[0]["d_hat"] - target.distance)

# %%  Plot
import matplotlib.pyplot as plt

mean = np.mean(abs_errors, axis=0)
std = np.std(abs_errors, axis=0)

mean_db = 10 * np.log10(mean)
std_db = 10 * np.log10(mean + std) - mean_db

plt.figure(figsize=(8, 5))

plt.plot(bandwidths, mean_db, label='Mean absolute error', color='blue')

plt.fill_between(
    bandwidths,
    mean_db - std_db,
    mean_db + std_db,
    color="blue",
    alpha=0.3,
    label="±1 std"
)

plt.xlabel("Bandwidth [MHz]")
plt.ylabel(r"$|e_d|~[dB]$")
plt.title("Distance Error vs Bandwidth")
plt.legend()
plt.grid(linestyle=':')

plt.tight_layout()
plt.savefig("results/N_evaluation.png")
plt.show()        

# %%
