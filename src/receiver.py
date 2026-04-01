import numpy as np
from scipy.signal import windows

def ofdm_demodulation(rx_signal, CP_LEN, N_fft, F_tx):
    M, N = F_tx.shape
    rx_frame_cp = rx_signal.reshape(M, N_fft + CP_LEN) 
    rx_frame = rx_frame_cp[:, CP_LEN:] 

    F_rx = np.fft.fft(rx_frame, n=N_fft, axis=1)

    F = F_rx[:, :N] / F_tx
    return F



def periodogram(F, N_per, M_per, window="boxcar"):
    M, N = F.shape
    F_nm = F.T

    if window == "hamming":
        w_range = np.hamming(N)
        w_doppler = np.hamming(M)
        W = 1 / (np.linalg.norm(w_range)**2 * np.linalg.norm(w_doppler)**2) * np.outer(w_range, w_doppler)
        F_nm *= W
    elif window == "blackman-harris":
        w_range = windows.blackmanharris(N)
        w_doppler = windows.blackmanharris(M)
        W = 1 / (np.linalg.norm(w_range)**2 * np.linalg.norm(w_doppler)**2) * np.outer(w_range, w_doppler)
        F_nm *= W

    F_dopp = np.fft.fft(F_nm, n=M_per, axis=1)
    F_rd = N_per * np.fft.ifft(F_dopp, n=N_per, axis=0)

    per = np.abs(F_rd)**2 / (N * M)
    per = np.fft.fftshift(per, axes=1)

    n_idx = np.arange(N_per)
    if M_per % 2 == 0:
        m_idx = np.arange(-M_per // 2, M_per // 2)
    else:
        m_idx = np.arange(-(M_per // 2), M_per // 2 + 1)

    return per, n_idx, m_idx




def crop_periodogram(F, N_per, M_per, N_max, M_max, window="boxcar"):
    M, N = F.shape
    F_nm = F.T

    if window == "hamming":
        w_range = np.hamming(N)
        w_doppler = np.hamming(M)
        W = 1 / (np.linalg.norm(w_range)**2 * np.linalg.norm(w_doppler)**2) * np.outer(w_range, w_doppler)
        F_nm *= W
    elif window == "blackman-harris":
        w_range = windows.blackmanharris(N)
        w_doppler = windows.blackmanharris(M)
        W = 1 / (np.linalg.norm(w_range)**2 * np.linalg.norm(w_doppler)**2) * np.outer(w_range, w_doppler)
        F_nm *= W

    F_range = N_per * np.fft.ifft(F_nm, n=N_per, axis=0) # N_per scaling because of numpy's ifft implementation

    # Noise power estimation ----------------------------------
    noise_rows = F_range[-1:, :]
    noise_dopp = np.fft.fft(noise_rows, n=M_per, axis=1)
    noise_per = np.abs(noise_dopp)**2 / (N * M)
    sigma2_hat = noise_per.mean()
    # -----------------------------------------------------------

    F_range = F_range[:N_max, :]

    F_dopp = np.fft.fft(F_range, n=M_per, axis=1)
    F_dopp = np.fft.fftshift(F_dopp, axes=1)
    F_dopp = F_dopp[:, (M_per // 2) - M_max : (M_per // 2) + M_max + 1]
    

    per = np.abs(F_dopp)**2 / (N * M)

    n_idx = np.arange(N_max)
    m_idx = np.arange(-M_max, M_max + 1)

    peak_corr = (N * M / np.sum(W))**2 # peak correction because of windowing usage

    return per, n_idx, m_idx, sigma2_hat, peak_corr



