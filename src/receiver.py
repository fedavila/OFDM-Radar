import numpy as np

def ofdm_demodulation(rx_signal, CP_LEN, N_fft, F_tx):
    M, N = F_tx.shape
    rx_frame_cp = rx_signal.reshape(M, N + CP_LEN)
    rx_frame = rx_frame_cp[:, CP_LEN:]

    F_rx = np.fft.fft(rx_frame, n=N_fft, axis=1)
    F = F_rx / F_tx
    return F

def periodogram(F, N_per, M_per):
    M, N = F.shape
    F_nm = F.T
    F_dopp = np.fft.fft(F_nm, n=M_per, axis=1)
    F_rd = np.fft.ifft(F_dopp, n=N_per, axis=0)

    per = np.abs(F_rd)**2 / (N * M)
    per = np.fft.fftshift(per, axes=1)

    n_idx = np.arange(N_per)
    if M_per % 2 == 0:
        m_idx = np.arange(-M_per // 2, M_per // 2)
    else:
        m_idx = np.arange(-(M_per // 2), M_per // 2 + 1)

    return per, n_idx, m_idx


    


