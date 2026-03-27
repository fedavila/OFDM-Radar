import numpy as np

import numpy as np


def cfar_detector(
    per: np.ndarray,
    noise_power: float,
    FAR: float,
    N_win: int = 3,
    M_win: int = 3,
):
    """
    Braun-style iterative peak detection on a cropped periodogram,
    without explicit peak interpolation.

    Parameters
    ----------
    per : ndarray, shape (N_max, 2*M_max+1)
        Cropped periodogram in linear scale.
    noise_power : float
        Estimated periodogram-bin noise power.
    FAR : float
        Required false alarm rate over the cropped map.
    N_win : int
        Suppression window size in range bins.
    M_win : int
        Suppression window size in Doppler bins.

    Returns
    -------
    detections : list of dict
        Each dict contains:
            - 'n_bin': range-bin index
            - 'm_col': Doppler column index inside cropped periodogram
            - 'm_bin': centered Doppler-bin index
            - 'peak_power': detected peak value
    eta : float
        Detection threshold.
    B : ndarray
        Final binary map (True = available, False = suppressed).
    """
    if per.ndim != 2:
        raise ValueError("per must be 2D")

    N_max, M_crop = per.shape

    if noise_power <= 0:
        raise ValueError("noise_power must be positive")
    if FAR <= 0:
        raise ValueError("FAR must be positive")

    n_cells = N_max * M_crop
    if FAR >= n_cells:
        raise ValueError("FAR must be smaller than number of cells")

    # Braun FAR-based threshold on cropped map
    eta = -noise_power * np.log(FAR / n_cells)

    B = np.ones_like(per, dtype=bool)
    detections = []

    r_n = N_win // 2
    r_m = M_win // 2
    M_max = (M_crop - 1) // 2

    while True:
        masked_per = np.where(B, per, -np.inf)
        flat_idx = np.argmax(masked_per)
        peak = masked_per.flat[flat_idx]

        if not np.isfinite(peak) or peak < eta:
            break

        n0, m0 = np.unravel_index(flat_idx, per.shape)
        m_bin = m0 - M_max

        detections.append(
            {
                "n_bin": n0,
                "m_col": m0,
                "m_bin": m_bin,
                "peak_power": float(peak),
            }
        )

        n_start = max(0, n0 - r_n)
        n_stop  = min(N_max, n0 + r_n + 1)
        m_start = max(0, m0 - r_m)
        m_stop  = min(M_crop, m0 + r_m + 1)

        B[n_start:n_stop, m_start:m_stop] = False

    return detections, eta, B