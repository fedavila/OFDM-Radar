import yaml
import matplotlib.pyplot as plt
import numpy as np

C0 = 299792458.0
KB = 1.38064852e-23

def load_config(path="configs/simulation_parameters.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def plot_periodogram(per, d, v, v_lim=None, d_lim=None, title="Range-Doppler Map"):
    """ 
    Plot periodogram as a colormap (range-Doppler map).

    Parameters
    ----------
    per : ndarray (N_per, M_per)
        Periodogram
    n_idx : ndarray
        Range bin indices
    m_idx : ndarray
        Doppler bin indices (already centered if fftshift was used)
    """
    
    per_db = 10 * np.log10((per + 1e-15) * 1000)

    if d_lim is not None:
        d_mask = (d >= d_lim[0]) & (d <= d_lim[1])
        per_db = per_db[d_mask, :]
        d = d[d_mask]

    if v_lim is not None:
        v_mask = (v >= v_lim[0]) & (v <= v_lim[1])
        per_db = per_db[:, v_mask]
        v = v[v_mask]

    plt.figure(figsize=(8, 5))

    vmax = per_db.max()
    vmin = per_db.min()

    plt.imshow(
        per_db,
        aspect='auto',
        cmap='viridis',
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        extent=[v[0], v[-1], d[0], d[-1]]
    )

    plt.colorbar(label="Received power (dBm)")
    plt.xlabel("Relative speed (m/s)")
    plt.ylabel("Distance (m)")
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_binary_map_with_detections(
    B,
    detections,
    n_idx,
    m_idx,
    delta_f,
    T_sym,
    fc,
    nper,
    mper,
    v_lim=None,
    d_lim=None,
    title="Binary Map with Detections",
):
    """
    Plot the binary map B in physical coordinates and overlay detections.

    Parameters
    ----------
    B : ndarray, shape (N_max, 2*M_max+1)
        Binary map (True/1 = available, False/0 = suppressed)
    detections : list of dict
        Output of detect_targets_braun_no_interp(...)
    n_idx : ndarray
        Range bin indices
    m_idx : ndarray
        Doppler bin indices (centered)
    delta_f : float
        Subcarrier spacing (Hz)
    T_sym : float
        OFDM symbol duration incl. CP (s)
    fc : float
        Carrier frequency (Hz)
    nper : int
        Zero-padded range FFT size
    mper : int
        Zero-padded Doppler FFT size
    v_lim : list[float] | None
        Velocity limits [vmin, vmax] in m/s
    d_lim : list[float] | None
        Distance limits [dmin, dmax] in m
    """

    # Bin -> physical axes
    d = n_idx * C0 / (2 * nper * delta_f)
    v = m_idx * C0 / (2 * fc * T_sym * mper)

    # Make numeric image
    B_plot = B.astype(int)

    # Build masks exactly like in your periodogram plot
    d_mask = np.ones_like(d, dtype=bool)
    v_mask = np.ones_like(v, dtype=bool)

    if d_lim is not None:
        d_mask = (d >= d_lim[0]) & (d <= d_lim[1])

    if v_lim is not None:
        v_mask = (v >= v_lim[0]) & (v <= v_lim[1])

    # Apply masks to image and axes
    B_plot = B_plot[np.ix_(d_mask, v_mask)]
    d_plot = d[d_mask]
    v_plot = v[v_mask]

    plt.figure(figsize=(8, 5))

    plt.imshow(
        B_plot,
        aspect="auto",
        cmap="gray_r",      # 1 -> white, 0 -> black
        origin="lower",
        vmin=0,
        vmax=1,
        extent=[v_plot[0], v_plot[-1], d_plot[0], d_plot[-1]],
    )

    # Overlay detections
    for det in detections:
        n_bin = det["n_bin"]
        m_bin = det["m_bin"]

        d_det = n_bin * C0 / (2 * nper * delta_f)
        v_det = m_bin * C0 / (2 * fc * T_sym * mper)

        # Only plot if inside displayed limits
        if d_lim is not None and not (d_lim[0] <= d_det <= d_lim[1]):
            continue
        if v_lim is not None and not (v_lim[0] <= v_det <= v_lim[1]):
            continue

        plt.plot(v_det, d_det, "ro", markersize=6, markeredgecolor="k")

    plt.colorbar(label="1 = available, 0 = suppressed")
    plt.xlabel("Relative speed (m/s)")
    plt.ylabel("Distance (m)")
    plt.title(title)

    plt.tight_layout()
    plt.show()
