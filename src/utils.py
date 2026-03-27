import yaml
import matplotlib.pyplot as plt
import numpy as np

C0 = 299792458.0
KB = 1.38064852e-23

def load_config(path="configs/simulation_parameters.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def plot_periodogram(per, n_idx, m_idx, delta_f, T_sym, fc, nper, mper, v_lim, d_lim, title="Range-Doppler Map"):
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

    # Convert to dB scale (avoid log(0))
    

    d = n_idx * C0 / (2 * nper * delta_f)
    v = m_idx * C0 / (2 * fc * T_sym * mper)

    d_mask = (d >= d_lim[0]) & (d <= d_lim[1])
    v_mask = (v >= v_lim[0]) & (v <= v_lim[1])

    per_crop = per[np.ix_(d_mask, v_mask)]
    per_crop_db = 10 * np.log10(per_crop + 1e-12)

    d_crop = d[d_mask]
    v_crop = v[v_mask]

    plt.figure(figsize=(8, 5))

    vmax = per_crop_db.max()
    vmin = per_crop_db.min()

    plt.imshow(
        per_crop_db,
        aspect='auto',
        cmap='viridis',
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        extent=[v_crop[0], v_crop[-1], d_crop[0], d_crop[-1]]
    )

    plt.colorbar(label="Power (dB)")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.title(title)

    plt.tight_layout()
    plt.show()


