import yaml
import matplotlib.pyplot as plt
import numpy as np

C0 = 299792458.0
KB = 1.38064852e-23

def load_config(path="configs/simulation_parameters.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def plot_periodogram(per, n_idx, m_idx, title="Range-Doppler Map"):
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
    per_db = 10 * np.log10(per + 1e-12)

    plt.figure(figsize=(8, 5))

    # extent maps indices to axes
    extent = [
        m_idx[0], m_idx[-1],   # Doppler axis (x)
        n_idx[0], n_idx[-1]    # Range axis (y, flipped)
    ]

    plt.imshow(
        per_db,
        aspect='auto',
        cmap='viridis',
        extent=extent
    )

    plt.colorbar(label="Power (dB)")
    plt.xlabel("Doppler bins")
    plt.ylabel("Range bins")
    plt.title(title)

    plt.tight_layout()
    plt.show()


