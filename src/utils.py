import yaml
import matplotlib.pyplot as plt
import numpy as np

C0 = 299792458.0
KB = 1.38064852e-23

def load_config(path="configs/simulation_parameters.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def plot_periodogram(per, eta, d, v, v_lim=None, d_lim=None, title="Range-Doppler Map"):
    """ 
    Plot periodogram as a colormap (range-Doppler map).

    """
    
    per_db = 10 * np.log10((per) * 1000)

    if d_lim is not None:
        d_mask = (d >= d_lim[0]) & (d <= d_lim[1])
        per_db = per_db[d_mask, :]
        d = d[d_mask]

    if v_lim is not None:
        v_mask = (v >= v_lim[0]) & (v <= v_lim[1])
        per_db = per_db[:, v_mask]
        v = v[v_mask]

    plt.figure(figsize=(8, 5))

    
    eta_dbm = 10 * np.log10(eta * 1000)
    im = plt.imshow(
        per_db,
        aspect='auto',
        cmap='viridis',
        origin='lower',
        vmin= eta_dbm,
        extent=[v[0], v[-1], d[0], d[-1]]
    )

    # Overlay threshold contour
    # plt.contour(
    #     v,
    #     d,
    #     per_db,
    #     levels=[eta_dbm],
    #     colors="red",
    #     linewidths=1
    # )

    plt.colorbar(im, label="Received power (dBm)")
    plt.xlabel("Relative speed (m/s)")
    plt.ylabel("Distance (m)")
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_detections(B, detections, d ,v, v_lim=None, d_lim=None, title="Binary Map with Detections",):
    """
    Plot the binary map B in physical coordinates and overlay detections.

    """

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
        1 - B_plot,
        aspect="auto",
        cmap="gray_r",      # 1 -> white, 0 -> black
        origin="lower",
        vmin=0,
        vmax=1,
        extent=[v_plot[0], v_plot[-1], d_plot[0], d_plot[-1]],
    )

    # Overlay detections
    for det in detections:
        d_hat = det["d_hat"]
        v_hat = det["v_hat"]

        # Only plot if inside displayed limits
        if d_lim is not None and not (d_lim[0] <= d_hat <= d_lim[1]):
            continue
        if v_lim is not None and not (v_lim[0] <= v_hat <= v_lim[1]):
            continue

        plt.plot(v_hat, d_hat, "x", markersize=6, color="red")

    plt.colorbar(label="0 = available, 1 = suppressed")
    plt.xlabel("Relative speed (m/s)")
    plt.ylabel("Distance (m)")
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_periodogram_and_detections(per, B, detections, eta, d, v,
                          v_lim=None, d_lim=None,
                          title1="Range-Doppler Map",
                          title2="Binary Map with Detections"):
    """
    Plot Range-Doppler map and Binary detection map side-by-side.
    """

    fig, axs = plt.subplots(2, 1, figsize=(6, 7))

    # --- Periodogram ---
    per_db = 10 * np.log10(per * 1000)

    d_plot = d.copy()
    v_plot = v.copy()
    per_plot = per_db.copy()

    if d_lim is not None:
        d_mask = (d_plot >= d_lim[0]) & (d_plot <= d_lim[1])
        per_plot = per_plot[d_mask, :]
        d_plot = d_plot[d_mask]

    if v_lim is not None:
        v_mask = (v_plot >= v_lim[0]) & (v_plot <= v_lim[1])
        per_plot = per_plot[:, v_mask]
        v_plot = v_plot[v_mask]

    eta_dbm = 10 * np.log10(eta * 1000)
    im0 = axs[0].imshow(
        per_plot,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        vmin = eta_dbm,
        extent=[v_plot[0], v_plot[-1], d_plot[0], d_plot[-1]],
    )

    # Overlay threshold contour
    # axs[0].contour(
    #     v_plot,
    #     d_plot,
    #     per_plot,
    #     levels=[eta_dbm],
    #     colors="red",
    #     linewidths=1
    # )

    axs[0].set_title(title1)
    axs[0].set_xlabel("Relative speed (m/s)")
    axs[0].set_ylabel("Distance (m)")
    fig.colorbar(im0, ax=axs[0], label="Received power (dBm)")

    # --- Binary map ---
    B_plot = B.astype(int)

    d_mask = np.ones_like(d, dtype=bool)
    v_mask = np.ones_like(v, dtype=bool)

    if d_lim is not None:
        d_mask = (d >= d_lim[0]) & (d <= d_lim[1])

    if v_lim is not None:
        v_mask = (v >= v_lim[0]) & (v <= v_lim[1])

    B_plot = B_plot[np.ix_(d_mask, v_mask)]
    d_plot = d[d_mask]
    v_plot = v[v_mask]

    im1 = axs[1].imshow(
        1 - B_plot,
        aspect="auto",
        cmap="gray_r",
        origin="lower",
        vmin=0,
        vmax=1,
        extent=[v_plot[0], v_plot[-1], d_plot[0], d_plot[-1]],
    )

    # Overlay detections
    for det in detections:
        d_hat = det["d_hat"]
        v_hat = det["v_hat"]

        if d_lim is not None and not (d_lim[0] <= d_hat <= d_lim[1]):
            continue
        if v_lim is not None and not (v_lim[0] <= v_hat <= v_lim[1]):
            continue

        axs[1].plot(v_hat, d_hat, "x", color="red", markersize=6)

    axs[1].set_title(title2)
    axs[1].set_xlabel("Relative speed (m/s)")
    axs[1].set_ylabel("Distance (m)")
    axs[1].grid(linestyle=":")
    fig.colorbar(im1, ax=axs[1], label="0 = available, 1 = suppressed")

    plt.tight_layout()
    plt.savefig("results/2D_periodogram.png")
    plt.show()



def plot_periodogram_3d(per, eta, d, v, v_lim=None, d_lim=None,
                        title="3D Range-Doppler Map"):
    """
    Plot 3D do periodograma.

    Parameters
    ----------
    per : ndarray
        Periodograma em escala linear, shape (len(d), len(v))
    d : ndarray
        Eixo de distância
    v : ndarray
        Eixo de velocidade
    v_lim : tuple/list, optional
        Limites de velocidade [vmin, vmax]
    d_lim : tuple/list, optional
        Limites de distância [dmin, dmax]
    floor_dbm : float
        Piso apenas para visualização
    """

    per_db = 10 * np.log10(per * 1000)
    # per_db[~np.isfinite(per_db)] = floor_dbm
    # per_db = np.maximum(per_db, floor_dbm)

    d_plot = d.copy()
    v_plot = v.copy()
    per_plot = per_db.copy()

    if d_lim is not None:
        d_mask = (d_plot >= d_lim[0]) & (d_plot <= d_lim[1])
        per_plot = per_plot[d_mask, :]
        d_plot = d_plot[d_mask]

    if v_lim is not None:
        v_mask = (v_plot >= v_lim[0]) & (v_plot <= v_lim[1])
        per_plot = per_plot[:, v_mask]
        v_plot = v_plot[v_mask]

    V, D = np.meshgrid(v_plot, d_plot)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        V, D, per_plot,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )

    # --- Threshold plane ---
    eta_dbm = 10 * np.log10(eta * 1000)
    # Z_eta = np.full_like(per_plot, eta_dbm)

    # ax.plot_surface(
    #     V, D, Z_eta,
    #     color='red',
    #     alpha=0.3
    # )

    # mask = per_plot >= eta_dbm

    # ax.scatter(
    #     V[mask],
    #     D[mask],
    #     per_plot[mask],
    #     color='red',
    #     s=5
    # )

    # ax.contour(
    #     V, D, per_plot,
    #     levels=[eta_dbm],
    #     colors='red',
    #     offset=eta_dbm
    # )

    ax.set_xlabel("Relative speed (m/s)")
    ax.set_ylabel("Distance (m)")
    ax.set_zlabel("Received power (dBm)")
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label="Received power (dBm)")

    plt.tight_layout()
    plt.savefig("results/3D_periodogram.png")
    plt.show()


def plot_distance_error_db(error_matrix, bandwidths):

    mean = np.mean(error_matrix, axis=0)
    std = np.std(error_matrix, axis=0)

    mean_db = 10 * np.log10(mean)
    std_db = 10 * np.log10(mean + std) - mean_db

    plt.figure(figsize=(8, 5))

    plt.plot(bandwidths, mean, label='Mean absolute error', color='blue')

    plt.fill_between(
        bandwidths,
        mean_db - std_db,
        mean_db + std_db,
        color="blue",
        alpha=0.3,
        label="±1 std"
    )

    plt.xlabel("Bandwidth [MHz]")
    plt.ylabel(r"$|e_d|~[m]$")
    plt.title("Distance Error vs Bandwidth")
    plt.legend()
    plt.grid(linestyle=':')

    plt.tight_layout()
    plt.show()