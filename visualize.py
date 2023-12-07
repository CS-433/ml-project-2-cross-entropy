from utils.energy_util import *
import matplotlib.pyplot as plt
import scipy

def visualize_energy(
    dimers_data,
    y_pred_dimers,
    trimers_data=None,
    y_pred_trimers=None,
    show_trimer=True,
    fname=None,
):
    """
    Plots energies for dimers and optionally for trimers, comparing reference data with predictions.

    Args:
        dimers_data (Dataset): Dimer dataset.
        y_pred_dimers (1D array): Predicted dimer energies.
        trimers_data (Dataset, optional): Trimer (equilateral) dataset. Required if show_trimer is True.
        y_pred_trimers (1D array, optional): Predicted (equilateral) trimer energies.
        show_trimer (bool): Whether to include trimers in the plot.
        fname (str, optional): Filename to save the plot. If None, plot is not saved.

    """
    if show_trimer and (trimers_data is None or y_pred_trimers is None):
        raise ValueError("Trimer data must be provided when show_trimer is True")

    n_ax = 2 if show_trimer else 1
    fig, ax = plt.subplots(n_ax, sharex=True, dpi=200)

    if not show_trimer:
        ax = [ax]

    distances_dimers = dimers_data.get_distance()
    energies_dimers = dimers_data.get_energy(zero_point_idx=-1)
    distances_trimers = trimers_data.get_distance()
    energies_trimers = trimers_data.get_energy(zero_point_idx=-1)
    align_to_end = lambda array: array - array[-1]
    y_pred_dimers = y_pred_dimers.flatten()
    y_pred_trimers = y_pred_trimers.flatten()

    for a in ax:
        a.axhline(0, ls="dashed", c="gray")
        a.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
        a.minorticks_on()

    # Plotting for dimers
    step = 2
    ax[0].plot(
        distances_dimers[::step],
        energies_dimers[::step],
        "s",
        markerfacecolor="None",
        label="dimer ref.",
        c="blue",
    )
    ax[0].plot(
        distances_dimers, align_to_end(y_pred_dimers), label="dimer pred.", c="blue"
    )

    # Plotting for trimers
    if show_trimer:
        ax[0].plot(
            distances_trimers,
            energies_trimers,
            "s",
            markerfacecolor="None",
            label="trimer ref.",
            c="red",
        )
        ax[0].plot(
            distances_trimers,
            align_to_end(y_pred_trimers),
            label="trimer pred.",
            c="red",
        )

    ax[0].set_xlim(3.3, 14)
    ax[0].set_ylim(-8e-2, 11e-2)
    ax[0].set_ylabel("$E$ / eV")

    energy_inter = scipy.interpolate.CubicSpline(
        distances_dimers, dimers_data.get_energy(), extrapolate=False
    )
    energies_trimers_drimers = calc_dimer_trimer_energies(
        trimers_data.raw_data, energy_inter
    )
    y_pred_inter = scipy.interpolate.CubicSpline(
        distances_dimers, y_pred_dimers, extrapolate=False
    )
    y_trimers_dimers = calc_dimer_trimer_energies(trimers_data.raw_data, y_pred_inter)
    if show_trimer:
        ax[1].plot(
            distances_trimers[::step],
            align_to_end(energies_trimers)[::step]
            - align_to_end(energies_trimers_drimers)[::step],
            "s",
            markerfacecolor="None",
            label="ref.",
            c="k",
        )
        ax[1].plot(
            distances_trimers,
            align_to_end(y_pred_trimers) - align_to_end(y_trimers_dimers),
            label="pred.",
            c="k",
        )
        ax[1].set_xlabel("d / Å")
        ax[1].set_ylabel("$E_\mathrm{trimer} - 3E_\mathrm{dimer}$ / eV")

    for a in ax:
        a.legend(
            ncol=2, handlelength=1, columnspacing=0.5, frameon=True, edgecolor="None"
        )

    fig.align_labels()

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight", transparent=False)
        
    plt.show()