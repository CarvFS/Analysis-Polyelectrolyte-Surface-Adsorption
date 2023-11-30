"""
This module provides functions to render figures and movies of the free energy surface
and how it converges over time.
| Author: Alec Glisman (GitHub: @alec-glisman)
| Date: 2023-11-24

Functions
---------
plt_pmf(cv_grid: np.ndarray, pmf: np.ndarray, ymax: float = None,
dir_fig: str = "figures", tag: str = None) -> plt.Figure:
    Render a figure of the free energy surface. This figure should be used to assess
    the convergence of the free energy surface. The figure contains 1 subplot:
    - PMF: curve should be smooth and converged

plt_pmf_diff_conv(times: np.ndarray, differences: np.ndarray,
lower_well: tuple[float, float], upper_well: tuple[float, float], tag: str = None,
ymax: float = None) -> plt.Figure:
    Render a figure of the free energy difference between two wells. This figure should
    be used to assess the convergence of the free energy surface. The figure contains
    1 subplot:
    - PMF difference: curve should be converged to a constant within an error of 1 kT

mov_pmf_conv(times: np.ndarray, cv_grid: np.ndarray, pmfs: np.ndarray, tag: str = None,
ymax: float = None, dir_fig: str = "movies") -> plt.Figure:
    Render a movie of the free energy surface. This figure should be used to assess the
    convergence of the free energy surface. The figure contains 1 subplot:
    - PMF: curve should be smooth and converged

Raises
------
AssertionError
    If the PMF and CV do not have the same size.

Notes
-----
This module requires the following columns in the DataFrame:
- time: time in picoseconds
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# import local modules
from .util import save_fig


def plt_pmf(
    cv_grid: np.ndarray,
    pmf: np.ndarray,
    ymax: float = None,
    dir_fig: str = "figures",
    tag: str = None,
) -> plt.Figure:
    """
    Render a figure of the free energy surface. This figure should be used to
    assess the convergence of the free energy surface. The figure contains 1
    subplot:
    - PMF: curve should be smooth and converged

    Parameters
    ----------
    cv_grid : np.ndarray
        Array of collective variable values.
    pmf : np.ndarray
        Array of free energies as a function of collective variable, assumed to be
        unitless.
    ymax : float, optional
        Maximum value for the y-axis, by default None
    dir_fig : str, optional
        Directory to save the figure, by default "figures"
    tag : str, optional
        Tag to append to the figure name, by default None

    Returns
    -------
    plt.Figure
        Figure of the free energy surface.
    """
    fname = "pmf"
    if tag is not None:
        fname += f"_{tag}"

    fig = plt.figure()

    # PMF: curve should be smooth and converged
    ax1 = fig.add_subplot(111)
    ax1.set_title("Potential of Mean Force")
    ax1.set_xlabel(r"$z$ [nm]")
    ax1.set_ylabel(r"$\Delta F$ [$k_\mathrm{B}T$]")
    ax1.plot(cv_grid, pmf, linewidth=3)
    if ymax is not None:
        ax1.set_ylim((0, ymax))

    # save figure
    save_fig(fig, fname, dir_fig)
    return fig


def plt_pmf_diff_conv(
    times: np.ndarray,
    differences: np.ndarray,
    lower_well: tuple[float, float],
    upper_well: tuple[float, float],
    tag: str = None,
    ymax: float = None,
) -> plt.Figure:
    """
    Render a figure of the free energy difference between two wells. This figure
    should be used to assess the convergence of the free energy surface. The
    figure contains 1 subplot:
    - PMF difference: curve should be converged to a constant within an error of
      1 kT

    Parameters
    ----------
    times : np.ndarray
        Array of times in ns.
    differences : np.ndarray
        Array of free energy differences in kT.
    lower_well : tuple[float, float]
        Tuple of (min, max) values for the lower well.
    upper_well : tuple[float, float]
        Tuple of (min, max) values for the upper well.
    tag : str, optional
        Tag to append to the figure name, by default None
    ymax : float, optional
        Maximum value for the y-axis, by default None

    Returns
    -------
    plt.Figure
        Figure of the free energy difference between two wells.
    """
    fname = "pmf_diff_conv"
    if tag is not None:
        fname += f"_{tag}"

    fig = plt.figure()

    # PMF difference: curve should be converged to a constant within an error of 1 kT
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(r"$\Delta F$ [$k_\mathrm{B} T$]")
    ax.set_title(
        f"$F({lower_well[0]}"
        + r" \leq z \leq"
        + f" {lower_well[1]}) - F({upper_well[0]}"
        + r" \leq z \leq"
        + f" {upper_well[1]})$",
        fontsize=20,
    )

    # add horizontal lines that are delta_fes[-1] +- 0.5
    ax.axhline(differences[-1] - 0.5, linestyle="--", color="k")
    ax.axhline(differences[-1] + 0.5, linestyle="--", color="k")
    # fill in horizontal region between lines with opacity 0.2
    ax.fill_between(
        [times[0] / 1e3, times[-1] / 1e3],
        differences[-1] - 0.5,
        differences[-1] + 0.5,
        alpha=0.1,
        color="k",
    )
    # plot delta F
    ax.plot(times / 1e3, differences, linewidth=3)

    # set ylim top to ymax if it is greater than ymax
    if ymax is not None:
        if max(differences) > ymax:
            ax.set_ylim(top=ymax)
        if min(differences) < -ymax:
            ax.set_ylim(bottom=-ymax)

    # save figure
    save_fig(fig, fname)
    return fig


def mov_pmf_conv(
    times: np.ndarray,
    cv_grid: np.ndarray,
    pmfs: np.ndarray,
    tag: str = None,
    ymax: float = None,
    dir_fig: str = "movies",
) -> plt.Figure:
    """
    Render a movie of the free energy surface. This figure should be used to
    assess the convergence of the free energy surface. The figure contains 1
    subplot:
    - PMF: curve should be smooth and converged

    Parameters
    ----------
    times : np.ndarray
        Array of times in ps.
    cv_grid : np.ndarray
        Array of collective variable values.
    pmfs : np.ndarray
        Array of free energies as a function of collective variable, assumed to be
        unitless.
    tag : str, optional
        Tag to append to the figure name, by default None
    ymax : float, optional
        Maximum value for the y-axis, by default None
    dir_fig : str, optional
        Directory to save the figure, by default "movies"

    Returns
    -------
    plt.Figure
        Figure of the free energy surface.
    """
    fname = "pmf_conv_movie"
    if tag is not None:
        fname += f"_{tag}"

    Path(dir_fig).mkdir(parents=True, exist_ok=True)

    fig = plt.figure()

    # setup figure
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$z$ [nm]", labelpad=10)
    ax.set_ylabel(r"$\Delta F$ [$k_\mathrm{B}T$]", labelpad=10)
    ax.set_ylim((0, ymax))
    ax.set_title("Potential of Mean Force", pad=10)

    # initialize plot elements
    idx = 0
    (curve,) = ax.plot(cv_grid, np.zeros_like(cv_grid), linewidth=3)
    text = ax.text(
        0.5,
        0.9,
        f"$t_f = ${times[idx]/1e3:.2f} ns",
        transform=ax.transAxes,
        ha="center",
    )

    def animate(frame_num: int, *fargs) -> tuple:
        idx = frame_num
        times = fargs[0]
        pmfs = fargs[1]
        curve = fargs[2]

        curve.set_ydata(pmfs[idx])
        time = times[idx] / 1e3
        text.set_text(f"$t_f = $ {time:.1f} ns")
        idx += 1

        return (curve, text)

    # animate
    n_frames = len(times) - 1
    anim = animation.FuncAnimation(
        fig,
        animate,
        blit=True,
        frames=n_frames,
        fargs=(times, pmfs, curve),
    )

    # save animation as mp4 and output tqdm progress bar
    anim.save(
        f"{dir_fig}/{fname}.mp4",
        writer="ffmpeg",
        fps=20,
        dpi=300,
    )

    return fig
