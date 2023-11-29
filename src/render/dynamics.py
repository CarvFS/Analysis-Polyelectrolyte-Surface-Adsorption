"""
Module to render figures for time series data.
| Author: Alec Glisman (GitHub: @alec-glisman)
| Date: 2023-11-29

Functions
---------
plt_opes_dynamics(df: pd.DataFrame, cv: str = "dist_chain.z", dir_fig: str = "figures",
tag: str = None) -> plt.Figure:
    Render a figure of the OPES bias dynamics.

plt_metad_dynamics(df: pd.DataFrame, cv: str = "dist_chain.z", dir_fig: str = "figures",
tag: str = None) -> plt.Figure:
    Render a figure of the MetaD bias dynamics.

plt_md_dynamics(df: pd.DataFrame, cv: str = "dist_chain.z", dir_fig: str = "figures",
tag: str = None) -> plt.Figure:
    Render a figure of the molecular dynamics CV.

Raises
------
AssertionError
    If the DataFrame does not contain the required columns.

Notes
-----
This module requires the following columns in the DataFrame:
- time: time in picoseconds
"""

# import modules
import pandas as pd
import matplotlib.pyplot as plt

# import local modules
from .util import save_fig


def plt_opes_dynamics(
    df: pd.DataFrame,
    cv: str = "dist_chain.z",
    dir_fig: str = "figures",
    tag: str = None,
) -> plt.Figure:
    """
    Render a figure of the OPES bias dynamics. This figure should be used to
    assess the convergence of the OPES bias. The figure contains 8 subplots:
    - CV: curve should show rapid sampling of the entire CV space
    - OPES Bias: bias should be slowly increasing over time
    - Lower wall bias: should hopefully be near zero for most of the simulation
    - Upper wall bias: should hopefully be near zero for most of the simulation
    - OPES time constant: should increase and converge to a constant as the bias
      becomes quasi-static
    - OPES normalization constant: starts from 1 an changes significantly when
        a new region of CV space is explored
    - OPES number of effective samples: should be increasing over time with
        a fixed ratio to the number of samples
    - OPES number of compressed kernels: should increase and plateau

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the OPES bias dynamics.
    cv : str, optional
        Name of the collective variable, by default "dist_chain.z"
    dir_fig : str, optional
        Directory to save the figure, by default "figures"
    tag : str, optional
        Tag to append to the figure name, by default None

    Returns
    -------
    plt.Figure
        Figure of the OPES bias dynamics.
    """
    fname = "opes_bias_dynamics"
    if tag is not None:
        fname += f"_{tag}"
    fig = plt.figure(figsize=(22, 9))

    # CV: curve should show rapid sampling of the entire CV space
    ax1 = fig.add_subplot(241)
    ax1.set_xlabel("Time [ns]")
    ax1.set_ylabel("$z$ [nm]")
    ax1.set_title("Collective Variable")
    ax1.scatter(df["time"] / 1e3, df[cv], s=2, alpha=0.2)

    # OPES Bias: bias should be slowly increasing over time
    ax2 = fig.add_subplot(242)
    ax2.set_xlabel("Time [ns]")
    ax2.set_ylabel("OPES Bias [kJ/mol]")
    ax2.set_title("OPES Bias")
    ax2.scatter(df["time"] / 1e3, df["opes.bias"], s=2, alpha=0.2)

    # Lower wall bias: should hopefully be near zero for most of the simulation
    ax7 = fig.add_subplot(243)
    ax7.set_xlabel("Time [ns]")
    ax7.set_ylabel("Bias [kJ/mol]")
    ax7.set_title("Lower Wall Bias")
    ax7.scatter(df["time"] / 1e3, df["lower_wall.bias"], s=2, alpha=0.2)

    # Upper wall bias: should hopefully be near zero for most of the simulation
    ax8 = fig.add_subplot(244)
    ax8.set_xlabel("Time [ns]")
    ax8.set_ylabel("Bias [kJ/mol]")
    ax8.set_title("Upper Wall Bias")
    ax8.scatter(df["time"] / 1e3, df["upper_wall.bias"], s=2, alpha=0.2)

    # OPES time constant: should increase and converge to a constant as the bias
    # becomes quasi-static
    ax3 = fig.add_subplot(245)
    ax3.set_xlabel("Time [ns]")
    ax3.set_ylabel("$c{(t)}$")
    ax3.set_title("Quasi-Static Bias")
    ax3.plot(df["time"] / 1e3, df["opes.rct"], linewidth=3)

    # OPES normalization constant: starts from 1 an changes significantly when
    # a new region of CV space is explored
    ax4 = fig.add_subplot(246)
    ax4.set_xlabel("Time [ns]")
    ax4.set_ylabel("$Z_n$")
    ax4.set_title("CV Exploration")
    ax4.plot(df["time"] / 1e3, df["opes.zed"], linewidth=3)

    # OPES number of effective samples: should be increasing over time with
    # a fixed ratio to the number of samples
    ax5 = fig.add_subplot(247)
    ax5.set_xlabel("Time [ns]")
    ax5.set_ylabel(r"$n_\mathrm{eff}$")
    ax5.set_title("Effective Sample Size")
    ax5.plot(df["time"] / 1e3, df["opes.neff"], linewidth=3)

    # OPES number of compressed kernels: should increase and plateau
    ax6 = fig.add_subplot(248)
    ax6.set_xlabel("Time [ns]")
    ax6.set_ylabel(r"$n_\mathrm{ker}$")
    ax6.set_title("Compressed Kernels")
    ax6.plot(df["time"] / 1e3, df["opes.nker"], linewidth=3)

    # save figure
    save_fig(fig, fname, dir_fig)
    return fig


def plt_metad_dynamics(
    df: pd.DataFrame,
    cv: str = "dist_chain.z",
    dir_fig: str = "figures",
    tag: str = None,
) -> plt.Figure:
    """
    Render a figure of the MetaD bias dynamics. This figure should be used to
    assess the convergence of the MetaD bias. The figure contains 6 subplots:
    - CV: curve should show rapid sampling of the entire CV space
    - MetaD Bias: bias should be slowly increasing over time
    - Lower wall bias: should hopefully be near zero for most of the simulation
    - Upper wall bias: should hopefully be near zero for most of the simulation
    - MetaD time constant: should increase over time
    - MetaD bias: should increase and plateau

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the MetaD bias dynamics.
    cv : str, optional
        Name of the collective variable, by default "dist_chain.z"
    dir_fig : str, optional
        Directory to save the figure, by default "figures"
    tag : str, optional
        Tag to append to the figure name, by default None

    Returns
    -------
    plt.Figure
        Figure of the MetaD bias dynamics.
    """
    fname = "metad_bias_dynamics"
    if tag is not None:
        fname += f"_{tag}"
    fig = plt.figure(figsize=(9, 15))

    # CV: curve should show rapid sampling of the entire CV space
    ax1 = fig.add_subplot(321)
    ax1.set_xlabel("Time [ns]")
    ax1.set_ylabel("$z$ [nm]")
    ax1.set_title("Collective Variable")
    ax1.scatter(df["time"] / 1e3, df[cv], s=2, alpha=0.2)

    ax2 = fig.add_subplot(322)
    ax2.set_xlabel("Time [ns]")
    ax2.set_ylabel("MetaD Bias [kJ/mol]")
    ax2.set_title("MetaD Bias")
    ax2.scatter(df["time"] / 1e3, df["metad.bias"], s=2, alpha=0.2)

    # Lower wall bias: should hopefully be near zero for most of the simulation
    ax7 = fig.add_subplot(323)
    ax7.set_xlabel("Time [ns]")
    ax7.set_ylabel("Bias [kJ/mol]")
    ax7.set_title("Lower Wall Bias")
    ax7.scatter(df["time"] / 1e3, df["lower_wall.bias"], s=2, alpha=0.2)

    # Upper wall bias: should hopefully be near zero for most of the simulation
    ax8 = fig.add_subplot(324)
    ax8.set_xlabel("Time [ns]")
    ax8.set_ylabel("Bias [kJ/mol]")
    ax8.set_title("Upper Wall Bias")
    ax8.scatter(df["time"] / 1e3, df["upper_wall.bias"], s=2, alpha=0.2)

    # MetaD time constant: should increase over time
    ax3 = fig.add_subplot(325)
    ax3.set_xlabel("Time [ns]")
    ax3.set_ylabel("$c{(t)}$")
    ax3.set_title("Quasi-Static Bias")
    ax3.plot(df["time"] / 1e3, df["metad.rct"], linewidth=3)

    # MetaD bias: should increase and plateau
    ax2 = fig.add_subplot(326)
    ax2.set_xlabel("Time [ns]")
    ax2.set_ylabel("MetaD R-Bias [kJ/mol]")
    ax2.set_title("MetaD R-Bias")
    ax2.scatter(df["time"] / 1e3, df["metad.rbias"], s=2, alpha=0.2)

    # save figure
    save_fig(fig, fname, dir_fig)
    return fig


def plt_md_dynamics(
    df: pd.DataFrame,
    cv: str = "dist_chain.z",
    dir_fig: str = "figures",
    tag: str = None,
) -> plt.Figure:
    """
    Render a figure of the molecular dynamics CV. This figure should be used to
    assess the sampling of the unconstrained simulation. The figure contains 1
    subplot:
    - CV: curve should show rapid sampling of the entire CV space

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the MetaD bias dynamics.
    cv : str, optional
        Name of the collective variable, by default "dist_chain.z"
    dir_fig : str, optional
        Directory to save the figure, by default "figures"
    tag : str, optional
        Tag to append to the figure name, by default None

    Returns
    -------
    plt.Figure
        Figure of the MetaD bias dynamics.
    """
    fname = "md_dynamics"
    if tag is not None:
        fname += f"_{tag}"

    fig = plt.figure()

    # CV: curve should show rapid sampling of the entire CV space
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("Time [ns]")
    ax1.set_ylabel("$z$ [nm]")
    ax1.set_title("Collective Variable")
    ax1.scatter(df["time"] / 1e3, df[cv], s=2, alpha=0.2)

    # save figure
    save_fig(fig, fname, dir_fig)
    return fig
