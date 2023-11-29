"""
This module provides helper functions for general plotting. Matplotlib is used for
the plotting backend.
| Author: Alec Glisman (GitHub: @alec-glisman)
| Date: 2023-11-29

Functions
---------
set_style()
    Set the style of the plots.
save_fig(fig: plt.Figure, fname: str, dir_fig: str = "figures") -> None:
    Save a figure to the output directory.
close_fig(fig: plt.figure) -> None:
    Close the figure and clear the memory.

Raises
------
FileNotFoundError
    If the output directory does not exist.
"""

# Standard library
import gc
from pathlib import Path

# Third-party packages
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_style():
    """
    Set the style of the plots.
    """
    # Pyplot parameters
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["agg.path.chunksize"] = 10000
    plt.style.use(["seaborn-v0_8-colorblind"])

    mpl.rcParams.update(
        {
            "axes.formatter.use_mathtext": True,
            "axes.labelpad": 10,
            "axes.titlepad": 10,
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "axes.linewidth": 1.5,
            "axes.unicode_minus": False,
            "figure.autolayout": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            "font.size": 14,
            "legend.columnspacing": 1,
            "legend.fontsize": 14,
            "legend.handlelength": 1.25,
            "legend.labelspacing": 0.25,
            "legend.loc": "best",
            "legend.title_fontsize": 16,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "k",
            "lines.linewidth": 2,
            "lines.markersize": 10,
            "mathtext.fontset": "cm",
            "savefig.dpi": 1200,
            "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amsbsy,"
            + r"amssymb,bm,amsthm,mathrsfs,fixmath,gensymb}",
            "text.usetex": True,
            "xtick.labelsize": 16,
            "xtick.major.size": 5,
            "xtick.major.width": 1.2,
            "xtick.minor.size": 3,
            "xtick.minor.width": 0.9,
            "xtick.minor.visible": True,
            "xtick.direction": "in",
            "xtick.bottom": True,
            "xtick.top": True,
            "ytick.labelsize": 16,
            "ytick.major.size": 5,
            "ytick.major.width": 1.2,
            "ytick.minor.size": 3,
            "ytick.minor.width": 0.9,
            "ytick.minor.visible": True,
            "ytick.direction": "in",
            "ytick.left": True,
            "ytick.right": True,
        }
    )


def save_fig(fig: plt.Figure, fname: str, dir_fig: str = "figures") -> None:
    """
    Save a figure to the output directory.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    fname : str
        Name of the figure.
    dir_fig : str, optional
        Directory to save the figure, by default "figures"
    """
    # make output directory if it does not exist
    dir_png = f"{dir_fig}/png"
    dir_pdf = f"{dir_fig}/pdf"
    for dir_ in [dir_png, dir_pdf]:
        Path(dir_).mkdir(parents=True, exist_ok=True)

    # save figure
    fig.tight_layout()
    fig.savefig(f"{dir_png}/{fname}.png", dpi=600, bbox_inches="tight")
    fig.savefig(
        f"{dir_pdf}/{fname}.pdf", dpi=1200, bbox_inches="tight", transparent=True
    )


def close_fig(fig: plt.figure) -> None:
    """
    Close the figure and clear the memory.

    Parameters
    ----------
    fig : plt.figure
        Figure to close.
    """
    fig.clear()
    plt.figure(fig.number)
    plt.clf()
    plt.close()

    del fig
    gc.collect()
