# ########################################################################### #
# Imports                                                                     #
# ########################################################################### #

# standard library
import sys
from pathlib import Path
import warnings

# third-party packages
import colorcet as cc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get absolute path to file's parent directory
dir_proj_base = Path.cwd().resolve().parents[1]
sys.path.insert(0, f"{dir_proj_base}/src")

# Internal dependencies
from render.util import set_style, close_fig, save_fig  # noqa: E402

# ########################################################################### #
# Global variables                                                            #
# ########################################################################### #

# general
VERBOSE: bool = False

# data I/O
DIR_INPUT: str = Path.cwd().resolve() / "output"
DIR_OUTPUT: str = Path.cwd().resolve() / "output/combined"

# simulation parameters
TAGS: list = ["AA", "An", "VAc", "VAl"]

# ########################################################################### #
# Functions                                                                   #
# ########################################################################### #


def load_data(directory: str, file_pattern: str) -> tuple:
    """Load data from files in directory.

    Parameters
    ----------
    directory : str
        Directory to search for files.
    file_pattern : str
        File pattern to match.

    Returns
    -------
    files : list
        List of files matching pattern.
    data : pd.DataFrame or np.ndarray
        Data from files.
    """
    # recursively search subdirectories for files
    files = list(Path(directory).rglob(file_pattern))
    files = sorted(files)

    # load data from files
    if file_pattern.endswith(".csv"):
        data = [pd.read_csv(file) for file in files]
    elif file_pattern.endswith(".npy"):
        data = [np.load(file) for file in files]
    elif file_pattern.endswith(".parquet"):
        data = [pd.read_parquet(file) for file in files]

    return files, data


def plt_all_pmf(data: list, labels: list, tag: str) -> tuple:
    """Plot all PMFs on same figure.

    Input data are plotted with low opacity to show all data and the mean is
    plotted with a solid line of full opacity. 95% confidence intervals are
    plotted as a shaded region.

    Parameters
    ----------
    data : list
        List of data to plot.
    labels : list
        List of labels for data.
    tag : str
        Tag for figure name.

    Returns
    -------
    x : np.ndarray
        CV grid.
    y_mean : np.ndarray
        Mean of PMFs.
    """
    fname = "pmf_all"
    if tag:
        fname = f"{fname}_{tag}"

    cmap = cc.glasbey_dark
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$z$ [nm]")
    ax.set_ylabel(r"$\Delta F$ [$k_\mathrm{B} T$]")
    ax.set_title(f"PMF of Monomer Binding ({tag})")

    # add black line at zero
    ax.axhline(0, color="k", linestyle="-")

    # plot data
    x = np.zeros((len(data), len(data[0]["cv"])))
    y = np.zeros_like(x)
    for i, (d, label) in enumerate(zip(data, labels)):
        x[i] = d["cv"].values
        y[i] = d["pmf"].values

        # shift y data by mean of last 10% of data
        final_mean = np.nanmean(y[i, -int(0.10 * len(y[i])) :])
        y[i] -= final_mean

        # drop simulation if final mean is too large
        if final_mean > 1e2:
            warnings.warn(
                f"Dropping simulation {i + 1} of {label} with"
                + f" final mean > 1e2: {final_mean:.2e}"
            )
            y[i] = np.nan
            continue

        ax.plot(x[i], y[i], color=cmap[i], alpha=0.4, label=f"Run {i + 1}")

    # assert x rows are equal
    for i in range(1, len(x)):
        assert np.allclose(x[i], x[0])

    # compute mean and confidence intervals
    y_mean = np.nanmean(y, axis=0)

    # plot mean and confidence intervals
    ax.set_ylim(-4, 12)
    ax.plot(x[0], y_mean, color=cmap[len(x) + 2], label="Mean", linewidth=3.5)

    # add legend
    ax.legend(loc="upper right", ncol=2)
    save_fig(fig, fname=fname, dir_fig=DIR_OUTPUT)
    close_fig(fig)

    return x[0], y_mean


def plt_mean_pmf(cv_grid: np.ndarray, pmfs: list, labels: list) -> None:
    assert len(pmfs) == len(labels)

    fname = "pmf_mean"
    cmap = cc.glasbey_dark
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$z$ [nm] (Monomer CoM to Frozen Crystal)")
    ax.set_ylabel(r"$\Delta F$ [$k_\mathrm{B} T$]")
    ax.set_title("Average PMF of Monomer Binding")

    # add black line at zero
    ax.axhline(0, color="k", linestyle="-")

    # plot mean PMFs
    for i, (pmf, label) in enumerate(zip(pmfs, labels)):
        ax.plot(cv_grid, pmf, color=cmap[i], label=label, linewidth=3)

    # add legend
    ax.legend(loc="upper right")
    ax.set_ylim(-2, 4)
    ax.set_xlim(0.38, 1.5)
    save_fig(fig, fname=fname, dir_fig=DIR_OUTPUT)
    close_fig(fig)


# ########################################################################### #
# Main script                                                                 #
# ########################################################################### #
if __name__ == "__main__":
    # set style
    set_style()

    files, data = load_data(DIR_INPUT, "*pmf_dist_chain.z.csv")
    if VERBOSE:
        print(f"Loaded {len(data)} files from {DIR_INPUT}")
        for f in files:
            print(f"  - {f}")

    # plot all PMFs
    idx = 0
    cv_grid, pmfs = [], []
    idx_per_tag = len(data) // len(TAGS)
    for i, tag in enumerate(TAGS):
        cv, pmf = plt_all_pmf(
            data[idx : idx + idx_per_tag], files[idx : idx + idx_per_tag], tag
        )
        cv_grid.append(cv)
        pmfs.append(pmf)
        idx += idx_per_tag

    # plot mean PMFs
    plt_mean_pmf(cv_grid[0], pmfs, TAGS)
