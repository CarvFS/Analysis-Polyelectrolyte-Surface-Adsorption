"""
:Authors: Alec Glisman
:Year: 2022
:Copyright: GNU Public License v2

This module implements the calculation of the atom pair relaxation
correlation function (APC). APCs are used to characterize the association
dynamics between pairs of atoms in a system. The function is also commonly
known as the mean residence time (MRT) and is used to describe the water
residence time next to a given solute.

Long relaxation times are associated with strong interactions between atoms
in the specified groups. For a full explanation of the theory and derivation
behind the atom pair relaxation function (`_survival_imm`), please read
[Impey1983]_.

Note that the IMM correlation function is highly sensitive to the maximum
time (equivalent to number of successive frames) that a given pair is
allowed to continuously unpair during a sub-trajectory. This time accounts
for brief "recrossing events" that are not considered pair breaks. For a
discussion of these issues, see [Laage2008]_.

We also use an alternative implementation of the APC first derived in
[Northrop1980]_ (`_survival_ssp`). This function does not look at the time
spent in the paired state, but rather the time required to go from a reactant
(paired) to product (unpaired) and does not need to explicitly account for a
recrossing time. This function used with `n_frames_recross = None`.


References
----------

.. [Northrop1980] Northrup, S. H.; Hynes, J. T. The Stable States Picture of
                  Chemical Reactions. I. Formulation for Rate Constants and
                  Initial Condition Effects. The Journal of Chemical Physics
                  1980, 73 (6), 2700-2714. https://doi.org/10.1063/1.440484.

.. [Impey1983]  Impey, R. W.; Madden, P. A.; McDonald, I. R. Hydration and
                Mobility of Ions in Solution. J. Phys. Chem. 1983, 87 (25),
                5071-5083. https://doi.org/10.1021/j150643a008.

.. [Laage2008]  Laage, D.; Hynes, J. T. On the Residence Time for Water in a
                Solute Hydration Shell: Application to Aqueous Halide
                Solutions. Journal of Physical Chemistry B 2008, 112 (26),
                7697-7701. https://doi.org/10.1021/jp802033r.
"""

# Standard library
from pathlib import Path
import sys
from typing import Callable

# Third-party packages
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# MDAnalysis package
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.log import ProgressBar

# Internal dependencies
from .base import ParallelAnalysisBase

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from utils.logs import setup_logging  # noqa: E402


def mon_exp(x, c1, b1):
    return c1 * np.exp(b1 * x)


def bi_exp(x, c1, b1, c2, b2):
    return c1 * np.exp(b1 * x) + c2 * np.exp(b2 * x)


def tri_exp(x, c1, b1, c2, b2, c3, b3):
    return c1 * np.exp(b1 * x) + c2 * np.exp(b2 * x) + c3 * np.exp(b3 * x)


class AtomPair(ParallelAnalysisBase):
    """
    This class implements the calculation of the atom pair relaxation
    correlation function (APC)
    """

    def __init__(
        self,
        select: AtomGroup,
        n_pairs: int,
        label: str,
        tau_max: int,
        window_step: int = 0,
        fit_func: Callable = None,
        **kwargs,
    ):
        """
        Initialize AtomPair analysis class. Method calls parent
        class initializer.

        Parameters
        ----------
        select : AtomGroup
            AtomGroup to analyze
        n_pairs : int
            Maximum number of pairs to analyze
        label : str
            Label for the AtomGroup
        tau_max : int
            Maximum number of frames to calculate the survival probability
        window_step : int, optional
            Maximum number of frames an atom pair can be continuously unpaired
            and be considered a recrossing event, by default 0
        fit_func : Callable, optional
            Function to fit the survival probability, by default None
        """
        super().__init__(select.universe.trajectory, (select,), label=label, **kwargs)
        self._logger = setup_logging(log_file=f"logs/{__name__}.log")

        # Verify that the atomgroups are of type AtomGroup
        if not isinstance(select, AtomGroup):
            raise TypeError("selection must be of type AtomGroup")

        self.window_step: int = window_step
        self.tau_max: int = tau_max

        # iterate through trajectory and find maximum number of pairs
        self.n_pairs = n_pairs

        self._dir_out: Path = Path("./mdanalysis_survivalprobability")
        self._df_filename = f"sp_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df: pd.DataFrame = None

        self.coeffs = None
        self.fit_func = mon_exp if fit_func is None else fit_func

    def _prepare(self) -> None:
        """
        Initialize data structures dependent on `self.n_frames` and prepare
        class for frame-by-frame trajectory analysis.
        """
        # calculate number of frames to analyze
        self.n_frames_analysis = (self.stop - self.start) // self.step
        self._logger.debug(f"n_frames_analysis: {self.n_frames_analysis}")

        # output data
        self._results = np.zeros((self.n_frames_analysis, self.n_pairs))

    def _single_frame(self, idx_frame: int) -> np.ndarray:
        """
        Analyze a single frame in the trajectory.

        Parameters
        ----------
        idx_frame : int
            The index of the current frame.

        Returns
        -------
        np.array
            The weight, the volume of the box, and the histogram of the
            pair distances at the current frame.
        """
        # update frame and atomgroup
        _ = self._universe.trajectory[idx_frame]
        ag = self._atomgroups[0]

        # return atom indices for each atom in the selection
        indices = ag.indices
        results = np.zeros((self.n_pairs))
        for i, idx in enumerate(indices):
            results[i] = idx
        return results

    def _conclude(self) -> None:
        """
        Calculate the survival probability correlation function
        for current trajectory.
        """
        # convert results to list of sets and drop zeros
        # now, we have the atom pairs that exist at each frame
        results = [set(i) for i in self._results if i.any()]
        unique_pairs = set.union(*results)

        # create a 2D array of booleans indicating whether the atom pair exits
        # size: (unique_pairs, n_frames)
        atom_pairs = np.zeros((len(unique_pairs), len(results)), dtype=bool)
        for i, pair in enumerate(unique_pairs):
            for j, frame in enumerate(results):
                atom_pairs[i, j] = pair in frame

        # calculate correlation function
        max_frames = min(self.tau_max, self.n_frames)
        sp = np.zeros((max_frames, 3))
        for lag in ProgressBar(range(max_frames), verbose=self._verbose):
            sp[lag, 0] = lag * self.step
            sp[lag, 1] = lag * self._universe.trajectory.dt * self.step
            sp[lag, 2] = _survival_imm(lag, atom_pairs, self.window_step)
        # normalize probability to start at 1
        sp[:, 2] /= sp[0, 2]
        self.results = sp

        cols = ["Frame", "Time[ps]", "Survival_Probability"]
        self._df = pd.DataFrame(sp, columns=cols)

        # Fit the ACF survival probability
        t0 = min(self._df["Time[ps]"])
        self.coeff, _, _, _, _ = curve_fit(
            self.fit_func,
            self._df["Time[ps]"][: self.tau_max] - t0,
            self._df["Survival_Probability"][: self.tau_max],
            p0=[1, -0.01],
            full_output=True,
            maxfev=100000,
        )
        self._df["Survival_Probability_Fit"] = self.fit_func(
            self._df["Time[ps]"] - t0, *self.coeff
        )

    def save(self, dir_out: str = None) -> None:
        """
        Save key data structures for later analysis.

        :param dir_out: directory path to output files, defaults to None
        :type dir_out: str, optional
        """
        if dir_out is None:
            dir_out = self._dir_out / "data"
        Path(dir_out).mkdir(parents=True, exist_ok=True)

        # save dataframes
        self._df.to_parquet(f"{dir_out}/sp_{self._tag}.parquet", index=False)

    def figures(self, title: str = None, ext: str = "png"):
        figs = []
        axs = []

        fig, ax = self.plt_apc(title=title, ext=ext)
        figs.append(fig)
        axs.append(ax)

        return figs, axs

    def plt_apc(self, title: str = None, ext: str = "png"):
        d = self._dir_out / "figures"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cf = self.coeff
        prefactor = cf[0]
        timescale = 1.0 / (cf[1] * 1000.0)

        ax.plot(
            self._df["Time[ps]"] / 1000,
            self._df["Survival_Probability"],
            label="ACF",
        )
        ax.plot(
            self._df["Time[ps]"] / 1000,
            self._df["Survival_Probability_Fit"],
            "--",
            label=f"Fit: ${prefactor:0.3f} e^{{t/{timescale:0.3f}}}$",
        )

        ax.set_xlabel(r"Time [ns]")
        ax.set_ylabel("Survival Probability")
        ax.legend(loc="best")
        if title is not None:
            ax.set_title(title, y=1.05)
        Path(d).mkdir(parents=True, exist_ok=True)
        ax.set_xscale("log")
        fig.savefig(f"{d}/sp_{self._tag}.{ext}", dpi=300, bbox_inches="tight")
        return fig, ax


@nb.jit(nopython=True, parallel=True, nogil=True)
def _survival_imm(lag: int, atom_pairs: np.ndarray, window_step: int = 0) -> int:
    """
    Calculate the un-normalized survival probability of all possible atom pairs
    at all frames with range of the trajectory for a specific frame lag time.
    The atom pair must exist in the frame for the lag time at both end points
    and can be unpaired continuously for at most n_frame_unpair frames.

    The survival probability is calculated using the algorithm in [Impey1983]_.

    :param lag: Number of frames between the start and end frames of the
    atom pair
    :type lag: int
    :param atom_pairs: 2D array (atom, n_frames) of booleans indicating
    whether the atom pair exists at the frame
    :type atom_pairs: np.ndarray
    :return: Total number of atom pairs that existed in the trajectory for the
    specified lag time
    :rtype: int

    """
    # initialize number of surviving atom pairs to zero
    survive: int = 0
    n_ag, n_frames = np.shape(atom_pairs)

    # loop over all possible subsets of trajectory with given lag time
    for start in nb.prange(n_frames - lag):  # pylint: disable=not-an-iterable
        end: int = start + lag

        # iterate over all atom pairs
        for ag in nb.prange(n_ag):  # pylint: disable=not-an-iterable
            # skip if atom pair does not exist at both ends of trajectory
            if not atom_pairs[ag, start] or not atom_pairs[ag, end]:
                continue

            # iterate through trajectory and verify that atom pair exists
            n_frames_unpair: int = 0
            for frame in range(start, end + 1):
                if not atom_pairs[frame, ag]:
                    n_frames_unpair += 1
                    if n_frames_unpair > window_step:
                        break

            # if atom pair survived, increment counter
            if n_frames_unpair <= window_step:
                survive += 1

    return survive
