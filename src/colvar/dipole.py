# Standard library
from pathlib import Path
import sys
import warnings

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as st

# MDAnalysis inheritance
from MDAnalysis.core.groups import AtomGroup

# Internal dependencies
from .base import ParallelAnalysisBase
from stats.block_error import BlockError

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from utils.logs import setup_logging  # noqa: E402


class Dipole(ParallelAnalysisBase):

    def __init__(
        self, atomgroup: AtomGroup, label: str = None, verbose: bool = False, **kwargs
    ):
        """
        Initialize the dipole analysis object.

        Parameters
        ----------
        atomgroup : AtomGroup
            AtomGroup for polymer chain.
        label : str, optional
            text label for system. default is none.
        verbose : bool, optional
            If True, print additional information. Default is False.
        **kwargs
            Additional keyword arguments for :class:`ParallelAnalysisBase`.
        """
        super().__init__(
            atomgroup.universe.trajectory,
            (atomgroup,),
            label=label,
            verbose=verbose,
            **kwargs,
        )
        self._logger = setup_logging(verbose=verbose, log_file=f"logs/{__name__}.log")

        # output data
        self._dir_out: Path = Path("./mdanalysis_dipole")
        self._df_filename = f"dipole_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df: pd.DataFrame = None

        # set output data structures
        self._columns: list[str] = [
            "frame",
            "time",
            "dipole_x",
            "dipole_y",
            "dipole_z",
        ]

        self._logger.info(f"Initialized dipole analysis for {self._tag}.")

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
        ts = self._universe.trajectory[idx_frame]
        ag = self._atomgroups[0]

        results = np.empty(len(self._columns), dtype=np.float64)
        results.fill(np.nan)

        # save frame index, time
        results[0] = ts.frame
        results[1] = ag.universe.trajectory.time

        # calculate dipole vector
        results[2:] = ag.dipole_vector(
            wrap=True,
            unwrap=False,
            compound="group",
            center="mass",
        )  # [e * Ang]

        return results

    def figures(
        self, title: str = None, ext: str = "png"
    ) -> tuple[plt.figure, plt.axes]:
        """
        Plot the radial distribution function and the potential of mean
        force. The figures are saved to the `figures` directory in the
        `dir_out` directory. This is a wrapper for all plotting methods.

        This method should only be called after the analysis has been
        run.

        Parameters
        ----------
        title : str, optional
            The title of the plots.
        ext : str, optional
            The file extension of the saved figures. Default is "png".

        Returns
        -------
        tuple[plt.figure, plt.axes]
            The figures and axes of the plots.
        """
        self._logger.info(f"Plotting dipole analysis for {self._tag}.")

        figs = []
        axs = []

        for d in ["x", "y", "z"]:
            fig, ax = self.plt_dipole_dyn(dim=d, title=title, ext=ext)
            figs.append(fig)
            axs.append(ax)

        self._logger.info(f"Finished plotting dipole analysis for {self._tag}.")
        return figs, axs

    def plt_dipole_dyn(self, dim, title: str = None, ext: str = "png"):
        self._logger.info(f"Plotting dipole dynamics for {self._tag} along {dim}.")
        d = self._dir_out / "figures"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        x = self._df["time"] / 1e3
        y = self._df[f"dipole_{dim}"]

        ax.scatter(
            x,
            y,
            s=2,
            label=r"$\mu$",
            alpha=0.5,
        )
        ax.set_xlabel(r"Time [ns]")
        ax.set_ylabel(r"$\mu$ [D]")

        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            f"{d}/plt_mu_dyn_{dim}_{self._tag}.{ext}", dpi=300, bbox_inches="tight"
        )
        self._logger.debug("Saved figure")

        return fig, ax
