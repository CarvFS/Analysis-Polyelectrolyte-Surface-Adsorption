# Standard library
from pathlib import Path
import sys

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# MDAnalysis inheritance
from MDAnalysis.core.groups import AtomGroup

# Internal dependencies
from .base import ParallelAnalysisBase

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


class DipoleField(ParallelAnalysisBase):
    def __init__(
        self,
        atomgroup: AtomGroup,
        compound,
        axis: str = "z",
        axis_bin_width: float = 0.2,
        cos_theta_bin_width: float = 0.02,
        label: str = None,
        df_weights: pd.DataFrame = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the dipole analysis object.

        Parameters
        ----------
        atomgroup : AtomGroup
            AtomGroup for dipole field.
        compound : str
            The grouping for the dipole field.
                Can be {"group", "segments", "residues", "molecules"}.
        axis : str, optional
            The axis for the dipole field. Default is "z".
        axis_bin_width : float, optional
            The bin width for the axis. Default is 0.2.
        label : str, optional
            text label for system. default is none.
        df_weights : pd.DataFrame, optional
            The weights for the dipole field. Default is None.
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
        self._dir_out: Path = Path("./mdanalysis_dipolefield")
        self._df_filename = f"dipolefield_{axis}_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df: pd.DataFrame = None

        # dataframe containing weight of each frame from biasing potential
        if df_weights is not None:
            self._weighted: bool = True
            self._df_weights = df_weights[["time", "weight"]].copy()
        else:
            self._weighted: bool = False
            self._df_weights: pd.DataFrame = None
        self._logger.debug(f"weighted: {self._weighted}")

        self.compound = compound
        self.axis_label = str(axis).lower()

        if self.axis_label == "x":
            self.axis = 0
        elif self.axis_label == "y":
            self.axis = 1
        elif self.axis_label == "z":
            self.axis = 2
        else:
            raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

        self._logger.debug(f"axis: {self.axis}")
        box = self._universe.dimensions[:3]

        self.axis_bin_width = axis_bin_width
        self.axis_bins = np.arange(
            0 - self.axis_bin_width,
            box[self.axis] + self.axis_bin_width,
            self.axis_bin_width,
        )
        self._logger.debug(f"Number of axis bins: {len(self.axis_bins)}")

        self.cos_theta_bin_width = cos_theta_bin_width
        self.cos_theta_bins = np.arange(
            -1 - cos_theta_bin_width,
            1 + cos_theta_bin_width,
            self.cos_theta_bin_width,
        )
        self._logger.debug(f"Number of cos(theta) bins: {len(self.cos_theta_bins)}")

        self.mesh_bins_axis, self.mesh_bins_cos_theta = np.meshgrid(
            self.axis_bins, self.cos_theta_bins
        )
        self.n_bins = (len(self.axis_bins) - 1) * (len(self.cos_theta_bins) - 1)
        self._logger.debug(f"n_bins: {self.n_bins}")

        # set output data structures
        self._columns: list[str] = [
            "frame",
            "time",
        ]
        for i in range(3):
            for j in range(self.n_bins):
                self._columns.append(f"hist_{i}_{j}")

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
        dipoles = ag.dipole_vector(
            wrap=False,
            unwrap=True,
            compound=self.compound,
            center="mass",
        )  # [e * Ang]
        dipoles /= np.linalg.norm(dipoles, axis=1)[:, np.newaxis]

        centers = ag.center_of_mass(
            wrap=False,
            unwrap=True,
            compound=self.compound,
        )  # [Ang]

        # 2D histogram of dipole field along axis and cos(theta)
        hist_cos_x, _, _ = np.histogram2d(
            centers[:, self.axis],
            dipoles[:, 0],
            bins=[self.axis_bins, self.cos_theta_bins],
            density=True,
        )
        hist_cos_y, _, _ = np.histogram2d(
            centers[:, self.axis],
            dipoles[:, 1],
            bins=[self.axis_bins, self.cos_theta_bins],
            density=True,
        )
        hist_cos_z, _, _ = np.histogram2d(
            centers[:, self.axis],
            dipoles[:, 2],
            bins=[self.axis_bins, self.cos_theta_bins],
            density=True,
        )

        # flatten 2D histogram and save to results
        results[2 : (2 + self.n_bins)] = hist_cos_x.flatten()
        results[(2 + self.n_bins) : (2 + 2 * self.n_bins)] = hist_cos_y.flatten()
        results[(2 + 2 * self.n_bins) : (2 + 3 * self.n_bins)] = hist_cos_z.flatten()

        return results

    def _conclude(self) -> None:
        # time-averaged dipole field
        if self._weighted:
            # merge the rdf results with the weights
            self._df = pd.DataFrame(self._results, columns=self._columns)
            self._df = pd.merge(self._df, self._df_weights, how="inner", on="time")

            weight = self._df["weight"].values

        else:
            weight = np.ones(len(self._results[:, 0]))

        # average the histograms over time
        histo_x_cols = self._df.columns[2 : (2 + self.n_bins)]
        histo_y_cols = self._df.columns[(2 + self.n_bins) : (2 + 2 * self.n_bins)]
        histo_z_cols = self._df.columns[(2 + 2 * self.n_bins) : (2 + 3 * self.n_bins)]

        histo_2d_x_mean = np.average(
            self._df[histo_x_cols], axis=0, weights=weight
        ).reshape(self.axis_bins.size - 1, self.cos_theta_bins.size - 1)
        histo_2d_y_mean = np.average(
            self._df[histo_y_cols], axis=0, weights=weight
        ).reshape(self.axis_bins.size - 1, self.cos_theta_bins.size - 1)
        histo_2d_z_mean = np.average(
            self._df[histo_z_cols], axis=0, weights=weight
        ).reshape(self.axis_bins.size - 1, self.cos_theta_bins.size - 1)

        self._logger.debug(f"histo_2d_x_mean: {histo_2d_x_mean.shape}")

        xx_bins, yy_bins = np.meshgrid(self.axis_bins, self.cos_theta_bins)

        # convert 2D histograms to average cos(theta) histograms vs axis
        histo_1d_x_mean = np.zeros(self.axis_bins.size - 1)
        histo_1d_y_mean = np.zeros(self.axis_bins.size - 1)
        histo_1d_z_mean = np.zeros(self.axis_bins.size - 1)
        cos_avg = 0.5 * (self.cos_theta_bins[1:] + self.cos_theta_bins[:-1])
        for i in range(self.axis_bins.size - 1):
            histo_1d_x_mean[i] = np.average(
                cos_avg,
                weights=(
                    histo_2d_x_mean[i, :] if np.sum(histo_2d_x_mean[i, :]) > 0 else None
                ),
            )
            histo_1d_y_mean[i] = np.average(
                cos_avg,
                weights=(
                    histo_2d_y_mean[i, :] if np.sum(histo_2d_y_mean[i, :]) > 0 else None
                ),
            )
            histo_1d_z_mean[i] = np.average(
                cos_avg,
                weights=(
                    histo_2d_z_mean[i, :] if np.sum(histo_2d_z_mean[i, :]) > 0 else None
                ),
            )

        self.results = {
            "bins_axis": self.axis_bins,
            "bins_cos_theta": self.cos_theta_bins,
            "histo_2d_x_mean": histo_2d_x_mean,
            "histo_2d_y_mean": histo_2d_y_mean,
            "histo_2d_z_mean": histo_2d_z_mean,
            "histo_1d_x_mean": histo_1d_x_mean,
            "histo_1d_y_mean": histo_1d_y_mean,
            "histo_1d_z_mean": histo_1d_z_mean,
        }

        # save results as compressed numpy arrays
        Path(self._dir_out / "data").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._dir_out / f"data/histo_dipolefield_{self.axis_label}_{self._tag}.npz",
            bins_axis=self.axis_bins,
            bins_cos_theta=self.cos_theta_bins,
            histo_2d_x_mean=histo_2d_x_mean,
            histo_2d_y_mean=histo_2d_y_mean,
            histo_2d_z_mean=histo_2d_z_mean,
            histo_1d_x_mean=histo_1d_x_mean,
            histo_1d_y_mean=histo_1d_y_mean,
            histo_1d_z_mean=histo_1d_z_mean,
        )

    def figures(
        self, title: str = None, ext: str = "png"
    ) -> tuple[plt.figure, plt.axes]:
        self._logger.info(f"Plotting DipoleField analysis for {self._tag}.")

        figs = []
        axs = []

        for d in ["x", "y", "z"]:
            fig, ax = self.plt_2d_histo(title=title, ext=ext, dim=d)
            figs.append(fig)
            axs.append(ax)

            fig, ax = self.plt_1d_histo(title=title, ext=ext, dim=d)
            figs.append(fig)
            axs.append(ax)

        self._logger.info(f"Finished plotting DipoleField analysis for {self._tag}.")
        return figs, axs

    def plt_2d_histo(self, title: str = None, ext: str = "png", dim: str = "z"):
        self._logger.info(f"Plotting 2D histogram for {self._tag} along {dim}.")
        d = self._dir_out / "figures"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if dim == "x":
            histo_2d = self.results["histo_2d_x_mean"]
        elif dim == "y":
            histo_2d = self.results["histo_2d_y_mean"]
        elif dim == "z":
            histo_2d = self.results["histo_2d_z_mean"]
        else:
            raise ValueError("Invalid dimension. Choose 'x', 'y', or 'z'.")

        im = ax.imshow(
            histo_2d.T,
            origin="lower",
            extent=[
                self.axis_bins[0] / 10.0,
                self.axis_bins[-1] / 10.0,
                self.cos_theta_bins[0],
                self.cos_theta_bins[-1],
            ],
            aspect="auto",
            interpolation="gaussian",
        )
        fig.colorbar(im, ax=ax, label=r"Density")
        ax.set_xlabel(f"{self.axis_label} [nm]")
        ax.set_ylabel(r"$\cos(\theta)$ from $" + dim + r"$")

        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            f"{d}/plt_2d_histo_{dim}_{self._tag}.{ext}", dpi=600, bbox_inches="tight"
        )
        self._logger.debug("Saved figure")

        return fig, ax

    def plt_1d_histo(self, title: str = None, ext: str = "png", dim: str = "z"):
        self._logger.info(f"Plotting 1D histogram for {self._tag} along {dim}.")
        d = self._dir_out / "figures"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if dim == "x":
            histo_1d = self.results["histo_1d_x_mean"]
        elif dim == "y":
            histo_1d = self.results["histo_1d_y_mean"]
        elif dim == "z":
            histo_1d = self.results["histo_1d_z_mean"]
        else:
            raise ValueError("Invalid dimension. Choose 'x', 'y', or 'z'.")

        ax.plot(
            self.axis_bins[:-1] / 10.0,
            histo_1d,
            label=f"{dim}-axis",
        )
        ax.set_ylabel(r"$\langle \cos(\theta) \rangle$ from $" + dim + r"$")
        ax.set_xlabel(f"{self.axis_label} [nm]")

        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            f"{d}/plt_1d_histo_{dim}_{self._tag}.{ext}", dpi=600, bbox_inches="tight"
        )
        self._logger.debug("Saved figure")

        return fig, ax
