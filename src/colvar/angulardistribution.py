# Standard library
import concurrent.futures
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
from colvar.lineardensity import Results  # noqa: E402


class AngularDistribution(ParallelAnalysisBase):

    def __init__(
        self,
        atomgroup: AtomGroup,
        grouping: str = "segments",
        label: str = None,
        df_weights: pd.DataFrame = None,
        axis: str = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            atomgroup.universe.trajectory,
            (atomgroup,),
            label=label,
            verbose=verbose,
            **kwargs,
        )
        self._logger = setup_logging(verbose=verbose, log_file=f"logs/{__name__}.log")

        # bins for angle distribution
        self.grouping = grouping
        self.n_bins = 200
        self.cos_bins = np.linspace(-1, 1, self.n_bins + 1)
        self.cos_bins_centers = (self.cos_bins[1:] + self.cos_bins[:-1]) / 2.0
        self.angle_bins = np.arccos(self.cos_bins)
        self.angle_bins_centers = (self.angle_bins[1:] + self.angle_bins[:-1]) / 2.0

        # output data
        self._dir_out: Path = Path("./mdanalysis_angulardistribution")
        self._df_filename = f"pl_{self._tag}.parquet"
        self._df_weights: pd.DataFrame = df_weights
        self._df: pd.DataFrame = None

        # initiate result instances
        if axis is not None:
            self.dims = [axis]
        else:
            self.dims = ["x", "y", "z"]
        for idx, dim in enumerate(self.dims):
            self.results[dim] = Results(dim=idx)
        # add keys to results
        self.keys = [
            "bin_centers",
            "angle_density",
            "cos_bin_centers",
            "cos_angle_density",
        ]
        for dim in self.results:
            idx = self.results[dim]["dim"]
            for key in self.keys:
                self.results[dim][key] = np.zeros(self.n_bins)

        # axes for calculating angle
        self.x_axis = np.array([1, 0, 0])
        self.y_axis = np.array([0, 1, 0])
        self.z_axis = np.array([0, 0, 1])
        if axis == "x":
            self.axes = [self.x_axis]
        elif axis == "y":
            self.axes = [self.y_axis]
        elif axis == "z":
            self.axes = [self.z_axis]
        elif axis is None:
            self.axes = [self.x_axis, self.y_axis, self.z_axis]
        else:
            raise ValueError(f"Invalid axis: {axis}")

        # set output data structures
        self._columns: list[str] = [
            "frame",
            "time",
        ]
        for dim in self.dims:
            self._columns.extend(
                [f"cos_angle_{dim}_{i}" for i in range(len(self.angle_bins) - 1)]
            )
        self.n_columns = len(self._columns)

    def _single_frame(self, idx_frame: int) -> np.ndarray:
        # update frame and atomgroup
        ts = self._universe.trajectory[idx_frame]
        ag = self._atomgroups[0]

        results = np.empty(len(self._columns))
        results.fill(np.nan)

        # save frame index, time
        results[0] = ts.frame
        results[1] = ag.universe.trajectory.time
        if len(ag) == 0:
            return results

        # calculate dipole vector of all compounds [n_groups, 3]
        dipoles = ag.dipole_vector(compound=self.grouping)
        norms = np.linalg.norm(dipoles, axis=1)
        dipoles_norm = np.empty_like(dipoles)
        dipoles_norm.fill(np.nan)
        with np.errstate(divide="ignore"):
            dipoles_norm = dipoles / norms[:, np.newaxis]

        # calculate angle of dipole vector with respect to axis
        idx_start = 2
        for axis in self.axes:
            # calculate cosine of angle
            cos_angle = np.clip(np.dot(dipoles_norm, axis), -1, 1)
            hist, _ = np.histogram(cos_angle, bins=self.cos_bins, density=True)
            # replace hist where cos_angle is 0 with nearest neighbor average for smoothness
            idx_zero = np.where(cos_angle == 0)[0]
            hist[idx_zero] = (hist[idx_zero - 1] + hist[idx_zero + 1]) / 2
            # save results
            results[idx_start : idx_start + self.n_bins] = hist
            idx_start += self.n_bins

        return results

    def _conclude(self) -> None:
        # get weights
        if self._df_weights is not None:
            self._logger.debug("Merge weights with results")
            self.merge_external_data(self._df_weights)
            weights = self._df_weights["weight"].to_numpy()
        else:
            self._logger.debug("No weights provided")
            weights = np.ones(len(self._results[:, 0]))

        # drop indices containing NaN
        idx_drop = np.where(np.isnan(self._results[:, 0]))[0]
        self._logger.info(f"Dropping {len(idx_drop)} frames with NaN")
        self._results = np.delete(self._results, idx_drop, axis=0)
        weights = np.delete(weights, idx_drop)

        # compute results
        self._logger.debug(f"Computing results from {self._results.shape[0]} frames")
        idx_start = 2
        for dim in self.dims:
            self._logger.debug(f"Computing results for dimension {dim}")
            self.results[dim]["cos_bin_centers"] = self.cos_bins_centers
            self.results[dim]["bin_centers"] = self.angle_bins_centers
            self.results[dim]["cos_angle_density"] = np.average(
                self._results[:, idx_start : idx_start + self.n_bins],
                axis=0,
                weights=weights,
            )
            self.results[dim]["angle_density"] = self.results[dim]["cos_angle_density"]
            idx_start += self.n_bins

    def save(self, dir_out: Path = None) -> None:
        # set output directory
        if dir_out is None:
            dir_out = self._dir_out / "data"
        self._logger.info(f"Saving results for {self._tag} to {dir_out}")
        Path(dir_out).mkdir(parents=True, exist_ok=True)

        # save results to a compressed numpy file
        def save_results(dim: str) -> None:
            self._logger.debug(f"Saving results for dimension {dim}")
            np.savez_compressed(
                dir_out / f"solventorientation_{dim}-{self._tag}.npz",
                bin_centers=self.results[dim]["bin_centers"],
                angle_density=self.results[dim]["angle_density"],
                cos_bin_centers=self.results[dim]["cos_bin_centers"],
                cos_angle_density=self.results[dim]["cos_angle_density"],
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(save_results, self.dims)

    def figures(self, title: str = None, ext: str = "png") -> None:
        def figure_dim(dim: str, title: str, ext: str) -> None:
            self.plt_ang_dist(dim=dim, title=title, ext=ext)
            self.plt_cos_ang_dist(dim=dim, title=title, ext=ext)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(
                figure_dim, self.dims, [title] * len(self.dims), [ext] * len(self.dims)
            )

    def plt_ang_dist(self, dim: str = "z", title: str = None, ext: str = "png"):
        if dim not in ["x", "y", "z"]:
            raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")
        if title is None:
            title = f"${dim}$-axis"
        dir_out = self._dir_out / "figures"
        if dir_out not in list(self._dir_out.iterdir()):
            dir_out.mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            np.degrees(self.results[dim]["bin_centers"]),
            self.results[dim]["angle_density"],
            label=f"{dim}-axis",
        )
        ax.set_xlabel("Angle [°]")
        ax.set_ylabel("Probability Density")
        ax.set_title(title, y=1.05)
        fig.savefig(dir_out / f"angular_distribution_{dim}_{self._tag}.{ext}")

        return fig, ax

    def plt_cos_ang_dist(self, dim: str = "z", title: str = None, ext: str = "png"):
        if dim not in ["x", "y", "z"]:
            raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")
        if title is None:
            title = f"${dim}$-axis"
        dir_out = self._dir_out / "figures"
        if dir_out not in list(self._dir_out.iterdir()):
            dir_out.mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            self.results[dim]["cos_bin_centers"],
            self.results[dim]["cos_angle_density"],
            label=f"{dim}-axis",
        )
        ax.set_xlabel(r"$\cos(\theta)$")
        ax.set_ylabel("Probability Density")
        ax.set_title(title, y=1.05)
        fig.savefig(dir_out / f"cosine_angular_distribution_{dim}_{self._tag}.{ext}")

        return fig, ax
