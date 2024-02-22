"""
Linear Density --- :mod:`colvar.lineardensity`
===========================================================

A tool to compute mass and charge density profiles along the three
cartesian axes [xyz] of the simulation cell. Works only for orthorombic,
fixed volume cells (thus for simulations in canonical NVT ensemble).

Modified from the original MDAnalysis.analysis.lineardensity module to
allow for parallelization as well as weighted data from a biased
simulation.
"""

# Standard library
import concurrent.futures
from pathlib import Path
import sys
import warnings

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate

import numpy as np
import warnings

# MDAnalysis inheritance
from MDAnalysis.analysis.base import Results
from MDAnalysis.units import constants

# Internal dependencies
from .base import ParallelAnalysisBase

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from utils.logs import setup_logging  # noqa: E402


class Results(Results):
    """From version 3.0.0 onwards, some entries in Results will be renamed. See
    the docstring for LinearDensity for details. The Results class is defined
    here to implement deprecation warnings for the user."""

    _deprecation_dict = {
        "pos": "mass_density",
        "pos_std": "mass_density_stddev",
        "char": "charge_density",
        "char_std": "charge_density_stddev",
    }

    def _deprecation_warning(self, key):
        warnings.warn(
            f"`{key}` is deprecated and will be removed in version 3.0.0. "
            f"Please use `{self._deprecation_dict[key]}` instead.",
            DeprecationWarning,
        )

    def __getitem__(self, key):
        if key in self._deprecation_dict.keys():
            self._deprecation_warning(key)
            return super(Results, self).__getitem__(self._deprecation_dict[key])
        return super(Results, self).__getitem__(key)

    def __getattr__(self, attr):
        if attr in self._deprecation_dict.keys():
            self._deprecation_warning(attr)
            attr = self._deprecation_dict[attr]
        return super(Results, self).__getattr__(attr)


class LinearDensity(ParallelAnalysisBase):
    r"""Linear density profile

    Parameters
    ----------
    select : AtomGroup
          any atomgroup
    grouping : str {'atoms', 'residues', 'segments', 'fragments'}
          Density profiles will be computed either on the atom positions (in
          the case of 'atoms') or on the center of mass of the specified
          grouping unit ('residues', 'segments', or 'fragments').
    binsize : float
          Bin width in Angstrom used to build linear density
          histograms. Defines the resolution of the resulting density
          profile (smaller --> higher resolution)
    verbose : bool, optional
          Show detailed progress of the calculation if set to ``True``

    Attributes
    ----------
    results.x.dim : int
           index of the [xyz] axes
    results.x.mass_density : numpy.ndarray
           mass density in :math:`g \cdot cm^{-3}` in [xyz] direction
    results.x.mass_density_stddev : numpy.ndarray
           standard deviation of the mass density in [xyz] direction
    results.x.charge_density : numpy.ndarray
           charge density in :math:`\mathrm{e} \cdot mol \cdot cm^{-3}` in
           [xyz] direction
    results.x.charge_density_stddev : numpy.ndarray
           standard deviation of the charge density in [xyz] direction
    results.x.pos: numpy.ndarray
        Alias to the :attr:`results.x.mass_density` attribute.

        .. deprecated:: 2.2.0
           Will be removed in MDAnalysis 3.0.0. Please use
           :attr:`results.x.mass_density` instead.
    results.x.pos_std: numpy.ndarray
        Alias to the :attr:`results.x.mass_density_stddev` attribute.

        .. deprecated:: 2.2.0
           Will be removed in MDAnalysis 3.0.0. Please use
           :attr:`results.x.mass_density_stddev` instead.
    results.x.char: numpy.ndarray
        Alias to the :attr:`results.x.charge_density` attribute.

        .. deprecated:: 2.2.0
           Will be removed in MDAnalysis 3.0.0. Please use
           :attr:`results.x.charge_density` instead.
    results.x.char_std: numpy.ndarray
        Alias to the :attr:`results.x.charge_density_stddev` attribute.

        .. deprecated:: 2.2.0
           Will be removed in MDAnalysis 3.0.0. Please use
           :attr:`results.x.charge_density_stddev` instead.
    results.x.slice_volume : float
           volume of bin in [xyz] direction
    results.x.hist_bin_edges : numpy.ndarray
           edges of histogram bins for mass/charge densities, useful for, e.g.,
           plotting of histogram data.
    results.x.hist_bin_centers : numpy.ndarray
            centers of histogram bins for mass/charge densities, useful for, e.g.,
            plotting of histogram data.
    Note: These density units are likely to be changed in the future.

    Example
    -------
    First create a :class:`LinearDensity` object by supplying a selection,
    then use the :meth:`run` method. Finally access the results
    stored in results, i.e. the mass density in the x direction.

    .. code-block:: python

       ldens = LinearDensity(selection)
       ldens.run()
       print(ldens.results.x.mass_density)


    Alternatively, other types of grouping can be selected using the
    ``grouping`` keyword. For example to calculate the density based on
    a grouping of the :class:`~MDAnalysis.core.groups.ResidueGroup`
    of the input :class:`~MDAnalysis.core.groups.AtomGroup`.

    .. code-block:: python

       ldens = LinearDensity(selection, grouping='residues', binsize=1.0)
       ldens.run()



    .. versionadded:: 0.14.0

    .. versionchanged:: 1.0.0
       Support for the ``start``, ``stop``, and ``step`` keywords has been
       removed. These should instead be passed to :meth:`LinearDensity.run`.
       The ``save()`` method was also removed, you can use ``np.savetxt()`` or
       ``np.save()`` on the :attr:`LinearDensity.results` dictionary contents
       instead.

    .. versionchanged:: 1.0.0
       Changed `selection` keyword to `select`

    .. versionchanged:: 2.0.0
       Results are now instances of
       :class:`~MDAnalysis.core.analysis.Results` allowing access
       via key and attribute.

    .. versionchanged:: 2.2.0

       *  Fixed a bug that caused LinearDensity to fail if grouping="residues"
          or grouping="segments" were set.
       *  Residues, segments, and fragments will be analysed based on their
          centre of mass, not centre of geometry as previously stated.
       *  LinearDensity now works with updating atom groups.
       *  Added new result container :attr:`results.x.hist_bin_edges`.
          It contains the bin edges of the histrogram bins for calculated
          densities and can be used for easier plotting of histogram data.


    .. deprecated:: 2.2.0
       The `results` dictionary has been changed and the attributes
       :attr:`results.x.pos`, :attr:`results.x.pos_std`, :attr:`results.x.char`
       and :attr:`results.x.char_std` are now deprecated. They will be removed
       in 3.0.0. Please use :attr:`results.x.mass_density`,
       :attr:`results.x.mass_density_stddev`, :attr:`results.x.charge_density`,
       and :attr:`results.x.charge_density_stddev` instead.
    """

    def __init__(
        self,
        select,
        grouping="atoms",
        bins=0.25,
        label=None,
        df_weights=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(select.universe.trajectory, (select,), label=label, **kwargs)
        self._logger = setup_logging(log_file=f"logs/{__name__}.log", verbose=verbose)
        self._verbose = verbose

        # allows use of run(parallel=True)
        self._ags = [select]
        self._universe = select.universe

        # Box sides
        self.dimensions = self._universe.dimensions[:3]
        self.volume = np.prod(self.dimensions)
        if not isinstance(bins, (int, float)):
            self.binsize = bins[1] - bins[0]
            self.bins = bins
        else:
            self.binsize = bins
            self.bins = (self.dimensions // self.binsize).astype(int)
        self.nbins = len(self.bins) - 1

        # group of atoms on which to compute the COM (same as used in
        # AtomGroup.wrap())
        self.grouping = grouping

        # Initiate result instances
        self.results["x"] = Results(dim=0)
        self.results["y"] = Results(dim=1)
        self.results["z"] = Results(dim=2)

        # Here we choose a number of bins of the largest cell side so that
        # x, y and z values can use the same "coord" column in the output file
        slices_vol = self.volume / (self.bins[1:] - self.bins[:-1])

        self.keys = [
            "number_density",
            "number_density_stddev",
            "mass_density",
            "mass_density_stddev",
            "charge_density",
            "charge_density_stddev",
        ]

        # Initialize results array with zeros
        for dim in self.results:
            idx = self.results[dim]["dim"]
            self.results[dim]["slice_volume"] = slices_vol[idx]
            for key in self.keys:
                self.results[dim][key] = np.zeros(self.nbins)

        # Variables later defined in _single_frame() method
        self.masses = None
        self.charges = None
        self.totalmass = None

        # dataframe containing weight of each frame from biasing potential
        if df_weights is not None:
            self._weighted: bool = True
            self._df_weights = df_weights[["time", "weight"]].copy()
        else:
            self._weighted: bool = False
            self._df_weights: pd.DataFrame = None

        # output data
        self._dir_out: Path = Path(
            "./mdanalysis_lineardensity"
            + f"-{bins[0]:.3f}_min-{bins[-1]:.3f}_max-{bins[1]-bins[0]:.3f}_delta"
        )
        self._df_filename = f"lineardensity_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df = None
        self._columns = ["frame", "time"]
        for dim in self.results:
            for key in self.keys:
                for idx in range(self.nbins):
                    self._columns.append(f"{dim}_{key}_{idx}")
            for idx in range(self.nbins + 1):
                self._columns.append(f"{dim}_hist_bin_edge_{idx}")

        self._ncols = len(self._columns)
        self._columns = None  # ths avoids creation of dataframe

    def _single_frame(self, idx_frame: int) -> None:
        # get current frame
        ts = self._universe.trajectory[idx_frame]
        ag = self._ags[0]

        # initialize result array
        result = np.zeros(self._ncols)
        result[0] = ts.frame
        result[1] = ts.time

        # Get masses and charges for the selection
        if self.grouping == "atoms":
            self.masses = ag.masses
            self.charges = ag.charges

        elif self.grouping in ["residues", "segments", "fragments"]:
            self.masses = ag.total_mass(compound=self.grouping)
            self.charges = ag.total_charge(compound=self.grouping)

        else:
            raise AttributeError(f"{self.grouping} is not a valid value for grouping.")

        self.totalmass = np.sum(self.masses)

        self.group = getattr(ag, self.grouping)
        ag.wrap(compound=self.grouping)

        # Find position of atom/group of atoms
        if self.grouping == "atoms":
            positions = ag.positions  # faster for atoms
        else:
            # Centre of mass for residues, segments, fragments
            positions = ag.center_of_mass(compound=self.grouping)

        idx_start = 2  # skip frame and time columns
        for dim in ["x", "y", "z"]:
            idx = self.results[dim]["dim"]

            # histogram for positions
            hist, _ = np.histogram(
                positions[:, idx],
                bins=self.bins,
            )
            result[idx_start : idx_start + self.nbins] = hist
            idx_start += self.nbins
            result[idx_start : idx_start + self.nbins] = np.square(hist)
            idx_start += self.nbins

            # histogram for positions weighted on masses
            hist, _ = np.histogram(
                positions[:, idx],
                weights=self.masses,
                bins=self.bins,
            )
            result[idx_start : idx_start + self.nbins] = hist
            idx_start += self.nbins
            result[idx_start : idx_start + self.nbins] = np.square(hist)
            idx_start += self.nbins

            # histogram for positions weighted on charges
            hist, bin_edges = np.histogram(
                positions[:, idx],
                weights=self.charges,
                bins=self.bins,
            )
            result[idx_start : idx_start + self.nbins] = hist
            idx_start += self.nbins
            result[idx_start : idx_start + self.nbins] = np.square(hist)
            idx_start += self.nbins

            # output bin edges
            result[idx_start : idx_start + self.nbins + 1] = bin_edges
            idx_start += self.nbins + 1

        idx_final = idx_start
        if idx_final != self._ncols:
            msg = f"dim: {dim}, idx_final: {idx_final}, ncols: {self._ncols}"
            self._logger.error(msg)
            raise ValueError(msg)

        return result

    def _conclude(self):
        # merge weights with results
        if self._weighted:
            self._logger.debug("Merging weights with results")
            self.merge_external_data(self._df_weights)
            weights = self._df["weight"].to_numpy()
        else:
            self._logger.debug("No weights provided")
            weights = np.ones(len(self._results[:, 0]))

        # drop self._df for memory
        self._df = None
        self._logger.info(f"Memory of results: {self._results.nbytes / 1024**2} MB")

        # output data from self._results to self.results
        # weighted average over all frames (rows of self._results)
        self._logger.debug(f"Computing results from {self._results.shape[0]} frames")
        idx_start = 2  # skip frame and time columns
        for dim in ["x", "y", "z"]:
            self._logger.debug(f"Computing results for {dim}")
            # number density
            key = "number_density"
            key_std = "number_density_stddev"
            try:
                self.results[dim][key] = np.average(
                    self._results[:, idx_start : (idx_start + self.nbins)],
                    axis=0,
                    weights=weights,
                )
                idx_start += self.nbins
            except ValueError as e:
                self._logger.error(f"Error in np.average for {key}: {e}")
                self._logger.error(f"idx_start: {idx_start}, nbins: {self.nbins}")
                self._logger.error(f"weights shape: {weights.shape}")
                self._logger.error(
                    f"results shape: {self._results[:, idx_start : (idx_start + self.nbins)].shape}"
                )
                raise e

            self.results[dim][key_std] = np.average(
                self._results[:, idx_start : (idx_start + self.nbins)],
                axis=0,
                weights=weights,
            )
            idx_start += self.nbins

            # mass density
            key = "mass_density"
            key_std = "mass_density_stddev"
            self.results[dim][key] = np.average(
                self._results[:, idx_start : (idx_start + self.nbins)],
                axis=0,
                weights=weights,
            )
            idx_start += self.nbins
            self.results[dim][key_std] = np.average(
                self._results[:, idx_start : (idx_start + self.nbins)],
                axis=0,
                weights=weights,
            )
            idx_start += self.nbins

            # charge density
            key = "charge_density"
            key_std = "charge_density_stddev"
            self.results[dim][key] = np.average(
                self._results[:, idx_start : (idx_start + self.nbins)],
                axis=0,
                weights=weights,
            )
            idx_start += self.nbins
            self.results[dim][key_std] = np.average(
                self._results[:, idx_start : (idx_start + self.nbins)],
                axis=0,
                weights=weights,
            )
            idx_start += self.nbins

            # bin edges
            self.results[dim]["hist_bin_edges"] = self._results[
                :, idx_start : (idx_start + self.nbins + 1)
            ][0]
            idx_start += self.nbins + 1
            self.results[dim]["hist_bin_centers"] = (
                self.results[dim]["hist_bin_edges"][:-1]
                + self.results[dim]["hist_bin_edges"][1:]
            ) / 2.0

        idx_final = idx_start
        if idx_final != self._ncols:
            msg = f"dim: {dim}, idx_final: {idx_final}, ncols: {self._ncols}"
            self._logger.error(msg)
            raise ValueError(msg)

        # add electrostatic potential

        # Compute standard deviation for the error
        # For certain tests in testsuite, floating point imprecision
        # can lead to negative radicands of tiny magnitude (yielding nan).
        # radicand_mass and radicand_charge are therefore calculated first
        # and negative values set to 0 before the square root
        # is calculated.
        self._logger.debug("Computing standard deviation")
        radicand_number = self.results[dim]["number_density_stddev"] - np.square(
            self.results[dim]["number_density"]
        )
        radicand_number[radicand_number < 0] = 0
        self.results[dim]["number_density_stddev"] = np.sqrt(radicand_number)

        radicand_mass = self.results[dim]["mass_density_stddev"] - np.square(
            self.results[dim]["mass_density"]
        )
        radicand_mass[radicand_mass < 0] = 0
        self.results[dim]["mass_density_stddev"] = np.sqrt(radicand_mass)

        radicand_charge = self.results[dim]["charge_density_stddev"] - np.square(
            self.results[dim]["charge_density"]
        )
        radicand_charge[radicand_charge < 0] = 0
        self.results[dim]["charge_density_stddev"] = np.sqrt(radicand_charge)

        self._logger.debug("Normalizing results")
        for dim in ["x", "y", "z"]:
            # norming factor, units of A^3
            norm = self.results[dim]["slice_volume"]
            for key in self.keys:
                self.results[dim][key] /= norm

        # calculate potential
        self._logger.debug("Calculating potential")
        self.calculate_potential()

    def calculate_potential(
        self, sigma_e: float = 0.0, dielectric: float = 1.0
    ) -> None:
        """Calculate the electrostatic potential from the charge density profile.

        Parameters
        ----------
        sigma_e : float, optional
            Surface charge density [e/A^2]. Default is 0.0.
        dielectric : float, optional
            Relative dielectric constant [unitless]. Default is 1.0.
        """
        # constants
        elementary_charge = 1.60217663e-19  # [C]
        vacuum_permittivity = 8.85418782e-12  # [F/m] = [C/(V*m)]
        angstrom = 1e-10  # [m]
        permittivity = vacuum_permittivity * dielectric * angstrom  # [C/(V*A)]
        prefactor = elementary_charge / permittivity  # [V]

        def cumulative_trapezoidal_error(yerr, x):
            """Calculate the error of the cumulative trapezoidal rule."""
            prefactor = (x[1:] - x[:-1]) / 2
            std_dev = np.sqrt(np.cumsum((np.square(yerr[1:]) + np.square(yerr[:-1]))))
            return prefactor * std_dev

        for dim in ["x", "y", "z"]:
            self._logger.debug(f"Calculating potential for {dim}")
            density = self.results[dim]["charge_density"]  # [e/A^3]
            err = self.results[dim]["charge_density_stddev"]  # [e/A^3]
            bins = self.results[dim]["hist_bin_centers"]  # [A]

            # Calculate the first integral of the charge density profile [e/A^2]
            potential = integrate.cumulative_trapezoid(
                density,  # [e/A^3]
                bins,  # [A]
                initial=0,  # [e/A^2] arbitrary reference
            )
            potential_var = cumulative_trapezoidal_error(err, bins)  # [e/A^2]

            # Calculate the second integral of the charge density profile [e/A]
            potential = -integrate.cumulative_trapezoid(
                potential - sigma_e,  # [e/A^2]
                bins,  # [A]
                initial=0,  # [e/A] no voltage at dim=0 (arbitrary reference)
            )
            potential_var = cumulative_trapezoidal_error(potential_var, bins)  # [e/A]

            # Add units to potential to convert to [V]
            potential *= prefactor
            potential_var *= prefactor
            self.results[dim]["potential"] = potential
            self.results[dim]["potential_stddev"] = potential_var

    def save(self, dir_out: str = None) -> None:
        """
        Save the results of the analysis to a parquet file. The results
        are saved to the `data` directory in the `dir_out` directory.

        This method should only be called after the analysis has been
        run.

        Parameters
        ----------
        dir_out : str, optional
            The directory to save the results to. If not specified, the
            results are saved to the directory specified in the
            `dir_out` attribute.
        """
        self._logger.info(f"Saving results for {self._tag} to {dir_out}")
        if dir_out is None:
            dir_out = self._dir_out / "data"
        Path(dir_out).mkdir(parents=True, exist_ok=True)

        # save the dataframe to a file
        # self._logger.debug(f"Saving results to {dir_out / self._df_filename}")
        # self._df.to_parquet(dir_out / self._df_filename)

        # save the results to a compressed numpy file
        def save_results(dim: str) -> None:
            self._logger.debug(f"Saving results for {dim}")
            np.savez_compressed(
                dir_out / f"lineardensity_{dim}_{self._tag}.npz",
                hist_bin_edges=self.results[dim]["hist_bin_edges"],
                hist_bin_centers=self.results[dim]["hist_bin_centers"],
                number_density=self.results[dim]["number_density"],
                number_density_stddev=self.results[dim]["number_density_stddev"],
                mass_density=self.results[dim]["mass_density"],
                mass_density_stddev=self.results[dim]["mass_density_stddev"],
                charge_density=self.results[dim]["charge_density"],
                charge_density_stddev=self.results[dim]["charge_density_stddev"],
                potential=self.results[dim]["potential"],
                potential_stddev=self.results[dim]["potential_stddev"],
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(save_results, ["x", "y", "z"])

    def figures(
        self, dim: str = None, title: str = "Linear Density", ext: str = "png"
    ) -> tuple[plt.figure, plt.axes]:
        """
        Plot the mass and charge density profiles.

        Parameters
        ----------
        dim : str, optional
            Axis to plot. Default is None. Must be one of 'x', 'y', or 'z'.
        title : str, optional
            Title of the plot
        ext : str, optional
            Extension of the plot file. Default is 'png'.

        Returns
        -------
        tuple[plt.figure, plt.axes]
            The figures and axes of the plots.
        """
        figs = []
        axs = []

        if dim is None:
            dim = ["x", "y", "z"]

        for d in dim:
            fig, ax = self.plt_number_density(dim=d, title=title, ext=ext)
            figs.append(fig)
            axs.append(ax)

            fig, ax = self.plt_mass_density(dim=d, title=title, ext=ext)
            figs.append(fig)
            axs.append(ax)

            fig, ax = self.plt_charge_density(dim=d, title=title, ext=ext)
            figs.append(fig)
            axs.append(ax)

        return figs, axs

    def plt_number_density(self, dim: str = "z", title: str = None, ext: str = "png"):
        """
        Plot the number density profiles.

        Parameters
        ----------
        dim : str, optional
            Dimension to plot. Default is 'z'. Must be one of 'x', 'y', or 'z'.
        title : str, optional
            Title of the plot
        ext : str, optional
            Extension of the plot file. Default is 'png'.

        Returns
        -------
        tuple[plt.figure, plt.axes]
            The figure and axes of the plot.

        Raises
        ------
        ValueError
            If `dim` is not one of 'x', 'y', or 'z'.
        """
        if dim not in ["x", "y", "z"]:
            raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")
        if title is None:
            title = f"${dim}$-axis"
        if self._dir_out / "figures" not in list(self._dir_out.iterdir()):
            (self._dir_out / "figures").mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            self.results[dim]["hist_bin_centers"] / 10,
            self.results[dim]["number_density"] / (10**3),
            label=f"{dim}-axis",
        )
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Number density [nm$^{-3}$]")
        ax.set_title(title, y=1.05)
        fig.savefig(self._dir_out / f"figures/number_density_{dim}_{self._tag}.{ext}")

        return fig, ax

    def plt_mass_density(self, dim: str = "z", title: str = None, ext: str = "png"):
        """
        Plot the mass density profiles.

        Parameters
        ----------
        dim : str, optional
            Dimension to plot. Default is 'z'. Must be one of 'x', 'y', or 'z'.
        title : str, optional
            Title of the plot
        ext : str, optional
            Extension of the plot file. Default is 'png'.

        Returns
        -------
        tuple[plt.figure, plt.axes]
            The figure and axes of the plot.

        Raises
        ------
        ValueError
            If `dim` is not one of 'x', 'y', or 'z'.
        """
        if dim not in ["x", "y", "z"]:
            raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")
        if title is None:
            title = f"${dim}$-axis"
        if self._dir_out / "figures" not in list(self._dir_out.iterdir()):
            (self._dir_out / "figures").mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            self.results[dim]["hist_bin_centers"] / 10,
            self.results[dim]["mass_density"] / (10**3),
            label=f"{dim}-axis",
        )
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Mass density [g/nm$^3$]")
        ax.set_title(title, y=1.05)
        fig.savefig(self._dir_out / f"figures/mass_density_{dim}_{self._tag}.{ext}")

        return fig, ax

    def plt_charge_density(self, dim: str = "z", title: str = None, ext: str = "png"):
        """
        Plot the charge density profiles.

        Parameters
        ----------
        dim : str, optional
            Dimension to plot. Default is 'z'. Must be one of 'x', 'y', or 'z'.
        title : str, optional
            Title of the plot
        ext : str, optional
            Extension of the plot file. Default is 'png'.

        Returns
        -------
        tuple[plt.figure, plt.axes]
            The figure and axes of the plot.

        Raises
        ------
        ValueError
            If `dim` is not one of 'x', 'y', or 'z'.
        """
        if dim not in ["x", "y", "z"]:
            raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")
        if title is None:
            title = f"${dim}$-axis"
        if self._dir_out / "figures" not in list(self._dir_out.iterdir()):
            (self._dir_out / "figures").mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            self.results[dim]["hist_bin_centers"] / 10,
            self.results[dim]["charge_density"] / (10**3),
            label=f"{dim}-axis",
        )
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Charge density [$e$/nm$^3$]")
        ax.set_title(title, y=1.05)
        fig.savefig(self._dir_out / f"figures/charge_density_{dim}_{self._tag}.{ext}")

        return fig, ax
