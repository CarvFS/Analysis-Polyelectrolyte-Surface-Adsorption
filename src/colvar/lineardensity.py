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
from scipy import interpolate
from scipy import linalg
from scipy.signal import find_peaks

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


def cumsimps(
    x, y, y_err=None, initial_condition: float = 0, reverse_order=False
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative integral of y using Simpson's rule.

    Parameters
    ----------
    x : array_like
        1-D array of x values.
    y : array_like
        1-D array of y values.
    y_err : array_like, optional
        1-D array of y errors. Default is None.
    initial_condition : float, optional
        Initial condition for the cumulative integral. Default is 0.
    reverse_order : bool, optional
        If True, the cumulative integral will be calculated in reverse order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The cumulative integral and its error.

    Raises
    ------
    ValueError
        If the length of x and y do not match.
    ValueError
        If the length of x and y_err do not match.
    ValueError
        If x is not equally spaced.
    """

    if y_err is None:
        y_err = np.zeros_like(y)
    if len(x) != len(y):
        raise ValueError("Length of x and y do not match.")
    if len(x) != len(y_err):
        raise ValueError("Length of x and y_err do not match.")

    dx = x[1:] - x[:-1]
    if not np.allclose(dx, dx[0]):
        raise ValueError("x is not equally spaced.")
    dx = dx[0]

    # calculate the cumulative integral
    y_var = np.square(y_err)
    integral = np.zeros_like(y)
    integral_var = np.zeros_like(y)

    if reverse_order:
        x = x[::-1].copy()
        y = y[::-1].copy()
        y_var = y_var[::-1].copy()

    # set initial condition
    integral[0] = initial_condition

    for i in range(1, len(y)):
        x_sub, y_sub, y_var_sub = x[: i + 1], y[: i + 1], y_var[: i + 1]
        integral[i] = integrate.simps(y_sub, x_sub)

        # 2 points: trapezoidal rule for the error
        if i == 1:
            integral_var[i] = np.square(dx / 2.0) * (y_var_sub[0] + y_var_sub[1])
        # 3 points: 1 interval of Simpson's rule
        elif i == 2:
            integral_var[i] = np.square(dx / 6.0) * (
                y_var_sub[0] + 4 * y_var_sub[1] + y_var_sub[2]
            )
        # 4+ points: general case
        else:
            integral_var[i] = np.square(dx / 3.0) * (
                y_var_sub[0]
                + 16 * np.sum(y_var_sub[1::2])
                + 4 * np.sum(y_var_sub[2::2])
                + y_var_sub[-1]
            )

    if reverse_order:
        integral = -integral[::-1].copy()
        integral_var = integral_var[::-1].copy()

    return integral, np.sqrt(integral_var)


def cumtrapz(
    x, y, y_err=None, initial_condition: float = 0.0, reverse_order=False
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative integral of y using the trapezoidal rule.

    Parameters
    ----------
    x : array_like
        1-D array of x values.
    y : array_like
        1-D array of y values.
    y_err : array_like, optional
        1-D array of y errors. Default is None.
    initial_condition : float, optional
        Initial condition for the cumulative integral. Default is 0.
    reverse_order : bool, optional
        If True, the cumulative integral will be calculated in reverse order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The cumulative integral and its error.

    Raises
    ------
    ValueError
        If the length of x and y do not match.
    ValueError
        If the length of x and y_err do not match.
    """

    if y_err is None:
        y_err = np.zeros_like(y)
    if len(x) != len(y):
        raise ValueError("Length of x and y do not match.")
    if len(x) != len(y_err):
        raise ValueError("Length of x and y_err do not match.")

    dx = x[1:] - x[:-1]
    if not np.allclose(dx, dx[0]):
        raise ValueError("x is not equally spaced.")
    dx = dx[0]

    # calculate the cumulative integral
    y_var = np.square(y_err)
    integral = np.zeros_like(y)
    integral_var = np.zeros_like(y)

    if reverse_order:
        x = x[::-1].copy()
        y = y[::-1].copy()
        y_var = y_var[::-1].copy()

    # set initial condition
    integral[0] = initial_condition

    for i in range(1, len(y)):
        x_sub, y_sub, y_var_sub = x[: i + 1], y[: i + 1], y_var[: i + 1]
        integral[i] = integrate.trapz(y_sub, x_sub)

        # 2 points: trapezoidal rule for the error
        if i == 1:
            integral_var[i] = np.square(dx / 2.0) * (y_var_sub[0] + y_var_sub[1])
        # 3+ points: general case
        else:
            integral_var[i] = np.square(dx / 2.0) * (
                y_var_sub[0] + 2 * np.sum(y_var_sub[1:-1]) + y_var_sub[-1]
            )

    if reverse_order:
        integral = -integral[::-1].copy()
        integral_var = integral_var[::-1].copy()

    return integral, np.sqrt(integral_var)


def periodic_bvp(
    x: np.ndarray, y: np.ndarray, y_err: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    # distance array for 2nd order differences
    dx = x[1] - x[0]
    assert np.allclose(x[1:] - x[:-1], dx), "x is not equally spaced"

    # Laplacian matrix
    N = len(x)
    A = (
        np.diag(-2 * np.ones(N), 0)
        + np.diag(np.ones(N - 1), -1)
        + np.diag(np.ones(N - 1), 1)
    )
    # periodic boundary conditions
    A[0, N - 1] = 1
    A[N - 1, 0] = 1
    # scale by dx^2
    A = A / dx**2

    # solve the linear system
    b = -y
    u = np.zeros(N)
    u[1:] = linalg.solve(A[1:, 1:], b[1:])
    u[0] = u[-1]
    return u, np.zeros_like(u)


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
    results.x.position : numpy.ndarray
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
        nbins=1000,
        label=None,
        df_weights=None,
        dims=["x", "y", "z"],
        densities=["number", "mass", "charge"],
        verbose=False,
        **kwargs,
    ):
        super().__init__(select.universe.trajectory, (select,), label=label, **kwargs)
        self._logger = setup_logging(log_file=f"logs/{__name__}.log", verbose=verbose)
        self._verbose = verbose

        # dimensions for the analysis
        self.dims = dims
        self._logger.debug(f"dims: {self.dims}")
        for dim in self.dims:
            if dim not in self.dims:
                raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")

        # allows use of run(parallel=True)
        self._ags = [select]
        self._universe = select.universe

        # box dimensions
        offset = 0.0  # [Angstrom] to avoid numerical issues
        a, b, c, alpha, beta, gamma = self._universe.dimensions
        # convert to radians
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)
        # volume of the parallelepiped
        angle_scale = np.sqrt(
            1.0
            - np.cos(alpha) ** 2
            - np.cos(beta) ** 2
            - np.cos(gamma) ** 2
            + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        self._logger.debug(f"angle_scale: {angle_scale}")
        self._volume = a * b * c * angle_scale
        self._logger.debug(f"volume: {self._volume} [Angstrom^3]")
        # discretization
        self.nbins = nbins
        self._logger.debug(f"nbins: {self.nbins}")
        self.x_bins = np.linspace(0 + offset, a - offset, nbins + 1)
        self.y_bins = np.linspace(0 + offset, b - offset, nbins + 1)
        self.z_bins = np.linspace(0 + offset, c - offset, nbins + 1)
        self.bins = [self.x_bins, self.y_bins, self.z_bins]

        # calculate volume of each slice
        self.x_slice_volume = b * c * angle_scale * (self.x_bins[1:] - self.x_bins[:-1])
        self.y_slice_volume = a * c * angle_scale * (self.y_bins[1:] - self.y_bins[:-1])
        self.z_slice_volume = a * b * angle_scale * (self.z_bins[1:] - self.z_bins[:-1])

        # group of atoms on which to compute the COM (same as used in
        # AtomGroup.wrap())
        self.grouping = grouping

        # Initiate result instances
        if "x" in dims:
            self.results["x"] = Results(dim=0)
            self.results["x"]["slice_volume"] = self.x_slice_volume
            self.results["x"]["position"] = (self.x_bins[1:] + self.x_bins[:-1]) / 2
        if "y" in dims:
            self.results["y"] = Results(dim=1)
            self.results["y"]["slice_volume"] = self.y_slice_volume
            self.results["y"]["position"] = (self.y_bins[1:] + self.y_bins[:-1]) / 2
        if "z" in dims:
            self.results["z"] = Results(dim=2)
            self.results["z"]["slice_volume"] = self.z_slice_volume
            self.results["z"]["position"] = (self.z_bins[1:] + self.z_bins[:-1]) / 2

        # properties to calculate
        self.calc_props = densities
        self.calc_mass = True if "mass" in self.calc_props else False
        self.calc_charge = True if "charge" in self.calc_props else False
        self.calc_number = True if "number" in self.calc_props else False
        self.num_props = len(self.calc_props)
        self.keys = []
        for prop in self.calc_props:
            self.keys.append(f"{prop}_density")
            self.keys.append(f"{prop}_density_stddev")
            self._logger.debug(f"Calculating {prop} density in {self.dims} directions")

        # Initialize results array with zeros
        for dim in self.results:
            self._logger.debug(f"Initializing results for {dim}")
            for key in self.keys:
                self.results[dim][key] = np.zeros(self.nbins)

        # dataframe containing weight of each frame from biasing potential
        if df_weights is not None:
            self._weighted: bool = True
            self._df_weights = df_weights[["time", "weight"]].copy()
        else:
            self._weighted: bool = False
            self._df_weights: pd.DataFrame = None

        # output data
        self._dir_out: Path = Path("./mdanalysis_lineardensity")
        self._df_filename = f"lineardensity_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df = None
        self._columns = ["frame", "time"]
        for dim in self.results:
            for key in self.keys:
                for idx in range(self.nbins):
                    self._columns.append(f"{dim}_{key}_{idx}")

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
            masses = ag.masses
            charges = ag.charges

        elif self.grouping in ["residues", "segments", "fragments"]:
            masses = ag.total_mass(compound=self.grouping)
            charges = ag.total_charge(compound=self.grouping)

        else:
            raise AttributeError(f"{self.grouping} is not a valid value for grouping.")

        # wrap the atomgroup to the unit cell
        self.group = getattr(ag, self.grouping)
        ag.wrap(compound=self.grouping)

        # Find position of atom/group of atoms
        if self.grouping == "atoms":
            positions = ag.positions  # faster for atoms
        else:
            # Centre of mass for residues, segments, fragments
            positions = ag.center_of_mass(compound=self.grouping)

        idx_start = 2  # skip frame and time columns
        for dim in self.dims:
            idx = self.results[dim]["dim"]
            bins = self.bins[idx]

            # histogram for positions [# / A]
            if self.calc_number:
                hist, _ = np.histogram(
                    positions[:, idx],
                    bins=bins,
                )
                result[idx_start : idx_start + self.nbins] = hist
                idx_start += self.nbins
                result[idx_start : idx_start + self.nbins] = np.square(hist)
                idx_start += self.nbins

            # histogram for positions weighted on masses [g / A]
            if self.calc_mass:
                hist, _ = np.histogram(
                    positions[:, idx],
                    weights=masses,
                    bins=bins,
                )  # [# * g / mol / A^3]
                hist /= 6.02214076e23  # Avogadro number [mol/#]
                result[idx_start : idx_start + self.nbins] = hist
                idx_start += self.nbins
                result[idx_start : idx_start + self.nbins] = np.square(hist)
                idx_start += self.nbins

            # histogram for positions weighted on charges [e / A]
            if self.calc_charge:
                hist, _ = np.histogram(
                    positions[:, idx],
                    weights=charges,
                    bins=bins,
                )
                result[idx_start : idx_start + self.nbins] = hist
                idx_start += self.nbins
                result[idx_start : idx_start + self.nbins] = np.square(hist)
                idx_start += self.nbins

        if idx_start != self._ncols:
            msg = f"dim: {dim}, idx_start: {idx_start}, ncols: {self._ncols}"
            self._logger.error(msg)
            raise ValueError(msg)

        return result

    def _conclude(self):
        # merge weights with results
        if self._weighted:
            self._logger.debug("Merging weights with results")
            weights_plumed = self._df_weights["weight"].to_numpy()
            times_plumed = self._df_weights["time"].to_numpy()
            times_mda = self._results[:, 1]

            # find weights_mda as the closest time in times_plumed
            weights_mda = np.zeros_like(times_mda)
            for idx, time in enumerate(times_mda):
                idx_closest = np.argmin(np.abs(time - times_plumed))
                if np.abs(time - times_plumed[idx_closest]) > 1e-3:
                    self._logger.warning(
                        f"Closest time in plumed file: {times_plumed[idx_closest]}, "
                        f"MDAnalysis time: {time}"
                    )
                weights_mda[idx] = weights_plumed[idx_closest]

            weights = weights_mda

        else:
            self._logger.debug("No weights provided")
            weights = np.ones(len(self._results[:, 0]))

        # drop self._df for memory
        self._df = None
        self._logger.info(f"Memory of results: {self._results.nbytes / 1024**2:.0f} MB")

        # output data from self._results to self.results
        # weighted average over all frames (rows of self._results)
        self._logger.debug(f"Computing results from {self._results.shape[0]} frames")
        idx_start = 2  # skip frame and time columns
        for dim in self.dims:

            # number density
            if self.calc_number:
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
            if self.calc_mass:
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
            if self.calc_charge:
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

        # delete self._results to free memory
        self._results = None

        idx_final = idx_start
        if idx_final != self._ncols:
            msg = f"dim: {dim}, idx_final: {idx_final}, ncols: {self._ncols}"
            self._logger.error(msg)
            raise ValueError(msg)

        # Compute standard deviation for the error
        # For certain tests in testsuite, floating point imprecision
        # can lead to negative radicands of tiny magnitude (yielding nan).
        # radicand_mass and radicand_charge are therefore calculated first
        # and negative values set to 0 before the square root
        # is calculated.
        self._logger.debug("Computing standard deviation")
        if self.calc_number:
            radicand_number = self.results[dim]["number_density_stddev"] - np.square(
                self.results[dim]["number_density"]
            )
            radicand_number[radicand_number < 0] = 0
            self.results[dim]["number_density_stddev"] = np.sqrt(radicand_number)

        if self.calc_mass:
            radicand_mass = self.results[dim]["mass_density_stddev"] - np.square(
                self.results[dim]["mass_density"]
            )
            radicand_mass[radicand_mass < 0] = 0
            self.results[dim]["mass_density_stddev"] = np.sqrt(radicand_mass)

        if self.calc_charge:
            radicand_charge = self.results[dim]["charge_density_stddev"] - np.square(
                self.results[dim]["charge_density"]
            )
            radicand_charge[radicand_charge < 0] = 0
            self.results[dim]["charge_density_stddev"] = np.sqrt(radicand_charge)

        self._logger.debug("Normalizing results")
        for dim in self.dims:
            # norming factor, units of A^3
            norm = self.results[dim]["slice_volume"]
            for key in self.keys:
                self.results[dim][key] /= norm

        # calculate potential
        if self.calc_charge:
            self._logger.debug("Calculating potential")
            self.calculate_potential()

    def calculate_potential(
        self,
        sigma_e: float = 0.0,
        dielectric: float = 1.0,
        method: str = "periodic_bvp",
        n_layers: int = 4,
    ) -> None:
        """Calculate the electrostatic potential from the charge density profile.

        Parameters
        ----------
        sigma_e : float, optional
            Surface charge density [e/A^2]. Default is 0.0.
        dielectric : float, optional
            Relative dielectric constant [unitless]. Default is 1.0.
        method : str, optional
            Method to use for numerical integration. Must be one of 'cumtrapz', 'cumsimps', or 'periodic_bvp'.
            o'cumsimps'. Default is 'cumtrapz'.
        n_layers : int, optional
            Number of crystal layers to consider. Default is 4.
        """
        # constants
        elementary_charge = 1.60217663e-19  # [C]
        vacuum_permittivity = 8.85418782e-12  # [F/m] = [C/(V*m)]
        angstrom = 1e-10  # [m]
        permittivity = vacuum_permittivity * dielectric * angstrom  # [C/(V*Angstrom)]
        prefactor = elementary_charge / permittivity  # [V]

        for dim in self.dims:
            density = self.results[dim]["charge_density"]  # [e/A^3]
            err = self.results[dim]["charge_density_stddev"]  # [e/A^3]
            bins = self.results[dim]["position"]  # [A]

            if len(bins) != len(density):
                raise ValueError(
                    "Length of bins and density do not match. "
                    f"bins: {len(bins)}, density: {len(density)}"
                )
            if len(bins) != len(err):
                raise ValueError(
                    "Length of bins and error do not match. "
                    f"bins: {len(bins)}, error: {len(err)}"
                )

            # upsample the density by a factor of 10 with cubic spline interpolation
            # to improve the accuracy of the numerical integration
            if method in ["cumsimps", "cumtrapz"]:
                upsample = 10
            elif method == "periodic_bvp":
                upsample = 1

            bins_upsampled = np.linspace(bins[0], bins[-1], len(bins) * upsample)
            density_upsampled = (
                interpolate.interp1d(bins, density, kind="cubic")(bins_upsampled)
                * prefactor
            )
            err_upsampled = (
                interpolate.interp1d(bins, err, kind="cubic")(bins_upsampled)
                * prefactor
            )

            if dim == "x":
                symmetry_center = self._universe.dimensions[0] / 2.0
            elif dim == "y":
                symmetry_center = self._universe.dimensions[1] / 2.0
            elif dim == "z":
                # find n largest peaks in the density profile
                peaks, _ = find_peaks(density_upsampled, distance=1000)
                peaks = peaks[np.argsort(density_upsampled[peaks])[-n_layers:]]
                peak_locs = bins_upsampled[peaks]
                # crystal center is average position of the n largest peaks
                crystal_center = np.mean(peak_locs)
                # crystal width is the distance between the n largest peaks
                crystal_width = np.max(peak_locs) - np.min(peak_locs)
                self._logger.debug(f"Four largest peaks locations: {peak_locs} [A]")
                self._logger.debug(f"Crystal center: {crystal_center} [A]")
                self._logger.debug(f"Crystal width: {crystal_width} [A]")

                # water width is the box length minus the crystal width
                water_width = self._universe.dimensions[2] - crystal_width
                self._logger.debug(f"Water width: {water_width} [A]")

                # symmetry center is box_half_length away from the crystal center
                symmetry_center = crystal_center + 0.5 * water_width

            idx_symmetry = np.argmin(np.abs(bins_upsampled - symmetry_center))
            self._logger.debug(f"Symmetry center: {symmetry_center} [A]")
            self._logger.debug(f"Symmetry index: {idx_symmetry}")

            if method == "cumsimps":
                potential_upsampled, potential_err_upsampled = cumsimps(
                    bins_upsampled,
                    density_upsampled,
                    err_upsampled,
                    reverse_order=False,
                    initial_condition=0.0,
                )
                # set potential to zero at the symmetry center
                potential_upsampled -= potential_upsampled[idx_symmetry]

                potential_upsampled, potential_err_upsampled = cumsimps(
                    bins_upsampled,
                    potential_upsampled - sigma_e,
                    potential_err_upsampled,
                    reverse_order=False,
                    initial_condition=0.0,
                )
                potential_upsampled -= potential_upsampled[idx_symmetry]

            elif method == "cumtrapz":
                potential_upsampled, potential_err_upsampled = cumtrapz(
                    bins_upsampled,
                    density_upsampled,
                    err_upsampled,
                    reverse_order=False,
                    initial_condition=0.0,
                )
                potential_upsampled -= potential_upsampled[idx_symmetry]

                potential_upsampled, potential_err_upsampled = cumtrapz(
                    bins_upsampled,
                    potential_upsampled - sigma_e,
                    potential_err_upsampled,
                    reverse_order=False,
                    initial_condition=0.0,
                )
                potential_upsampled -= potential_upsampled[idx_symmetry]

            elif method == "periodic_bvp":
                potential_upsampled, potential_err_upsampled = periodic_bvp(
                    bins_upsampled,
                    density_upsampled,
                    err_upsampled,
                )

            else:
                raise ValueError(f"Method {method} is not supported.")

            # downsample the potential to the original number of bins
            potential = interpolate.interp1d(
                bins_upsampled, potential_upsampled, kind="cubic"
            )(bins)
            potential_err = interpolate.interp1d(
                bins_upsampled, potential_err_upsampled, kind="cubic"
            )(bins)

            self.results[dim]["potential"] = potential
            self.results[dim]["potential_stddev"] = potential_err

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
        if dir_out is None:
            dir_out = self._dir_out / "data"
        self._logger.info(f"Saving results for {self._tag} to {dir_out}")
        Path(dir_out).mkdir(parents=True, exist_ok=True)

        # save the dataframe to a file
        # self._logger.debug(f"Saving results to {dir_out / self._df_filename}")
        # self._df.to_parquet(dir_out / self._df_filename)

        # save the results to a compressed numpy file
        def save_results(dim: str) -> None:
            np.savez_compressed(
                dir_out / f"lineardensity_{dim}_{self._tag}.npz",
                position=self.results[dim]["position"],
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
            executor.map(save_results, self.dims)

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
            dim = self.dims

        for d in dim:
            if self.calc_number:
                fig, ax = self.plt_number_density(dim=d, title=title, ext=ext)
                figs.append(fig)
                axs.append(ax)

            if self.calc_mass:
                fig, ax = self.plt_mass_density(dim=d, title=title, ext=ext)
                figs.append(fig)
                axs.append(ax)

            if self.calc_charge:
                fig, ax = self.plt_charge_density(dim=d, title=title, ext=ext)
                figs.append(fig)
                axs.append(ax)

                fig, ax = self.plt_potential(dim=d, title=title, ext=ext)
                figs.append(fig)
                axs.append(ax)

                fig, ax = self.plt_potential_nondim(dim=d, title=title, ext=ext)
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
        if dim not in self.dims:
            raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")
        if title is None:
            title = f"${dim}$-axis"
        if self._dir_out / "figures" not in list(self._dir_out.iterdir()):
            (self._dir_out / "figures").mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            self.results[dim]["position"] / 10,
            self.results[dim]["number_density"] / (10**3),
            label=f"{dim}-axis",
        )
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Number density [nm$^{-3}$]")
        ax.set_title(title, y=1.05)
        fig.savefig(self._dir_out / f"figures/{dim}_number_density_{self._tag}.{ext}")

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
        if dim not in self.dims:
            raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")
        if title is None:
            title = f"${dim}$-axis"
        if self._dir_out / "figures" not in list(self._dir_out.iterdir()):
            (self._dir_out / "figures").mkdir(parents=True, exist_ok=True)

        kg_per_g = 0.001
        nm_per_angstrom = 0.1
        m_per_angstrom = 1e-10

        x_nm = self.results[dim]["position"] * nm_per_angstrom
        y_kg_m3 = self.results[dim]["mass_density"] * kg_per_g / (m_per_angstrom**3)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            x_nm,
            y_kg_m3,
            label=f"{dim}-axis",
        )
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Mass density [kg/m$^3$]")
        ax.set_title(title, y=1.05)
        fig.savefig(self._dir_out / f"figures/{dim}_mass_density_{self._tag}.{ext}")

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
        if dim not in self.dims:
            raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")
        if title is None:
            title = f"${dim}$-axis"
        if self._dir_out / "figures" not in list(self._dir_out.iterdir()):
            (self._dir_out / "figures").mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            self.results[dim]["position"] / 10,
            self.results[dim]["charge_density"] / (10**3),
            label=f"{dim}-axis",
        )
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Charge density [$e$/nm$^3$]")
        ax.set_title(title, y=1.05)
        fig.savefig(self._dir_out / f"figures/{dim}_charge_density_{self._tag}.{ext}")

        return fig, ax

    def plt_potential(self, dim: str = "z", title: str = None, ext: str = "png"):
        """
        Plot the electrostatic potential profiles.

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
        if dim not in self.dims:
            raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")
        if title is None:
            title = f"${dim}$-axis"
        if self._dir_out / "figures" not in list(self._dir_out.iterdir()):
            (self._dir_out / "figures").mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            self.results[dim]["position"] / 10,
            self.results[dim]["potential"],
            label=f"{dim}-axis",
        )
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Electrostatic Potential [V]")
        ax.set_title(title, y=1.05)
        fig.savefig(self._dir_out / f"figures/{dim}_potential_{self._tag}.{ext}")

        return fig, ax

    def plt_potential_nondim(self, dim: str = "z", title: str = None, ext: str = "png"):
        """
        Plot the electrostatic potential profiles in non-dimensional units.

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
        if dim not in self.dims:
            raise ValueError(f"dim must be one of 'x', 'y', or 'z'. Got {dim}.")
        if title is None:
            title = f"${dim}$-axis"
        if self._dir_out / "figures" not in list(self._dir_out.iterdir()):
            (self._dir_out / "figures").mkdir(parents=True, exist_ok=True)

        kB = 1.38064852e-23  # [J/K]
        T = 300.0  # [K]
        e_electron = 1.602176634e-19  # [C]
        dimensionless_factor = e_electron / (kB * T)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # add major and minor grid
        ax.grid(which="major", linestyle="-", linewidth="0.5", color="black")
        ax.grid(which="minor", linestyle=":", linewidth="0.5", color="black")

        ax.plot(
            self.results[dim]["position"] / 10,
            self.results[dim]["potential"] * dimensionless_factor,
            label=f"{dim}-axis",
        )
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Electrostatic Potential [k$_B$T/e]")
        ax.set_title(title, y=1.05)
        fig.savefig(self._dir_out / f"figures/{dim}_potential_nondim_{self._tag}.{ext}")

        return fig, ax
