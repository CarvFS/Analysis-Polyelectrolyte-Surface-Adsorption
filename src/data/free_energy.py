"""
| Author: Alec Glisman (GitHub: @alec-glisman)
| Date: 2023-11-08
| Description: This module provides functions to calculate the free energy
| surface for a given distance collective variable and to |calculate the
| difference in free energy between two wells.

Functions
---------
fes_1d(x: np.ndarray, weights: np.ndarray = None, bandwidth: float = None,
d3_coord: bool = False, cv_grid: np.ndarray = None, domain: tuple[float, float] = None,
n_grid: int = 300) -> tuple[np.ndarray, np.ndarray]:
    Calculate the free energy surface for a given distance collective variable.

diff_fes_1d(cv: np.ndarray, pmf: np.ndarray, lower_well: tuple[float, float],
upper_well: tuple[float, float]) -> float:
    Calculate the difference in free energy between the two wells.

Raises
------
ValueError
    If `min_val` is not less than `max_val`.
AssertionError
    If the lower or upper well domains do not have upper and lower bounds.
AssertionError
    If the PMF and CV do not have the same size.
"""

# import modules
import numpy as np
from scipy import integrate, stats


def fes_1d(
    x: np.ndarray,
    weights: np.ndarray = None,
    eqbm_percent: float = 0.1,
    final_percent: float = 1.0,
    bandwidth: float = None,
    d3_coord: bool = False,
    cv_grid: np.ndarray = None,
    domain: tuple[float, float] = (None, None),
    n_grid: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the free energy surface for a given distance collective variable.

    Parameters
    ----------
    x : np.ndarray
        Array of collective variable values.
    weights : np.ndarray, optional
        Array of weights for each collective variable value, by default None
    eqbm_percent : float, optional
        Percent of the simulation to use for equilibration, by default 0.1
    final_percent : float, optional
        Percent of the simulation to use for the final free energy surface, by default
        1.0
    bandwidth : float, optional
        Bandwidth for kernel density estimation, by default None
    d3_coord : bool, optional
        Whether the distance is a 3D coordinate so that the 2 log(r) term is added, by
        default False
    cv_grid : np.ndarray, optional
        Array of grid points to use for KDE, by default None
    domain : tuple[float, float], optional
        Tuple of (min, max) values for the domain of the free energy surface, by
        default None
    n_grid : int, optional
        Number of grid points to use for KDE, by default 300

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of arrays of distances and free energies.

    Raises
    ------
    ValueError
        If `min_val` is not less than `max_val`.
    """
    # input checking
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if weights is None:
        weights = np.ones_like(x)
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    # drop equilibration data
    idx_eqbm = int(eqbm_percent * len(x))
    idx_final = int(final_percent * len(x))
    x = x.copy()[idx_eqbm:idx_final]
    weights = weights.copy()[idx_eqbm:idx_final]

    # set kde domain
    min_val, max_val = domain
    if cv_grid is None:
        if min_val is None:
            min_val = np.nanmin(x)
        if max_val is None:
            max_val = np.nanmax(x)
        if min_val >= max_val:
            raise ValueError(
                f"min_val ({min_val}) must be less than max_val ({max_val})"
            )
        cv_grid = np.linspace(min_val, max_val, n_grid)

    # calculate KDE of x weighted by weights
    kde = stats.gaussian_kde(x, weights=weights, bw_method=bandwidth)
    fes = -kde.logpdf(cv_grid)

    # apply distance correction
    if d3_coord:
        fes += 2.0 * np.log(cv_grid)

    # set minimum to zero
    fes -= np.nanmin(fes)

    return cv_grid, fes


def diff_fes_1d(
    cv_grid: np.ndarray,
    pmf: np.ndarray,
    lower_well: tuple[float, float],
    upper_well: tuple[float, float],
) -> float:
    """
    Calculate the difference in free energy between the two wells.

    Parameters
    ----------
    cv : np.ndarray
        Array of collective variable values.
    pmf : np.ndarray
        Array of free energies as a function of collective variable, assumed to be
        unitless.
    lower_well : tuple[float, float]
        Tuple of (min, max) values for the lower well.
    upper_well : tuple[float, float]
        Tuple of (min, max) values for the upper well.

    Returns
    -------
    float
        Difference in free energy between the two wells. Unitless.

    Raises
    ------
    AssertionError
        If the lower or upper well domains do not have upper and lower bounds.
    AssertionError
        If the PMF and CV do not have the same size.
    """

    assert len(lower_well) == 2, "Lower well domain must have upper and lower bounds"
    assert len(upper_well) == 2, "Upper well domain must have upper and lower bounds"
    assert len(pmf) == len(cv_grid), "PMF and CV must have the same size"

    # get indices of lower and upper wells
    lower_well_idx = np.where((cv_grid > lower_well[0]) & (cv_grid < lower_well[1]))
    upper_well_idx = np.where((cv_grid > upper_well[0]) & (cv_grid < upper_well[1]))

    # integrate boltzmann factors of wells to get probabilities
    boltzmann = np.exp(-pmf)
    prob_upper = integrate.simpson(boltzmann[upper_well_idx], x=cv_grid[upper_well_idx])
    prob_lower = integrate.simpson(boltzmann[lower_well_idx], x=cv_grid[lower_well_idx])

    # calculate free energy difference as log of ratio of probabilities
    delta_fe = -np.log(prob_lower / prob_upper)
    return delta_fe
