# import modules
import numpy as np
from pathlib import Path
import pytest
from scipy import stats
import sys

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from bias.free_energy import fes_1d, diff_fes_1d


# Tests for fes_1d
def test_fes_1d():
    # Test with minimum input for fes_1d
    x = np.array([1, 2, 3])
    x_grid, fes = fes_1d(x)
    assert len(fes) == 300  # default n_grid
    assert len(x_grid) == 300

    # Test with weights
    weights = np.array([1, 1, 1])
    x_grid, fes = fes_1d(x, weights=weights)
    assert len(fes) == 300

    # Test with domain
    x_grid, fes = fes_1d(x, domain=(0, 4))
    assert x_grid[0] == 0
    assert x_grid[-1] == 4

    # Test ValueError for invalid domain
    with pytest.raises(ValueError):
        fes_1d(x, domain=(5, 4))

    # More tests can be added as needed


# Tests for diff_fes_1d
def test_diff_fes_1d():
    cv = np.linspace(0, 10, 100)
    pmf = np.exp(-cv)
    lower_well = (1, 2)
    upper_well = (3, 4)

    delta_fe = diff_fes_1d(cv, pmf, lower_well, upper_well)
    assert isinstance(delta_fe, float)

    # Test AssertionError for mismatched sizes
    with pytest.raises(AssertionError):
        diff_fes_1d(cv, pmf[:-1], lower_well, upper_well)

    # Test AssertionError for incorrect well domain
    with pytest.raises(AssertionError):
        diff_fes_1d(cv, pmf, (1,), upper_well)
