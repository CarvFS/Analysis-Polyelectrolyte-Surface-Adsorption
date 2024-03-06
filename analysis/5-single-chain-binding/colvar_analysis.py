# ########################################################################### #
# Imports                                                                     #
# ########################################################################### #

# standard library
import argparse
import datetime
from pathlib import Path
import os
import sys

# third-party packages
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm

# get absolute path to file's parent directory
dir_proj_base = Path.cwd().resolve().parents[1]
sys.path.insert(0, f"{dir_proj_base}/src")

# Internal dependencies
from data.pipeline import DataPipeline  # noqa: E402
from data.free_energy import fes_1d, diff_fes_1d  # noqa: E402
from render.util import set_style, close_fig  # noqa: E402
from render.dynamics import (  # noqa: E402
    plt_opes_dynamics,
    plt_metad_dynamics,
    plt_md_dynamics,
)
from render.thermo import plt_pmf, plt_pmf_diff_conv, mov_pmf_conv  # noqa: E402


# ########################################################################### #
# Global variables                                                            #
# ########################################################################### #

# general
VERBOSE: bool = True
N_JOBS: int = 1

# data I/O
CWDIR: Path = Path.cwd()
OUTPUT_SDIR: Path = Path("data")
SIMULATION_SDIR: str = "data"
INPUT_BASE_DIR: Path = Path(
    "/nfs/zeal_nas/home_mount/aglisman/GitHub" + "/Polyelectrolyte-Surface-Adsorption"
)

# set thermal energy
TEMPERATURE_K: float = 300  #: simulation temperature in Kelvin

# calculate free energy surface at the end of the simulation
CV: str = "dist_chain.z"
EQBM_PERCENT: float = 0.10
BANDWIDTH: float = 0.08

# calculate free energy difference between the two states
CV_GRID: float = np.linspace(0.2, 5.0, 250)
CV_LOWER_WELL: tuple[float, float] = (0.7, 1.0)
CV_UPPER_WELL: tuple[float, float] = (3.0, 3.2)
CV_PLATEAU: tuple[float, float] = (2.2, 2.6)

# plotting parameters
N_TIMESTEP_RENDER: int = 200
PMF_YRANGE: tuple[float, float] = (-20, 10)
PMF_XRANGE: tuple[float, float] = (0.2, 3.5)


# ########################################################################### #
# Functions                                                                   #
# ########################################################################### #


def load_data(
    pipeline: DataPipeline, method: str, output_subdir: Path
) -> tuple[str, pd.DataFrame]:
    """Load data for a given method.

    Parameters
    ----------
    pipeline : DataPipeline
        Data pipeline object.
    method : str
        Method name.
    output_subdir : Path
        Output subdirectory.

    Returns
    -------
    output_dir : str
        Output directory.
    colvar : pd.DataFrame
        Pandas DataFrame containing the colvar data.
    """

    # create output directory
    if "replica" in method:
        method_no_replica = method.split("-replica")[0] + "_replex"
    else:
        method_no_replica = method
    output_dir = CWDIR / Path(output_subdir) / pipeline.tag / method_no_replica
    output_dir.mkdir(parents=True, exist_ok=True)

    # load plumed data
    colvar = pipeline.load_plumed_colvar(method)

    return output_dir, colvar


def dynamics(df: pd.DataFrame, cv: str, method: str) -> None:
    """Plot dynamics of a collective variable.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of collective variable.
    method : str
        Method used for sampling the simulation. Must include
        "opes", "metad", or "md" in the string.
    cv : str
        Collective variable to plot.

    Raises
    ------
    ValueError
        If an invalid method is provided.
    """
    if "opes" in method:
        fig = plt_opes_dynamics(df, cv, tag=method)
    elif "metad" in method:
        fig = plt_metad_dynamics(df, cv, tag=method)
    elif "md" in method:
        fig = plt_md_dynamics(df, cv, tag=method)
    else:
        raise ValueError(f"Invalid method: {method}")

    close_fig(fig)


def free_energy_surface(
    df: pd.DataFrame,
    cv: str,
    cv_grid: np.ndarray,
    bandwidth: float,
    eqbm_percent: float,
    method: str,
) -> None:
    """Calculate and plot free energy surface.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of collective variable.
    cv : str
        Collective variable to plot.
    cv_grid : np.ndarray
        Grid for free energy surface.
    bandwidth : float
        Bandwidth for kernel density estimation.
    eqbm_percent : float
        Percent of simulation to use for equilibration.
    method : str
        Method used for sampling in the simulation.
    """
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # calculate free energy surface and save to csv
    cv_grid, pmf = fes_1d(
        x=df[cv],
        cv_grid=cv_grid,
        bandwidth=bandwidth,
        weights=df["weight"],
        eqbm_percent=eqbm_percent,
        plateau_domain=CV_PLATEAU,
    )
    df = pd.DataFrame({"cv": cv_grid, "pmf": pmf})
    df.to_csv(output_dir / f"{method}_pmf_{cv}.csv", index=False)

    # plot free energy surface and save
    fig = plt_pmf(
        cv_grid=cv_grid,
        pmf=pmf,
        tag=f"{method}_cv_{cv}",
        xrange=PMF_XRANGE,
        yrange=PMF_YRANGE,
    )
    close_fig(fig)


def free_energy_difference(
    df: pd.DataFrame,
    cv: str,
    cv_grid: np.ndarray,
    bandwidth: float,
    eqbm_percent: float,
    lower_well: tuple[float, float],
    upper_well: tuple[float, float],
    n_timestep_plot: int,
    method: str,
    output_dir: Path,
) -> None:
    """Calculate and plot free energy difference between two states.
    This function is used to calculate the convergence of the free energy difference
    between two states as a function of time as well as the overall free energy
    surface.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of collective variable.
    cv : str
        Collective variable to plot.
    cv_grid : np.ndarray
        Grid for free energy surface.
    bandwidth : float
        Bandwidth for kernel density estimation.
    eqbm_percent : float
        Percent of simulation to use for equilibration.
    lower_well : tuple[float, float]
        Lower well of the free energy surface.
    upper_well : tuple[float, float]
        Upper well of the free energy surface.
    n_timestep_plot : int
        Number of timesteps to plot.
    method : str
        Method used for sampling in the simulation.
    """
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # calculate free energy difference between the two states as a function of time
    percentage_grid = np.linspace(eqbm_percent + 0.01, 1, n_timestep_plot)
    t_max = df["time"].max()
    t_grid = np.array([t_max * perc for perc in percentage_grid])
    pmfs = np.zeros((len(percentage_grid), len(cv_grid)))
    pmf_diffs = np.zeros_like(t_grid)

    for i, p in enumerate(percentage_grid):
        cv_grid, pmfs[i] = fes_1d(
            x=df[cv],
            cv_grid=cv_grid,
            bandwidth=bandwidth,
            weights=df["weight"],
            eqbm_percent=eqbm_percent,
            final_percent=p,
            plateau_domain=CV_PLATEAU,
        )
        pmf_diffs[i] = diff_fes_1d(
            pmf=pmfs[i],
            cv_grid=cv_grid,
            lower_well=lower_well,
            upper_well=upper_well,
        )

    # save free energy profiles to csv
    header = ["final_time"] + cv_grid.tolist()
    dat = np.hstack((t_grid.reshape(-1, 1), pmfs))
    df = pd.DataFrame(dat, columns=header)
    df.to_csv(output_dir / f"{method}_pmf_convergence_{cv}.csv", index=False)

    # save free energy difference to csv
    df = pd.DataFrame(
        {
            "time": t_grid,
            "percentage": percentage_grid,
            "free_energy_diff": pmf_diffs,
        }
    )
    df.to_csv(output_dir / f"{method}_pmf_diff_{cv}.csv", index=False)

    # plot free energy difference between the two states as a function of time
    fig = plt_pmf_diff_conv(
        times=t_grid,
        differences=pmf_diffs,
        lower_well=lower_well,
        upper_well=upper_well,
        tag=f"{method}_cv_{cv}",
    )
    close_fig(fig)

    # render movie of free energy surface convergence
    mov_pmf_conv(
        times=t_grid,
        cv_grid=cv_grid,
        pmfs=pmfs,
        tag=f"{method}_cv_{cv}",
        xrange=(0, max(CV_GRID)),
        yrange=PMF_YRANGE,
    )
    close_fig(fig)


def all_analysis(method: str, pipeline: DataPipeline) -> None:
    """Perform all analysis for a given method.

    Parameters
    ----------
    method : str
        Method used for sampling in the simulation.
    pipeline : DataPipeline
        Data pipeline object.
    """
    # set-up
    set_style()

    # load data
    os.chdir(CWDIR)
    output_dir, colvar = load_data(pipeline, method, OUTPUT_SDIR)
    os.chdir(output_dir)
    pipeline.save_plumed_colvar(method, directory="data")

    # figures
    dynamics(
        df=colvar,
        cv=CV,
        method=method,
    )
    free_energy_surface(
        df=colvar,
        cv=CV,
        cv_grid=CV_GRID,
        bandwidth=BANDWIDTH,
        eqbm_percent=EQBM_PERCENT,
        method=method,
    )
    free_energy_difference(
        df=colvar,
        cv=CV,
        cv_grid=CV_GRID,
        bandwidth=BANDWIDTH,
        eqbm_percent=EQBM_PERCENT,
        lower_well=CV_LOWER_WELL,
        upper_well=CV_UPPER_WELL,
        n_timestep_plot=N_TIMESTEP_RENDER,
        method=method,
        output_dir=output_dir,
    )


# ########################################################################### #
# Main script                                                                 #
# ########################################################################### #

# if name main
if __name__ == "__main__":
    # command line input
    parser = argparse.ArgumentParser(description="Analyze COLVAR data")
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        help="Subdirectory for the input data",
        default=SIMULATION_SDIR,
    )
    args = parser.parse_args()

    # find subdirectory for input data
    dir_list = [d for d in (INPUT_BASE_DIR / args.dir).iterdir() if d.is_dir()]
    # remove directories that have pattern "1-energy-minimization", "2-equilibration", "3-*"
    patterns = ["1-energy-minimization", "2-equilibration", "3-"]
    dir_list = [d for d in dir_list if not any(p in str(d) for p in patterns)]
    if len(dir_list) == 0:
        dir_list = [INPUT_BASE_DIR / args.dir]
    if VERBOSE:
        print(f"Found {len(dir_list)} subdirectories for input data")

    for i, d in tqdm(enumerate(dir_list), desc="Directories", total=len(dir_list)):
        if VERBOSE:
            print(f"{i}: {d}")

        try:
            # create data pipeline
            pipeline = DataPipeline(
                data_path_base=d,
                temperature=TEMPERATURE_K,
                verbose=VERBOSE,
            )

            # perform analysis in parallel using joblib with tqdm progress bar
            methods = pipeline.sampling_methods
            Parallel(n_jobs=N_JOBS, verbose=10)(
                delayed(all_analysis)(method, pipeline) for method in methods
            )

            # write complete file to directory
            with open(
                CWDIR / OUTPUT_SDIR / "analysis_complete.txt", "w+", encoding="utf-8"
            ) as f:
                f.write(
                    f"Colvar analysis for {d} complete at {datetime.datetime.now()}.\n"
                )

        except Exception as e:
            print(f"Error in directory {d}: {e}")
            with open(
                CWDIR / OUTPUT_SDIR / "analysis_error.txt", "w+", encoding="utf-8"
            ) as f:
                f.write(
                    f"Error in colvar analysis for {d} at {datetime.datetime.now()}.\n"
                )
                f.write(f"{e}\n")
                # raise e

    # clean-up
    print("Script complete!")
