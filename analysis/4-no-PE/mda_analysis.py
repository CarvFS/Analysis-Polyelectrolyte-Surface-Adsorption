"""
| Author: Alec Glisman (GitHub: @alec-glisman)
| Date: 2021-08-31
| Description: This script generates data for the surface binding analysis.

This script is designed to be run from the command line. The following
arguments are required:
| -d, --dir: Base directory for the input data
| -f, --fname: File name for the input data
| -t, --tag: Subdirectory tag for the output data

Multiple simulations can be analyzed by running this script multiple times
and they can be analyzed in parallel by running multiple instances of this
script at the same time.
"""


# #############################################################################
# Imports
# #############################################################################

# Standard library
import argparse
import json
import os
from pathlib import Path
import time
import sys

# External dependencies
import MDAnalysis as mda
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from data.pipeline import DataPipeline  # noqa: E402
from colvar.rdf import InterRDF  # noqa: E402
from render.util import set_style  # noqa: E402
from utils.logs import setup_logging  # noqa: E402


# #############################################################################
# Globals
# #############################################################################

# ANCHOR: Script variables and setup
# system information
TEMPERATURE_K: float = 300  # [K] System temperature
# File I/O
FIG_EXT: str = "png"  # Figure file extension
DEFAULT_PATH: Path = Path(
    "/nfs/zeal_nas/home_mount/aglisman/GitHub/Polyelectrolyte-Surface-Adsorption/data_archive/4_many_monomer_binding/4.2.0-calcite-104surface-9nm_surface-10nm_vertical-0chain-PAcr-0mer-0Crb-64Ca-0Na-128Cl-300K-1bar-NVT"
)
# MDAnalysis trajectory parameters
START: int = 0  # First frame to read
STOP: int = None  # Last frame to read
STEP: int = 1  # Step between frames to read
N_JOBS: int = 24  # Number of parallel jobs
N_BLOCKS: int = 5 * N_JOBS  # Number of blocks to split trajectory into
UNWRAP: bool = False  # Unwrap trajectory before analysis
# Data processing parameters
VERBOSE: bool = True  # Verbose output
RELOAD_DATA: bool = False  # if True, remake all data
REFRESH_OFFSETS: bool = False  # if True, remake all offsets on trajectory files


# #############################################################################
# Functions
# #############################################################################


def rdf_wrapper(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:
    """
    Wrapper function for RDF calculation.

    Parameters
    ----------
    uni : mda.Universe
        Universe object to analyze
    df_weights : pd.DataFrame
        Dataframe with column "weights" containing statistical weights for each frame
    sel_dict : dict
        Dictionary of selection strings

    Returns
    -------
    None

    Notes
    -----
    This function is designed to be called from the universe_analysis function.

    The following collective variables are calculated:
    | - RDF(O_water, Ca_ion)

    """

    # set output path and information for analysis section
    label_groups, label_references, updating, exclusions = [], [], [], []
    output_path = Path("mdanalysis_rdf/data")

    # {O_water, Ca_ion}
    label_references.append(sel_dict["O_water"])
    label_groups.append(sel_dict["Ca_ion"])
    updating.append((False, False))
    exclusions.append(None)

    for group, reference, update, exclude in tqdm(
        zip(label_groups, label_references, updating, exclusions),
        total=len(label_groups),
        desc="RDF",
    ):
        log.info(f"Collective variable: RDF({group}, {reference})")
        label = f"{group.replace(' ', '_')}-{reference.replace(' ', '_')}"

        # see if output file exists, and if so, load it
        file_gr = f"rdf_{label}.parquet"
        output_np = output_path / file_gr
        if output_np.exists() and not RELOAD_DATA:
            log.debug("Skipping calculation")
        elif len(uni.select_atoms(reference, updating=update[0])) == 0:
            log.warning(f"No reference atoms found for reference {reference}")
        elif len(uni.select_atoms(group, updating=update[0])) == 0:
            log.warning(f"No reference atoms found for group {group}")
        else:
            mda_rdf = InterRDF(
                uni.select_atoms(reference, updating=update[0]),
                uni.select_atoms(group, updating=update[1]),
                nbins=500,
                domain=(0, 50),
                exclusion_block=exclude,
                label=label,
                df_weights=df_weights if df_weights is not None else None,
                verbose=VERBOSE,
            )
            t_start = time.time()
            mda_rdf.run(
                start=START,
                stop=STOP,
                step=STEP,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            t_end = time.time()
            log.debug(f"RDF with {N_JOBS} threads took {(t_end - t_start)/60:.2f} min")
            log.debug(
                f"[frames, A atoms, B atoms] = [{mda_rdf.n_frames}, "
                + f"{uni.select_atoms(reference, updating=update[0]).n_atoms}, "
                + f"{uni.select_atoms(group, updating=update[1]).n_atoms}]"
            )
            mda_rdf.save()
            mda_rdf.figures(ext=FIG_EXT)
            mda_rdf = None
            plt.close("all")


def universe_analysis(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:
    """
    Perform analysis on the universe object.

    Parameters
    ----------
    uni : mda.Universe
        Universe object to analyze
    df_weights : pd.DataFrame
        Dataframe with column "weights" containing statistical weights for each frame
    sel_dict : dict
        Dictionary of selection strings
    """
    rdf_wrapper(uni, df_weights, sel_dict)


# #############################################################################
# Script
# #############################################################################

if __name__ == "__main__":
    # command line input
    parser = argparse.ArgumentParser(description="Analyze MD data")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="Base directory for the input data",
        default=DEFAULT_PATH,
    )
    args = parser.parse_args()

    # #########################################################################
    # Data processing
    # #########################################################################
    dir_out_base = Path(f"{os.getcwd()}/data")
    data_dir = Path(f"{args.dir}")
    set_style()
    with open("selections.json", "r") as f:
        sel = json.load(f)

    # load data pipeline
    pipeline = DataPipeline(
        data_path_base=data_dir, temperature=TEMPERATURE_K, verbose=VERBOSE
    )
    if VERBOSE:
        print(f"Found {len(pipeline.sampling_methods)} sampling methods")

    # iterate over all simulation methods
    for i, method in tqdm(
        enumerate(pipeline.sampling_methods),
        total=len(pipeline.sampling_methods),
        desc="Sampling Methods",
    ):
        # create output directory
        dir_out = dir_out_base / f"{pipeline.tag}/{method}"
        dir_out.mkdir(parents=True, exist_ok=True)
        os.chdir(dir_out)
        log = setup_logging(log_file="mda_data_gen.log", verbose=VERBOSE, stream=True)
        pipeline._init_log(log_file="data_pipeline.log")

        # load data for method
        df_plumed = pipeline.load_plumed_colvar(method)
        pipeline.save_plumed_colvar(method, directory=dir_out / "plumed")
        universe = pipeline.load_universe(method)

        # perform analysis
        universe_analysis(universe, df_plumed, sel)
