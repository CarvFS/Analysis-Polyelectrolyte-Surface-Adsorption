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
from colvar.atompair import SurvivalProbability  # noqa: E402
from colvar.lineardensity import LinearDensity  # noqa: E402
from colvar.rdf import InterRDF  # noqa: E402
from render.util import set_style  # noqa: E402
from utils.logs import setup_logging  # noqa: E402
from parameters.globals import *  # noqa: E402


# #############################################################################
# Functions
# #############################################################################


def wrapper_rdf(
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
    | - RDF(Ca_ion, Ca_ion)
    | - RDF(O_water, Ca_ion)
    | - RDF(O_water, Carbonate Carbon atoms)
    """
    # set output path and information for analysis section
    label_groups, label_references, updating, exclusions = [], [], [], []
    output_path = Path("mdanalysis_rdf/data")

    # {Ca_ion, Ca_ion}
    label_references.append(sel_dict["Ca_ion"])
    label_groups.append(sel_dict["Ca_ion"])
    updating.append((False, False))
    exclusions.append((1, 1))

    # {O_water, Ca_ion}
    label_references.append(sel_dict["O_water"])
    label_groups.append(sel_dict["Ca_ion"])
    updating.append((False, False))
    exclusions.append(None)

    # {O_water, Carbonate C}
    label_references.append(sel_dict["O_water"])
    label_groups.append(sel_dict["C_crb_ion"])
    updating.append((False, False))
    exclusions.append(None)

    for group, reference, update, exclude in tqdm(
        zip(label_groups, label_references, updating, exclusions),
        total=len(label_groups),
        desc="RDF",
        dynamic_ncols=True,
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


def wrapper_lineardensity(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:
    """
    Wrapper function for linear density calculation.

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
    | - LinearDensity of Ca_ion
    | - LinearDensity of Na_ion
    | - LinearDensity of Cl_ion
    | - LinearDensity of Carbonate Carbon atoms
    | - LinearDensity of non-solvent atoms
    | - LinearDensity of water oxygen atoms
    | - LinearDensity of water hydrogen atoms
    """
    # set output path and information for analysis section
    label_groups, groupings = [], []
    output_path = Path("mdanalysis_lineardensity/data")
    binsize_ang = 0.2

    # Polyelectrolyte monomers
    label_groups.append(sel_dict["polyelectrolyte"])
    groupings.append("residues")

    # Na_ion
    label_groups.append(sel_dict["Na_ion"])
    groupings.append("atoms")

    # Ca_ion
    label_groups.append(sel_dict["Ca_ion"])
    groupings.append("atoms")

    # Cl_ion
    label_groups.append(sel_dict["Cl_ion"])
    groupings.append("atoms")

    # Carbonate C
    label_groups.append(sel_dict["C_crb_ion"])
    groupings.append("atoms")

    # not solvent
    label_groups.append(sel_dict["not_sol"])
    groupings.append("atoms")

    # O_water
    label_groups.append(sel_dict["O_water"])
    groupings.append("atoms")

    # H_water
    label_groups.append(sel_dict["H_water"])
    groupings.append("atoms")

    for group, grouping in tqdm(
        zip(label_groups, groupings),
        total=len(label_groups),
        desc="LinearDensity",
        dynamic_ncols=True,
    ):
        log.info(f"Collective variable: LinearDensity({group})")
        label = f"{group.replace(' ', '_')}"
        select = uni.select_atoms(group)
        file_gr = f"lineardensity_{label}.parquet"
        output_np = output_path / file_gr

        if output_np.exists() and not RELOAD_DATA:
            log.debug("Skipping calculation")
        elif len(select) == 0:
            log.warning(f"No atoms found for group {group}")
        else:
            mda_ld = LinearDensity(
                select=select,
                grouping=grouping,
                binsize=binsize_ang,
                label=label,
                df_weights=df_weights if df_weights is not None else None,
                verbose=VERBOSE,
            )
            t_start = time.time()
            mda_ld.run(
                start=START,
                stop=STOP,
                step=STEP,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            t_end = time.time()
            log.debug(f"LD with {N_JOBS} threads took {(t_end - t_start)/60:.2f} min")
            log.debug(
                f"[frames, atoms, grouping] = [{mda_ld.n_frames}, "
                + f"{select.n_atoms}, {grouping}]"
            )
            mda_ld.save()
            mda_ld.figures(ext=FIG_EXT)
            mda_ld = None
            plt.close("all")


def wrapper_survivalprobability(
    uni: mda.Universe,
    sel_dict: dict,
) -> None:
    """
    Wrapper function for survival probability calculation.

    Parameters
    ----------
    uni : mda.Universe
        Universe object to analyze
    sel_dict : dict
        Dictionary of selection strings

    Returns
    -------
    None

    Notes
    -----
    This function is designed to be called from the universe_analysis function.

    The following collective variables are calculated:
    | - SurvivalProbability of O_water around Ca_ion
    | - SurvivalProbability of O_water around Carbonate Carbon atoms
    """
    # set output path and information for analysis section
    label_groups, label_references, r_cuts_ang, window_steps, tau_maxs = (
        [],
        [],
        [],
        [],
        [],
    )
    output_path = Path("mdanalysis_survivalprobability/data")

    # {O_water, Ca_ion}
    label_groups.append(sel_dict["O_water"])
    label_references.append(sel_dict["Ca_ion"])
    r_cuts_ang.append(3.2)
    window_steps.append(1)
    tau_maxs.append(100)

    # {O_water, Carbonate C}
    label_groups.append(sel_dict["O_water"])
    label_references.append(sel_dict["C_crb_ion"])
    r_cuts_ang.append(5.0)
    window_steps.append(1)
    tau_maxs.append(100)

    for group, reference, r_cut, window, tau_max in tqdm(
        zip(label_groups, label_references, r_cuts_ang, window_steps, tau_maxs),
        total=len(label_groups),
        desc="SurvivalProbability",
        dynamic_ncols=True,
    ):
        log.info(f"Collective variable: SurvivalProbability({group}, {reference})")
        label = f"{group.replace(' ', '_')}-{reference.replace(' ', '_')}-r{r_cut:.1f}"
        select_str = f"(around {r_cut} ({reference})) and {group}"
        log.debug(f"Selection string: {select_str}")
        select = uni.select_atoms(select_str, updating=True)
        file_gr = f"sp_{label}.parquet"
        output_np = output_path / file_gr

        if output_np.exists() and not RELOAD_DATA:
            log.debug("Skipping calculation")
        else:
            mda_sp = SurvivalProbability(
                select=select,
                n_pairs=uni.select_atoms(group).n_atoms,
                label=label,
                tau_max=tau_max,
                window_step=window,
                verbose=VERBOSE,
            )
            t_start = time.time()
            mda_sp.run(
                start=START,
                stop=STOP,
                step=STEP,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            t_end = time.time()
            log.debug(f"SP with {N_JOBS} threads took {(t_end - t_start)/60:.2f} min")
            log.debug(
                f"[frames, pairs] = [{mda_sp.n_frames}, {uni.select_atoms(group).n_atoms}]"
            )
            mda_sp.save()
            mda_sp.figures(ext=FIG_EXT)
            mda_sp = None
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
    t_start_uni = time.time()
    wrapper_rdf(uni, df_weights, sel_dict)
    wrapper_lineardensity(uni, df_weights, sel_dict)
    wrapper_survivalprobability(uni, sel_dict)
    t_end_uni = time.time()
    log.debug(f"Analysis took {(t_end_uni - t_start_uni)/60:.2f} min")


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

    # I/O and parameters
    dir_out_base = Path(f"{os.getcwd()}/data")
    data_dir = Path(f"{args.dir}")
    set_style()
    with open("parameters/selections.json", "r") as f:
        sel = json.load(f)

    # #########################################################################
    # Data processing
    # #########################################################################
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

        # calculate number of frames to analyze and frames per block
        last_frame = (
            STOP if STOP < len(universe.trajectory) else len(universe.trajectory)
        )
        n_frames_analyze = (last_frame - START) // STEP
        n_frames_block = n_frames_analyze // N_BLOCKS
        log.info(f"Frames to analyze: {n_frames_analyze}")
        log.info(f"Frames per block: {n_frames_block}")

        # print final simulation time
        log.info(
            f"Final simulation time: {universe.trajectory.n_frames * universe.trajectory.dt / 1e3:.2f} ns"
        )
        log.info(
            f"Final analysis time: {last_frame * universe.trajectory.dt / 1e3:.2f} ns"
        )

        # perform analysis
        universe_analysis(universe, df_plumed, sel)
