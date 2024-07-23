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
from functools import partial
import json
import os
from pathlib import Path
import pickle
import time
import sys

# External dependencies
import gromacs as gmx
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import WaterBridgeAnalysis
from MDAnalysis.analysis.base import AnalysisFromFunction
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from data.pipeline import DataPipeline  # noqa: E402
from colvar.angulardistribution import AngularDistribution  # noqa: E402
from colvar.atompair import AtomPair  # noqa: E402
from colvar.contacts import Contacts  # noqa: E402
from colvar.dipole import Dipole, DipoleField  # noqa: E402
from colvar.dihedrals import Dihedral  # noqa: E402
from colvar.lineardensity import LinearDensity  # noqa: E402
from colvar.polymerlengths import PolymerLengths  # noqa: E402
from colvar.rdf import InterRDF  # noqa: E402
from render.util import set_style  # noqa: E402
from utils.logs import setup_logging  # noqa: E402
from parameters.globals import (  # noqa: E402
    START,
    STOP,
    STEP,
    N_JOBS,
    N_BLOCKS,
    MODULE,
    SOLVENT,
    VERBOSE,
    RELOAD_DATA,
    REFRESH_OFFSETS,
    ALL_REPLICAS,
    CONCATENATED,
    TEMPERATURE_K,
    FIG_EXT,
    DEFAULT_PATH,
)


# #############################################################################
# Functions
# #############################################################################


def wrapper_waterbridge(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:

    def analysis_order(current, output, u, order=1):
        r"""This function defines how the type of water bridge should be
        specified.

        Parameters
        ----------
        current : list
            The current water bridge being analysed is a list of hydrogen bonds
            from selection 1 to selection 2.
        output : dict
            A dictionary which is modified in-place where the key is the type
            of the water bridge and the value is the weight of this type of
            water bridge.
        u : MDAnalysis.universe
            The current Universe for looking up atoms.
        """
        # decompose the first hydrogen bond.
        sele1_index, sele1_heavy_index, atom2, heavy_atom2, dist, angle = current[0]
        # decompose the last hydrogen bond.
        atom1, heavy_atom1, sele2_index, sele2_heavy_index, dist, angle = current[-1]
        # expand the atom index to the resname, resid, atom names
        sele1 = u.atoms[sele1_index]
        sele2 = u.atoms[sele2_index]
        (s1_resname, s1_resid, s1_name) = (sele1.resname, sele1.resid, sele1.name)
        (s2_resname, s2_resid, s2_name) = (sele2.resname, sele2.resid, sele2.name)
        # bridge order
        bridge_order = len(current) - 1

        # output the water bridge number
        key = (
            s1_resname, s1_resid, s1_name, s2_resname, s2_resid, s2_name, bridge_order,
        )
        if bridge_order == order:
            output[key] += 1

    # default selections for H-bonds
    donors_sel = ("OW", "OB2", "OA")
    acceptors_sel = ("OW", "OB1", "OB2", "OA", "OB", "OX1", "OX2", "OX3")

    # set output path and information for analysis section
    label_groups, label_references, kwargs = [], [], []
    output_path = Path("mdanalysis_waterbridge")

    # find topology file for universe to get crystal atoms
    tpr_file = Path(uni.filename)
    ndx_file = tpr_file.parents[1] / "index.ndx"
    ndx = gmx.fileformats.ndx.NDX()
    ndx.read(ndx_file)
    crystal = np.array(ndx["Crystal"]) - 1

    # {Polyelectrolyte, CaCO3 Crystal} 2nd order
    label_references.append(sel_dict["polyelectrolyte"])
    label_groups.append(f"index {' '.join(str(i) for i in crystal)}")
    kwargs.append(
        {
            "order": 2,
            "forcefield": "other",
            "donors": donors_sel,
            "acceptors": acceptors_sel,
            "update_selection": True,
        }
    )

    # {Polyelectrolyte, CaCO3 Crystal} 3rd order
    label_references.append(sel_dict["polyelectrolyte"])
    label_groups.append(f"index {' '.join(str(i) for i in crystal)}")
    kwargs.append(
        {
            "order": 3,
            "forcefield": "other",
            "donors": donors_sel,
            "acceptors": acceptors_sel,
            "update_selection": True,
        }
    )

    # {Polyelectrolyte, CaCO3 Crystal} 4th order
    label_references.append(sel_dict["polyelectrolyte"])
    label_groups.append(f"index {' '.join(str(i) for i in crystal)}")
    kwargs.append(
        {
            "order": 4,
            "forcefield": "other",
            "donors": donors_sel,
            "acceptors": acceptors_sel,
            "update_selection": True,
        }
    )

    for group, reference, kwarg in tqdm(
        zip(label_groups, label_references, kwargs),
        total=len(label_groups),
        desc="Bridges",
        dynamic_ncols=True,
    ):

        log.info(f"Collective variable: WaterBridge({kwarg['order']})")
        label = (
            f"{kwarg['order']}"
            + f"-{group.replace(' ', '_')}-{reference.replace(' ', '_')}"
        )

        # shorten label if it is too long
        if len(label) > 50:
            label = label[:50]

        output_pd = output_path / "data" / f"wb_{label}.parquet"

        if output_pd.exists() and not RELOAD_DATA:
            log.debug("Skipping calculation")
            continue
        else:
            log.debug("Number of atoms in group: " + str(len(uni.select_atoms(group))))
            log.debug(
                "Number of atoms in reference: " + str(len(uni.select_atoms(reference)))
            )

        Path(output_path / "data").mkdir(parents=True, exist_ok=True)

        # run the water bridge analysis
        wb = WaterBridgeAnalysis(uni, group, reference, **kwarg, verbose=VERBOSE)
        wb.run(
            start=START,
            stop=STOP,
            step=STEP,
        )

        # save the results
        results = wb.results
        with open(output_path / "data" / f"results_{label}.pkl", "wb") as f:
            pickle.dump(results, f)

        data = {}
        for i in range(kwarg["order"]):
            counts = wb.count_by_time(analysis_func=partial(analysis_order, order=i+1))
            if i == 0:
                data["time"] = [c[0] for c in counts]
            data[f"order_{i+1}"] = [c[1] for c in counts]

        df = pd.DataFrame(data)
        df = pd.merge(df, df_weights, on="time", how="inner")
        df.to_parquet(output_pd)

        # calculate the average number of each type of bridge weighted
        avg_counts = {}
        for i in range(kwarg["order"]):
            avg_counts[f"order_{i+1}"] = np.average(
                df[f"order_{i+1}"],
                weights=df["weight"],
            )
        log.info(f"Average counts: {avg_counts}")

        with open(output_path / "data" / f"avg_counts_{label}.json", "w") as f:
            json.dump(avg_counts, f)


def wrapper_saltbridge(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:

    def n_atoms(atomgroup) -> list:
        return [atomgroup.universe.trajectory.time, len(atomgroup)]

    cutoff_ang = 3.5

    # find topology file for universe to get crystal atoms
    tpr_file = Path(uni.filename)
    ndx_file = tpr_file.parents[1] / "index.ndx"
    ndx = gmx.fileformats.ndx.NDX()
    ndx.read(ndx_file)
    crystal = ndx["Crystal"]
    crystal = np.array(crystal) - 1
    sel_crystal = f"index {' '.join(str(i) for i in crystal)}"

    # check if output file exists
    output_path = Path("mdanalysis_saltbridge")
    Path(output_path / "data").mkdir(parents=True, exist_ok=True)
    output_file = output_path / "data" / "saltbridge.parquet"
    if output_file.exists() and not RELOAD_DATA:
        log.debug("Skipping calculation")
        return
    else:
        log.debug("Calculating salt bridge data")

    # select the polymer atoms
    sel_pe = sel_dict["polyelectrolyte"]

    # calculate the number of sodium bridges
    sel_na = sel_dict["Na_ion"]
    atomgroup_na = uni.select_atoms(
        f"{sel_na}"
        + f" and (around {cutoff_ang} {sel_pe})"
        + f" and (around {cutoff_ang} {sel_crystal})",
        updating=True,
    )
    analysis_na = AnalysisFromFunction(n_atoms, uni.trajectory, atomgroup_na)
    analysis_na.run(start=START, stop=STOP, step=STEP, verbose=VERBOSE)
    df_na = pd.DataFrame(analysis_na.results.timeseries, columns=["time", "n_na"])

    # calculate the number of calcium bridges
    sel_ca = f"({sel_dict["Ca_ion"]}) and not ({sel_crystal})"
    atomgroup_ca = uni.select_atoms(
        f"{sel_ca}"
        + f" and (around {cutoff_ang} {sel_pe})"
        + f" and (around {cutoff_ang} {sel_crystal})",
        updating=True,
    )
    analysis_ca = AnalysisFromFunction(n_atoms, uni.trajectory, atomgroup_ca)
    analysis_ca.run(start=START, stop=STOP, step=STEP, verbose=VERBOSE)
    df_ca = pd.DataFrame(analysis_ca.results.timeseries, columns=["time", "n_ca"])

    # merge with the weights
    df = pd.merge(df_na, df_ca, on="time", how="inner")
    df = pd.merge(df, df_weights, on="time", how="inner")

    # save the data
    df.to_parquet(output_file)

    # calculate the average number of each type of bridge weighted
    avg_counts = np.zeros(2)
    for i, counts in enumerate([df["n_na"], df["n_ca"]]):
        avg_counts[i] = np.average(counts, weights=df["weight"])

    with open(output_path / "data" / "avg_counts_saltbridge.json", "w") as f:
        json.dump({"n_na": avg_counts[0], "n_ca": avg_counts[1]}, f)


def wrapper_pebind(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:

    def n_residues(atomgroup) -> list:
        resids = atomgroup.residues.resids
        return [atomgroup.universe.trajectory.time, len(set(resids))]

    cutoff_ang = 3.5

    # check if output file exists
    output_path = Path("mdanalysis_pebind")
    Path(output_path / "data").mkdir(parents=True, exist_ok=True)
    output_file = output_path / "data" / "pebind.parquet"
    if output_file.exists() and not RELOAD_DATA:
        log.debug("Skipping calculation")
        return
    else:
        log.debug("Calculating polyelectrolyte binding data")

    # find topology file for universe to get crystal atoms
    tpr_file = Path(uni.filename)
    ndx_file = tpr_file.parents[1] / "index.ndx"
    ndx = gmx.fileformats.ndx.NDX()
    ndx.read(ndx_file)
    crystal = ndx["Crystal"]
    crystal = np.array(crystal) - 1
    sel_crystal = f"index {' '.join(str(i) for i in crystal)}"

    # calculate the number of direct binding residues
    sel_pe = sel_dict["polyelectrolyte"]
    atomgroup_pe = uni.select_atoms(
        f"{sel_pe}" + f" and (around {cutoff_ang} {sel_crystal})",
        updating=True,
    )
    analysis_pe = AnalysisFromFunction(n_residues, uni.trajectory, atomgroup_pe)
    analysis_pe.run(start=START, stop=STOP, step=STEP, verbose=VERBOSE)
    df_pe = pd.DataFrame(analysis_pe.results.timeseries, columns=["time", "n_pe"])

    # merge with the weights
    df = pd.merge(df_pe, df_weights, on="time", how="inner")

    # save the data
    df.to_parquet(output_file)

    # calculate the average number of each type of bridge weighted
    avg_counts = np.zeros(1)
    for i, counts in enumerate([df["n_pe"]]):
        avg_counts[i] = np.average(counts, weights=df["weight"])

    with open(output_path / "data" / "avg_counts_pebind.json", "w") as f:
        json.dump({"n_bind": avg_counts[0]}, f)


def wrapper_polymerlength(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:
    """
    Wrapper function for polymer length calculation.

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
    | - PolymerLength of polyelectrolyte
    """
    # set output path and information for analysis section
    label_groups = []
    output_path = Path("mdanalysis_polymer_lengths")

    # Polyelectrolyte
    label_groups.append(sel_dict["polyelectrolyte"])

    for group in tqdm(
        label_groups,
        total=len(label_groups),
        desc="PolymerLength",
        dynamic_ncols=True,
    ):
        log.info(f"Collective variable: PolymerLength({group})")
        label = f"{group.replace(' ', '_')}"
        select = uni.select_atoms(group)

        file_gr = f"pl_{label}.parquet"
        output_np = output_path / "data" / file_gr
        if output_np.exists() and not RELOAD_DATA:
            log.debug("Skipping calculation")
        elif len(select) == 0:
            log.warning(f"No atoms found for group {group}")
        else:
            mda_pl = PolymerLengths(
                atomgroup=select,
                label=label,
                verbose=VERBOSE,
            )
            t_start = time.time()
            mda_pl.run(
                start=START,
                stop=STOP,
                step=STEP,
                module=MODULE,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            t_end = time.time()
            log.debug(f"PL with {N_JOBS} threads took {(t_end - t_start)/60:.2f} min")

            mda_pl.merge_external_data(df_weights)
            mda_pl.save()
            mda_pl.figures(ext=FIG_EXT)
            mda_pl = None
            plt.close("all")


def wrapper_dihedrals(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:
    """
    Wrapper function for dihedral angle calculation.

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
    | - Dihedrals of polyelectrolyte
    """
    # set output path and information for analysis section
    label_groups = []
    output_path = Path("mdanalysis_dihedrals")

    # Polyelectrolyte
    label_groups.append(sel_dict["polyelectrolyte"])

    for group in tqdm(
        label_groups,
        total=len(label_groups),
        desc="Dihedrals",
        dynamic_ncols=True,
    ):
        log.info(f"Collective variable: Dihedrals({group})")
        label = f"{group.replace(' ', '_')}"
        select = uni.select_atoms(group)

        file_gr = f"dihedral_{label}.parquet"
        output_np = output_path / "data" / file_gr
        if output_np.exists() and not RELOAD_DATA:
            log.debug("Skipping calculation")
        elif len(select) == 0:
            log.warning(f"No atoms found for group {group}")
        else:
            mda_dh = Dihedral(
                atomgroup=select,
                label=label,
                verbose=VERBOSE,
            )
            t_start = time.time()
            mda_dh.run(
                start=START,
                stop=STOP,
                step=STEP,
                module=MODULE,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            t_end = time.time()
            log.debug(
                f"Dihedrals with {N_JOBS} threads took {(t_end - t_start)/60:.2f} min"
            )

            mda_dh.merge_external_data(df_weights)
            mda_dh.save()
            # mda_dh.figures(ext=FIG_EXT)
            mda_dh = None
            plt.close("all")


def wrapper_contacts(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:
    """
    Wrapper function for contact calculation.

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
    | - Contacts of polyelectrolyte
    """
    methods = ["hard_cut", "6_12"]

    # set output path and information for analysis section
    label_groups, label_references, updating = [], [], []
    kwargs_hard_cut, kwargs_rational = [], []
    output_path = Path("mdanalysis_contacts")

    # {Ca, O_carboxylate}
    label_references.append(sel_dict["O_carb"])
    label_groups.append(sel_dict["Ca_ion"])
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 3.5})
    kwargs_rational.append({"radius": 0.98, "d0": 2.80})

    # {Na, O_carboxylate}
    label_references.append(sel_dict["O_carb"])
    label_groups.append(sel_dict["Na_ion"])
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 3.5})
    kwargs_rational.append({"radius": 0.98, "d0": 2.80})

    for group, reference, update, kw_hc, kw_r in tqdm(
        zip(label_groups, label_references, updating, kwargs_hard_cut, kwargs_rational),
        total=len(label_groups),
        desc="Contacts",
        dynamic_ncols=True,
    ):
        for method in methods:
            log.info(
                f"Collective variable: Contacts({group}, {reference}) with {method}"
            )
            label = f"{group.replace(' ', '_')}-{reference.replace(' ', '_')}-{method}"
            file_gr = f"contact_{label}.parquet"
            output_np = output_path / "data" / file_gr
            if output_np.exists() and not RELOAD_DATA:
                log.debug("Skipping calculation")
                continue

            coord = uni.select_atoms(reference, updating=update[0])
            ref = uni.select_atoms(group, updating=update[1])
            if len(coord) == 0:
                log.warning(f"No reference atoms found for reference {reference}")
                continue
            elif len(ref) == 0:
                log.warning(f"No reference atoms found for group {group}")
                continue

            if method == "hard_cut":
                mda_cn = Contacts(
                    uni,
                    (coord, ref),
                    method=method,
                    radius=kw_hc["radius"],
                    label=label,
                    verbose=VERBOSE,
                )
            elif method == "6_12":
                mda_cn = Contacts(
                    uni,
                    (coord, ref),
                    method=method,
                    radius=kw_r["radius"],
                    label=label,
                    verbose=VERBOSE,
                    kwargs={"d0": kw_r["d0"]},
                )
            else:
                raise ValueError(f"Unknown method {method}")

            t_start = time.time()
            mda_cn.run(
                start=START,
                stop=STOP,
                step=STEP,
                module=MODULE,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            t_end = time.time()
            log.debug(
                f"Contacts with {N_JOBS} threads took {(t_end - t_start)/60:.2f} min"
            )

            mda_cn.merge_external_data(df_weights)
            mda_cn.save()
            mda_cn.figures(ext=FIG_EXT)
            mda_cn = None
            plt.close("all")


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
    | - RDF(Cl_ion, Ca_ion)
    | - RDF(Ca_ion, O_carbonate)
    | - RDF(C_alpha, C_alpha)
    | - RDF(O_water, Ca_ion)
    | - RDF(O_water, Carbonate Carbon atoms)
    | - RDF(O_water, O_carb)
    """
    # set output path and information for analysis section
    label_groups, label_references, updating, exclusions = [], [], [], []
    output_path = Path("mdanalysis_rdf/data")

    # {Cl_ion, Ca_ion}
    label_groups.append(sel_dict["Cl_ion"])
    label_references.append(sel_dict["Ca_ion"])
    updating.append((False, False))
    exclusions.append(None)

    # {Ca_ion, O_carbonate}
    label_groups.append(sel_dict["Ca_ion"])
    label_references.append(sel_dict["O_carb"])
    updating.append((False, False))
    exclusions.append(None)

    # {C_carbonate, O_carboxylate}
    label_groups.append(sel_dict["C_crb_ion"])
    label_references.append(sel_dict["O_carb"])
    updating.append((False, False))
    exclusions.append(None)

    # {O_carbonate, O_carboxylate}
    label_groups.append(sel_dict["O_crb_ion"])
    label_references.append(sel_dict["O_carb"])
    updating.append((False, False))
    exclusions.append(None)

    # {Ca_ion, O_carboxylate}
    label_groups.append(sel_dict["Ca_ion"])
    label_references.append(sel_dict["O_carb"])
    updating.append((False, False))

    # {C_alpha, C_alpha}
    label_groups.append(sel_dict["C_alpha"])
    label_references.append(sel_dict["C_alpha"])
    updating.append((False, False))
    exclusions.append((1, 1))

    # {C_carbonate, Polyelectrolyte}
    label_groups.append(sel_dict["C_crb_ion"])
    label_references.append(sel_dict["polyelectrolyte"])
    updating.append((False, False))
    exclusions.append(None)

    if SOLVENT:
        # {O_water, O_carboxylate}
        label_groups.append(sel_dict["O_water"])
        label_references.append(sel_dict["O_carb"])
        updating.append((False, False))
        exclusions.append(None)

        # {O_water, PE}
        label_groups.append(sel_dict["O_water"])
        label_references.append(sel_dict["polyelectrolyte"])
        updating.append((False, False))
        exclusions.append(None)

        # {Water, PE}
        label_groups.append(sel_dict["sol"])
        label_references.append(sel_dict["polyelectrolyte"])
        updating.append((False, False))
        exclusions.append(None)

        # {O_water, Ca_ion}
        label_groups.append(sel_dict["O_water"])
        label_references.append(sel_dict["Ca_ion"])
        updating.append((False, False))
        exclusions.append(None)

        # {O_water, Carbonate C}
        label_groups.append(sel_dict["O_water"])
        label_references.append(sel_dict["C_crb_ion"])
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
                module=MODULE,
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
    | - LinearDensity of polyelectrolyte monomers
    | - LinearDensity of polyelectrolyte chains
    | - LinearDensity of polyelectrolyte C_alpha atoms
    | - LinearDensity of polyelectrolyte C_carb atoms
    | - LinearDensity of polyelectrolyte O_carb atoms
    | - LinearDensity of Na_ion
    | - LinearDensity of Cl_ion
    | - LinearDensity of Ca_ion
    | - LinearDensity of Carbonate Carbon atoms
    | - LinearDensity of Carbonate Oxygen atoms
    | - LinearDensity of non-solvent atoms
    | - LinearDensity of water oxygen atoms
    | - LinearDensity of water hydrogen atoms
    | - LinearDensity of all atoms
    """
    # set output path and information for analysis section
    output_path = Path("mdanalysis_lineardensity")
    label_groups, groupings, bins, dims, props = [], [], [], [], []

    bins_tight = int(10e3)
    bins_normal = int(1e3)

    dims_all = ["x", "y", "z"]
    dims_x = ["x"]
    dims_y = ["y"]
    dims_z = ["z"]

    props_all = ["number", "mass", "charge"]
    props_charge = ["charge"]

    # Polyelectrolyte monomers
    label_groups.append(sel_dict["polyelectrolyte"])
    groupings.append("residues")
    dims.append(dims_all)
    bins.append(bins_normal)
    props.append(props_all)
    # Polyelectrolyte chain
    label_groups.append(sel_dict["polyelectrolyte"])
    groupings.append("fragments")
    dims.append(dims_all)
    bins.append(bins_normal)
    props.append(props_all)
    # Polyelectrolyte C_alpha atoms
    label_groups.append(sel_dict["C_alpha"])
    groupings.append("atoms")
    bins.append(bins_normal)
    dims.append(dims_all)
    props.append(props_all)
    # Polyelectrolyte C_carb atoms
    label_groups.append(sel_dict["C_carb"])
    groupings.append("atoms")
    bins.append(bins_normal)
    dims.append(dims_all)
    props.append(props_all)
    # Polyelectrolyte O_carb atoms
    label_groups.append(sel_dict["O_carb"])
    groupings.append("atoms")
    bins.append(bins_normal)
    dims.append(dims_all)
    props.append(props_all)

    # Na_ion
    label_groups.append(sel_dict["Na_ion"])
    groupings.append("atoms")
    bins.append(bins_normal)
    dims.append(dims_all)
    props.append(props_all)
    # Cl_ion
    label_groups.append(sel_dict["Cl_ion"])
    groupings.append("atoms")
    bins.append(bins_normal)
    dims.append(dims_all)
    props.append(props_all)
    # Ca_ion
    label_groups.append(sel_dict["Ca_ion"])
    groupings.append("atoms")
    bins.append(bins_normal)
    dims.append(dims_all)
    props.append(props_all)
    # Carbonate C
    label_groups.append(sel_dict["C_crb_ion"])
    groupings.append("atoms")
    bins.append(bins_normal)
    dims.append(dims_all)
    props.append(props_all)
    # not solvent
    label_groups.append(sel_dict["not_sol"])
    groupings.append("atoms")
    bins.append(bins_normal)
    dims.append(dims_all)
    props.append(props_all)

    if SOLVENT:
        # solvent
        label_groups.append(sel_dict["sol"])
        groupings.append("atoms")
        bins.append(bins_tight)
        dims.append(dims_x)
        props.append(props_charge)

        label_groups.append(sel_dict["sol"])
        groupings.append("atoms")
        bins.append(bins_tight)
        dims.append(dims_y)
        props.append(props_charge)

        label_groups.append(sel_dict["sol"])
        groupings.append("atoms")
        bins.append(bins_tight)
        dims.append(dims_z)
        props.append(props_charge)

        # O_water
        label_groups.append(sel_dict["O_water"])
        groupings.append("atoms")
        bins.append(bins_normal)
        dims.append(dims_all)
        props.append(props_all)

        # all atoms
        label_groups.append("all")
        groupings.append("atoms")
        bins.append(bins_tight)
        dims.append(dims_x)
        props.append(props_charge)

        label_groups.append("all")
        groupings.append("atoms")
        bins.append(bins_tight)
        dims.append(dims_y)
        props.append(props_charge)

        label_groups.append("all")
        groupings.append("atoms")
        bins.append(bins_tight)
        dims.append(dims_z)
        props.append(props_charge)

    for group, grouping, bin, dim in tqdm(
        zip(label_groups, groupings, bins, dims),
        total=len(label_groups),
        desc="LinearDensity",
        dynamic_ncols=True,
    ):
        log.info(f"Collective variable: LinearDensity({group})")
        label = f"{group.replace(' ', '_')}_{grouping}"
        select = uni.select_atoms(group)

        file_gr = f"lineardensity_{dim[0]}_{label}.npz"
        output_np = output_path / "data" / file_gr

        if output_np.exists() and not RELOAD_DATA:
            log.debug("Skipping calculation")
        elif len(select) == 0:
            log.warning(f"No atoms found for group {group}")
        else:
            mda_ld = LinearDensity(
                select=select,
                grouping=grouping,
                nbins=bin,
                label=label,
                df_weights=df_weights if df_weights is not None else None,
                dims=dim,
                verbose=VERBOSE,
            )
            t_start = time.time()
            mda_ld.run(
                start=START,
                stop=STOP,
                step=STEP,
                module=MODULE,
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

            t_start = time.time()
            mda_ld.save()
            t_end = time.time()
            log.debug(f"Saving took {(t_end - t_start)/60:.2f} min")

            t_start = time.time()
            mda_ld.figures(ext=FIG_EXT)
            t_end = time.time()
            log.debug(f"Figures took {(t_end - t_start)/60:.2f} min")
            mda_ld = None
            plt.close("all")


def wrapper_dipole_field(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:
    """
    Wrapper function for dipole field calculation.

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
    | - TODO
    """

    output_path = Path("mdanalysis_dipolefield")
    label_groups, axes, compounds, updating = [], [], [], []

    # {Polyelectrolyte}
    label_groups.append(sel_dict["polyelectrolyte"])
    compounds.append("molecules")
    updating.append(False)
    axes.append("z")

    # {Solvent}
    if SOLVENT:
        label_groups.append(sel_dict["sol"])
        compounds.append("molecules")
        updating.append(False)
        axes.append("z")

    # {Polyelectrolyte solvation shell}
    if SOLVENT:
        first_solvation_shell = (
            f"same resid as (around 3.1 {sel_dict['polyelectrolyte']})"
            + f" and ({sel_dict['sol']})"
        )
        second_solvation_shell = (
            f"same resid as (around 5.6 {first_solvation_shell})"
            + f" and ({sel_dict['sol']})"
            + f" and not ({first_solvation_shell})"
        )
        third_solvation_shell = (
            f"same resid as (around 8.1 {second_solvation_shell})"
            + f" and ({sel_dict['sol']})"
            + f" and not ({first_solvation_shell})"
            + f" and not ({second_solvation_shell})"
        )

        # first solvation shell
        label_groups.append(first_solvation_shell)
        compounds.append("molecules")
        updating.append(True)
        axes.append("z")

        # second solvation shell
        label_groups.append(second_solvation_shell)
        compounds.append("molecules")
        updating.append(True)
        axes.append("z")

        # third solvation shell
        label_groups.append(third_solvation_shell)
        compounds.append("molecules")
        updating.append(True)
        axes.append("z")

    for group, compound, axis, update in tqdm(
        zip(label_groups, compounds, axes, updating),
        total=len(label_groups),
        desc="DipoleField",
        dynamic_ncols=True,
    ):
        log.info(f"Collective variable: DipoleField({group})")
        label = f"{group.replace(' ', '_')}"
        select = uni.select_atoms(group, updating=update)

        tag = f"histo_dipolefield_{axis}_{label}"
        if len(tag) > 128:
            log.warning(f"Filename too long: {tag}")
            log.warning(f"Truncating to {tag[:128]}")
            tag = tag[:128]

        file_gr = f"{tag}.npz"
        output_np = output_path / "data" / file_gr

        if output_np.exists() and not RELOAD_DATA:
            log.debug("Skipping calculation")
            continue

        mda_df = DipoleField(
            atomgroup=select,
            compound=compound,
            axis=axis,
            label=label,
            df_weights=df_weights if df_weights is not None else None,
            verbose=VERBOSE,
        )
        t_start = time.time()
        mda_df.run(
            start=START,
            stop=STOP,
            step=STEP,
            module=MODULE,
            verbose=VERBOSE,
            n_jobs=N_JOBS,
            n_blocks=N_BLOCKS,
        )
        t_end = time.time()
        log.debug(f"DF with {N_JOBS} threads took {(t_end - t_start)/60:.2f} min")

        t_start = time.time()
        mda_df.save()
        t_end = time.time()
        log.debug(f"Saving took {(t_end - t_start)/60:.2f} min")

        t_start = time.time()
        mda_df.figures(ext=FIG_EXT)
        t_end = time.time()
        log.debug(f"Figures took {(t_end - t_start)/60:.2f} min")
        mda_df = None
        plt.close("all")


def wrapper_solvent_orientation(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:
    """
    Wrapper function for solvent orientation calculation.

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
    | - SolventOrientation of water molecules along the z-axis
    """
    if not SOLVENT:
        log.warning("Skipping solvent orientation calculation")
        return

    output_path = Path("mdanalysis_angulardistribution")
    min_dist = 25 - 13  # [Angstrom]
    max_dist = 36 - 13  # [Angstrom]
    bin_width = 0.2  # [Angstrom]
    bins = np.arange(min_dist, max_dist + bin_width, bin_width)
    nbins_angle = 100

    # all water
    label_groups = [sel_dict["sol"]]
    groupings = ["residues"]

    # iterate over all groups
    for group, grouping in zip(label_groups, groupings):
        log.info(f"Collective variable: SolventOrientation({group})")

        # iterate over all bins
        for dim_min, dim_max in zip(bins[:-1], bins[1:]):
            # select atoms
            label = f"{group.replace(' ', '_')}-{dim_min:.3f}_min-{dim_max:.3f}_max"
            selection = (
                f"same resid as ({group} and "
                + f"(prop z > {dim_min:.3f} and prop z <= {dim_max:.3f}))"
            )
            ag = uni.select_atoms(selection, updating=True)

            # check if output file exists or if no atoms are found
            file_gr = f"angulardistribution_z-{label}.npz"
            output_np = f"{output_path}/data/{file_gr}"
            if Path(output_np).exists() and not RELOAD_DATA:
                log.debug("Skipping calculation")
                continue
            elif len(uni.select_atoms(selection)) % 3 != 0:
                raise ValueError(
                    f"Number of atoms not divisible by 3 for selection {selection}"
                )
            elif len(uni.select_atoms(selection)) < 3 * 10 and group == sel_dict["sol"]:
                log.warning(f"Not enough atoms found for selection {selection}")
                continue
            else:
                log.debug(f"Selection string: {selection}")
                log.debug(f"Number of residues: {ag.n_residues}")

            # perform calculation
            mda_so = AngularDistribution(
                atomgroup=ag,
                grouping=grouping,
                label=label,
                nbins=nbins_angle,
                df_weights=df_weights if df_weights is not None else None,
            )
            t_start = time.time()
            mda_so.run(
                start=START,
                stop=STOP,
                step=STEP,
                module=MODULE,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            t_end = time.time()
            log.debug(f"SO took {(t_end - t_start)/60:.2f} min")

            t_start = time.time()
            mda_so.save()
            t_end = time.time()
            log.debug(f"Saving took {(t_end - t_start)/60:.2f} min")

            t_start = time.time()
            mda_so.figures(ext=FIG_EXT)
            t_end = time.time()
            log.debug(f"Figures took {(t_end - t_start)/60:.2f} min")
            mda_so = None
            plt.close("all")


def wrapper_dipole(
    uni: mda.Universe,
    df_weights: pd.DataFrame,
    sel_dict: dict,
) -> None:
    """
    Wrapper function for dipole moment calculation.

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
    | - DipoleMoment of water molecules
    | - DipoleMoment of CaCO3
    | - DipoleMoment of polyelectrolyte
    | - DipoleMoment of all atoms
    """
    groups = []
    output_path = Path("mdanalysis_dipole")

    # water
    groups.append(sel_dict["sol"])
    # caco3
    groups.append(sel_dict["CaCO3"])
    # polymer
    groups.append(sel_dict["polyelectrolyte"])
    # all atoms
    groups.append("all")

    for group in groups:
        log.info(f"Collective variable: DipoleMoment({group})")
        label = f"{group.replace(' ', '_')}"
        select = uni.select_atoms(group)

        # see if output file exists, and if so, load it
        file_gr = f"dipole_{label}.parquet"
        output_np = output_path / "data" / file_gr
        if output_np.exists() and not RELOAD_DATA:
            log.debug("Skipping calculation")
        elif len(select) == 0:
            log.warning(f"No atoms found for group {group}")
        else:
            mda_d = Dipole(
                atomgroup=select,
                label=label,
                verbose=VERBOSE,
                df_weights=df_weights if df_weights is not None else None,
            )
            t_start = time.time()
            mda_d.run(
                start=START,
                stop=STOP,
                step=STEP,
                module=MODULE,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            t_end = time.time()
            log.debug(
                f"Dipole with {N_JOBS} threads took {(t_end - t_start)/60:.2f} min"
            )
            mda_d.save()
            mda_d.figures(ext=FIG_EXT)
            mda_d = None
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

    if SOLVENT:
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
            mda_sp = AtomPair(
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
                module=MODULE,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            t_end = time.time()
            log.debug(f"SP with {N_JOBS} threads took {(t_end - t_start)/60:.2f} min")
            log.debug(
                f"[frames, pairs] = [{mda_sp.n_frames}, "
                + f"{uni.select_atoms(group).n_atoms}]"
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
    wrapper_polymerlength(uni, df_weights, sel_dict)
    wrapper_dihedrals(uni, df_weights, sel_dict)
    wrapper_contacts(uni, df_weights, sel_dict)
    wrapper_lineardensity(uni, df_weights, sel_dict)
    # wrapper_solvent_orientation(uni, df_weights, sel_dict)
    wrapper_pebind(uni, df_weights, sel_dict)
    wrapper_saltbridge(uni, df_weights, sel_dict)
    wrapper_waterbridge(uni, df_weights, sel_dict)
    wrapper_dipole(uni, df_weights, sel_dict)
    wrapper_dipole_field(uni, df_weights, sel_dict)
    wrapper_rdf(uni, df_weights, sel_dict)
    # wrapper_survivalprobability(uni, sel_dict)
    t_end_uni = time.time()
    log.debug(f"Analysis took {(t_end_uni - t_start_uni)/60:.2f} min")


# #############################################################################
# Script
# #############################################################################
if __name__ == "__main__":
    Path("logs").mkdir(parents=True, exist_ok=True)
    log = setup_logging(log_file="logs/mda_data_gen.log", verbose=VERBOSE, stream=True)

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
    mda_tag = f"START_{START}-STOP_{STOP}-STEP_{STEP}"
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
        data_path_base=data_dir,
        temperature=TEMPERATURE_K,
        concatenated=CONCATENATED,
        verbose=VERBOSE,
    )
    if VERBOSE:
        print(f"Found {len(pipeline.sampling_methods)} sampling methods")

    step_init = STEP
    start_init = START

    # iterate over all simulation methods
    for i, method in tqdm(
        enumerate(pipeline.sampling_methods),
        total=len(pipeline.sampling_methods),
        desc="Sampling Methods",
    ):
        if (method.split("_")[1] != "00") and (not ALL_REPLICAS):
            log.critical("Skipping non-base replica")
            continue

        # create output directory
        dir_out = dir_out_base / f"{pipeline.tag}/{method}"
        dir_out.mkdir(parents=True, exist_ok=True)
        os.chdir(dir_out)
        pipeline._init_log(log_file="data_pipeline.log")

        # load data for method
        df_plumed = pipeline.load_plumed_colvar(method)
        pipeline.save_plumed_colvar(method, directory=dir_out / "plumed")
        universe = pipeline.load_universe(method, refresh_offsets=REFRESH_OFFSETS)

        # non-base replicas are down-sampled by a factor of 10
        if ("replica" in method) and (method.split("_")[1] != "00"):
            log.critical(f"Reducing STEP by factor of 10 to {STEP//10}")
            STEP = step_init // 10
            log.critical(f"Reducing START by factor of 10 to {START//10}")
            START = start_init // 10
        else:
            STEP = step_init
            START = start_init

        # calculate number of frames to analyze and frames per block
        last_frame = (
            STOP if STOP < len(universe.trajectory) else len(universe.trajectory)
        )
        n_frames_analyze = (last_frame - START) // STEP
        n_frames_block = n_frames_analyze // N_BLOCKS
        log.info(f"Slicing frames: {START} to {last_frame} with step {STEP}")
        log.info(f"Frames to analyze: {n_frames_analyze}")
        log.info(f"Frames per block: {n_frames_block}")

        # print final simulation time
        tf = universe.trajectory.n_frames * universe.trajectory.dt / 1e3
        log.info(f"Final simulation time: {tf:.2f} ns")
        tf_anl = last_frame * universe.trajectory.dt / 1e3
        log.info(f"Final analysis time: {tf_anl:.2f} ns")

        # perform analysis
        universe_analysis(universe, df_plumed, sel)
