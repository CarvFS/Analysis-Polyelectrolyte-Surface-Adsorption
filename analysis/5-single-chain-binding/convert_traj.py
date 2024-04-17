import MDAnalysis as mda
from pathlib import Path


def convert_traj(
    top: str, trajs: list[str], out: str, selection: str = "all", slice: tuple = None
):
    """
    Convert trajectory to GRO format

    Parameters
    ----------
    top : str
        Path to topology file
    traj : list[str]
        List of paths to trajectory files
    out : str
        Path to output file
    selection : str
        Selection string for atoms to keep in output file
    slice : tuple
        Slice of trajectory to keep (start, stop, step)
    """
    u = mda.Universe(top, trajs)
    ag = u.select_atoms(selection)
    if slice is None:
        ag.write(out, frames="all")
    else:
        ag.write(out, frames=u.trajectory[slice[0] : slice[1] : slice[2]])


if __name__ == "__main__":
    dir_base = Path(
        "/media/aglisman/Data/Single-Chain-Adsorption/cleaned/6.3.1-calcite-104surface-12nm_surface-13nm_vertical-1chain-PAcr-32mer-0Crb-0Ca-32Na-0Cl-300K-1bar-NVT/3-sampling-opes-one/replica_00/2-concatenated"
    )
    file_stem = "prod_opes_one_multicv"
    slice = (0, 10000, 10)

    convert_traj(
        top=dir_base / f"{file_stem}.tpr",
        trajs=[dir_base / f"{file_stem}.xtc"],
        out=dir_base / f"{file_stem}.ncdf",
        slice=slice,
    )
