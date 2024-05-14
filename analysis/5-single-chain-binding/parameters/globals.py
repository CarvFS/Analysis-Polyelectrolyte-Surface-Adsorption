"""
Global parameters for analysis scripts

Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2024-01-22
Description: This script contains global parameters for the analysis scripts.
"""

# imports
from pathlib import Path

# MDAnalysis trajectory parameters
START: int = int(25e3)  # First frame to read
STOP: int = int(500e3)  # Last frame to read
STEP: int = 10  # Step between frames to read
MODULE: str = "multiprocessing"  # parallel processing {joblib, multiprocessing, dask}
N_JOBS: int = 30  # Number of parallel jobs
N_BLOCKS: int = 30 * 8  # Number of blocks to split trajectory into
SOLVENT: bool = True  # Whether or not to include solvent in the analysis

# Data processing parameters
VERBOSE: bool = True  # Verbose output
RELOAD_DATA: bool = False  # if True, remake all data
REFRESH_OFFSETS: bool = False  # if True, remake all offsets on trajectory files
ALL_REPLICAS: bool = False  # if True, process all replicas
CONCATENATED: bool = True  # if True, use concatenated trajectory files

# system information
TEMPERATURE_K: float = 300  # [K] System temperature

# File I/O
FIG_EXT: str = "png"  # Figure file extension
DEFAULT_PATH: Path = Path(
    "/nfs/zeal_nas/data_mount/aglisman-data/1-electronic-continuum-correction"
    + "/7-single-chain-surface-binding/6_single_chain_binding/cleaned"
    + "/6.1.0-calcite-104surface-9nm_surface-10nm_vertical-1chain-PAcr"
    + "-16mer-0Crb-0Ca-16Na-0Cl-300K-1bar-NVT"
)
