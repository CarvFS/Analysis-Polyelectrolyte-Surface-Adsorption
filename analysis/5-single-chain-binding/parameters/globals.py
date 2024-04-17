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
STEP: int = 5  # Step between frames to read
MODULE: str = "multiprocessing"  # parallel processing {joblib, multiprocessing, dask}
N_JOBS: int = 20  # Number of parallel jobs
N_BLOCKS: int = 240  # Number of blocks to split trajectory into
SOLVENT: bool = True  # Whether or not to include solvent in the analysis

# Data processing parameters
VERBOSE: bool = True  # Verbose output
RELOAD_DATA: bool = False  # if True, remake all data
REFRESH_OFFSETS: bool = False  # if True, remake all offsets on trajectory files
ALL_REPLICAS: bool = False  # if True, process all replicas

# system information
TEMPERATURE_K: float = 300  # [K] System temperature

# File I/O
FIG_EXT: str = "png"  # Figure file extension
DEFAULT_PATH: Path = Path(
    "/nfs/zeal_nas/home_mount/aglisman/GitHub/Polyelectrolyte-Surface-Adsorption"
    + "/data_archive/6_single_chain_binding/cleaned/6.5.4-calcite-104surface-12nm_"
    + "surface-13nm_vertical-1chain-PAcr-32mer-0Crb-32Ca-32Na-64Cl-300K-1bar-NVT"
)
