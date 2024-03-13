"""
Module for a pipeline to load, clean, and analyze data from the molecular
dynamics simulation.
| Author: Alec Glisman (GitHub: @alec-glisman)
| Date: 2023-11-08

This module defines a class `DataPipeline` that provides methods to load and
process data from molecular dynamics simulations. The class takes a base
directory containing the simulation data and optional parameters such as
temperature, file extensions, and verbosity. The class provides methods to
load plumed collective variables, molecular dynamics universe, and
statistical weights.
"""

# standard library
import logging
from pathlib import Path
import warnings

# third party
import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    import panedr as edr
    import MDAnalysis as mda


class DataPipeline:
    """
    A class to load, clean, and analyze data from molecular dynamics simulations.

    The class takes a base directory containing the simulation data and optional
    parameters such as temperature, file extensions, and verbosity. The class
    provides methods to load plumed collective variables, molecular dynamics
    universe, and statistical weights.
    """

    def __init__(
        self,
        data_path_base: Path,
        temperature: float = 300.0,
        verbose: bool = False,
        logger: logging.Logger = None,
        ext_top: str = "tpr",
        ext_traj: str = "xtc",
        ext_energy: str = "edr",
        ext_plumed: str = "data",
    ) -> None:
        """
        Initialize a Pipeline object.

        Parameters
        ----------
        data_path_base : Path
            The base directory containing the simulation data.
        temperature : float, optional
            The temperature of the simulation in Kelvin, by default 300.0.
        verbose : bool, optional
            Whether to print verbose logging messages, by default False.
        logger : logging.Logger, optional
            A logger object for logging messages, by default None.
        ext_top : str, optional
            The file extension for topology files, by default "tpr".
        ext_traj : str, optional
            The file extension for trajectory files, by default "xtc".
        ext_energy : str, optional
            The file extension for energy files, by default "edr".
        ext_plumed : str, optional
            The file extension for plumed files, by default "data".

        Attributes
        ----------
        tag : str
            The name of the directory containing the simulation data.
        data_path_base : Path
            The base directory containing the simulation data.
        temperature : float
            The temperature of the simulation in Kelvin.
        sampling_methods : list of str
            The names of the sampling methods used in the simulation.
        sampling_paths : list of Path
            The paths to the directories containing the sampling data.
        data_files : dict
            A dictionary containing the data files for each sampling method.
        _kb : float
            The Boltzmann constant in kJ/mol/K.
        _beta : float
            The inverse temperature in 1/kJ/mol.
        _verbose : bool
            Whether to print verbose logging messages.
        _ext_top : str
            The file extension for topology files.
        _ext_traj : str
            The file extension for trajectory files.
        _ext_energy : str
            The file extension for energy files.
        _ext_plumed : str
            The file extension for plumed files.
        _sampling_prefix : str
            The prefix for directories containing sampling data.
        _repl_prefix : str
            The prefix for directories containing replica data.
        """

        # external parameters
        self.tag = data_path_base.parts[-1]
        self.data_path_base = data_path_base
        self.temperature = temperature

        # internal parameters
        self._kb = 8.314462618e-3  # [kJ/mol/K]
        self._beta = 1.0 / (self._kb * self.temperature)
        self._verbose = verbose
        self._ext_top = ext_top
        self._ext_traj = ext_traj
        self._ext_energy = ext_energy
        self._ext_plumed = ext_plumed
        self._sampling_prefix = "3-sampling-"
        self._repl_prefix = "replica_"

        # loaded data
        self.sampling_methods = None
        self.sampling_paths = None
        self.data_files = None

        # setup class object
        self._init_log(logger=logger)
        self._log.info(
            f"Initializing data pipeline with data path: {self.data_path_base}"
        )
        self._find_data_files()

    def __iter__(self) -> "DataPipeline":
        """
        Returns an iterator over the sampling methods and data files.

        Parameters
        ----------
        None

        Returns
        -------
        DataPipeline
            An iterator over the sampling methods and data files.

        Raises
        ------
        ValueError
            If sampling methods or data files are not initialized.
        ValueError
            If sampling methods and data files do not have the same length.
        """
        if self.sampling_methods is None or self.data_files is None:
            raise ValueError("Sampling methods or data files not initialized")
        if len(self.sampling_methods) != len(self.data_files):
            raise ValueError(
                "Sampling methods and data files must have the same length"
            )

        self._iter_index = 0
        return self

    def __next__(self):
        """
        Returns the next sampling method and data files.

        Parameters
        ----------
        None

        Returns
        -------
        tuple of str and dict
            The next sampling method and data files.
        """
        if self._iter_index < len(self.sampling_methods):
            method = self.sampling_methods[self._iter_index]
            files = self.data_files[method]
            self._iter_index += 1
            return method, files
        else:
            raise StopIteration

    def __repr__(self) -> str:
        """
        Returns a string representation of the Pipeline object.

        Parameters
        ----------
        None

        Returns
        -------
        str
            A string representation of the Pipeline object.
        """
        return (
            f"DataPipeline(data_path_base={self.data_path_base}, "
            + f"temperature={self.temperature}, verbose={self._verbose}, "
            + f"ext_top={self._ext_top}, ext_traj={self._ext_traj}, "
            + f"ext_energy={self._ext_energy}, ext_plumed={self._ext_plumed}), "
            + f"sampling_methods={self.sampling_methods}, "
        )

    def __len__(self) -> int:
        """
        Returns the number of sampling methods.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of sampling methods.
        """
        return len(self.sampling_methods)

    def _init_log(self, logger: logging.Logger = None, log_file: Path = None) -> None:
        """
        Initializes the logger for the pipeline.

        The logger is initialized with a StreamHandler that outputs to stdout.
        If the logger already has a handler, the existing handler is used
        instead. The logging level is set to DEBUG if the pipeline is in
        verbose mode, and to WARNING otherwise.

        Parameters
        ----------
        logger : logging.Logger
            A logger object for logging messages, by default None.
        log_file : Path
            The path to a log file, by default None.

        Returns
        -------
        None
        """
        if logger is not None:
            self._log = logger
        else:
            self._log = logging.getLogger(__name__)

        add_handler = False
        if not self._log.hasHandlers():
            # add handler if not already present to stdout
            handler = logging.StreamHandler()
            self._log.addHandler(handler)

            # add file handler to log file
            if log_file is None:
                file = f"{self.data_path_base}/pipeline.log"
            else:
                file = log_file
            fhandler = logging.FileHandler(filename=file, mode="w")
            add_handler = True
            self._log.addHandler(fhandler)

            # set logging format
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %I:%M:%S",
            )

            handler.setFormatter(formatter)
            fhandler.setFormatter(formatter)

        # if log already has handlers, modify write file handler to write to log_file
        else:
            for handler in self._log.handlers:
                if isinstance(handler, logging.FileHandler):
                    file = log_file
                    handler.baseFilename = file
                    handler.stream = open(file, "w")
                    handler.acquire()
                    add_handler = True

        # set logging level of logger and handler
        try:
            if self._verbose:
                self._log.setLevel(logging.DEBUG)
                for handler in self._log.handlers:
                    handler.setLevel(logging.DEBUG)
            else:
                self._log.setLevel(logging.WARNING)
                for handler in self._log.handlers:
                    handler.setLevel(logging.WARNING)
        except Exception as e:
            self._log.error(f"Failed to set logging level: {e}")

        # print log file path to stdout
        if add_handler:
            self._log.info(f"Logging to file: {self.data_path_base}/pipeline.log")

    def _find_data_files(self) -> None:
        """
        Find all data files for each sampling method and create a dictionary of
        data files where the key is the sampling method. Check that there are
        no duplicate sampling methods and all sampling methods have a
        corresponding sampling path.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If there are duplicate sampling methods.
        ValueError
            If the number of sampling methods does not match the number of sampling
            paths.
        ValueError
            If a sampling path does not exist.
        ValueError
            If there are no topology files for a sampling method.
        ValueError
            If there are no trajectory files for a sampling method.
        ValueError
            If there are no plumed files for a sampling method.
        ValueError
            If there is more than one topology file for a sampling method.
        ValueError
            If there is more than one plumed file for a sampling method.
        """

        # find all sampling methods: subdirectories of the data path starting with
        # "3-sampling-"
        self.sampling_paths = [
            x
            for x in self.data_path_base.iterdir()
            if x.is_dir() and x.name.startswith(self._sampling_prefix)
        ]
        # remove prefix from sampling method names and remove "eqbm" sampling methods
        self.sampling_methods = [
            x.name[len(self._sampling_prefix) :]
            for x in self.sampling_paths
            if "eqbm" not in x.name
        ]
        # remove "-bad-" from sampling method names
        self.sampling_methods = [x for x in self.sampling_methods if "-bad-" not in x]

        # if sampling method has multiple replicas (subdirectories of the sampling
        # method directory of the form "replica-#") then append the replica number
        # to the sampling method name (e.g. "hremd" -> "hremd-1") for each replica
        self._log.debug(
            f"Found {len(self.sampling_methods)} "
            + f"sampling methods: {self.sampling_methods}"
        )
        orig_sampling_methods = self.sampling_methods.copy()
        for method in orig_sampling_methods:
            replicas = [
                x
                for x in (
                    self.data_path_base / f"{self._sampling_prefix}{method}"
                ).iterdir()
                if x.is_dir() and x.name.startswith(self._repl_prefix)
            ]
            self._log.debug(f"Found {len(replicas)} replicas for method {method}")
            if len(replicas) > 0:
                # remove the sampling method name from the list of sampling methods and
                # add the sampling method name with the replica number
                self.sampling_methods.remove(method)
                self.sampling_methods += [f"{method}-{x.name}" for x in replicas]
                # remove the sampling method directory from the list of sampling paths
                # and add the replica directories
                repl_base = self.data_path_base / f"{self._sampling_prefix}{method}"
                repl_paths = [repl_base / x.name for x in replicas]
                self.sampling_paths.remove(repl_base)
                self.sampling_paths += repl_paths

        # sort sampling methods and paths by sampling method name in ascending order
        if len(self.sampling_paths) > 0:
            self.sampling_methods, self.sampling_paths = zip(
                *sorted(zip(self.sampling_methods, self.sampling_paths))
            )

        # check that there are no duplicate sampling methods and all sampling methods
        # have a corresponding sampling path
        if len(self.sampling_methods) != len(set(self.sampling_methods)):
            raise ValueError("Found duplicate sampling methods")
        if len(self.sampling_methods) != len(self.sampling_paths):
            raise ValueError(
                "Number of sampling methods does not match number of sampling paths"
            )
        for p in self.sampling_paths:
            if not p.exists():
                raise ValueError(f"Sampling path {p} does not exist")
        self._log.debug("All sampling methods and paths are valid")

        # find all data files
        self.top_files = list(self.data_path_base.rglob(f"*.{self._ext_top}"))
        self.traj_files = list(self.data_path_base.rglob(f"*.{self._ext_traj}"))
        self.energy_files = list(self.data_path_base.rglob(f"*.{self._ext_energy}"))
        self.plumed_files = list(
            self.data_path_base.rglob(f"COLVAR*.{self._ext_plumed}")
        )

        # drop any files that are not in a sampling method directory
        self.top_files = [x for x in self.top_files if x.parent in self.sampling_paths]
        self.traj_files = [
            x for x in self.traj_files if x.parent in self.sampling_paths
        ]
        self.energy_files = [
            x for x in self.energy_files if x.parent in self.sampling_paths
        ]
        self.plumed_files = [
            x for x in self.plumed_files if x.parent in self.sampling_paths
        ]

        # create a dictionary of data files where the key is the sampling method
        self.data_files = {}
        for method, path in zip(self.sampling_methods, self.sampling_paths):
            self.data_files[method] = {}
            # find all files for this sampling method and add them to the dictionary
            self.data_files[method]["top"] = [
                x for x in self.top_files if x.parent == path
            ]
            self.data_files[method]["traj"] = [
                x for x in self.traj_files if x.parent == path
            ]
            self.data_files[method]["energy"] = [
                x for x in self.energy_files if x.parent == path
            ]
            self.data_files[method]["plumed"] = [
                x for x in self.plumed_files if x.parent == path
            ]
            self.data_files[method]["df_colvar"] = None
            self.data_files[method]["df_energy"] = None
            self.data_files[method]["universe"] = None

            # sort files by name
            self.data_files[method]["top"].sort()
            self.data_files[method]["traj"].sort()
            self.data_files[method]["energy"].sort()
            self.data_files[method]["plumed"].sort()

            # log the number of files found for this sampling method
            self._log.debug(
                "Found "
                + str(len(self.data_files[method]["top"]))
                + " topology files for method "
                + str(method)
            )
            self._log.debug(
                "Found "
                + str(len(self.data_files[method]["traj"]))
                + " trajectory files for method "
                + str(method)
            )
            self._log.debug(
                "Found "
                + str(len(self.data_files[method]["energy"]))
                + " energy files for method "
                + str(method)
            )

            self._log.debug(
                "Found "
                + str(len(self.data_files[method]["plumed"]))
                + " plumed files for method "
                + str(method)
            )

            if len(self.data_files[method]["top"]) == 0:
                raise ValueError(
                    f"No topology files found for sampling method {method} "
                    + f"in path {path}"
                )
            elif len(self.data_files[method]["top"]) > 1:
                raise ValueError(
                    f"Found {len(self.data_files[method]['top'])} topology files "
                    + f"for sampling method {method}, expected 1, in path {path}. "
                    + f"Found files: {self.data_files[method]['top']}"
                )

            if len(self.data_files[method]["traj"]) == 0:
                raise ValueError(
                    f"No trajectory files found for sampling method {method} "
                    + f"in path {path}"
                )

            if len(self.data_files[method]["plumed"]) == 0 and method != "md":
                raise ValueError(
                    f"No plumed files found for sampling method {method} in path {path}"
                )
            elif len(self.data_files[method]["plumed"]) == 0 and method == "md":
                self._log.warning(
                    f"No plumed files found for sampling method {method} in path {path}"
                )
            elif len(self.data_files[method]["plumed"]) > 1:
                raise ValueError(
                    f"Found {len(self.data_files[method]['plumed'])} plumed files "
                    + f"for sampling method {method}, expected 1 in path {path}. "
                    + f"Found files: {self.data_files[method]['plumed']}"
                )

    def _statistical_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the statistical weight of each sample in the input DataFrame based
        on the bias values.
        The statistical weight is calculated as exp(-beta * (bias - max(bias))).
        The statistical weight is normalized so that the sum of all weights is 1.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the bias values.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with the statistical weight added as a column.

        Raises
        ------
        ValueError
            If the input DataFrame does not contain any bias columns.
        ValueError
            If all statistical weights are zero.
        """
        # get all columns ending in ".bias"
        cols = [col for col in df.columns if col.endswith(".bias")]
        if "metad.bias" in df.columns and "metad.rbias" in df.columns:
            cols.remove("metad.bias")
        if "metad.rbias" in df.columns:
            cols.append("metad.rbias")
        self._log.info(f"Found bias columns: {cols}")

        if len(cols) == 0:
            raise ValueError("No bias columns found in DataFrame")

        # calculate statistical weight
        df["bias"] = df[cols].sum(axis=1)
        df["bias_nondim"] = (df["bias"] - np.nanmax(df["bias"])) * self._beta
        df["weight"] = np.exp(df["bias_nondim"])
        df["weight"] /= df["weight"].sum()

        # check if all weights are zero
        if np.all(df["weight"] == 0.0):
            raise ValueError("All weights are zero")

    def load_plumed_colvar(self, method: str) -> pd.DataFrame:
        """
        Load the collective variables (colvar) data from the plumed file associated
        with the given method.

        Parameters
        ----------
        method : str
            The name of the sampling method.

        Returns
        -------
        pd.DataFrame
            The collective variables data.

        Raises
        ------
        ValueError
            If there is not exactly one plumed file for the given method.
        """

        # check that there is only one plumed file for this method
        if len(self.data_files[method]["plumed"]) == 0 and method != "md":
            raise ValueError(f"No plumed files found for method {method}, expected 1")
        elif len(self.data_files[method]["plumed"]) > 1:
            raise ValueError(
                f"Found {len(self.data_files[method]['plumed'])} plumed files "
                + f"for method {method}, expected 1"
            )

        if method == "md" and len(self.data_files[method]["plumed"]) == 0:
            self._log.warning(f"No plumed file for method: {method}")
            self.data_files[method]["df_colvar"] = None
            return None

        file = self.data_files[method]["plumed"][0]
        self._log.info(f"Loading plumed file for method: {method}")

        # check if file already loaded
        if self.data_files[method]["df_colvar"] is not None:
            self._log.debug("Plumed file already loaded")
            return self.data_files[method]["df_colvar"].copy()

        # first line of file contains column names
        with open(str(file), encoding="utf8") as f:
            header = f.readline()
        header = header.split()[2:]  # remove "#!" FIELDS
        n_cols = len(header)
        self._log.debug(f"Found {n_cols} columns in plumed file")
        self._log.debug(f"Columns: {header}")

        # read in data
        df = pd.read_csv(
            str(file),
            names=header,
            comment="#",
            sep=r"\s+",
            skipinitialspace=True,
            usecols=list(range(n_cols)),
        )

        # drop any null columns
        nrows = len(df)
        df = df.dropna(axis=1, how="all")
        self._log.warning(
            f"Dropped {n_cols - len(df.columns)} columns "
            + f"({(n_cols - len(df.columns))/n_cols*100:.2f}%) with NaN values"
        )

        # convert columns to numeric and drop any rows with NaN values
        nrows = len(df)
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        self._log.warning(
            f"Dropped {nrows - len(df)} rows ({(nrows - len(df))/nrows*100:.2f}%) that "
            + "could not be converted to numeric"
        )

        # drop rows where abs(*.bias) is > bias_max
        bias_max = 100
        cols = [
            col for col in df.columns if col.endswith(".bias") or col.endswith(".rbias")
        ]
        nrows = len(df)
        df = df[~(df[cols].abs() > bias_max).any(axis=1)]
        self._log.warning(
            f"Dropped {nrows - len(df)} rows ({(nrows - len(df))/nrows*100:.2f}%) "
            + f"where bias > {bias_max}"
        )

        # sort by time
        df = df.sort_values(by="time")

        # if duplicate "time" rows, keep only the last one
        df = df.drop_duplicates(subset="time", keep="last")
        self._statistical_weight(df)
        self.data_files[method]["df_colvar"] = df.copy()
        self._log.debug(f"Number of rows in plumed file: {len(df)}")
        return df.copy()

    def save_plumed_colvar(
        self, method: str = None, file: Path = None, directory: Path = None
    ) -> Path:
        """
        Save the collective variables (colvar) data to a parquet file for a given
        sampling method or all sampling methods.

        Parameters
        ----------
        method : str, optional
            The name of the sampling method, by default None. If None, save the
            collective variables for all sampling methods.
        file : Path, optional
            The name of the output file, by default None
        directory : Path, optional
            The directory to save the output file, by default None

        Returns
        -------
        Path
            The path to the output file.
        """
        if directory is None:
            directory = self.data_path_base / "analysis"
        if not isinstance(directory, Path):
            directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # return colvar path of a single method
        if method is not None:
            if file is None:
                file = f"colvar_{method}.parquet"
            file = directory / file
            self._log.debug(f"Saving plumed colvar file: {file}")
            if self.data_files[method]["df_colvar"] is None:
                self.load_plumed_colvar(method)

            if self.data_files[method]["df_colvar"] is not None:
                # ignore FutureWarning about is_sparse
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                    self.data_files[method]["df_colvar"].to_parquet(file)

            return file

        # return colvar paths of all methods
        files = []
        for method in self.sampling_methods:
            if file is None:
                file = f"colvar_{method}.parquet"
            file = directory / file
            files.append(file)
            self._log.debug(f"Saving plumed colvar file: {file}")
            if self.data_files[method]["df_colvar"] is None:
                self.load_plumed_colvar(method)

                # ignore FutureWarning about is_sparse
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                    self.data_files[method]["df_colvar"].to_parquet(file)

            file = None

        return files

    def load_universe(self, method: str, **kwargs) -> mda.Universe:
        """
        Load the MDAnalysis universe for a given sampling method.

        Parameters
        ----------
        method : str
            The name of the sampling method.

        Returns
        -------
        mda.Universe
            The MDAnalysis universe.

        Raises
        ------
        ValueError
            If there is not exactly one topology file for the given method.
        """
        # check that there is only one topology file for this method
        num_top_files = len(self.data_files[method]["top"])
        if num_top_files != 1:
            raise ValueError(
                f"Expected 1 topology file for method {method}. "
                + f"Found {num_top_files} topology files."
            )

        self._log.debug(f"Loading MDA universe for method: {method}")
        self._log.debug(
            f"Found {len(self.data_files[method]['traj'])} trajectory files"
        )
        self._log.debug(f"Topology file: {self.data_files[method]['top'][0]}")
        self._log.debug(f"Trajectory files: {self.data_files[method]['traj']}")
        universe = mda.Universe(
            self.data_files[method]["top"][0],
            self.data_files[method]["traj"],
            verbose=self._verbose,
            **kwargs,
        )
        self._log.debug(f"Loaded MDA universe for method: {method}")
        self._log.info(f"Number of frames in universe: {universe.trajectory.n_frames}")
        self.data_files[method]["universe"] = universe
        return universe

    def load_energy(self, method: str) -> pd.DataFrame:
        """
        Load the energy data for a given sampling method.

        Parameters
        ----------
        method : str
            The name of the sampling method.

        Returns
        -------
        pd.DataFrame
            The energy data.

        Raises
        ------
        ValueError
            If there are no energy files for the given method.
        """
        self._log.debug(f"Loading energy files for method: {method}")
        self._log.debug(f"Found {len(self.data_files[method]['energy'])} energy files")
        self._log.debug(f"Energy files: {self.data_files[method]['energy']}")

        if len(self.data_files[method]["energy"]) == 0:
            raise ValueError(f"No energy files found for sampling method {method}")

        # read in all energy files
        energy = []
        for file in self.data_files[method]["energy"]:
            energy.append(edr.edr_to_df(file))

        # concatenate all energy files
        df_energy = pd.concat(energy)

        # save energy data to internal dictionary
        self.data_files[method]["df_energy"] = df_energy.copy()

        return df_energy
