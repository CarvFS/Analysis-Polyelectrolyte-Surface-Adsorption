"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-09
Description: Utility functions to use in analysis scripts.
"""

# Standard library
import logging


def setup_logging(
    log_file: str = "default.log",
    stream: bool = False,
    verbose: bool = False,
    overwrite: bool = False,
) -> logging.Logger:
    """
    Setup logging for the script.

    Parameters
    ----------
    log_file : str, optional
        Name of the log file, by default "default.log"
    stream : bool, optional
        If True, log to the console, by default False
    verbose : bool, optional
        If True, the logging level is set to DEBUG, by default False
    overwrite : bool, optional
        If True, overwrite the log file, by default False

    Returns
    -------
    logging.Logger
        Logger object
    """
    log = logging.getLogger(__name__)

    if log.hasHandlers() and not overwrite:
        log.debug("Logger already initialized, returning log")
        return log

    # remove existing handlers
    if log.hasHandlers() and overwrite:
        for handler in log.handlers[:]:
            log.removeHandler(handler)

        # fmt="%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : "
        # + "%(lineno)d : Log : %(message)s",

    # add formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S",
    )
    # add file handler
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    if verbose:
        log.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.WARNING)
        handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    log.addHandler(handler)

    # add stream handler
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
        stream_handler.setFormatter(formatter)
        log.addHandler(stream_handler)

    log.info(f"Initializing logger file with name: {log_file.split('.')[0]}")
    return log
