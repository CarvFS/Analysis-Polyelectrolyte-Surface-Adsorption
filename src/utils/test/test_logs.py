import logging
import os
import sys
from pathlib import Path
import pytest  # noqa: F401

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from utils.logs import setup_logging  # noqa: E402


def test_setup_logging_default():
    # Test default behavior
    log = setup_logging("test.log", overwrite=True)
    assert isinstance(log, logging.Logger)
    assert log.getEffectiveLevel() == logging.WARNING
    assert len(log.handlers) == 1
    assert isinstance(log.handlers[0], logging.FileHandler)
    assert log.handlers[0].level == logging.WARNING


def test_setup_logging_file_creation():
    # Test log file creation
    log_file = "test.log"
    log = setup_logging(log_file=log_file, overwrite=True)
    assert os.path.exists(log_file)

    # Test log file name
    assert log_file.split(".")[0] in log.handlers[0].baseFilename


def test_setup_logging_stream():
    # Test stream logging
    log = setup_logging("test3.log", stream=True, overwrite=True)
    assert len(log.handlers) == 2
    assert isinstance(log.handlers[1], logging.StreamHandler)
    assert log.handlers[1].level == logging.WARNING


def test_setup_logging_verbose():
    # Test verbose logging
    log = setup_logging("test4.log", verbose=True, overwrite=True)
    assert log.level == logging.DEBUG
    assert log.handlers[0].level == logging.DEBUG
    assert (
        log.handlers[1].level == logging.DEBUG
        if len(log.handlers) == 2
        else logging.WARNING
    )
