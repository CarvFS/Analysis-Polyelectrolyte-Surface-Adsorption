from dask.distributed import Client
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from utils.dask_helper import get_client  # noqa: E402


@pytest.fixture(scope="module")
def scheduler() -> Client:
    return get_client(4, memory_limit="1GB")


def test_get_client_new_client(scheduler: Client):
    assert isinstance(scheduler, Client)
