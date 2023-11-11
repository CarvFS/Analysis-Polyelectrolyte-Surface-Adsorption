"""
| Author: Alec Glisman (GitHub: @alec-glisman)
| Date: 2021-08-31
| Description: This script generates data for the 2-chain system using MDAnalysis
and Plumed.
"""

import dask
from dask.distributed import Client, LocalCluster


def get_client(n_workers: int, memory_limit: str = "60GB") -> Client:
    """
    Get a Dask client for parallel processing. If a client is already running,
    return that client. Otherwise, start a new client.

    Parameters
    ----------
    n_workers : int
        Number of workers to use for parallel processing.
    memory_limit : str, optional
        Memory limit for each worker, by default "60GB" (60 GB).

    Returns
    -------
    Client
        Dask client for parallel processing.
    """

    try:
        client = Client("tcp://localhost:8785", timeout=2)

    except OSError:
        dask.config.set(
            {
                "distributed.worker.memory.target": 0.6,
                "distributed.worker.memory.spill": 0.7,
                "distributed.worker.memory.pause": 1,
                "distributed.worker.memory.terminate": 1,
            }
        )
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit=memory_limit,
            processes=True,
            scheduler_port=8785,
        )
        client = Client(cluster)

    return client
