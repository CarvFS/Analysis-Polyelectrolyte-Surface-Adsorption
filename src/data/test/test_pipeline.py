# import modules
import logging
import os
from pathlib import Path
import pytest
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from data.pipeline import DataPipeline  # noqa: E402


class TestPipeline(object):
    # add data_path and pipeline fixtures
    @pytest.fixture(scope="module")
    def data_path(self):
        return Path(
            "/media/aglisman/Linux_Overflow/home/aglisman/VSCodeProjects"
            + "/Polyelectrolyte-Surface-Adsorption/data/completed"
            + "/sjobid_0-calcite-104surface-5nm_surface-8nm_vertical-1chain"
            + "-PAcr-8mer-0Crb-0Ca-8Na-0Cl-300K-1bar-NVT"
        )

    @pytest.fixture(scope="module")
    def data_path_empty(self):
        dir_path = Path(os.getcwd()) / "temp"
        return dir_path

    @pytest.fixture(scope="module")
    def pipeline(self, data_path):
        return DataPipeline(data_path)

    @pytest.fixture(scope="module")
    def pipeline_empty(self, data_path_empty):
        if not data_path_empty.exists():
            data_path_empty.mkdir()
        return DataPipeline(data_path_empty)

    def test_init(self, pipeline, data_path):
        assert pipeline.tag == data_path.parts[-1]
        assert pipeline.data_path_base == data_path
        assert pipeline.temperature == 300.0
        assert pipeline._kb == pytest.approx(8.314462618e-3)
        assert pipeline._beta == pytest.approx(
            1.0 / (pipeline._kb * pipeline.temperature)
        )
        assert pipeline._verbose is False
        assert pipeline._ext_top == "tpr"
        assert pipeline._ext_traj == "xtc"
        assert pipeline._ext_energy == "edr"
        assert pipeline._ext_plumed == "data"
        assert pipeline._sampling_prefix == "3-sampling-"
        assert pipeline._repl_prefix == "replica_"
        assert pipeline.universe is None
        assert pipeline.energy is None

    def test_find_data_files(self, pipeline, data_path):
        pipeline._find_data_files()
        assert len(pipeline.sampling_methods) == 42
        assert len(pipeline.sampling_paths) == 42
        assert len(pipeline.top_files) == 42
        assert len(pipeline.traj_files) == 83
        assert len(pipeline.energy_files) == 83
        assert len(pipeline.plumed_files) == 42
        assert len(pipeline.data_files) == 42
        assert set(pipeline.sampling_methods) == set(
            [
                "hremd-metadynamics-replica_00",
                "hremd-metadynamics-replica_01",
                "hremd-metadynamics-replica_02",
                "hremd-metadynamics-replica_03",
                "hremd-metadynamics-replica_04",
                "hremd-metadynamics-replica_05",
                "hremd-metadynamics-replica_06",
                "hremd-metadynamics-replica_07",
                "hremd-opes-explore-replica_00",
                "hremd-opes-explore-replica_01",
                "hremd-opes-explore-replica_02",
                "hremd-opes-explore-replica_03",
                "hremd-opes-explore-replica_04",
                "hremd-opes-explore-replica_05",
                "hremd-opes-explore-replica_06",
                "hremd-opes-explore-replica_07",
                "hremd-replica_00",
                "hremd-replica_01",
                "hremd-replica_02",
                "hremd-replica_03",
                "hremd-replica_04",
                "hremd-replica_05",
                "hremd-replica_06",
                "hremd-replica_07",
                "metadynamics",
                "opes-explore",
                "opes-one-0-replica_00",
                "opes-one-0-replica_01",
                "opes-one-0-replica_02",
                "opes-one-0-replica_03",
                "opes-one-0-replica_04",
                "opes-one-0-replica_05",
                "opes-one-0-replica_06",
                "opes-one-0-replica_07",
                "opes-one-replica_00",
                "opes-one-replica_01",
                "opes-one-replica_02",
                "opes-one-replica_03",
                "opes-one-replica_04",
                "opes-one-replica_05",
                "opes-one-replica_06",
                "opes-one-replica_07",
            ]
        )

    def test_init_log(self, pipeline):
        pipeline._init_log()
        assert pipeline._log.level == logging.WARNING

        pipeline._verbose = True
        pipeline._init_log()
        assert pipeline._log.level == logging.DEBUG

    def test_find_data_files_errors(self, pipeline_empty, data_path_empty):
        # test for duplicate sampling methods
        (data_path_empty / "3-sampling-hremd-1").mkdir(exist_ok=True)
        with pytest.raises(ValueError):
            pipeline_empty._find_data_files()

        # test for mismatch between number of sampling methods and paths
        (data_path_empty / "3-sampling-hremd-1").rmdir()
        (data_path_empty / "3-sampling-tempering-1").mkdir(exist_ok=True)
        with pytest.raises(ValueError):
            pipeline_empty._find_data_files()

        # test for missing topology file
        (data_path_empty / "3-sampling-tempering-1" / "replica_1").mkdir(exist_ok=True)
        with pytest.raises(ValueError):
            pipeline_empty._find_data_files()

        # test for missing trajectory file
        (data_path_empty / "3-sampling-tempering-1" / "replica_1").joinpath(
            "tempering.xtc"
        ).touch()
        with pytest.raises(ValueError):
            pipeline_empty._find_data_files()

        # test for missing plumed file
        (data_path_empty / "3-sampling-tempering-1" / "tempering.tpr").touch()
        (data_path_empty / "3-sampling-tempering-1" / "tempering.xtc").touch()
        with pytest.raises(ValueError):
            pipeline_empty._find_data_files()

        # test for multiple topology files
        (data_path_empty / "3-sampling-tempering-1" / "tempering-1.tpr").touch()
        with pytest.raises(ValueError):
            pipeline_empty._find_data_files()

        # test for multiple plumed files
        (data_path_empty / "3-sampling-tempering-1" / "COLVAR-1.data").touch()
        with pytest.raises(ValueError):
            pipeline_empty._find_data_files()

        # delete temp directories
        shutil.rmtree(data_path_empty)

    def test_find_data_files_replicas(self, pipeline_empty, data_path_empty):
        # test for multiple replicas
        (data_path_empty / "3-sampling-hremd-1" / "replica_1").mkdir(parents=True)
        (data_path_empty / "3-sampling-hremd-1" / "replica_2").mkdir()
        (data_path_empty / "3-sampling-hremd-1" / "replica_1" / "hremd-1.tpr").touch()
        (data_path_empty / "3-sampling-hremd-1" / "replica_2" / "hremd-2.tpr").touch()
        (data_path_empty / "3-sampling-hremd-1" / "replica_1" / "hremd-1.xtc").touch()
        (data_path_empty / "3-sampling-hremd-1" / "replica_2" / "hremd-2.xtc").touch()
        (data_path_empty / "3-sampling-hremd-1" / "replica_1" / "hremd-1.edr").touch()
        (data_path_empty / "3-sampling-hremd-1" / "replica_2" / "hremd-2.edr").touch()
        (data_path_empty / "3-sampling-hremd-1" / "replica_1" / "COLVAR.1.data").touch()
        (data_path_empty / "3-sampling-hremd-1" / "replica_2" / "COLVAR.2.data").touch()

        pipeline_empty._find_data_files()
        assert len(pipeline_empty.sampling_methods) == 2
        assert len(pipeline_empty.sampling_paths) == 2
        assert set(pipeline_empty.sampling_methods) == set(
            ["hremd-1-replica_1", "hremd-1-replica_2"]
        )

    def test_remove_temp(self, pipeline_empty, data_path_empty):
        shutil.rmtree(data_path_empty)
        assert not data_path_empty.exists()
