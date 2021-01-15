import os
import numpy as np

from terra.utils import ensure_dir_exists
from terra.io import Artifact
from terra.settings import TERRA_CONFIG
import terra.database as tdb

TERRA_CONFIG["notify"] = False


def test_artifact_dump_and_load(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    tdb.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir
    
    run_dir = os.path.join(tmpdir, str(1))
    ensure_dir_exists(run_dir)
    x = np.random.rand(100)
    artifact = Artifact.dump(value=x, run_dir=run_dir)

    x_loaded = artifact.load(run_id=1)
    assert(np.allclose(x, x_loaded))

    # test row added to artifact table
    df = tdb.get_artifact_dumps()
    assert(len(df) == 1)
    assert(df.iloc[0].type == "<class 'numpy.ndarray'>")

    # test row added to artifact table
    df = tdb.get_artifact_loads()
    assert(len(df) == 1)


def test_artifact_rm(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    tdb.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir
    
    run_dir = os.path.join(tmpdir, str(1))
    ensure_dir_exists(run_dir)
    x = np.random.rand(100)
    artifact = Artifact.dump(value=x, run_dir=run_dir)
    
    artifact.rm()
    







