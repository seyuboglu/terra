import os
import numpy as np

from terra.utils import ensure_dir_exists
from terra.io import Artifact
from terra.settings import TERRA_CONFIG
import terra.database as tdb

TERRA_CONFIG["notify"] = True


def test_artifact_dump(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    tdb.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    run_dir = os.path.join(tmpdir, str(1))
    ensure_dir_exists(run_dir)
    x = np.random.rand(100)
    artifact = Artifact.dump(value=x, run_dir=run_dir)

    x_loaded = artifact.load()
    assert(np.allclose(x, x_loaded))
