import os

from terra.settings import TERRA_CONFIG
from terra.database import get_session

TERRA_CONFIG["notify"] = False


def test_get_session(tmpdir):
    TERRA_CONFIG["storage_dir"] = tmpdir

    get_session(create=True)
    db_path = os.path.join(TERRA_CONFIG["storage_dir"], "terra.sqlite")
    assert os.path.exists(db_path)
    
    storage_dir = os.path.join(tmpdir, "non_default")
    get_session(storage_dir=storage_dir, create=True)
    db_path = os.path.join(storage_dir, "terra.sqlite")
    assert os.path.exists(db_path)
    

def test_dont_create_database(tmpdir):
    TERRA_CONFIG["storage_dir"] = tmpdir

    try:
        get_session(create=False)
    except ValueError:
        pass