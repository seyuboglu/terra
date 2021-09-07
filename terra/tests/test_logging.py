import os

from terra import Task
from terra.logging import Logger, init_logging
from terra.settings import TERRA_CONFIG
import terra.database as tdb


TERRA_CONFIG["notify"] = False


def test_logging(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    tdb.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    stdout_str = "qwertyuiop"
    stderr_str = "asdfghjkl"

    @Task
    def fn_a(run_dir=None):
        print(stdout_str)
        raise ValueError(stderr_str)
    try:
        fn_a()
    except ValueError:
        pass 
    log = fn_a.get_log()

    assert(stdout_str in log)
    assert(stderr_str in log)

