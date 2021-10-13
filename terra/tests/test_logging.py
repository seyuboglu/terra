import pytest

from terra import Task
from .testbed import BaseTestBed


@pytest.fixture()
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@BaseTestBed.parametrize()
def test_logging(testbed: BaseTestBed):
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

    assert stdout_str in log
    assert stderr_str in log
