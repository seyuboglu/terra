import numpy as np
import pandas as pd
import tempfile


from terra import Task, TerraSettings

TerraSettings.notify = False


def test_make_task(tmpdir):
    TerraSettings.storage_dir = tmpdir

    @Task.make_task
    def fn_a(run_dir=None):
        return None

    assert isinstance(fn_a, Task)


def test_np_pipeline(tmpdir):
    TerraSettings.storage_dir = tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        return np.ones(4) * x

    @Task.make_task
    def fn_b(x, run_dir=None):
        return np.ones(4) * (x ** 2)

    @Task.make_task
    def fn_c(x, y, run_dir=None):
        return x + y

    fn_a(2)
    fn_b(2)
    out_c = fn_c(fn_a.out(), fn_b.out())

    assert np.all(out_c == np.full(4, 6))


def test_pandas_pipeline(tmpdir):
    TerraSettings.storage_dir = tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        df = pd.DataFrame([{"a": idx, "b": idx ** 2} for idx in range(1, x + 1)])
        return df

    fn_a(10)

    @Task.make_task
    def fn_b(x, run_dir=None):
        df = pd.DataFrame([{"a": idx, "c": idx ** 3} for idx in range(1, x + 1)])
        return df

    fn_b(10)

    @Task.make_task
    def fn_c(x, y, run_dir=None):
        return x.merge(right=y, on="a")

    out_c = fn_c(fn_a.out(), fn_b.out())

    assert isinstance(out_c, pd.DataFrame)
    assert len(out_c) == 10
    assert (out_c.a == out_c.c / out_c.b).all()


def test_scalar_pipeline(tmpdir):
    TerraSettings.storage_dir = tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        return np.ones(4) * x

    @Task.make_task
    def fn_b(x, run_dir=None):
        return x ** 2

    @Task.make_task
    def fn_c(x, y, run_dir=None):
        return x * y

    fn_a(2)
    fn_b(4)
    out_c = fn_c(fn_a.out(), fn_b.out())

    assert np.all(out_c == np.full(4, 32))


def test_out_scalar(tmpdir):
    TerraSettings.storage_dir = tmpdir

    @Task.make_task
    def fn_b(x, run_dir=None):
        return x ** 2

    fn_b(4)
    assert fn_b.out() == 16

    @Task.make_task
    def fn_b(x, run_dir=None):
        return x ** 2, x ** 3

    fn_b(4)
    a, b = fn_b.out()
    assert a == 16
    assert b == 64


def test_out_np(tmpdir):
    TerraSettings.storage_dir = tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        return np.ones(4) * x

    fn_a(3)
    assert np.all(fn_a.out().load() == np.full(4, 3))

    @Task.make_task
    def fn_a(x, run_dir=None):
        return np.ones(4) * x, np.ones(4) / x

    fn_a(3)
    assert np.all(fn_a.out()[0].load() == np.full(4, 3))
    assert np.all(fn_a.out()[1].load() == np.full(4, 1 / 3))


def test_out_pandas(tmpdir):
    TerraSettings.storage_dir = tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        df = pd.DataFrame([{"a": idx, "b": idx ** 2} for idx in range(x)])
        return df

    fn_a(10)
    assert isinstance(fn_a.out().load(), pd.DataFrame)
    assert len(fn_a.out().load()) == 10


def test_inp_scalar(tmpdir):
    TerraSettings.storage_dir = tmpdir

    @Task.make_task
    def fn_b(x, run_dir=None):
        return x ** 2

    fn_b(4)
    assert fn_b.inp()["x"] == 4


def test_inp_np(tmpdir):
    TerraSettings.storage_dir = tmpdir
    x = np.ones(4)

    @Task.make_task
    def fn_a(x, run_dir=None):
        return x

    fn_a(x)
    assert np.all(fn_a.inp()["x"].load() == x)


def test_inp_pandas(tmpdir):
    TerraSettings.storage_dir = tmpdir

    df = pd.DataFrame([{"a": idx, "b": idx ** 2} for idx in range(10)])

    @Task.make_task
    def fn_a(x, run_dir=None):
        return df

    fn_a(df)
    assert len(fn_a.inp()["x"].load()) == 10
