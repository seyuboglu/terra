import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import terra.database as tdb
from terra import Task

from .testbed import BaseTestBed


@pytest.fixture()
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@BaseTestBed.parametrize()
def test_make_task(testbed: BaseTestBed):
    @Task
    def fn_a(run_dir=None):
        return None

    assert isinstance(fn_a, Task)


@BaseTestBed.parametrize()
def test_np_pipeline(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        return np.ones(4) * x

    @Task
    def fn_b(x, run_dir=None):
        return np.ones(4) * (x ** 2)

    @Task
    def fn_c(x, y, run_dir=None):
        return x + y

    fn_a(2)
    fn_b(2)
    out_c = fn_c(fn_a.out(), fn_b.out())

    assert np.all(out_c.load() == np.full(4, 6))


@BaseTestBed.parametrize()
def test_pandas_pipeline(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        df = pd.DataFrame([{"a": idx, "b": idx ** 2} for idx in range(1, x + 1)])
        return df

    fn_a(10)

    @Task
    def fn_b(x, run_dir=None):
        df = pd.DataFrame([{"a": idx, "c": idx ** 3} for idx in range(1, x + 1)])
        return df

    fn_b(10)

    @Task
    def fn_c(x, y, run_dir=None):
        return x.merge(right=y, on="a")

    out_c = fn_c(fn_a.out(), fn_b.out())
    out_c = out_c.load()
    assert isinstance(out_c, pd.DataFrame)
    assert len(out_c) == 10
    assert (out_c.a == out_c.c / out_c.b).all()


@BaseTestBed.parametrize()
def test_scalar_pipeline(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        return np.ones(4) * x

    @Task
    def fn_b(x, run_dir=None):
        return x ** 2

    @Task
    def fn_c(x, y, run_dir=None):
        return x * y

    fn_a(2)
    fn_b(4)
    out_c = fn_c(fn_a.out(), fn_b.out())

    assert np.all(out_c.load() == np.full(4, 32))


@BaseTestBed.parametrize()
def test_nested_np_pipeline(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        return {"a": np.ones(4) * x, "b": [np.ones(4) * 2 * x, np.ones(4) * 2 * x]}

    @Task
    def fn_c(x, run_dir=None):
        return x["a"] + x["b"][0] + x["b"][0]

    fn_a(1)
    out_c = fn_c(fn_a.out())

    assert np.all(out_c.load() == np.full(4, 5))


@BaseTestBed.parametrize()
def test_out_scalar(testbed: BaseTestBed):
    @Task
    def fn_b(x, run_dir=None):
        return x ** 2

    fn_b(4)
    assert fn_b.out() == 16

    @Task
    def fn_a(x, run_dir=None):
        return x ** 2, x ** 3

    fn_a(4)
    a, b = fn_a.out()
    assert a == 16
    assert b == 64


@BaseTestBed.parametrize()
def test_out_np(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        return np.ones(4) * x

    fn_a(3)
    assert np.all(fn_a.out().load() == np.full(4, 3))

    @Task
    def fn_b(x, run_dir=None):
        return np.ones(4) * x, np.ones(4) / x

    fn_b(3)
    assert np.all(fn_b.out()[0].load() == np.full(4, 3))
    assert np.all(fn_b.out()[1].load() == np.full(4, 1 / 3))


@BaseTestBed.parametrize()
def test_out_pandas(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        df = pd.DataFrame([{"a": idx, "b": idx ** 2} for idx in range(x)])
        return df

    fn_a(10)
    assert isinstance(fn_a.out().load(), pd.DataFrame)
    assert len(fn_a.out().load()) == 10


@BaseTestBed.parametrize()
def test_run_table(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        return {"a": np.ones(4) * x, "b": [np.ones(4) * 2 * x, np.ones(4) * 2 * x]}

    print(fn_a.__qualname__)

    @Task
    def fn_c(x, run_dir=None):
        return x["a"] + x["b"][0] + x["b"][0]

    fn_a(1)
    fn_c(fn_a.out())

    run_df = tdb.get_runs()
    assert len(run_df) == 2
    assert (run_df.status == "success").all()
    assert set(run_df.fn) == set(
        ["test_run_table.<locals>.fn_a", "test_run_table.<locals>.fn_c"]
    )


class ConstructorClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y


@BaseTestBed.parametrize()
def test_constructor_task(testbed: BaseTestBed):
    artifact = Task(ConstructorClass)(x=15, y=4)

    obj = artifact.load()
    assert isinstance(obj, ConstructorClass)
    assert obj.x == 15
    assert obj.y == 4

    run_df = tdb.get_runs()
    assert len(run_df) == 1
    assert (run_df.status == "success").all()
    print(ConstructorClass.__module__)
    assert set(run_df.module) == set(["terra.tests.test__init__"])
    assert set(run_df.fn) == set(["ConstructorClass"])


@BaseTestBed.parametrize()
def test_artifact_table(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        return {"a": np.ones(4) * x, "b": [np.ones(4) * 2 * x, np.ones(4) * 2 * x]}

    @Task
    def fn_c(x, run_dir=None):
        return x["a"] + x["b"][0] + x["b"][0]

    fn_a(np.ones(4) * 2)
    fn_c(fn_a.out())

    artifact_df = tdb.get_artifact_dumps()
    assert len(artifact_df) == 5
    assert (artifact_df.type == "<class 'numpy.ndarray'>").all()


@BaseTestBed.parametrize()
def test_artifact_load_table(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        return {"a": np.ones(4) * x, "b": [np.ones(4) * 2 * x, np.ones(4) * 2 * x]}

    @Task
    def fn_c(x, run_dir=None):
        return x["a"] + x["b"][0] + x["b"][0]

    fn_a(np.ones(4) * 2)
    fn_c(fn_a.out())
    fn_a(fn_c.out())

    run_df = tdb.get_runs()
    artifact_df = tdb.get_artifact_dumps()
    artifact_load_df = tdb.get_artifact_loads()
    df = artifact_load_df.merge(
        artifact_df[["creating_run_id", "id"]], left_on="artifact_id", right_on="id"
    )
    df = df.merge(run_df[["id", "fn"]], left_on="creating_run_id", right_on="id")

    assert len(df) == 4
    assert set(zip(df.creating_run_id, df.loading_run_id, df.artifact_id)) == set(
        [(1, 2, 2), (1, 2, 3), (1, 2, 4), (2, 3, 5)]
    )


class CustomClass:
    def __init__(self, attr: int):
        self.attr = attr

    @classmethod
    def __terra_read__(cls, path):
        with open(path, "r") as f:
            dct = json.load(f)
        return cls(dct["attr"])

    def __terra_write__(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)


@BaseTestBed.parametrize()
def test_out_custom(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        return CustomClass(attr=x)

    @Task
    def fn_b(x, run_dir=None):
        return x.attr * 2

    fn_a(4)
    fn_b(fn_a.out())

    assert fn_b.out() == 8


@BaseTestBed.parametrize()
def test_inp_scalar(testbed: BaseTestBed):
    @Task
    def fn_b(x, run_dir=None):
        return x ** 2

    fn_b(4)
    assert fn_b.inp()["x"] == 4


@BaseTestBed.parametrize()
def test_inp_np(testbed: BaseTestBed):

    x = np.ones(4)

    @Task
    def fn_a(x, run_dir=None):
        return x

    fn_a(x)
    assert np.all(fn_a.inp()["x"].load() == x)


@BaseTestBed.parametrize()
def test_inp_pandas(testbed: BaseTestBed):

    df = pd.DataFrame([{"a": idx, "b": idx ** 2} for idx in range(10)])

    @Task
    def fn_a(x, run_dir=None):
        return df

    fn_a(df)
    assert len(fn_a.inp()["x"].load()) == 10


@BaseTestBed.parametrize()
def test_inp_custom(testbed: BaseTestBed):
    @Task
    def fn_a(x, run_dir=None):
        return x

    fn_a(CustomClass(attr=4))

    assert fn_a.inp()["x"].load().attr == 4


@BaseTestBed.parametrize()
def test_kwargs_custom(testbed: BaseTestBed):
    """Functions with kwargs are a bit of an edge testbed: BaseTestBed"""

    def fn_b(x, y, z):
        return x * y + z

    @Task
    def fn_a(run_dir=None, **kwargs):
        return fn_b(**kwargs)

    fn_a(x=3, y=2, z=9)

    assert fn_a.inp()["x"] == 3
    assert fn_a.inp()["y"] == 2
    assert fn_a.inp()["z"] == 9

    assert fn_a.out() == 15


@BaseTestBed.parametrize()
def test_failure(testbed: BaseTestBed):
    @Task
    def fn_a(run_dir=None):
        raise ValueError("error")

    try:
        fn_a()
    except ValueError:
        run = tdb.get_runs(run_ids=1, df=False)[0]
        assert run.status == "failure"
