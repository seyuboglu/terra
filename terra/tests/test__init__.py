import numpy as np
import pandas as pd
import json
from importlib import reload

import ray

import terra
from terra import Task, init_remote
import terra.database as tdb
from terra.settings import TERRA_CONFIG

TERRA_CONFIG["notify"] = False


def test_make_task(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    @Task.make_task
    def fn_a(run_dir=None):
        return None

    assert isinstance(fn_a, Task)


def test_np_pipeline(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

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

    assert np.all(out_c.load() == np.full(4, 6))


def test_pandas_pipeline(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

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
    out_c = out_c.load()
    assert isinstance(out_c, pd.DataFrame)
    assert len(out_c) == 10
    assert (out_c.a == out_c.c / out_c.b).all()


def test_scalar_pipeline(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

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

    assert np.all(out_c.load() == np.full(4, 32))


def test_nested_np_pipeline(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        return {"a": np.ones(4) * x, "b": [np.ones(4) * 2 * x, np.ones(4) * 2 * x]}

    @Task.make_task
    def fn_c(x, run_dir=None):
        return x["a"] + x["b"][0] + x["b"][0]

    fn_a(1)
    out_c = fn_c(fn_a.out())

    assert np.all(out_c.load() == np.full(4, 5))


def test_out_scalar(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

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
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

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
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        df = pd.DataFrame([{"a": idx, "b": idx ** 2} for idx in range(x)])
        return df

    fn_a(10)
    assert isinstance(fn_a.out().load(), pd.DataFrame)
    assert len(fn_a.out().load()) == 10


def test_run_table(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        return {"a": np.ones(4) * x, "b": [np.ones(4) * 2 * x, np.ones(4) * 2 * x]}

    @Task.make_task
    def fn_c(x, run_dir=None):
        return x["a"] + x["b"][0] + x["b"][0]

    fn_a(1)
    fn_c(fn_a.out())

    run_df = tdb.get_runs()
    assert len(run_df) == 2
    assert (run_df.status == "success").all()
    assert set(run_df.fn) == set(["fn_a", "fn_c"])


def test_artifact_table(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        return {"a": np.ones(4) * x, "b": [np.ones(4) * 2 * x, np.ones(4) * 2 * x]}

    @Task.make_task
    def fn_c(x, run_dir=None):
        return x["a"] + x["b"][0] + x["b"][0]

    fn_a(np.ones(4) * 2)
    fn_c(fn_a.out())

    artifact_df = tdb.get_artifact_dumps()
    assert len(artifact_df) == 5
    assert (artifact_df.type == "<class 'numpy.ndarray'>").all()


def test_artifact_load_table(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        return {"a": np.ones(4) * x, "b": [np.ones(4) * 2 * x, np.ones(4) * 2 * x]}

    @Task.make_task
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
    assert (
        set(zip(df.creating_run_id, df.loading_run_id, df.artifact_id))
        == set([(1, 2, 2), (1, 2, 3), (1, 2, 4), (2, 3, 5)])
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


def test_out_custom(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        return CustomClass(attr=x)

    @Task.make_task
    def fn_b(x, run_dir=None):
        return x.attr * 2

    fn_a(4)
    fn_b(fn_a.out())

    assert fn_b.out() == 8


def test_inp_scalar(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    @Task.make_task
    def fn_b(x, run_dir=None):
        return x ** 2

    fn_b(4)
    assert fn_b.inp()["x"] == 4


def test_inp_np(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    x = np.ones(4)

    @Task.make_task
    def fn_a(x, run_dir=None):
        return x

    fn_a(x)
    assert np.all(fn_a.inp()["x"].load() == x)


def test_inp_pandas(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    df = pd.DataFrame([{"a": idx, "b": idx ** 2} for idx in range(10)])

    @Task.make_task
    def fn_a(x, run_dir=None):
        return df

    fn_a(df)
    assert len(fn_a.inp()["x"].load()) == 10


def test_inp_custom(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    @Task.make_task
    def fn_a(x, run_dir=None):
        return x

    fn_a(CustomClass(attr=4))

    assert fn_a.inp()["x"].load().attr == 4


def test_kwargs_custom(tmpdir):
    """Functions with kwargs are a bit of an edge case"""
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    def fn_b(x, y, z):
        return x * y + z

    @Task.make_task
    def fn_a(run_dir=None, **kwargs):
        return fn_b(**kwargs)

    fn_a(x=3, y=2, z=9)

    assert fn_a.inp()["x"] == 3
    assert fn_a.inp()["y"] == 2
    assert fn_a.inp()["z"] == 9

    assert fn_a.out() == 15


def test_failure(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    @Task.make_task
    def fn_a(run_dir=None):
        raise ValueError("error")

    try:
        fn_a()
    except ValueError:
        run = tdb.get_runs(run_ids=1, df=False)[0]
        assert run.status == "failure"


def test_parallel(tmpdir):
    TERRA_CONFIG["storage_dir"] = str(tmpdir)
    terra.database.Session = (
        tdb.get_session()
    )  # need to recreate Session with new tmpdir

    init_remote()

    @Task.make_task
    def fn_a(x, run_dir=None):
        return x

    obj_refs = []
    inps = list(range(1, 25))
    for x in inps:
        obj_refs.append(fn_a.remote(x, return_run_id=True, terra_config=TERRA_CONFIG))

    run_ids, artifacts = zip(*[ray.get(obj_ref) for obj_ref in obj_refs])

    # check that all inputs (inps) are returned, since (fn_a is identity)
    assert set(artifacts) == set(inps)

    # check that all inputs (inps) are returned, since (fn_a is identity)
    assert set([fn_a.out(run_id=run_id) for run_id in run_ids]) == set(inps)

    # check that we have a run_id for each input
    assert set([run.id for run in tdb.get_runs(df=False)]) == set(
        range(1, len(inps) + 1)
    )
