""" The `task` module provides a framework for running reproducible analyses."""

import importlib
import json
import os
import re
import traceback
from datetime import datetime
from functools import wraps
from inspect import getcallargs

import numpy as np
import pandas as pd

from terra.git import log_git_status
from terra.utils import ensure_dir_exists
from terra.logging import init_logging


class TerraSettings:

    storage_dir = "/Users/sabrieyuboglu/code/terra/test_storage_dir"


class Task:
    @classmethod
    def make_task(cls, fn):
        task = cls()
        task.fn = cls._get_wrapper(fn)
        return task

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    @staticmethod
    def _get_wrapper(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            silence_task = kwargs.pop("silence_task", False)

            args_dict = getcallargs(fn, *args, **kwargs)

            if not silence_task:
                process_dir = args_dict.get("process_dir", None)
                if process_dir is None:
                    process_dir = _get_process_dir(fn)
                run_dir = _get_next_run_dir(process_dir)

                params_dict = {
                    "git": log_git_status(run_dir),
                    "notebook": "get_ipython" in globals().keys(),
                    "start_time": datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"),
                    "module": fn.__module__,
                    "fn": fn.__name__,
                    "kwargs": args_dict,
                }

                write_input(run_dir, params_dict)

                logger = init_logging(os.path.join(run_dir, "process.log"))
                args_dict["process_dir"] = run_dir
                print(f"process: running in directory {run_dir}")

                # load node inputs
                for key, value in args_dict.items():
                    if isinstance(value, Task):
                        str_rep = json.dumps(value.serialize(), indent=2)
                        print(
                            f"Loading process output: {str_rep} \n and passing to parameter '{key}'"
                        )
                        args_dict[key] = value.load()
                    if isinstance(value, Task):
                        str_rep = json.dumps(value.serialize(), indent=2)
                        print(
                            f"Loading process output: {str_rep} \n and passing to parameter '{key}'"
                        )
                        args_dict[key] = value.load()

                try:
                    out = fn(**args_dict)
                except (Exception, KeyboardInterrupt) as e:
                    time = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
                    except_dir = os.path.join(process_dir, "_runs", "_except", time)
                    # TODO: decide what to do with exceptions
                    # shutil.move(run_dir, except_dir)
                    print(traceback.format_exc())
                    raise e
                else:
                    if out is not None:
                        _write_output(out, run_dir)
                del logger

                return out
            else:
                return fn(**args_dict)

        return wrapper


def _write_output(out, run_dir, suffix=0, _base=True):
    if _base and isinstance(out, tuple):
        for idx, sub_out in enumerate(out):
            _write_output(sub_out, run_dir, suffix=idx, _base=False)
    else:
        _generalized_write(out, path=os.path.join(run_dir, f"out_{suffix}"))


def read_output(fn=None, process_dir=None, run_idx=None):
    if (fn is None) == (process_dir is None):
        raise ValueError(
            "Must provide exactly one of fn and process_dir."
            "Cannot provide both. Cannot provide neither."
        )
    if fn is None:
        process_dir = process_dir
    else:
        process_dir = _get_process_dir(fn)
    runs_dir = os.path.join(process_dir, "_runs")
    run_idx = _get_latest_run_idx(runs_dir) if run_idx is None else run_idx

    if run_idx is None:
        raise ValueError(f"The process {process_dir} has not been run.")

    run_dir = os.path.join(runs_dir, str(run_idx))
    if not os.path.isdir(run_dir):
        raise ValueError(
            f"The run idx {run_idx} does not exist for process {process_dir}."
        )

    outs = sorted(
        [
            (
                int(re.search(r"out_(\d*).", out).group(1)),
                _generalized_read(os.path.join(run_dir, out)),
            )
            for out in os.listdir(run_dir)
            if out.startswith("out")
        ]
    )

    if len(outs) == 1:
        return outs[0][-1]
    else:
        return tuple(list(zip(*outs))[-1])


def write_input(run_dir, input_dict):
    # encode
    writeable_dict = input_dict.copy()
    writeable_dict["kwargs"] = {}

    for key, value in input_dict["kwargs"].items():
        if type(value) in writer_registry:
            path = os.path.join(run_dir, f"in_{key}")
            path = _generalized_write(value, path)
            writeable_dict["kwargs"][key] = {"__path__": path, "__type__": type(value)}

        else:
            writeable_dict["kwargs"][key] = value

    write_dict(writeable_dict, os.path.join(run_dir, "params.json"))


def read_input(process_dir, run_idx):
    run_dir = _get_run_dir(process_dir, run_idx)
    inp = read_dict(os.path.join(run_dir, "params.json"))
    return inp


class ProcessEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj) or isinstance(obj, type):
            return {"__module__": obj.__module__, "__name__": obj.__name__}
        if isinstance(obj, ProcessOutput):
            return obj.serialize()
        if isinstance(obj, ProcessInput):
            return obj.serialize()
        return json.JSONEncoder.default(self, obj)


class ProcessDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "__module__" in dct and "__name__" in dct:
            module = importlib.import_module(dct["__module__"])
            return getattr(module, dct["__name__"])
        if "__path__" in dct and "__type__" in dct:
            # TODO: check type
            return _generalized_read(dct["__path__"])
        if "__process_dir__" in dct and "__out_idx__" in dct:
            return ProcessOutput.deserialize(dct)
        if "__process_dir__" in dct and "__parameter_name__" in dct:
            return ProcessInput.deserialize(dct)
        return dct


def _get_next_run_dir(process_dir):
    """get the next available run directory (e.g. "_runs/0", "_runs/1", "_runs/2") in base_dir"""
    latest_idx = _get_latest_run_idx(process_dir)
    idx = latest_idx + 1 if latest_idx is not None else 0
    run_dir = _get_run_dir(process_dir, idx)
    ensure_dir_exists(run_dir)
    return run_dir


def _get_run_dir(process_dir, idx):
    print(process_dir)
    run_dir = os.path.join(process_dir, "_runs", str(idx))
    return run_dir


def _get_latest_run_idx(process_dir):
    base_dir = os.path.join(process_dir, "_runs")
    if not os.path.isdir(base_dir):
        return None

    existing_dirs = [
        int(idx) for idx in os.listdir(base_dir) if idx.split("_")[0].isdigit()
    ]

    if not existing_dirs:
        return None
    return max(existing_dirs)


def _get_process_dir(fn: callable):
    """Get """
    process_dir = os.path.join(
        TerraSettings.storage_dir,
        "tasks",
        *fn.__module__.split(".")[1:],
        fn.__name__,
    )
    return process_dir


reader_registry = {}


def reader(file_ext: str):
    def _reader(fn):
        reader_registry[file_ext] = fn
        return fn

    return _reader


def _generalized_read(path):
    _, ext = os.path.splitext(path)
    if ext not in reader_registry:
        raise ValueError(f"File extension {ext} not supported.")
    else:
        return reader_registry[ext](path)


writer_registry = {}


def writer(write_type: type):
    def _writer(fn):
        writer_registry[write_type] = fn
        return fn

    return _writer


def _generalized_write(out, path):
    if type(out) not in writer_registry:
        raise ValueError(f"Type {type(out)} not supported.")

    new_path = writer_registry[type(out)](out, path)

    if new_path is None:
        return path
    else:
        return new_path


@writer(pd.DataFrame)
def write_dataframe(out, path):
    path = path + ".csv" if not path.endswith(".csv") else path
    out.to_csv(path, index=False)
    return path


@reader(".csv")
def read_dataframe(path):
    return pd.read_csv(path)


@writer(dict)
def write_dict(out, path):
    path = path + ".json" if not path.endswith(".json") else path
    with open(path, "w") as f:
        json.dump(out, f, indent=4, cls=ProcessEncoder)
    return path


@reader(".json")
def read_dict(path):
    with open(path) as f:
        return json.load(f, cls=ProcessDecoder)


@writer(np.ndarray)
def write_nparray(out, path):
    path = path + ".npy" if not path.endswith(".npy") else path
    np.save(path, out)
    return path


@reader(".npy")
def read_nparray(path):
    return np.load(path)
