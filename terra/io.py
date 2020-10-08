import os
import json
import uuid
from typing import Union, Iterable, Mapping
import importlib
import pickle

import pandas as pd
import numpy as np

from terra.utils import ensure_dir_exists


class Artifact:
    def _get_path(self):
        ensure_dir_exists(os.path.join(self.run_dir, "artifacts"))
        return os.path.join(self.run_dir, "artifacts", self.id)

    def serialize(self):
        return {"__run_dir__": self.run_dir, "__id__": self.id, "__type__": self.type}

    @classmethod
    def deserialize(cls, dct):
        artifact = cls()
        artifact.run_dir = dct["__run_dir__"]
        artifact.id = dct["__id__"]
        artifact.type = dct["__type__"]
        return artifact

    def load(self):
        return generalized_read(self._get_path(), self.type)

    @classmethod
    def dump(cls, value, run_dir: str):
        artifact = cls()
        artifact.run_dir = run_dir
        artifact.id = uuid.uuid4().hex
        artifact.type = type(value)
        generalized_write(value, artifact._get_path())
        return artifact

    @staticmethod
    def is_serialized_artifact(dct: dict):
        return "__run_dir__" in dct and "__id__" in dct and "__type__" in dct

    def __str__(self):
        return str(self.serialize())


def load_nested_artifacts(obj: Union[list, dict]):
    if isinstance(obj, list):
        return [load_nested_artifacts(v) for v in obj]
    elif isinstance(obj, tuple):
        return (load_nested_artifacts(v) for v in obj)
    elif isinstance(obj, dict):
        return {k: load_nested_artifacts(v) for k, v in obj.items()}
    elif isinstance(obj, Artifact):
        return obj.load()
    else:
        return obj


def json_dump(obj: Union[dict, list], path: str, run_dir: str):
    with open(path, "w") as f:
        encoder = TerraEncoder(run_dir=run_dir, indent=4)
        encoded = encoder.encode(obj)
        f.write(encoded)
        decoder = TerraDecoder()
        return decoder.decode(encoded)


def json_load(path: str):
    with open(path) as f:
        decoder = TerraDecoder()
        return decoder.decode(f.read())


class TerraEncoder(json.JSONEncoder):
    def __init__(self, run_dir: str, *args, **kwargs):
        json.JSONEncoder.__init__(self, *args, **kwargs)
        self.run_dir = run_dir

    def default(self, obj):
        if (callable(obj) or isinstance(obj, type)) and hasattr(obj, "__name__"):
            return {"__module__": obj.__module__, "__name__": obj.__name__}
        elif isinstance(obj, Artifact):
            return obj.serialize()
        else:
            artifact = Artifact.dump(value=obj, run_dir=self.run_dir)
            return artifact.serialize()

        return json.JSONEncoder.default(self, obj)


class TerraDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "__module__" in dct and "__name__" in dct:
            module = importlib.import_module(dct["__module__"])
            return getattr(module, dct["__name__"])
        if Artifact.is_serialized_artifact(dct):
            return Artifact.deserialize(dct)
        return dct


reader_registry = {}


def reader(read_type: type):
    def register_reader(fn):
        reader_registry[read_type] = fn
        return fn

    return register_reader


def generalized_read(path, read_type: type):
    if hasattr(read_type, "__terra_read__"):
        return read_type.__terra_read__(path)

    elif read_type in reader_registry:
        return reader_registry[read_type](path)

    else:
        try:
            new_path = path + ".pkl"
            with open(new_path, "rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError as e:
            raise ValueError(f"Object type {read_type} not pickleable.")


writer_registry = {}


def writer(write_type: type):
    def register_writer(fn):
        writer_registry[write_type] = fn
        return fn

    return register_writer


def generalized_write(out, path):
    if hasattr(out, "__terra_write__"):
        new_path = out.__terra_write__(path)
    elif type(out) in writer_registry:
        new_path = writer_registry[type(out)](out, path)
    else:
        try:
            new_path = path + ".pkl"
            with open(new_path, "wb") as f:
                pickle.dump(out, f)
        except pickle.PicklingError as e:
            raise ValueError(f"Type {type(out)} not pickleable.")

    if new_path is None:
        return path
    else:
        return new_path


@writer(pd.DataFrame)
def write_dataframe(out, path):
    path = path + ".csv" if not path.endswith(".csv") else path
    out.to_csv(path, index=False)
    return path


@reader(pd.DataFrame)
def read_dataframe(path):
    path = path + ".csv" if not path.endswith(".csv") else path
    return pd.read_csv(path)


@writer(np.ndarray)
def write_nparray(out, path):
    path = path + ".npy" if not path.endswith(".npy") else path
    np.save(path, out)
    return path


@reader(np.ndarray)
def read_nparray(path):
    path = path + ".npy" if not path.endswith(".npy") else path
    return np.load(path)
