import importlib
import json
import os
import pickle
import shutil
import uuid
from functools import lru_cache
from typing import Union

import meerkat as mk
import numpy as np
from pandas import DataFrame, read_csv

import terra.database as tdb
from terra.utils import ensure_dir_exists


class Artifact:
    def _get_path(self):
        ensure_dir_exists(os.path.join(self.run_dir, "artifacts"))
        return os.path.join(self.run_dir, "artifacts", self.key)

    def load(self, run_id: int = None):

        if run_id is not None:
            # if a run_id is supplied, log the load to the loads table
            session = tdb.Session()
            entry = tdb.ArtifactLoad(artifact_id=self.id, loading_run_id=run_id)
            session.add(entry)
            session.commit()
            session.close()

        return generalized_read(self._get_path(), self.type)

    @classmethod
    def dump(cls, value, run_dir: str):
        artifact = cls()
        artifact.run_dir = run_dir
        artifact.key = uuid.uuid4().hex
        artifact.type = type(value)
        path = artifact._get_path()
        generalized_write(value, path)

        # add to artifacts table
        session = tdb.Session()
        entry = tdb.ArtifactDump(
            creating_run_id=artifact.run_id, path=path, type=str(artifact.type)
        )
        session.add(entry)
        session.commit()
        artifact.id = entry.id
        session.close()

        return artifact

    @staticmethod
    def is_serialized_artifact(dct: dict):
        return (
            "__id__" in dct
            and "__run_dir__" in dct
            and "__key__" in dct
            and "__type__" in dct
        )

    def serialize(self):
        return self.__getstate__()

    def __getstate__(self):
        return {
            "__run_dir__": self.run_dir,
            "__key__": self.key,
            "__type__": self.type,
            "__id__": self.id,
        }

    @classmethod
    def deserialize(cls, dct):
        artifact = cls()
        artifact.__setstate__(dct)
        return artifact

    def __setstate__(self, state):
        self.run_dir = state["__run_dir__"]
        self.key = state["__key__"]
        self.id = state["__id__"]
        self.type = state["__type__"]

    def rm(self):
        path = self._get_path()
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

        # update artifact_dumps table
        session = tdb.Session()
        session.query(tdb.ArtifactDump).filter(tdb.ArtifactDump.id == self.id).update(
            {"rm": True}
        )

        session.commit()

    def __str__(self):
        return str(self.serialize())

    def __repr__(self):
        return str(self.serialize())

    @property
    def run_id(self):
        return int(os.path.basename(self.run_dir))


def load_nested_artifacts(obj: Union[list, dict], run_id: int = None):
    if isinstance(obj, list):
        return [load_nested_artifacts(v, run_id=run_id) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(load_nested_artifacts(v, run_id=run_id) for v in obj)
    elif isinstance(obj, dict):
        return {k: load_nested_artifacts(v, run_id=run_id) for k, v in obj.items()}
    elif isinstance(obj, Artifact):
        return obj.load(run_id=run_id)
    else:
        return obj


def get_nested_artifact_paths(obj: Union[list, dict]):
    if isinstance(obj, list) or isinstance(obj, tuple):
        arts = []
        for v in obj:
            arts.extend(get_nested_artifact_paths(v))
        return arts
    elif isinstance(obj, dict):
        arts = []
        for v in obj.values():
            arts.extend(get_nested_artifact_paths(v))
        return arts
    elif isinstance(obj, Artifact):
        return [obj._get_path()]
    return []


def rm_nested_artifacts(obj: Union[list, dict]):
    if isinstance(obj, list):
        [rm_nested_artifacts(v) for v in obj]
    elif isinstance(obj, tuple):
        (rm_nested_artifacts(v) for v in obj)
    elif isinstance(obj, dict):
        {k: rm_nested_artifacts(v) for k, v in obj.items()}
    elif isinstance(obj, Artifact):
        obj.rm()


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


class TerraDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "__module__" in dct and "__name__" in dct:
            try:
                module = importlib.import_module(dct["__module__"])
                return getattr(module, dct["__name__"])
            except ModuleNotFoundError:
                # sometimes the names of modules, functions and classes change, we still
                #  want to be able to load Task inputs and outputs that reference them
                class ExtinctModule:
                    __name__ = dct["__name__"]
                    __module__ = dct["__module__"]

                return ExtinctModule

        if Artifact.is_serialized_artifact(dct):
            return Artifact.deserialize(dct)
        return dct


reader_registry = {}


def reader(read_type: type):
    def register_reader(fn):
        reader_registry[read_type] = fn
        return fn

    return register_reader


@lru_cache
def generalized_read(path, read_type: type):
    if hasattr(read_type, "__terra_read__"):
        return read_type.__terra_read__(path)

    elif read_type in reader_registry:
        return reader_registry[read_type](path)

    else:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError:
            raise ValueError(f"Object type {read_type} not pickleable.")


writer_registry = {}


def writer(write_type: type):
    def register_writer(fn):
        writer_registry[write_type] = fn
        return fn

    return register_writer


def generalized_write(out, path):
    if hasattr(out, "__terra_write__"):
        path = out.__terra_write__(path)
    elif type(out) in writer_registry:
        path = writer_registry[type(out)](out, path)
    else:
        try:
            with open(path, "wb") as f:
                pickle.dump(out, f)
        except pickle.PicklingError:
            raise ValueError(f"Type {type(out)} not pickleable.")

    return path


@writer(DataFrame)
def write_dataframe(out, path):
    out.to_csv(path, index=False)
    return path


@reader(DataFrame)
def read_dataframe(path):
    return read_csv(path)


@writer(np.ndarray)
def write_nparray(out, path):
    with open(path, "wb") as f:
        np.save(f, out)
    return path


@reader(np.ndarray)
def read_nparray(path):
    return np.load(path, allow_pickle=True)


@writer(mk.DataPanel)
def write_datapanel(out, path):
    out.write(path)
    return path


@reader(mk.DataPanel)
def read_datapanel(path):
    return mk.DataPanel.read(path)
