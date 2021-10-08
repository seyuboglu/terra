""" The `task` module provides a framework for running reproducible analyses."""
from __future__ import annotations

import os
import platform
import socket
import sys
import traceback
from datetime import datetime
from inspect import getcallargs
from typing import Collection

import terra.database as tdb
from terra.dependencies import get_dependencies
from terra.git import log_fn_source, log_git_status
from terra.logging import init_logging
from terra.notify import (
    init_task_notifications,
    notify_task_completed,
    notify_task_error,
)
from terra.settings import TERRA_CONFIG
from terra.utils import ensure_dir_exists


class Task:
    def __init__(
        self,
        fn: callable,
        no_dump_args: Collection[str] = None,
        no_load_args: Collection[str] = None,
    ):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__module__ = fn.__module__
        self.task_dir = self._get_task_dir(self)
        self.no_dump_args = no_dump_args
        self.no_load_args = no_load_args

    @classmethod
    def make(
        cls,
        no_dump_args: Collection[str] = None,
        no_load_args: Collection[str] = None,
    ) -> callable:
        def _make(fn: callable) -> Task:
            return cls(fn=fn, no_dump_args=no_dump_args, no_load_args=no_load_args)

        return _make

    @staticmethod
    def _get_task_dir(task: Task):
        return _get_task_dir(task.__module__, task.__name__)

    def run_dir(self, run_id: int = None):
        if run_id is None:
            return _get_run_dir(self.task_dir, self.last_run_id)
        return _get_run_dir(self.task_dir, run_id)

    @property
    def last_run_id(self):
        run_id = _get_latest_run_id(self.task_dir)
        return run_id

    def _get_latest_successful_run_id(self):
        return tdb.get_runs(
            fns=self.__name__,
            modules=self.__module__,
            statuses="success",
            df=False,
            limit=1,
        )[0].id

    def inp(self, run_id: int = None, load: bool = False):
        from terra.io import json_load, load_nested_artifacts

        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)
        inps = json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), "inputs.json"
            )
        )
        return load_nested_artifacts(inps) if load else inps

    def out(self, run_id: int = None, load: bool = False):
        from terra.io import json_load, load_nested_artifacts

        if run_id is None:
            # unsuccessful run_ids won't have an output
            run_id = self._get_latest_successful_run_id()
        outs = json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), "outputs.json"
            )
        )

        return load_nested_artifacts(outs) if load else outs

    def get(self, run_id: int = None, group_name: str = "outputs", load: bool = False):
        from terra.io import json_load, load_nested_artifacts

        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)
        artifacts = json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), f"{group_name}.json"
            )
        )
        return load_nested_artifacts(artifacts) if load else artifacts

    def get_artifacts(
        self, run_id: int = None, group_name: str = "outputs", load: bool = False
    ):
        return self.get(group_name=group_name, run_id=run_id, load=load)

    def get_log(self, run_id: int = None):
        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)

        log_path = os.path.join(
            _get_run_dir(task_dir=self.task_dir, idx=run_id), "task.log"
        )

        with open(log_path, mode="r") as f:
            return f.read()

    def get_meta(self, run_id: int = None):
        from terra.io import json_load

        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)

        artifacts = json_load(
            os.path.join(_get_run_dir(task_dir=self.task_dir, idx=run_id), "meta.json")
        )
        return artifacts

    def rm_artifacts(self, group_name: str, run_id: int):
        """Chose not to make static as that would be potentially dangerous."""
        from terra.io import json_load, rm_nested_artifacts

        artifacts = json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), f"{group_name}.json"
            )
        )
        rm_nested_artifacts(artifacts)

    def get_runs(self):
        return tdb.get_runs(fns=self.__name__)

    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)

    def _run(self, *args, **kwargs):
        from terra.io import (
            TerraEncoder,
            TerraEncodingError,
            json_dump,
            load_nested_artifacts,
        )

        # unpack optional Task modifiers
        # `return_run_id` instructs terra to return (run_id, returned_obj)
        return_run_id = kwargs.pop("return_run_id", False)
        # `skip_terra_cache` forces terra to rerun the task
        skip_terra_cache = kwargs.pop("skip_terra_cache", False)
        # `silence_task` instructs terra not to record the run
        silence_task = kwargs.pop("silence_task", False)

        args_dict = getcallargs(self.fn, *args, **kwargs)

        if "kwargs" in args_dict:
            args_dict.update(args_dict.pop("kwargs"))

        # distributed pytorch lightning (ddp) relies on rerunning the entire training
        # script for each node (see https://github.com/PyTorchLightning/pytorch-lightning/blob/3bdc0673ea5fcb10035d783df0d913be4df499b6/pytorch_lightning/plugins/training_type/ddp.py#L163). # noqa: E501
        # We do not want terra creating a separate task run for each process, so we
        # check if we're on node 0 and rank 0, and if not, we silence the task.
        if ("LOCAL_RANK" in os.environ and "NODE_RANK" in os.environ) and (
            os.environ["LOCAL_RANK"] != 0 or os.environ["NODE_RANK"] != 0
        ):
            silence_task = True
            args_dict["run_dir"] = os.environ["RANK_0_RUN_DIR"]

        if silence_task:
            args_dict = load_nested_artifacts(args_dict)
            return self.fn(**args_dict)

        # `terra_config` updates the terra config
        if "terra_config" in args_dict:
            TERRA_CONFIG.update(args_dict.pop("terra_config"))
            tdb.Session = tdb.get_session()  # need to recreate db session

        args_to_dump = (
            args_dict
            if self.no_dump_args is None
            else {
                k: ("__skipped__" if k in self.no_dump_args else v)
                for k, v in args_dict.items()
            }
        )
        # check cache for previous run
        if not skip_terra_cache:
            # try encoding inputs here so that we can
            try:
                encoder = TerraEncoder(indent=4)
                encoded_inputs = encoder.encode(args_to_dump)
            except TerraEncodingError:
                encoded_inputs = None
            else:
                input_hash = tdb.hash_inputs(encoded_inputs)
                cache_run_id = tdb.check_input_hash(
                    input_hash, fn=self.__name__, module=self.__module__
                )
                if cache_run_id is not None:
                    # cache hit – return the output of the previous run
                    print(
                        f"cache hit –> task: {self.fn.__name__}, run_id={cache_run_id}",
                        flush=True,
                    )
                    out = self.out(run_id=cache_run_id)
                    if return_run_id:
                        out = (cache_run_id, out) if out is not None else cache_run_id
                    return out
        else:
            encoded_inputs = None
        
        session = tdb.Session()

        meta_dict = {
            "notebook": "get_ipython" in globals().keys(),
            "start_time": datetime.now(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "module": self.fn.__module__,
            "fn": self.fn.__name__,
            "python_version": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", None),
        }

        # add run to terra db
        run = tdb.Run(status="in_progress", **meta_dict)
        session.add(run)
        tdb.safe_commit(session)
        try:
            run_id = run.id
            run_dir = _get_run_dir(self.task_dir, run_id)

            # distributed pytorch lightning (ddp) requires that the child processes
            # share the same directories for logging and checkpointing see
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/5319, so
            # we have to save the main run_dir as an environment variable
            os.environ["RANK_0_RUN_DIR"] = run_dir

            if os.path.exists(run_dir):
                raise ValueError(f"Run already exists at {run_dir}.")
            ensure_dir_exists(run_dir)
            run.run_dir = run_dir
            git_status = log_git_status(run_dir)

            run.git_commit = git_status["commit_hash"]
            run.git_dirty = len(git_status["dirty"]) > 0
            if self.fn.__module__ == "__main__":
                try:
                    log_fn_source(run_dir=run_dir, fn=self.fn)
                except OSError:
                    print("Could not log source code.")

            tdb.safe_commit(session)

            # write additional metadata
            meta_dict.update(
                {
                    "git": git_status,
                    "start_time": meta_dict["start_time"].strftime(
                        "%y-%m-%d_%H-%M-%S-%f"
                    ),
                    "dependencies": get_dependencies(),
                    "terra_config": TERRA_CONFIG,
                }
            )
            json_dump(meta_dict, os.path.join(run_dir, "meta.json"), run_dir=run_dir)

            # write inputs
            with open(os.path.join(run_dir, "inputs.json"), "w") as f:
                if encoded_inputs is None:
                    # if we couldn't encode the inputs before getting a run_dir
                    # (because there were artifacts that needed to be dumped) do it here
                    # and getthe input hash
                    encoder = TerraEncoder(indent=4, run_dir=run_dir)
                    encoded_inputs = encoder.encode(args_to_dump)
                    input_hash = tdb.hash_inputs(encoded_inputs)

                f.write(encoded_inputs)

            init_logging(os.path.join(run_dir, "task.log"))

            init_task_notifications(run_id=run_id)

            print(f"task: {self.fn.__name__}, run_id={run_id}", flush=True)

            if "run_dir" in args_dict:
                args_dict["run_dir"] = run_dir

            # load node inputs
            if self.no_load_args is not None:
                args_dict = {
                    **{k: args_dict[k] for k in self.no_load_args},
                    **load_nested_artifacts(
                        {
                            k: v
                            for k, v in args_dict.items()
                            if k not in self.no_load_args
                        },
                        run_id=run_id,
                    ),
                }
            else:
                args_dict = load_nested_artifacts(args_dict, run_id=run_id)

            # run function
            out = self.fn(**args_dict)

            # write outputs
            if out is not None:
                out = json_dump(
                    out, os.path.join(run_dir, "outputs.json"), run_dir=run_dir
                )

            # log success
            notify_task_completed(run_id)
            run.input_hash = input_hash
            run.status = "success"
            run.end_time = datetime.now()
            tdb.safe_commit(session)

        except (Exception, KeyboardInterrupt) as e:
            msg = traceback.format_exc()
            notify_task_error(run_id, msg)
            run.status = (
                "interrupted" if isinstance(e, KeyboardInterrupt) else "failure"
            )
            run.input_hash = input_hash
            run.end_time = datetime.now()
            tdb.safe_commit(session)
            print(msg)
            session.close()
            raise e

        if return_run_id:
            out = (int(run_id), out) if out is not None else int(run_id)
        session.close()
        return out

    @staticmethod
    def dump(artifacts: dict, run_dir: str, group_name: str, overwrite: bool = False):
        from terra.io import json_dump

        if group_name == "outputs" or group_name == "inputs":
            raise ValueError('"outputs" and "inputs" are reserved artifact group names')

        path = os.path.join(run_dir, f"{group_name}.json")
        if os.path.exists(path):
            if overwrite:
                # need to remove the artifacts in the group
                from terra.io import json_load, rm_nested_artifacts

                old_artifacts = json_load(path)
                rm_nested_artifacts(old_artifacts)
                os.remove(path)
            else:
                raise ValueError(f"Artifact group '{group_name}' already exists.")

        json_dump(artifacts, path, run_dir=run_dir)


def get_run_dir(run_id: int):
    runs = tdb.get_runs(run_ids=run_id, df=False)
    if not runs:
        raise ValueError("Could not find run with `run_id={run_id}`.")
    return runs[0].run_dir


def inp(run_id: int, load: bool = False):
    from terra.io import json_load, load_nested_artifacts

    run_dir = get_run_dir(run_id)
    inps = json_load(os.path.join(run_dir, "inputs.json"))
    return load_nested_artifacts(inps) if load else inps


def out(run_id: int, load: bool = False):
    from terra.io import json_load, load_nested_artifacts

    run_dir = get_run_dir(run_id)

    outs = json_load(os.path.join(run_dir, "outputs.json"))
    return load_nested_artifacts(outs) if load else outs


def get(run_id: int, group_name: str = "outputs", load: bool = False):
    # TODO: flip the order of `run_id` and groupname in the instance version
    from terra.io import json_load, load_nested_artifacts

    run_dir = get_run_dir(run_id)

    artifacts = json_load(os.path.join(run_dir, f"{group_name}.json"))
    return load_nested_artifacts(artifacts) if load else artifacts


def get_artifacts(run_id: int, group_name: str = "outputs", load: bool = False):
    return get(group_name=group_name, run_id=run_id, load=load)


def get_log(run_id: int):
    run_dir = get_run_dir(run_id)

    log_path = os.path.join(run_dir, "task.log")

    with open(log_path, mode="r") as f:
        return f.read()


def get_meta(run_id: int = None):
    from terra.io import json_load

    run_dir = get_run_dir(run_id)

    meta = json_load(os.path.join(run_dir, "meta.json"))
    return meta


def _get_task_dir(module_name: str, fn_name: str):
    module = module_name.split(".")
    if module[0] != "__main__":
        module = module[1:]  # TODO: take full path for everything

    task_dir = os.path.join(
        TERRA_CONFIG["storage_dir"],
        "tasks",
        *module,
        fn_name,
    )
    return task_dir


def _get_run_dir(task_dir, idx):
    run_dir = os.path.join(task_dir, "_runs", str(idx))
    return run_dir


def _get_latest_run_id(task_dir):
    base_dir = os.path.join(task_dir, "_runs")
    if not os.path.isdir(base_dir):
        return None

    existing_dirs = [
        int(idx) for idx in os.listdir(base_dir) if idx.split("_")[0].isdigit()
    ]

    if not existing_dirs:
        return None
    return max(existing_dirs)
