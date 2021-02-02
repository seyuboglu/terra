""" The `task` module provides a framework for running reproducible analyses."""
from __future__ import annotations
import os
from datetime import datetime
from inspect import getcallargs
import socket
import sys
import platform
import traceback

from terra.git import log_git_status, log_fn_source
from terra.utils import ensure_dir_exists
from terra.logging import init_logging
from terra.notify import (
    notify_task_completed,
    init_task_notifications,
    notify_task_error,
)
from terra.settings import TERRA_CONFIG
import terra.database as tdb


class Task:
    @classmethod
    def make_task(cls, fn: callable) -> Task:
        task = cls()
        task.task_dir = cls._get_task_dir(fn)
        task.fn = fn
        task.__name__ = fn.__name__
        task.__module__ = fn.__module__
        return task

    @staticmethod
    def _get_task_dir(task: Task):
        module = task.__module__.split(".")
        if task.__module__ != "__main__":
            module = module[1:]  # TODO: take full path for everything

        task_dir = os.path.join(
            TERRA_CONFIG["storage_dir"],
            "tasks",
            *module,
            task.__name__,
        )
        return task_dir

    def run_dir(self, run_id: int = None):
        if run_id is None:
            return _get_run_dir(self.task_dir, self.last_run_id)
        return _get_run_dir(self.task_dir, run_id)

    @property
    def last_run_id(self):
        run_id = _get_latest_run_id(self.task_dir)
        return run_id

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
            run_id = _get_latest_run_id(self.task_dir)
        outs = json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), "outputs.json"
            )
        )

        return load_nested_artifacts(outs) if load else outs

    def get_artifacts(
        self, group_name: str = "outputs", run_id=None, load: bool = False
    ):
        from terra.io import json_load, load_nested_artifacts

        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)
        artifacts = json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), f"{group_name}.json"
            )
        )
        return load_nested_artifacts(artifacts) if load else artifacts

    def rm_artifacts(self, group_name: str, run_id: int):
        from terra.io import json_load, rm_nested_artifacts
        artifacts = json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), f"{group_name}.json"
            )
        )
        rm_nested_artifacts(artifacts)

    def get_runs(self):
        return tdb.get_runs(fns=self.__name__)

    def get_log(self, run_id: int = None):
        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)

        log_path = os.path.join(
            _get_run_dir(task_dir=self.task_dir, idx=run_id), "task.log"
        )

        with open(log_path, mode="r") as f:
            return f.read()

    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)

    def remote(self, *args, **kwargs):
        """Warning: if you updated the TERRA_CONFIG, these changes will not persist
        into child tasks. You should pass `terra_config`=TERRA_CONFIG to `remote`in this
        case.
        """
        import ray

        @ray.remote
        def fn(task, *args, **kwargs):
            return task._run(*args, **kwargs)

        return fn.remote(self, *args, **kwargs)

    def _run(self, *args, **kwargs):
        from terra.io import json_dump, load_nested_artifacts

        # unpack optional Task modifiers
        # `silence_task` instructs terra not to record the run
        silence_task = kwargs.pop("silence_task", False)
        if silence_task:
            return self.fn(*args, **kwargs)
        # `return_run_id` instructs terra to return (run_id, returned_obj)
        return_run_id = kwargs.pop("return_run_id", False)

        # `terra_config` updates the terra config
        if "terra_config" in kwargs:
            TERRA_CONFIG.update(kwargs.pop("terra_config"))
            tdb.Session = tdb.get_session()  # neeed to recreate db session

        args_dict = getcallargs(self.fn, *args, **kwargs)

        if "kwargs" in args_dict:
            args_dict.update(args_dict.pop("kwargs"))

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
        session.commit()
        try:
            run_dir = _get_run_dir(self.task_dir, run.id)
            args_dict["run_dir"] = run_dir
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

                session.commit()

            # write additional metadata
            from pip._internal.operations import freeze  # lazy import to reduce startup
            meta_dict.update(
                {
                    "git": git_status,
                    "start_time": meta_dict["start_time"].strftime(
                        "%y-%m-%d_%H-%M-%S-%f"
                    ),
                    "dependencies": list(freeze.freeze()),
                    "terra_config": TERRA_CONFIG,
                }
            )
            json_dump(meta_dict, os.path.join(run_dir, "meta.json"), run_dir=run_dir)

            # write inputs
            json_dump(args_dict, os.path.join(run_dir, "inputs.json"), run_dir=run_dir)

            init_logging(os.path.join(run_dir, "task.log"))

            init_task_notifications(run_id=run.id)

            print(f"task: {self.fn.__name__}, run_id={run.id}", flush=True)

            # load node inputs
            args_dict = load_nested_artifacts(args_dict, run_id=run.id)

            # run function
            out = self.fn(**args_dict)

            # write outputs
            if out is not None:
                out = json_dump(
                    out, os.path.join(run_dir, "outputs.json"), run_dir=run_dir
                )

            # log success
            notify_task_completed(run.id)

            run.status = "success"
            run.end_time = datetime.now()
            session.commit()

        except (Exception, KeyboardInterrupt) as e:
            msg = traceback.format_exc()
            notify_task_error(run.id, msg)
            run.status = (
                "interrupted" if isinstance(e, KeyboardInterrupt) else "failure"
            )
            run.end_time = datetime.now()
            session.commit()
            print(msg)
            session.close()
            raise e

        if return_run_id:
            out = (int(run.id), out) if out is not None else int(run.id)
        session.close()
        return out


def init_remote():
    import ray
    ray.init()


def _get_next_run_dir(task_dir):
    """Get the next available run directory (e.g. "_runs/0", "_runs/1", "_runs/2")
    in base_dir"""
    latest_idx = _get_latest_run_id(task_dir)
    idx = latest_idx + 1 if latest_idx is not None else 0
    run_dir = _get_run_dir(task_dir, idx)
    ensure_dir_exists(run_dir)
    return run_dir


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
