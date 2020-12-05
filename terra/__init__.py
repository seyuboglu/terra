""" The `task` module provides a framework for running reproducible analyses."""
from __future__ import annotations
import os
from datetime import datetime
from functools import wraps
from inspect import getcallargs
from pip._internal.operations import freeze
import socket
import sys
import platform
import traceback

import pandas as pd

from terra.git import log_git_status, log_fn_source
from terra.utils import ensure_dir_exists
from terra.logging import init_logging
from terra.io import json_dump, json_load, load_nested_artifacts, rm_nested_artifacts
from terra.notify import (
    notify_task_completed,
    init_task_notifications,
    notify_task_error,
)
from terra.settings import TERRA_CONFIG
from terra.database import get_session, Run, TerraDatabase


class Task:
    @classmethod
    def make_task(cls, fn: callable) -> Task:
        task = cls()
        task.task_dir = cls._get_task_dir(fn)
        task.fn = task._get_wrapper(fn)
        task.__name__ = fn.__name__
        task.__module__ = fn.__module__
        return task

    @staticmethod
    def _get_task_dir(task: Task):
        module = task.__module__.split(".")
        if task.__module__ != "__main__":
            module = module[1:] # TODO: take full path for everything

        task_dir = os.path.join(
            TERRA_CONFIG["storage_dir"],
            "tasks",
            *module,  
            task.__name__,
        )
        return task_dir

    def run_dir(self, run_id: int):
        return _get_run_dir(self.task_dir, run_id)

    @property
    def last_run_id(self):
        run_id = _get_latest_run_id(self.task_dir)
        return run_id

    def inp(self, run_id: int = None, load: bool = False):
        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)
        inps = json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), "inputs.json"
            )
        )
        return load_nested_artifacts(inps) if load else inps

    def out(self, run_id: int = None, load: bool = False):
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
        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)
        artifacts = json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), f"{group_name}.json"
            )
        )
        return load_nested_artifacts(artifacts) if load else artifacts

    def rm_artifacts(
        self, group_name: str, run_id: int
    ):
        artifacts = json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), f"{group_name}.json"
            )
        )
        rm_nested_artifacts(artifacts)

    def get_runs(self):
        db = TerraDatabase()
        runs = db.get_runs(fns=self.__name__)
        df = pd.DataFrame([run.__dict__ for run in runs])
        return df[["id", "module", "fn", "run_dir", "status", "start_time", "end_time"]]

    def get_log(self, run_id: int = None):
        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)

        log_path = os.path.join(
            _get_run_dir(task_dir=self.task_dir, idx=run_id), "task.log"
        )

        with open(log_path, mode="r") as f:
            return f.read()

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def _get_wrapper(self, fn):            

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # `silence_task` is an optional parameter that when passed to a task call
            # it instructs Task to skip recording the run
            silence_task = kwargs.pop("silence_task", False)
            return_run_id = kwargs.pop("return_run_id", False)

            if not silence_task:
                args_dict = getcallargs(fn, *args, **kwargs)

                if "kwargs" in args_dict:
                    args_dict.update(args_dict.pop("kwargs"))

                Session = get_session()
                session = Session()

                meta_dict = {
                    "notebook": "get_ipython" in globals().keys(),
                    "start_time": datetime.now(),
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "module": fn.__module__,
                    "fn": fn.__name__,
                    "python_version": sys.version,
                }

                # add run to terra db
                run = Run(status="in_progress", **meta_dict)
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
                    if fn.__module__ == "__main__":
                        log_fn_source(run_dir=run_dir, fn=fn)
                    session.commit()

                    # write additional metadata
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
                    json_dump(
                        meta_dict, os.path.join(run_dir, "meta.json"), run_dir=run_dir
                    )

                    # write inputs
                    json_dump(
                        args_dict, os.path.join(run_dir, "inputs.json"), run_dir=run_dir
                    )

                    init_logging(os.path.join(run_dir, "task.log"))
                    init_task_notifications(run_id=run.id)
                    print(f"task: {fn.__name__}, run_id={run.id}", flush=True)

                    # load node inputs
                    args_dict = load_nested_artifacts(args_dict)

                    out = fn(**args_dict)
                except (Exception, KeyboardInterrupt) as e:
                    msg = traceback.format_exc()
                    notify_task_error(run.id, msg)
                    run.status = (
                        "interrupted" if isinstance(e, KeyboardInterrupt) else "failure"
                    )
                    run.end_time = datetime.now()
                    session.commit()
                    print(msg)
                    raise e
                else:
                    notify_task_completed(run.id)
                    run.status = "success"
                    run.end_time = datetime.now()
                    session.commit()
                    if out is not None:
                        out = json_dump(
                            out, os.path.join(run_dir, "outputs.json"), run_dir=run_dir
                        )

                if return_run_id:
                    out = (int(run.id), out) if out is not None else int(run.id)
                session.close()
                return out

            else:
                return fn(*args, **kwargs)

        return wrapper


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
