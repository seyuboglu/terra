""" The `task` module provides a framework for running reproducible analyses."""
from __future__ import annotations
import os
import traceback
from datetime import datetime
from functools import wraps
from inspect import getcallargs
import socket


from terra.git import log_git_status
from terra.utils import ensure_dir_exists
from terra.logging import init_logging
from terra.io import Artifact, json_dump, json_load
from terra.notify import (
    notify_task_completed,
    init_task_notifications,
    notify_task_error,
)
from terra.settings import TERRA_CONFIG
from terra.database import _get_session, Run


class Task:
    @classmethod
    def make_task(cls, fn: callable) -> Task:
        task = cls()
        task.task_dir = cls._get_task_dir(fn)
        task.fn = task._get_wrapper(fn)
        task.__name__ = fn.__name__
        return task

    @staticmethod
    def _get_task_dir(fn: callable):
        task_dir = os.path.join(
            TERRA_CONFIG["storage_dir"],
            "tasks",
            *fn.__module__.split(".")[1:],
            fn.__name__,
        )
        return task_dir

    def inp(self, run_id=None):
        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)
        return json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), "inputs.json"
            )
        )

    def out(self, run_id=None):
        if run_id is None:
            run_id = _get_latest_run_id(self.task_dir)
        return json_load(
            os.path.join(
                _get_run_dir(task_dir=self.task_dir, idx=run_id), "outputs.json"
            )
        )

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def _get_wrapper(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # `silence_task` is an optional parameter that when passed to a task call
            # it instructs Task to skip recording the run
            silence_task = kwargs.pop("silence_task", False)

            if not silence_task:
                args_dict = getcallargs(fn, *args, **kwargs)

                Session = _get_session()
                session = Session()

                meta_dict = {
                    "notebook": "get_ipython" in globals().keys(),
                    "start_time": datetime.now(),
                    "hostname": socket.gethostname(),
                    "module": fn.__module__,
                    "fn": fn.__name__,
                }
                run = Run(status="in_progress", **meta_dict)
                session.add(run)
                session.flush()
                run_dir = _get_run_dir(self.task_dir, run.id)
                args_dict["run_dir"] = run_dir
                if os.path.exists(run_dir):
                    raise ValueError(f"Run already exists at {run_dir}.")
                ensure_dir_exists(run_dir)
                run.run_dir = run_dir
                session.commit()

                # must write git status after getting run dir
                meta_dict["git"] = log_git_status(run_dir)
                meta_dict["start_time"] = meta_dict["start_time"].strftime("%y-%m-%d_%H-%M-%S-%f")

                # write metadata
                json_dump(
                    meta_dict, os.path.join(run_dir, "meta.json"), run_dir=run_dir
                )

                # write inputs
                json_dump(
                    args_dict, os.path.join(run_dir, "inputs.json"), run_dir=run_dir
                )

                init_logging(os.path.join(run_dir, "task.log"))
                init_task_notifications(run_dir=run_dir)
                print(f"task: {fn.__name__}, running in directory {run_dir}")

                # load node inputs
                for key, value in args_dict.items():
                    if isinstance(value, Artifact):
                        args_dict[key] = value.load()

                try:
                    out = fn(**args_dict)
                except (Exception, KeyboardInterrupt) as e:
                    msg = traceback.format_exc()
                    notify_task_error(run_dir, msg)
                    run.status = "failure"
                    run.end_time = datetime.now()
                    session.commit()
                    print(msg)
                    raise e
                else:
                    notify_task_completed(run_dir)
                    run.status = "success"
                    run.end_time = datetime.now()
                    session.commit()
                    if out is not None:
                        json_dump(
                            out, os.path.join(run_dir, "outputs.json"), run_dir=run_dir
                        )

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
