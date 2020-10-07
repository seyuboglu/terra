"""
"""
import importlib
import os
import subprocess
import pydoc
import shutil
from typing import List

import pandas as pd
import click

from terra import Task
from terra.database import TerraDatabase
from terra.utils import ensure_dir_exists


@click.group()
def cli():
    pass



@cli.command()
@click.option("--run_ids", "-r", type=str)
def tb(run_ids: str):
    run_ids = map(int, run_ids.split(","))
    db = TerraDatabase()
    specs = [f"{run.id}:{run.run_dir}" for run in db.get_runs(run_ids=run_ids)]
    subprocess.call(["tensorboard", "--logdir_spec", ",".join(specs)])


@cli.command()
@click.option("--module", default=None)
@click.option("--fn", default=None)
@click.option("--status", default=None)
@click.option("--run_ids", "-r", type=str, default=None)
def ls(module: str, fn: str, status: str, run_ids:str):
    if run_ids is not None:
        run_ids = map(int, run_ids.split(","))

    db = TerraDatabase()
    runs = db.get_runs(modules=module, fns=fn, statuses=status, run_ids=run_ids)
    if len(runs) == 0:
        print("Query returned no tasks.")
        return
    df = pd.DataFrame([run.__dict__ for run in runs])
    pydoc.pipepager(
        df[
            ["id", "module", "fn", "run_dir", "status", "start_time", "end_time"]
        ].to_string(index=False),
        "less -R",
    )


@cli.command()
def du():
    from terra.settings import TERRA_CONFIG

    working_dir = os.getcwd()
    os.chdir(TERRA_CONFIG["storage_dir"])
    subprocess.call(["du", "-sh", "--", os.path.join(TERRA_CONFIG["storage_dir"])])
    os.chdir(working_dir)


@cli.command()
@click.argument("run_id", type=int)
def rm(run_id: int):
    db = TerraDatabase()
    runs = db.get_runs(run_ids=run_id)
    if len(runs) == 0:
        raise ValueError(f"Could not find run with id {run_id}.")
    run = runs[0]

    if click.confirm(
        f"Are you sure you want to remove run with id {run.id}: \n {run.get_summary()}"
    ):
        db.rm_runs(run.id)
        shutil.rmtree(run.run_dir)
        print(f"Removed run with id {run_id}")


@cli.command()
@click.argument("module", type=str)
@click.argument("fn", type=str)
@click.option("--rerun_id", default=None, type=int)
def run(module: str, fn: str, rerun_id: int):
    print("importing module...")
    module_str, fn_str = module, fn
    module = importlib.import_module(module_str)
    fn = getattr(module, fn_str)

    if not isinstance(fn, Task):
        raise ValueError(
            f"The function {fn} is not a task. "
            "Use the `Task.make_task` decorator to turn it into a task."
        )

    task_dir = Task._get_task_dir(fn)

    if rerun_id is None:
        config_path = os.path.join(task_dir, "config.py")

        if not os.path.exists(config_path):
            _write_config_skeleton(config_path, module_str, fn_str)

        # this can be changed to vi or your preferred editor
        subprocess.call(["vi", config_path])
        #return_code = subprocess.call(["code", "--wait", config_path])
        #if return_code != 0:
        #    print("Using vim instead.")
        #    subprocess.call(["vi", config_path])

        # load config module
        config = _load_config(config_path)
    else:
        config = fn.inp(task_dir, run_id=rerun_id)
        config["kwargs"].pop("run_dir")

    module = importlib.import_module(config["module"])
    fn(**config["kwargs"])


def _load_config(config_path):
    """Load config module."""
    _, ext = os.path.splitext(config_path)
    if ext == ".py":
        spec = importlib.util.spec_from_file_location("config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config

    else:
        raise ValueError(f"The config file extension {ext} is not supported.")


def _write_config_skeleton(config_path, module, fn):
    ensure_dir_exists(os.path.split(config_path)[0])
    with open(config_path, "w") as f:
        f.write(
            " # INSTRUCTIONS: Edit process parameters below. Close this file (cmd-w) to run the process\n"
            "import terra\n"
            f"from {module} import {fn} \n"
            "\n"
            "config = {\n"
            f"   'module': '{module}',\n"
            f"   'fn': '{fn}',\n"
            "   'kwargs': {\n"
            "   }\n"
            "}\n"
        )
        f.flush()
        f.close()
