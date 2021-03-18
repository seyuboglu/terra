"""
"""
import importlib
from json.decoder import JSONDecodeError
import os
import subprocess
import pydoc
import shutil
from datetime import datetime

import click
from tqdm import tqdm

from terra import Task, _get_task_dir
import terra.database as tdb
from terra.utils import ensure_dir_exists


@click.group()
def cli():
    pass


@cli.command()
@click.option("--run_ids", "-r", type=str, default=None)
@click.option("--fn", default=None)
@click.option("--module", default=None)
def tb(run_ids: str, module: str, fn: str):

    if run_ids is not None:
        run_ids = map(int, run_ids.split(","))
        specs = [
            f"{run.id}:{run.run_dir}" for run in tdb.get_runs(run_ids=run_ids, df=False)
        ]
        subprocess.call(["tensorboard", "--logdir_spec", ",".join(specs)])
    elif fn is not None and module is not None:
        subprocess.call(["tensorboard", "--logdir", _get_task_dir(module, fn)])


@cli.command()
@click.option("--module", default=None)
@click.option("--fn", default=None)
@click.option("--status", default=None)
@click.option("--run_ids", "-r", type=str, default=None)
def ls(module: str, fn: str, status: str, run_ids: str):
    import pandas as pd

    if run_ids is not None:
        run_ids = map(int, run_ids.split(","))

    runs = tdb.get_runs(
        modules=module, fns=fn, statuses=status, run_ids=run_ids, df=False
    )
    if len(runs) == 0:
        print("Query returned no tasks.")
        return
    df = pd.DataFrame([run.__dict__ for run in runs])
    pydoc.pipepager(
        df[
            ["id", "module", "fn", "run_dir", "status", "slurm_job_id", "start_time", "end_time"]
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
    runs = tdb.get_runs(run_ids=run_id, df=False)
    if len(runs) == 0:
        raise ValueError(f"Could not find run with id {run_id}.")
    run = runs[0]

    if click.confirm(
        f"Are you sure you want to remove run with id {run.id}: \n {run.get_summary()}"
    ):
        tdb.rm_runs(run.id)
        shutil.rmtree(run.run_dir)
        print(f"Removed run with id {run_id}")


@cli.command()
@click.argument("module", type=str)
@click.argument("fn", type=str)
@click.option("--start_date", type=str)
@click.option("--end_date", type=str)
@click.option("--hanging_only", is_flag=True, default=False)
def rm_artifacts(
    module: str, fn: str, start_date: str, end_date: str, hanging_only: bool
):
    import pandas as pd
    from terra.io import json_load, get_nested_artifact_paths

    date_format = "%m-%d-%Y"
    date_range = (
        datetime.strptime(start_date, date_format),
        datetime.strptime(end_date, date_format),
    )
    runs = tdb.get_runs(fns=fn, modules=module, date_range=date_range, df=False)

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
    print(f"{hanging_only=}")
    print(f"Removing artifacts from {len(df)} runs of tasks {df['fn'].unique()}")
    if not click.confirm(
        f"Do you want to remove artifacts for the run_ids you just queried?"
    ):
        print("aborted")
        return
    for run_dir in tqdm(df.run_dir):
        artifacts_dir = os.path.join(run_dir, "artifacts")
        if os.path.isdir(artifacts_dir):
            if hanging_only:
                artifact_paths = set(
                    [os.path.join(artifacts_dir, x) for x in os.listdir(artifacts_dir)]
                )
                for group in ["checkpoint", "outputs", "inputs"]:
                    path = os.path.join(run_dir, f"{group}.json")
                    if os.path.isfile(path):
                        try:
                            out = json_load(path)
                        except JSONDecodeError:
                            print(f"Malformed JSON at {path}")
                            continue
                        not_hanging = get_nested_artifact_paths(out)
                        artifact_paths -= set(not_hanging)
                for path in artifact_paths:
                    if os.path.isfile(path):
                        os.remove(path)
            else:
                try:
                    shutil.rmtree(run_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))


@cli.command()
@click.argument("module", type=str)
@click.argument("fn", type=str)
@click.option("-s", "--slurm", is_flag=True, help="Submit job using sbatch.")
@click.option("--srun", is_flag=True, help="Submit job using srun. Must be used with --slurm.")
@click.option("-e", "--edit", is_flag=True, help="Edit the config prior to running.")
def run(module: str, fn: str, slurm: bool, srun: bool, edit: bool):
    if srun and not slurm:
        raise ValueError("--srun is only a valid option when using --slurm.")

    module_str, fn_str = module, fn
    module = importlib.import_module(module_str)
    fn = getattr(module, fn_str)

    if not isinstance(fn, Task):
        raise ValueError(
            f"The function {fn} is not a task. "
            "Use the `Task.make_task` decorator to turn it into a task."
        )

    task_dir = Task._get_task_dir(fn)

    config_path = os.path.join(task_dir, "config.py")
    if edit:
        if not os.path.exists(config_path):
            _write_config_skeleton(config_path, module_str, fn_str)

        # this can be changed to vi or your preferred editor
        print("Close config editor to continue...")
        return_code = subprocess.call(["code", "--wait", config_path])
        if return_code != 0:
            print("Using vim instead.")
            subprocess.call(["vi", config_path])
    # load config module
    config = _load_config(config_path)

    if slurm:
        sh_path = os.path.join(task_dir, f"{fn_str}.sh")
        _write_slurm_sh(
            sh_path=sh_path, slurm_config=config["slurm"], module=module_str, fn=fn_str
        )
        subprocess.call(["srun" if srun else "sbatch", sh_path])

    else:
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


def _write_slurm_sh(sh_path, slurm_config, module, fn):
    ensure_dir_exists(os.path.split(sh_path)[0])
    lines = ["#!/bin/bash"]
    for option, val in slurm_config.items():
        sep = "=" if option.startswith("--") else " "
        lines.append(f"#SBATCH {option}{sep}{val}")

    lines.extend(
        [
            "source $HOME/.bashrc",
            "conda activate win",
            f"cd {os.getcwd()}",
            f"terra run {module} {fn}",
        ]
    )
    with open(sh_path, "w") as f:
        f.write("\n".join(lines))
        f.flush()
        f.close()
    
    # need to provide execute permissions to the user
    subprocess.call(["chmod", "+rx", sh_path])


@cli.command()
@click.argument("module", type=str)
@click.argument("fn", type=str)
def config(module: str, fn: str):
    task_dir = _get_task_dir(module_name=module, fn_name=fn)

    config_path = os.path.join(task_dir, "config.py")

    if not os.path.exists(config_path):
        _write_config_skeleton(config_path, module, fn)

    print(f"config path: {config_path}")
