"""
"""
import importlib
from multiprocessing.sharedctypes import Value
import os
import pydoc
import shutil
import subprocess
from datetime import datetime
from json.decoder import JSONDecodeError
from multiprocessing import Pool

import click

import terra.database as tdb
from terra import Task, _get_task_dir
from terra.settings import TERRA_CONFIG, config_path
from terra.utils import bytes_fmt, ensure_dir_exists, to_abs_path



@click.group()
@click.option("--module", default=None)
@click.option("--fn", default=None)
@click.option("--status", default=None)
@click.option("--run_ids", "-r", type=str, default=None)
@click.option("--start_date", type=str, default=None)
@click.option("--end_date", type=str, default=None)
@click.option("--pushed", type=bool, default=None)
@click.pass_context
def cli(
    ctx,
    module: str,
    fn: str,
    run_ids: str,
    status: str,
    start_date: str,
    end_date: str,
    pushed: str  
):
    ctx.ensure_object(dict)
    ctx.obj["modules"] = module
    ctx.obj["fns"] = fn
    ctx.obj["statuses"] = status
    ctx.obj["pushed"] = pushed

    if run_ids is not None:
        run_ids = map(int, run_ids.split(","))

    ctx.obj["run_ids"] = run_ids

    ctx.obj["date_range"] = None
    date_format = "%m-%d-%Y"
    if start_date is not None and end_date is not None:
        ctx.obj["date_range"] = (
            datetime.strptime(start_date, date_format),
            datetime.strptime(end_date, date_format),
        )


@cli.command()
def config():
    import json
    print(f"config path: {config_path}")
    print(json.dumps(TERRA_CONFIG, indent=4))


@cli.command()
@click.option("--bucket_name", "-b", type=str, default=None)
@click.option("--force", "-f", is_flag=True, default=False)
@click.option("--num_workers", type=int, default=0)
@click.option("--warn_missing", "-w", is_flag=True, default=False)
@click.pass_context
def push(ctx, bucket_name: str, force: bool, num_workers: int, warn_missing: bool):
    from terra.remote import push

    push(**ctx.obj, bucket_name=bucket_name, force=force, num_workers=num_workers, warn_missing=warn_missing)


@cli.command()
@click.pass_context
@click.option("--bucket_name", "-b", type=str, default=None)
def pull(ctx, bucket_name: str):
    from terra.remote import pull

    pull(**ctx.obj, bucket_name=bucket_name)


@cli.command()
@click.option("--limit", type=int, default=1_000)
@click.pass_context
def ls(ctx, limit: int):
    import pandas as pd

    runs = tdb.get_runs(**ctx.obj, limit=limit, df=False)

    if len(runs) == 0:
        print("Query returned no tasks.")
        return
    df = pd.DataFrame([run.__dict__ for run in runs])
    pydoc.pipepager(
        df[
            [
                "id",
                "module",
                "fn",
                "run_dir",
                "status",
                "start_time",
                "end_time",
                "hostname",
                "git_commit",
            ]
        ].to_string(index=False),
        "less -R",
    )


def _rm_dir(run_dir):
    run_dir = to_abs_path(run_dir) 
    if run_dir is None:
        return
    try:
        assert run_dir.startswith(TERRA_CONFIG["storage_dir"])
        shutil.rmtree(run_dir)
    except FileNotFoundError:
        pass


@cli.command()
@click.option("--num_workers", type=int, default=0)
@click.option("--exclude_run_ids", type=str, default=None)
@click.pass_context
def rm_local(ctx, num_workers: int, exclude_run_ids: str):
    from tqdm import tqdm

    runs = tdb.get_runs(**ctx.obj, df=False)

    if not click.confirm(
        "Do you want to remove the directories for the run_ids you just queried?"
    ):
        print("aborted")
        return

    if exclude_run_ids is not None:
        if exclude_run_ids.endswith(".txt"):
            with open(exclude_run_ids, "r") as f:
                exclude_run_ids = f.read()
        exclude_run_ids = set(map(int, exclude_run_ids.split(",")))
        print(len(runs))
        run_dirs = [run.run_dir for run in runs if run.id not in exclude_run_ids]
        print(len(run_dirs))
    else:
        run_dirs = [run.run_dir for run in runs if run.run_dir is not None]

    if num_workers > 0:
        pool = Pool(processes=8)
        for _ in tqdm(pool.imap_unordered(_rm_dir, run_dirs), total=len(run_dirs)):
            pass

    else:
        [_rm_dir(run_dir) for run_dir in tqdm(run_dirs)]

    
@cli.command()
@click.option("--force", "-f",  is_flag=True, default=False)
@click.option("--interactive", "-i", is_flag=True, default=False)
@click.pass_context
def du(ctx, force: bool, interactive: bool):
    from terra.settings import TERRA_CONFIG
    from tqdm import tqdm
    import pandas as pd 

    SCHEMA = ["id", "size_bytes", "local"]

    du_dir = os.path.join(TERRA_CONFIG["storage_dir"], "du")
    os.makedirs(du_dir, exist_ok=True)

    run_df = tdb.get_runs(**ctx.obj, df=True)

    cache_path = os.path.join(du_dir, "du_cache.csv")
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
        assert set(SCHEMA) == set(cache_df.columns)
    else:
        cache_df = pd.DataFrame(columns=SCHEMA)

    if force:
        todo_df = run_df
        rows = []
        print(f"Skipping cache.")
    else:
        todo_df = run_df[~run_df["id"].isin(cache_df["id"])]
        rows = cache_df[cache_df["id"].isin(run_df["id"])].to_dict("records")
        print(f"{len(rows)}/{len(run_df)} runs cached (use -f to skip cache)")

    if len(todo_df) > 0:
        print("Running `du` for {} runs".format(len(todo_df)))
        for _, run in tqdm(todo_df.iterrows(), total=len(todo_df)):
            row = {"id": run["id"]}
            
            if run["run_dir"] is None:
                run_dir = None 
            else:
                run_dir = to_abs_path(run["run_dir"])
            if run_dir is None or not os.path.exists(run_dir):
                row["local"] = False
                row["size_bytes"] = 0
            else:   
                # get size of directory using du 
                out = subprocess.check_output(["du", "-s", run_dir])
                size_bytes = int(out.decode("utf-8").split("\t")[0])
                row["size_bytes"] = size_bytes  
                row["local"] = True
            rows.append(row)
    
    df = pd.DataFrame(rows)

    # need to make sure we don't write duplicate runs to the cache 
    new_cache_df = pd.concat([df, cache_df[~cache_df["id"].isin(df["id"])]], axis=0)
    assert new_cache_df["id"].is_unique
    new_cache_df.to_csv(cache_path, index=False)

    df = df.merge(run_df, on="id")


    print(
        "\n"
        "SUMMARY (use -i to interact with full results)\n"
        "----------------------------------------------\n"
        f"total size: {bytes_fmt(df['size_bytes'].sum())}\n"
        f"average size: {bytes_fmt(df['size_bytes'].mean())}\n"
        f"# of runs: {len(df)}\n"
        f"# of local runs: {df['local'].sum()}\n"
    )

    if interactive:
        import code
        code.interact(banner="Run sizes are stored in the `df` variable:", local=locals())



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

    from terra.io import get_nested_artifact_paths, json_load

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
        "Do you want to remove artifacts for the run_ids you just queried?"
    ):
        print("aborted")
        return
    for run_dir in df.run_dir:
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
                    shutil.rmtree(artifacts_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))


@cli.command()
@click.option(
    "-g",
    "--git_dir",
    type=str,
    help="Path to local git repository to be tracked.",
    default=None,
)
@click.option("-s", "--storage_dir", type=str, help="Directory to store ", default=None)
def init(git_dir: str, storage_dir: str):
    if "TERRA_CONFIG_PATH" in os.environ:
        print(
            "'TERRA_CONFIG_PATH' environment variable already set."
            "Skipping initialization."
        )
        #return

    from terra.settings import TERRA_CONFIG

    config = TERRA_CONFIG.copy()

    if storage_dir is None:
        storage_dir = os.path.join(os.getenv("HOME"), ".terra/default")
        os.makedirs(storage_dir, exist_ok=True)
    assert os.path.isdir(storage_dir)
    config["storage_dir"] = storage_dir

    if git_dir is None:
        git_dir = os.getcwd()

    if not os.path.exists(os.path.join(git_dir, ".git")):
        raise ValueError(
            "`git_dir` is not a git repository. Run `terra init` from a valid git "
            "repository or pass a valid git repo to `--git_dir`."
        )
    config["git_dir"] = git_dir

    config["local_db"] = True

    conda_environment = os.environ.get("CONDA_DEFAULT_ENV", None)
    if conda_environment is None or conda_environment == "base":
        raise ValueError(
            "Create and activate a conda environment before running `terra init`."
        )

    config_path = os.path.join(git_dir, "terra-config.json")
    subprocess.call(["conda", f"env config vars set TERRA_CONFIG_PATH={config_path}"])
    subprocess.call(["conda", "activate", conda_environment])


@cli.command()
@click.argument("module", type=str)
@click.argument("fn", type=str)
@click.option("-s", "--slurm", is_flag=True, help="Submit job using sbatch.")
@click.option(
    "--srun", is_flag=True, help="Submit job using srun. Must be used with --slurm."
)
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
            "Use the `Task` decorator to turn it into a task."
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
            " # INSTRUCTIONS: Edit process parameters below. Close this file (cmd-w) "
            " to run the process\n"
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


