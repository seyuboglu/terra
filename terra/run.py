"""
"""
import argparse
import importlib
import os
import subprocess

from terra import Task
from terra.utils import ensure_dir_exists


def load_config(config_path):
    """Load config module."""
    _, ext = os.path.splitext(config_path)
    if ext == ".py":
        spec = importlib.util.spec_from_file_location("config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config

    else:
        raise ValueError(f"The config file extension {ext} is not supported.")


def write_config_skeleton(config_path, process):
    module = ".".join(process.split(".")[:-1])
    fn = process.split(".")[-1]
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


def main():
    parser = argparse.ArgumentParser(description="Run a TPEX process.")
    parser.add_argument("process", type=str)
    parser.add_argument("--rerun_idx", default=None, type=int)
    args = parser.parse_args()

    print("importing module...")
    module = importlib.import_module(".".join(args.process.split(".")[:-1]))
    fn = getattr(module, args.process.split(".")[-1])

    if not isinstance(fn, Task):
        raise ValueError(
            f"The function {fn} is not a task. "
            "Use the `Task.make_task` decorator to turn it into a task."
        )

    task_dir = Task._get_task_dir(fn)

    if args.rerun_idx is None:
        config_path = os.path.join(task_dir, "config.py")

        if not os.path.exists(config_path):
            write_config_skeleton(config_path, args.process)

        # this can be changed to vi or your preferred editor
        return_code = subprocess.call(["code", "--wait", config_path])
        if return_code == 1:
            print("Using vim instead.")
            subprocess.call(["vi", config_path])

        # load config module
        config = load_config(config_path)
    else:
        config = fn.inp(task_dir, run_idx=args.rerun_idx)
        config["kwargs"].pop("run_dir")

    module = importlib.import_module(config["module"])
    fn(**config["kwargs"])


if __name__ == "__main__":
    main()
