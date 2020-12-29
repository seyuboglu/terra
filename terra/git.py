import os
import subprocess
import hashlib
import shutil
import inspect

from terra.utils import ensure_dir_exists
from terra.settings import TERRA_CONFIG


def _get_src_dump_path(run_dir, file_path):
    """ Hash the directory to avoid replicating folder structure of the repo within run_dir"""
    head, tail = os.path.split(file_path)
    file_name = f"{hashlib.sha256(head.encode('utf-8')).hexdigest()[:8]}_{tail}"
    dump_path = os.path.join(run_dir, "src", file_name)
    return dump_path


def log_fn_source(run_dir: str, fn: callable):
    src_dir = os.path.join(run_dir, "src")
    ensure_dir_exists(src_dir)
    src_path = os.path.join(src_dir, "__main__.py")
    with open(src_path, "w") as f:
        f.write(inspect.getsource(fn))


def log_git_status(run_dir: str, exts_to_dump=None) -> dict:
    """Check if git is dirty, dumping dirty files to run_dir if so. Also return dict with  commit_hash and a list of dirty
    files.
    """
    working_dir = os.getcwd()
    os.chdir(TERRA_CONFIG["git_dir"])
    commit_hash = subprocess.check_output(
        ["git", "log", "--pretty=format:%H", "-n", "1"]
    ).decode("utf-8")

    if exts_to_dump is None:
        exts_to_dump = [".py"]

    dirty = []
    dirty_files = [
        dirty_file
        for dirty_file in (
            subprocess.check_output(["git", "diff-files", "--name-status"])
            .decode("utf-8")
            .strip("\n")
            .split("\n")
        )
        if len(dirty_file) > 0
    ]
    top_level = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip("\n")
    )
    os.chdir(working_dir)
    for dirty_files in dirty_files:
        status, dirty_path = dirty_files.split("\t")

        _, ext = os.path.splitext(dirty_path)

        if status != "D" and ext in exts_to_dump:
            # hash the directory to avoid replicating folder structure of the repo within run_dir
            dst_path = _get_src_dump_path(run_dir, dirty_path)
            ensure_dir_exists(os.path.join(run_dir, "src"))
            shutil.copy(src=os.path.join(top_level, dirty_path), dst=dst_path)
            dirty.append({"file": dirty_path, "status": status, "dumped": dst_path})
        else:
            dirty.append({"file": dirty_path, "status": status, "dumped": False})

    return {"commit_hash": commit_hash, "dirty": dirty}
