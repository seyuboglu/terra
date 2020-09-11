import os
import subprocess
import hashlib
import shutil

from terra.utils import ensure_dir_exists


def _get_src_dump_path(run_dir, file_path):
    """ Hash the directory to avoid replicating folder structure of the repo within run_dir"""
    head, tail = os.path.split(file_path)
    file_name = f"{hashlib.sha256(head.encode('utf-8')).hexdigest()[:8]}_{tail}"
    dump_path = os.path.join(run_dir, "src", file_name)
    return dump_path


def log_git_status(run_dir, exts_to_dump=None) -> dict:
    """Check if git is dirty, dumping dirty files to run_dir if so. Also return dict with  commit_hash and a list of dirty
    files.
    """

    commit_hash = subprocess.check_output(
        ["git", "log", "--pretty=format:%H", "-n", "1"]
    ).decode("utf-8")

    if exts_to_dump is None:
        exts_to_dump = [".py"]

    dirty = []
    dirty_files = (
        subprocess.check_output(["git", "diff-files", "--name-status"])
        .decode("utf-8")
        .strip("\n")
        .split("\n")
    )
    top_level = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip("\n")
    )
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