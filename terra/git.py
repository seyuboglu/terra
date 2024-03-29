import hashlib
import inspect
import os
import shutil
import subprocess
import __main__

from terra.settings import TERRA_CONFIG
from terra.utils import ensure_dir_exists, to_abs_path

def to_rel_path_from_git(path: str) -> str:
    """Convert absolute path to relative path from git root"""
    git_dir = TERRA_CONFIG["git_dir"]
    abspath = os.path.abspath(path)
    if git_dir is None or not abspath.startswith(git_dir):
        return abspath  

    return os.path.relpath(path, git_dir)

def _get_src_dump_path(run_dir, file_path):
    """Hash the directory to avoid replicating folder structure of the repo within
    run_dir"""
    head, tail = os.path.split(file_path)
    file_name = f"{hashlib.sha256(head.encode('utf-8')).hexdigest()[:8]}_{tail}"
    dump_path = to_abs_path(os.path.join(run_dir, "src", file_name))
    return dump_path


def _log_src(run_dir: str, fn: callable):
    src_dir = to_abs_path(os.path.join(run_dir, "src"))
    ensure_dir_exists(src_dir)
    src_path = os.path.join(src_dir, "__task_src__.py")
    with open(src_path, "w") as f:
        f.write(inspect.getsource(fn))
    
def _get_src(run_dir: str):
    run_dir = to_abs_path(run_dir)
    log_path = os.path.join(
            run_dir,
            "src",
            "__task_src__.py",
    )

    with open(log_path, mode="r") as f:
        return f.read()

def _log_main_src(run_dir: str):
    if not (hasattr(__main__, "__file__") and os.path.exists(__main__.__file__)):
        return 
    src_dir = to_abs_path(os.path.join(run_dir, "src"))
    ensure_dir_exists(src_dir)
    src_path = os.path.join(src_dir, "__main_src__.py")
    
    shutil.copy(__main__.__file__, src_path)



git_status = None


def log_git_status(run_dir: str, exts_to_dump=None) -> dict:
    """Check if git is dirty, dumping dirty files to run_dir if so. Also return dict
    with commit_hash and a list of dirty files.
    """
    global git_status
    if git_status is None:
        git_dir = TERRA_CONFIG["git_dir"]
        if git_dir is None:
            return {"commit_hash": None, "dirty": []}
        working_dir = os.getcwd()
        os.chdir(git_dir)

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
                # hash the directory to avoid replicating folder structure of the repo
                # within run_dir
                dst_path = _get_src_dump_path(run_dir, dirty_path)
                ensure_dir_exists(os.path.dirname(dst_path))
                shutil.copy(src=os.path.join(top_level, dirty_path), dst=dst_path)
                dirty.append({"file": dirty_path, "status": status, "dumped": dst_path})
            else:
                dirty.append({"file": dirty_path, "status": status, "dumped": False})

        git_status = {"commit_hash": commit_hash, "dirty": dirty}
    return git_status
