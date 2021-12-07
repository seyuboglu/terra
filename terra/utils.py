import os
import random

from terra.settings import TERRA_CONFIG


def ensure_dir_exists(dir):
    """
    Ensures that a directory exists. Creates it if it does not.
    args:
        dir     (str)   directory to be created
    """
    if os.path.dirname(dir) and os.path.dirname(dir) != "/":
        ensure_dir_exists(os.path.dirname(dir))

    if not (os.path.exists(dir)):
        try:
            os.mkdir(dir)
        except FileExistsError:
            # it is possible that another process may have created the directory in the
            # intervening time between the check above – we want to ignore the error
            # that is raised in that case
            pass


def set_seed(seed: int, cudnn_deterministic: bool = False):
    """Set seed for packages that could introduce non-determinism to a task."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic

    import numpy as np

    np.random.seed(seed)


def to_rel_path(path: str):
    if os.path.isabs(path):
        return os.path.join("tasks", path.split("/tasks/")[-1])
    return path


def to_abs_path(path: str):
    if os.path.isabs(path):
        # if the abs path is on a different machine normalize it to this machine
        path = to_rel_path(path)
    return os.path.join(TERRA_CONFIG["storage_dir"], path)
