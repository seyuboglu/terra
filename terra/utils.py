import os
import random

from tqdm import tqdm as _tqdm


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


def tqdm(*args, **kwargs):
    """ Wrapper around tqdm that doesn't skip lines in Jupyter Notebooks."""
    getattr(_tqdm, "_instances", {}).clear()
    return _tqdm(*args, **kwargs)


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
