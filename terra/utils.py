import os
from tqdm import tqdm as _tqdm


def ensure_dir_exists(dir):
    """
    Ensures that a directory exists. Creates it if it does not.
    args:
        dir     (str)   directory to be created
    """
    if not os.path.dirname(dir):
        os.mkdir(dir)

    elif not (os.path.exists(dir)):
        ensure_dir_exists(os.path.dirname(dir))
        os.mkdir(dir)


def tqdm(*args, **kwargs):
    """ Wrapper around tqdm that doesn't skip lines in Jupyter Notebooks."""
    getattr(_tqdm, "_instances", {}).clear()
    return _tqdm(*args, **kwargs)