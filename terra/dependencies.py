from functools import lru_cache
import subprocess
import sys


@lru_cache
def get_dependencies():
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    return reqs.decode("utf-8").split("\n")
