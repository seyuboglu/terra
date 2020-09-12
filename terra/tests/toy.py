from terra import Task
from terra.notify import notify_task_checkpoint
import numpy as np

@Task.make_task
def circle_area(radius: int, run_dir=None):
    return np.pi * radius ** 2

@Task.make_task
def pythagoras(a: int, b: int, run_dir=None) -> int:
    notify_task_checkpoint(run_dir, "Computing...")
    return np.sqrt(a**2 + b**2)