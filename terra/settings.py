import json
import os

TERRA_CONFIG = {
    "storage_dir": "/home/sabri/terra/slice",
    "git_dir": "/pd/sabri/code/domino",
    "notify": False,
}

if "TERRA_CONFIG_PATH" in os.environ:
    config_path = os.getenv("TERRA_CONFIG_PATH")
    with open(config_path) as f:
        config = json.load(f)
    TERRA_CONFIG.update(config)


class TerraDatabaseSettings:
    num_commit_retries: int = 10
