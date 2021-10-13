import json
import os

TERRA_CONFIG = {
    "storage_dir": os.path.join(os.getenv("HOME"), ".terra/default"),
    "git_dir": None,
    "local_db": True, 
    "cloud_sql_connection": None, 
    "user": None,
    "password": None,
    "db": None,
    "repo_name": None
}

if "TERRA_CONFIG_PATH" in os.environ:
    config_path = os.getenv("TERRA_CONFIG_PATH")
    with open(config_path) as f:
        config = json.load(f)
    TERRA_CONFIG.update(config)


class TerraDatabaseSettings:
    num_commit_retries: int = 10
