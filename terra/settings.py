import json
import os

TERRA_CONFIG = {
    "storage_dir": os.path.join(os.getenv("HOME"), ".terra/default"),
    "git_dir": None,
    "default_package": None, 
    "local_db": True, 
    "cloud_sql_connection": None, 
    "user": None,
    "password": None,
    "db": None,
    "repo_name": None,
    "sort_args_before_hash": True
}

if "TERRA_CONFIG_PATH" in os.environ:
    config_path = os.getenv("TERRA_CONFIG_PATH")
    with open(config_path) as f:
        config = json.load(f)
    TERRA_CONFIG.update(config)
else:
    config_path = None

class TerraDatabaseSettings:
    num_commit_retries: int = 10
