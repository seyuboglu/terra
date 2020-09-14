import os
import json


TERRA_CONFIG = {
    "storage_dir": "/Users/sabrieyuboglu/code/terra/test_storage_dir",
    "slack_web_client_id": "xoxb-1190875346355-1203313826833-6ArzXNwOZ81aHdV95fxqbAcM",
    "notify": True,
}

if "TERRA_CONFIG_PATH" in os.environ:
    config_path = os.getenv("TERRA_CONFIG_PATH")
    with open(config_path) as f:
        config = json.load(f)
    TERRA_CONFIG.update(config)
