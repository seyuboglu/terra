import pytest
from functools import wraps
from itertools import product
from terra.settings import TERRA_CONFIG
import terra
import terra.database as tdb


class BaseTestBed:

    DEFAULT_CONFIG = {"local_db": [True]}

    def __init__(
        self,
        local_db: bool = True,
        tmpdir: str = None,
    ):
        TERRA_CONFIG["storage_dir"] = str(tmpdir)
        TERRA_CONFIG["local_db"] = local_db
        if not local_db:
            TERRA_CONFIG.update(
                {
                    "local_db": True,
                    "cloud_sql_connection": "hai-gcp-fine-grained:us-west1:terra",
                    "user": "postgres",
                    "password": "bhn6zph_wda0dqn_APH",
                    "db": "terra",
                    "repo_name": "terra-repo",
                }
            )
        terra.database.Session = (
            tdb.get_session()
        )  # need to recreate Session with new tmpdir

        self.tmpdir = tmpdir

    @classmethod
    def get_params(cls, config: dict = None, params: dict = None, single: bool = False):
        updated_config = cls.DEFAULT_CONFIG.copy()
        if config is not None:
            updated_config.update(config)
        configs = [
            (cls, config)
            for config in map(
                dict,
                product(*[[(k, v) for v in vs] for k, vs in updated_config.items()]),
            )
        ]
        if single:
            configs = configs[:1]
        if params is None:
            return {
                "argnames": "testbed",
                "argvalues": configs,
                "ids": [str(config) for config in configs],
            }
        else:
            argvalues = list(product(configs, *params.values()))
            return {
                "argnames": "testbed," + ",".join(params.keys()),
                "argvalues": argvalues,
                "ids": [",".join(map(str, values)) for values in argvalues],
            }

    @classmethod
    @wraps(pytest.mark.parametrize)
    def parametrize(
        cls, config: dict = None, params: dict = None, single: bool = False
    ):
        return pytest.mark.parametrize(
            **cls.get_params(config=config, params=params, single=single),
            indirect=["testbed"]
        )
