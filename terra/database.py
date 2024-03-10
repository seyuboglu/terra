import contextlib
from genericpath import exists
import hashlib
import os
from datetime import datetime
from time import sleep
from typing import TYPE_CHECKING, List, Tuple, Union, Dict

import sqlalchemy
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Float,
    desc,
    select,
    literal,
    union_all,
    cast,
    Table,
    exists,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from torch import chunk

from terra.settings import TERRA_CONFIG, TerraDatabaseSettings
from terra.utils import ensure_dir_exists, to_abs_path

if TYPE_CHECKING:
    import pandas as pd

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    module = Column(String)
    fn = Column(String)
    run_dir = Column(String)
    status = Column(String)
    notebook = Column(Boolean)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    hostname = Column(String)
    python_version = Column(String)
    platform = Column(String)
    git_commit = Column(String)
    git_dirty = Column(Boolean)
    input_hash = Column(String)
    slurm_job_id = Column(Integer)

    def get_summary(self):
        return (
            f"module={self.module}, fn={self.fn}, "
            "status={self.status}, run_dir={self.run_dir}"
        )


class ArtifactDump(Base):
    __tablename__ = "artifact_dumps"
    id = Column(Integer, primary_key=True)
    creating_run_id = Column(Integer, ForeignKey("runs.id"))
    path = Column(String)
    type = Column(String)
    dump_time = Column(DateTime, default=datetime.utcnow)
    rm = Column(Boolean, default=False)


class ArtifactLoad(Base):
    __tablename__ = "artifact_loads"
    id = Column(Integer, primary_key=True)
    artifact_id = Column(Integer, ForeignKey("artifact_dumps.id"))
    loading_run_id = Column(Integer, ForeignKey("runs.id"))
    load_time = Column(DateTime, default=datetime.utcnow)


class Ref(Base):
    __tablename__ = "refs"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    run_dir = Column(String)


def safe_commit(session):
    retries = 0
    while True:
        try:
            return session.commit()
        except sqlalchemy.exc.OperationalError as e:
            if (
                "(sqlite3.OperationalError) database is locked" in str(e)
            ) and retries < TerraDatabaseSettings.num_commit_retries:

                print(
                    "Retrying terra database commit "
                    f"({retries}/{TerraDatabaseSettings.num_commit_retries})."
                )
                # https://docs.sqlalchemy.org/en/13/faq/sessions.html#this-session-s-transaction-has-been-rolled-back-due-to-a-previous-exception-during-flush-or-similar
                session.rollback()
                sleep(0.5)
                retries += 1
            else:
                raise e


def get_runs(
    run_ids: Union[int, List[int]] = None,
    modules: Union[str, List[str]] = None,
    fns: Union[str, List[str]] = None,
    statuses: Union[str, List[str]] = None,
    date_range: Tuple[datetime] = None,
    pushed: bool = None,
    limit: int = None,
    df: bool = True,
) -> List[Run]:
    session = Session()
    query = session.query(Run)

    if run_ids is not None:
        run_ids = [run_ids] if isinstance(run_ids, int) else run_ids
        query = query.filter(Run.id.in_(run_ids))

    if modules is not None:
        modules = [modules] if isinstance(modules, str) else modules
        query = query.filter(Run.module.in_(modules))

    if fns is not None:
        fns = [fns] if isinstance(fns, str) else fns
        query = query.filter(Run.fn.in_(fns))

    if statuses is not None:
        statuses = [statuses] if isinstance(statuses, str) else statuses
        query = query.filter(Run.status.in_(statuses))

    if date_range is not None:
        query = query.filter(Run.start_time > date_range[0])
        query = query.filter(Run.start_time < date_range[1])

    if pushed is not None:
        from terra.remote import _get_pushed_run_ids
        import pandas as pd

        pushed_run_ids = _get_pushed_run_ids(TERRA_CONFIG["repo_name"])
        temp_table = temp_table_from_pandas(
            pd.DataFrame([{"id": id} for id in pushed_run_ids]), session=session
        )

        stmt = exists().where(temp_table.c.id == Run.id)
        if pushed:
            query = query.filter(stmt)
        else:
            query = query.filter(~stmt)
    else:
        temp_table = None

    query = query.order_by(desc(Run.start_time))
    if limit is not None:
        query = query.limit(limit)
    if df:
        from pandas import read_sql

        out = read_sql(query.statement, query.session.bind)
    else:
        out = query.all()

    session.close()
    return out


def get_artifact_dumps(
    run_ids: Union[int, List[int]] = None,
    artifact_ids: Union[int, List[int]] = None,
    date_range: Tuple[datetime] = None,
    df: bool = True,
):
    session = Session()
    query = session.query(ArtifactDump)

    if run_ids is not None:
        run_ids = [run_ids] if isinstance(run_ids, int) else run_ids
        query = query.filter(ArtifactDump.creating_run_id.in_(run_ids))

    if artifact_ids is not None:
        artifact_ids = [artifact_ids] if isinstance(artifact_ids, int) else artifact_ids
        query = query.filter(ArtifactDump.id.in_(artifact_ids))

    if date_range is not None:
        query = query.filter(ArtifactDump.dump_time > date_range[0])
        query = query.filter(ArtifactDump.dump_time < date_range[1])

    query = query.order_by(desc(ArtifactDump.dump_time))
    if df:
        from pandas import read_sql

        out = read_sql(query.statement, query.session.bind)
    else:
        out = query.all()

    session.close()
    return out


def get_artifact_loads(
    run_ids: Union[int, List[int]] = None,
    artifact_ids: Union[int, List[int]] = None,
    date_range: Tuple[datetime] = None,
    df: bool = True,
):
    session = Session()
    query = session.query(ArtifactLoad)

    if run_ids is not None:
        run_ids = [run_ids] if isinstance(run_ids, int) else run_ids
        query = query.filter(ArtifactLoad.loading_run_id.in_(run_ids))

    if artifact_ids is not None:
        artifact_ids = [artifact_ids] if isinstance(artifact_ids, int) else artifact_ids
        query = query.filter(ArtifactLoad.artifact_id.in_(artifact_ids))

    if date_range is not None:
        query = query.filter(ArtifactLoad.load_time > date_range[0])
        query = query.filter(ArtifactLoad.load_time < date_range[1])

    query = query.order_by(desc(ArtifactLoad.load_time))
    if df:
        from pandas import read_sql

        out = read_sql(query.statement, query.session.bind)
    else:
        out = query.all()

    session.close()
    return out


def rm_runs(run_ids: Union[int, List[int]]):
    run_ids = [run_ids] if isinstance(run_ids, int) else run_ids
    session = Session()
    query = session.query(Run).filter(Run.id.in_(run_ids))
    query.update({Run.status: "deleted"}, synchronize_session=False)
    session.commit()
    session.close()


def _init_google_cloud_sql_engine() -> sqlalchemy.engine.Engine:
    from google.cloud.sql.connector import Connector

    if TERRA_CONFIG.get("cloud_sql_connection", None) is None:
        raise ValueError(
            "Must provide `cloud_sql_connection` in config if using `local_db=False`"
        )
    

    # initialize Connector object
    connector = Connector()

    def getconn():
        conn = connector.connect(
            TERRA_CONFIG["cloud_sql_connection"],
            "pg8000",
            user=TERRA_CONFIG["user"],
            password=TERRA_CONFIG["password"],
            db=TERRA_CONFIG["db"],
        )
        return conn

    engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )
    engine.dialect.description_encoding = None
    return engine


def get_session(storage_dir: str = None, create: bool = True):

    storage_dir = TERRA_CONFIG["storage_dir"] if storage_dir is None else storage_dir
    ensure_dir_exists(storage_dir)
    if TERRA_CONFIG["local_db"]:
        db_path = os.path.join(storage_dir, "terra.sqlite")
        db_exists = os.path.exists(db_path)
        engine = sqlalchemy.create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            # https://docs.sqlalchemy.org/en/14/core/pooling.html#pooling-multiprocessing
            poolclass=NullPool,
            # increase the timeout to help avoid the database is locked error
            # https://stackoverflow.com/questions/15065037/how-to-increase-connection-timeout-using-sqlalchemy-with-sqlite-in-python
            connect_args={"timeout": 60},
        )
    else:
        engine = _init_google_cloud_sql_engine()
        db_exists = len(sqlalchemy.inspect(engine).get_table_names()) > 0

    if not db_exists:
        print("Creating database...")
        Base.metadata.create_all(engine)

    return sessionmaker(bind=engine)


Session = get_session()


def hash_inputs(encoded_inputs: str):
    return hashlib.sha1(encoded_inputs.encode("utf-8")).hexdigest()


def check_input_hash(input_hash: str, fn: str, module: str):
    import terra

    session = Session()

    query = (
        session.query(Run)
        .filter(Run.module == module)
        .filter(Run.fn == fn)
        .filter(Run.status == "success")
        .filter(Run.input_hash == input_hash)
        .order_by(desc(Run.start_time))
    )

    # if date_range is not None:
    #     query = query.filter(Run.start_time > date_range[0])
    #     query = query.filter(Run.start_time < date_range[1])
    if not terra.use_local:
        query = query.limit(1)

    runs = query.all()
    session.close()

    if len(runs) == 0:
        return None

    for run in runs:
        if os.path.isdir(to_abs_path(run.run_dir)):
            return run.id
    return runs[0].id


def migrate_local_db_to_cloud(
    storage_dir: str = None, batch_size: int = 1_000, tables: List[str] = None
):
    from tqdm import tqdm

    storage_dir = TERRA_CONFIG["storage_dir"] if storage_dir is None else storage_dir
    db_path = os.path.join(storage_dir, "terra.sqlite")
    sqlite_engine = sqlalchemy.create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        # https://docs.sqlalchemy.org/en/14/core/pooling.html#pooling-multiprocessing
        poolclass=NullPool,
        # increase the timeout to help avoid the database is locked error
        # https://stackoverflow.com/questions/15065037/how-to-increase-connection-timeout-using-sqlalchemy-with-sqlite-in-python
        connect_args={"timeout": 60},
    )

    cloud_engine = _init_google_cloud_sql_engine()

    Base.metadata.create_all(cloud_engine)

    with sqlite_engine.connect() as conn_lite:
        with cloud_engine.connect() as conn_cloud:
            for table in Base.metadata.sorted_tables:
                if tables is not None and str(table) not in tables:
                    continue

                print(f"Migrating table `{table}`...")
                data = [dict(row) for row in conn_lite.execute(select(table.c))]

                if str(table) == "artifact_loads":
                    # weird edge case ran into where some artifact_ids were none
                    # addressed by just dropping those rows
                    # TODO: remove this
                    data = [row for row in data if row["artifact_id"] != -1]
                for start_idx in tqdm(range(0, len(data), batch_size)):
                    conn_cloud.execute(
                        table.insert().values(data[start_idx : start_idx + batch_size])
                    )


def subquery_from_records(records: List[Dict], name: str = "temp"):
    # source: https://stackoverflow.com/questions/44140632/use-temp-table-with-sqlalchemy
    assert name not in ["runs", "artifact_dumps", "artifact_loads", "refs"]

    stmts = [
        select([literal(v).label(k) for k, v in row.items()])
        if idx == 0
        else select([literal(v) for v in row.values()])  # no cast on subsequent rowas
        for idx, row in enumerate(records)
    ]

    subquery = union_all(*stmts)
    subquery = subquery.cte(name=name)
    return subquery


def temp_table_from_pandas(df: "pd.DataFrame", session=None, name: str = "temp"):
    if session is None:
        session = Session()
    import pandas as pd

    type_to_sql_type = {
        pd.StringDtype: String,
        pd.Int64Dtype: Integer,
        pd.Int32Dtype: Integer,
        pd.Int16Dtype: Integer,
        pd.Int8Dtype: Integer,
        pd.Float64Dtype: Float,
        pd.Float32Dtype: Float,
        pd.BooleanDtype: Boolean,
        pd.DatetimeTZDtype: DateTime,
        pd.CategoricalDtype: String,
    }
    name = f"{name}#temp_table"
    columns = [
        Column(c, _pandas_type_to_sqlalchemy_type(t)) for c, t in df.dtypes.items()
    ]
    temp_table = Table(
        name,
        Base.metadata,
        *columns,
        prefixes=["TEMPORARY"],
    )
    temp_table.create(session.bind)
    num_rows = df.to_sql(
        name=name,
        con=session.bind,
        if_exists="append",
        method="multi",  # https://stackoverflow.com/questions/29706278/python-pandas-to-sql-with-sqlalchemy-how-to-speed-up-exporting-to-ms-sql
        chunksize=10_000,
        index=False,
    )

    return temp_table


def _pandas_type_to_sqlalchemy_type(pd_type):
    import pandas as pd
    import numpy as np

    if pd.StringDtype == pd_type:
        return String
    elif np.dtype(int) == pd_type:
        return Integer
    elif np.dtype(float) == pd_type:
        return Float
    return String
