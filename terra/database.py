import hashlib
import os
from datetime import datetime
from time import sleep
from typing import List, Tuple, Union

import sqlalchemy
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from terra.settings import TERRA_CONFIG, TerraDatabaseSettings
from terra.utils import ensure_dir_exists

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
    date_range: Tuple[datetime] = None,
    df: bool = True,
):
    session = Session()
    query = session.query(ArtifactDump)

    if run_ids is not None:
        run_ids = [run_ids] if isinstance(run_ids, int) else run_ids
        query = query.filter(ArtifactDump.creating_run_id.in_(run_ids))

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
    date_range: Tuple[datetime] = None,
    df: bool = True,
):
    session = Session()
    query = session.query(ArtifactLoad)

    if run_ids is not None:
        run_ids = [run_ids] if isinstance(run_ids, int) else run_ids
        query = query.filter(ArtifactLoad.loading_run_id.in_(run_ids))

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


def get_session(storage_dir: str = None, create: bool = True):

    storage_dir = TERRA_CONFIG["storage_dir"] if storage_dir is None else storage_dir
    ensure_dir_exists(storage_dir)
    db_path = os.path.join(storage_dir, "terra.sqlite")
    db_exists = os.path.exists(db_path)
    engine = sqlalchemy.create_engine(
        f"sqlite:///{os.path.join(storage_dir, 'terra.sqlite')}",
        echo=False,
        # https://docs.sqlalchemy.org/en/14/core/pooling.html#pooling-multiprocessing
        poolclass=NullPool,
        # increase the timeout to help avoid the database is locked error
        # https://stackoverflow.com/questions/15065037/how-to-increase-connection-timeout-using-sqlalchemy-with-sqlite-in-python
        connect_args={"timeout": 60},
    )
    if not db_exists:
        if create:
            Base.metadata.create_all(engine)
        else:
            raise ValueError(
                f"No database for storage dir {storage_dir}." "Set `create=True`"
            )

    return sessionmaker(bind=engine)


def hash_inputs(encoded_inputs: str):
    return hashlib.sha1(encoded_inputs.encode("utf-8")).hexdigest()


def check_input_hash(input_hash: str, fn: str, module: str):
    session = Session()

    query = (
        session.query(Run)
        .filter(Run.module == module)
        .filter(Run.fn == fn)
        .filter(Run.status == "success")
        .filter(Run.input_hash == input_hash)
        .order_by(desc(Run.start_time))
        .limit(1)
    )

    # if date_range is not None:
    #     query = query.filter(Run.start_time > date_range[0])
    #     query = query.filter(Run.start_time < date_range[1])

    out = query.all()
    session.close()
    return out[0].id if len(out) > 0 else None


Session = get_session()
