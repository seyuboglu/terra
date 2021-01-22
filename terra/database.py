import os
from typing import Union, List, Tuple
from datetime import datetime


import pandas as pd
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Boolean, DateTime, desc, ForeignKey
from sqlalchemy.orm import sessionmaker

from terra.settings import TERRA_CONFIG
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
    slurm_job_id = Column(Integer)

    def get_summary(self):
        return f"module={self.module}, fn={self.fn}, status={self.status}, run_dir={self.run_dir}"


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


def get_runs(
    run_ids: Union[int, List[int]] = None,
    modules: Union[str, List[str]] = None,
    fns: Union[str, List[str]] = None,
    statuses: Union[str, List[str]] = None,
    date_range: Tuple[datetime] = None,
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
    if df:
        out = pd.read_sql(query.statement, query.session.bind)
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
        out = pd.read_sql(query.statement, query.session.bind)
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
        out = pd.read_sql(query.statement, query.session.bind)
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
        f"sqlite:///{os.path.join(storage_dir, 'terra.sqlite')}", echo=False
    )
    if not db_exists:
        if create:
            Base.metadata.create_all(engine)
        else:
            raise ValueError(
                f"No database for storage dir {storage_dir}." "Set `create=True`"
            )

    return sessionmaker(bind=engine)


Session = get_session()
