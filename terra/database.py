import os
from typing import Union, List

import pandas as pd
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Boolean, DateTime
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


class Ref(Base):
    __tablename__ = "refs"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    run_dir = Column(String)


class TerraDatabase:
    def __init__(self, create: bool = True):
        self.Session = get_session(create=create)

    def get_runs(
        self,
        run_ids: Union[int, List[int]] = None,
        modules: Union[str, List[str]] = None,
        fns: Union[str, List[str]] = None,
    ) -> List[Run]:
        query = self.Session().query(Run)

        if run_ids is not None:
            run_ids = [run_ids] if isinstance(run_ids, int) else run_ids
            query = query.filter(Run.id.in_(run_ids))

        if modules is not None:
            modules = [modules] if isinstance(modules, str) else modules
            query = query.filter(Run.module.in_(modules))

        if fns is not None:
            fns = [fns] if isinstance(fns, str) else fns
            query = query.filter(Run.module.in_(fns))

        return query.all()

    def rm_runs(self, run_ids: Union[int, List[int]]):
        run_ids = [run_ids] if isinstance(run_ids, int) else run_ids
        session = self.Session()
        query = session.query(Run).filter(Run.id.in_(run_ids))
        query.update({"status": "deleted"})
        session.commit()


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
