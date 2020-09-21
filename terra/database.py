import os

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


class Ref(Base):
    __tablename__ = "refs"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    run_dir = Column(String)


def get_session(storage_dir: str = None, create: bool = True):

    storage_dir = (
        TERRA_CONFIG["storage_dir"] if storage_dir is None else storage_dir
    )

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


