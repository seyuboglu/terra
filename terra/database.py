import os

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String,Boolean, DateTime
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


def _get_session(storage_dir: str = TERRA_CONFIG["storage_dir"], create: bool = True):
    print(storage_dir)
    ensure_dir_exists(storage_dir)
    db_path = os.path.join(storage_dir, "terra.sqlite")
    engine = sqlalchemy.create_engine(
        f"sqlite://{os.path.join(storage_dir, 'terra.sqlite')}", echo=False
    )
    if not os.path.exists(db_path):
        if create:
            Base.metadata.create_all(engine)
        else:
            raise ValueError(
                f"No database for storage dir {storage_dir}." "Set `create=True`"
            )

    return sessionmaker(bind=engine)


class TerraDatabase:
    def __init__(self, storage_dir=TERRA_CONFIG["storage_dir"], create: bool = True):
        self.Session = _get_session(storage_dir=storage_dir, create=create)

    def add_run(self, run_dir="test/dir/1"):
        session = self.Session()
        session.add(Run(run_dir=run_dir, status="in_progress"))
        session.commit()
