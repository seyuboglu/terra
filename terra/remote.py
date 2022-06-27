import multiprocessing as mp
import os
import re
import subprocess
import io
from datetime import datetime
from logging import warn
from typing import Collection, List, Tuple, Union, Any
import tarfile

from tqdm import tqdm

import terra.database as tdb
from terra.settings import TERRA_CONFIG
from terra.tools.lazy import LazyLoader
from terra.utils import to_abs_path, to_rel_path

storage = LazyLoader("google.cloud.storage")
exceptions = LazyLoader("google.cloud.exceptions")


def _upload_dir_to_gcs(
    local_path: str,
    gcs_path: str,
    bucket: Any = None,
    bucket_name: str = None,
    compress: bool = False,
):
    if (bucket is None) == (bucket_name is None):
        raise ValueError("Either `bucket` or `bucket_name` must be specified.")

    if bucket is None:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

    assert os.path.isdir(local_path)

    tarball = io.BytesIO()
    with tarfile.open(fileobj=tarball, mode="w:gz" if compress else "w:") as tar:
        tar.add(local_path, arcname=os.path.basename(local_path), recursive=True)
        remote_path = gcs_path + ".tar.gz"
        blob = bucket.blob(remote_path)
        blob.upload_from_file(tarball, rewind=True)


def _download_dir_from_gcs(
    local_path: str, gcs_path: str, bucket: Any = None, bucket_name: str = None
):
    if (bucket is None) == (bucket_name is None):
        raise ValueError("Either `bucket` or `bucket_name` must be specified.")

    if bucket is None:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
    gcs_path + ".tar.gz"
    blob = bucket.blob(gcs_path + ".tar.gz")

    os.makedirs(local_path, exist_ok=True)

    tarball = io.BytesIO()
    blob.download_to_file(tarball)
    tarball.seek(0)  # need to rewind the buffer for tarfile to read it
    with tarfile.open(fileobj=tarball, mode="r:*") as tar:
        tar.extractall(path=os.path.dirname(local_path))


def _get_pushed_run_ids(bucket_name: str):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = " ".join(blob.name for blob in bucket.list_blobs())
    return set(map(int, re.findall(r"_runs\/(\d+).tar.gz", blobs)))


def push(
    run_ids: Union[int, Collection[int]] = None,
    modules: Union[str, List[str]] = None,
    fns: Union[str, List[str]] = None,
    statuses: Union[str, List[str]] = None,
    date_range: Tuple[datetime] = None,
    pushed: bool = None,
    limit: int = None,
    bucket_name: str = None,
    force: bool = False,
    warn_missing: bool = False, 
    num_workers: bool = 0,
):
    if bucket_name is None:
        bucket_name = TERRA_CONFIG["repo_name"]
        if bucket_name is None:
            raise ValueError(
                "Cannot push because no `repo_name` specified in terra config."
            )

    if not force:
        pushed_run_ids = _get_pushed_run_ids(bucket_name=bucket_name)

    runs = tdb.get_runs(
        run_ids=run_ids,
        modules=modules,
        fns=fns,
        statuses=statuses,
        date_range=date_range,
        pushed=pushed,
        limit=limit,
        df=False,
    )
    if num_workers > 0:
        pool = mp.Pool(processes=num_workers)
        async_results = []

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    for idx, run in enumerate(runs):
        if run.status == "in_progress":
            warn(
                f"({idx}/{len(runs)}) Skipping run_id={run.id} because the run's in progress."
            )
            continue
        if not force and (run.id in pushed_run_ids):
            print(
                f'({idx}/{len(runs)}) Skipping run_id={run.id}, already pushed to bucket "{bucket_name}".'
            )
            continue
        if run.run_dir is None:
            continue

        rel_path = to_rel_path(run.run_dir)
        abs_path = to_abs_path(run.run_dir)

        if not os.path.isdir(abs_path):
            msg = (
                f"Cannot push run_id={run.id}, it is not stored on this remote."
                f" Try pushing from host '{run.hostname}'."
            )
            if warn_missing:
                warn(msg)
                continue 
            else:
                raise ValueError(msg)
            

        if num_workers > 0:
            result = pool.apply_async(
                func=_upload_dir_to_gcs,
                kwds=dict(
                    local_path=abs_path,
                    bucket=bucket,
                    gcs_path=rel_path,
                ),
            )
            async_results.append(result)
        else:
            print(
                f'({idx}/{len(runs)}) Pushing run_id={run.id} to bucket "{bucket_name}" at "{abs_path}".'
            )
            _upload_dir_to_gcs(
                local_path=abs_path,
                bucket=bucket,
                gcs_path=rel_path,
            )

    if num_workers > 0:
        [result.get() for result in tqdm(async_results)]


def pull(
    run_ids: Union[int, Collection[int]] = None,
    modules: Union[str, List[str]] = None,
    fns: Union[str, List[str]] = None,
    statuses: Union[str, List[str]] = None,
    date_range: Tuple[datetime] = None,
    limit: int = None,
    bucket_name: str = None,
    force: bool = False,
):
    if bucket_name is None:
        bucket_name = TERRA_CONFIG["repo_name"]
        if bucket_name is None:
            raise ValueError(
                "Cannot push because no `repo_name` specified in terra config."
            )
    runs = tdb.get_runs(
        run_ids=run_ids,
        modules=modules,
        fns=fns,
        statuses=statuses,
        date_range=date_range,
        limit=limit,
        df=False,
    )

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    for run in runs:
        if run.status == "in_progress":
            warn(f"Skipping run_id={run.id} because the run's in progress.")
            continue

        rel_path = to_rel_path(run.run_dir)
        abs_path = to_abs_path(rel_path)

        if os.path.isdir(abs_path) and not force:
            print(
                f'Skipping run_id={run.id}, already pulled from bucket "{bucket_name}"'
                f' at path "{abs_path}".'
            )
            continue

        print(
            f'Pulling run_id={run.id} from bucket "{bucket_name} at path "{abs_path}".'
        )
        try:
            _download_dir_from_gcs(
                local_path=abs_path,
                bucket=bucket,
                gcs_path=rel_path,
            )
        except exceptions.NotFound:
            raise ValueError(
                f"Cannot pull run_id={run.id}, it is not stored in bucket"
                f" '{bucket_name}'. Try pushing from host '{run.hostname}'."
            )
