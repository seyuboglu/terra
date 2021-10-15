import glob
import os
from datetime import datetime
from typing import Collection, List, Tuple, Union

import terra.database as tdb
from terra.settings import TERRA_CONFIG
from terra.tools.lazy import LazyLoader
from terra.utils import to_abs_path, to_rel_path

storage = LazyLoader("google.cloud.storage")


def _upload_dir_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + "/**"):
        if not os.path.isfile(local_file):
            _upload_dir_to_gcs(
                local_file, bucket, gcs_path + "/" + os.path.basename(local_file)
            )
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path) :])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def _download_dir_from_gcs(local_path: str, bucket_name: str, gcs_path: str):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_path)
    for blob in blobs:
        path = os.path.join(local_path, os.path.relpath(blob.name, gcs_path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blob.download_to_filename(path)


def _gcs_dir_exists(gcs_path: str, bucket_name: str):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # it seems like this is the best way to check for existence of directory, using
    # weird for loop structure since we don't want to iterate over the whole list
    for _ in bucket.list_blobs(prefix=gcs_path):
        return True
    return False


def push(
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
    for run in runs:
        if run.status == "in_progress":
            raise ValueError(
                f"Cannot push run_id={run.id} because the run's in progress."
            )
        rel_path = to_rel_path(run.run_dir)
        abs_path = to_abs_path(run.run_dir)

        if _gcs_dir_exists(rel_path, bucket_name=bucket_name) and not force:
            print(
                f'Skipping run_id={run.id}, already pushed to bucket "{bucket_name}"'
                f' at path "{rel_path}".'
            )
            continue

        if not os.path.isdir(abs_path):
            raise ValueError(
                f"Cannot push run_id={run.id}, it is not stored on this remote."
                f" Try pushing from host '{run.hostname}'."
            )

        print(
            f'Pushing run_id={run.id} to bucket "{bucket_name}"" at path "{rel_path}".'
        )
        _upload_dir_to_gcs(
            abs_path,
            bucket_name=bucket_name,
            gcs_path=rel_path,
        )


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
    for run in runs:
        rel_path = to_rel_path(run.run_dir)
        abs_path = to_abs_path(rel_path)

        if os.path.isdir(abs_path) and not force:
            print(
                f'Skipping run_id={run.id}, already pulled from bucket "{bucket_name}"'
                f' at path "{rel_path}".'
            )
            continue

        if not _gcs_dir_exists(rel_path, bucket_name=bucket_name):
            raise ValueError(
                f"Cannot pull run_id={run.id}, it is not stored in bucket"
                f" '{bucket_name}'. Try pushing from host '{run.hostname}'."
            )
        print(
            f'Pulling run_id={run.id} from bucket "{bucket_name} at path "{rel_path}".'
        )
        _download_dir_from_gcs(
            local_path=abs_path,
            bucket_name=bucket_name,
            gcs_path=rel_path,
        )
