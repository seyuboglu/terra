import warnings
from typing import Dict, Mapping, Sequence, Tuple

import terra
from terra.io import Artifact
from terra.remote import pull


def get_nested_artifacts(data):
    """Recursively get DataPanels and Columns from nested collections."""
    objs = {}
    _get_nested_artifacts(objs, (), data)
    return objs


def _get_nested_artifacts(objs: Dict, key: Tuple[str], data: object):
    if isinstance(data, Sequence) and not isinstance(data, str):
        for idx, item in enumerate(data):
            _get_nested_artifacts(objs, key=(*key, idx), data=item)

    elif isinstance(data, Mapping):
        for curr_key, item in data.items():
            _get_nested_artifacts(objs, key=(*key, curr_key), data=item)
    elif isinstance(data, Artifact):
        objs[key] = data


def provenance(run_id: int, pull_missing_runs: bool = False):
    artifacts = {}
    runs = {}
    edges = []
    print(pull_missing_runs)
    _get_provenance(
        run_id=run_id,
        artifacts=artifacts,
        runs=runs,
        edges=edges,
        pull_missing_runs=pull_missing_runs,
    )
    return artifacts, runs, edges


def _get_provenance(
    run_id: int,
    artifacts: Dict[str, Dict],
    runs: Dict[str, Dict],
    edges: Dict[str, Dict],
    pull_missing_runs: bool,
):
    if pull_missing_runs:
        pull(run_id)

    meta = terra.get_meta(run_id)
    runs[run_id] = {"id": run_id, "module": meta["module"], "fn": meta["fn"]}
    for key, artifact in get_nested_artifacts(terra.out(run_id)).items():
        artifacts[artifact.id] = {
            "type": artifact.type,
            "id": artifact.id,
            "run_id": artifact.run_id,
        }
        edges.append({"source": run_id, "target": artifact.id, "key": key})

    for key, artifact in get_nested_artifacts(terra.inp(run_id)).items():
        edges.append({"source": artifact.id, "target": run_id, "key": key})
        if artifact.run_id not in runs:
            _get_provenance(
                artifact.run_id,
                artifacts=artifacts,
                runs=runs,
                edges=edges,
                pull_missing_runs=pull_missing_runs,
            )
    return artifacts, runs, edges


def visualize_provenance(
    artifacts: Dict[str, Dict],
    runs: Dict[str, Dict],
    edges: Dict[str, Dict],
    show_columns: bool = False,
    last_parent_only: bool = False,
):

    warnings.warn(  # pragma: no cover
        Warning(
            "The function `terra.provenance.visualize_provenance` is experimental and"
            " has limited test coverage. Proceed with caution."
        )
    )
    try:  # pragma: no cover
        import cyjupyter
    except ImportError:  # pragma: no cover
        raise ImportError(
            "`visualize_provenance` requires the `cyjupyter` dependency."
            "See https://github.com/cytoscape/cytoscape-jupyter-widget"
        )

    cy_nodes = [  # pragma: no cover
        {
            "data": {
                "id": artifact["id"],
                "name": f'{artifact["type"]}',
                "type": "artifact",
            }
        }
        for artifact in artifacts.values()
    ] + [
        {
            "data": {
                "id": run["id"],
                "name": f"{run['module']}.{run['fn']}, run_id={run['id']}",
                "type": "run",
            }
        }
        for run in runs.values()
    ]
    cy_edges = [{"data": edge} for edge in edges]

    cy_data = {"elements": {"nodes": cy_nodes, "edges": cy_edges}}  # pragma: no cover
    style = [  # pragma: no cover
        {
            "selector": "node",
            "css": {
                "content": "data(name)",
                "background-color": "#fc8d62",
                "border-color": "#252525",
                "border-opacity": 1.0,
                "border-width": 3,
            },
        },
        {
            "selector": "node[type = 'artifact']",
            "css": {
                "shape": "barrel",
                "background-color": "#8da0cb",
            },
        },
        # {
        #     # need to double index to access metadata (degree etc.)
        #     "selector": "node[[degree = 0]]",
        #     "css": {
        #         "visibility": "hidden",
        #     },
        # },
        {
            "selector": "edge",
            "css": {
                "content": "data(key)",
                "line-color": "#252525",
                "mid-target-arrow-color": "#8da0cb",
                "mid-target-arrow-shape": "triangle",
                "arrow-scale": 2.5,
                "text-margin-x": 10,
                "text-margin-y": 10,
            },
        },
    ]
    return cyjupyter.Cytoscape(  # pragma: no cover
        data=cy_data, visual_style=style, layout_name="breadthfirst"
    )
