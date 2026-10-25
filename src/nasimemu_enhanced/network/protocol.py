"""Shared vocabulary of the dm_env_rpc transport.

The :mod:`nasimemu_enhanced.network` server and client exchange the same
environment over gRPC using Google DeepMind's ``dm_env_rpc`` protocol. This
module holds everything both sides have to agree on: the names of the world
settings, the names and data types of the action and observation tensors, and
small helpers to pack and unpack them. Keeping it in one place means the two
sides cannot drift apart.

Notes
-----
**Why a raw matrix is always sent.** The observation an agent sees
(``observation`` in the matrix format, the ``graph_*`` tensors in a graph
format) is the environment's formatted observation. In addition the server
always sends ``s_raw``, the raw host matrix before augmentation or graph
conversion, so the client can expose :attr:`NASimEmuEnv.s_raw` unchanged. The
regression sweep compares ``s_raw``, so this keeps a remote run byte-for-byte
comparable with a local one.

**Why the info travels as JSON.** :meth:`NASimEmuEnv.step` already returns plain,
JSON-serialisable data (NASimEmu-Enhanced 2.0.0), and :meth:`reset` returns the
``layout`` and ``actions`` catalogue. Both are sent as a single ``STRING`` tensor
holding their JSON, rather than as a forest of typed tensors, because the server
and client are the same code base and gain nothing from re-typing it.

**Variable-length tensors.** Every tensor carries exactly one variable
dimension (the number of hosts, nodes or edges), which ``dm_env_rpc`` allows and
marks with a negative size in the spec.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from dm_env_rpc.v1 import dm_env_rpc_pb2, tensor_utils

# ---------------------------------------------------------------------------
# World-creation settings (CreateWorldRequest.settings), one tensor each.
# ---------------------------------------------------------------------------
SETTING_SCENARIO_NAME = 'scenario_name'          # base file name, e.g. 'uni.v2.yaml'
SETTING_SCENARIO_YAML = 'scenario_yaml'          # the scenario file content
SETTING_EMULATE = 'emulate'                      # 0 / 1
SETTING_STEP_LIMIT = 'step_limit'                # -1 means no limit
SETTING_RANDOM_INIT = 'random_init'              # 0 / 1
SETTING_OBSERVATION_FORMAT = 'observation_format'
SETTING_FULLY_OBS = 'fully_obs'                  # 0 / 1
SETTING_AUGMENT_WITH_ACTION = 'augment_with_action'  # 0 / 1

# Reset settings (ResetRequest.settings).
SETTING_SEED = 'seed'                            # -1 means no seed

# ---------------------------------------------------------------------------
# Tensor names. The action is a single (subnet, host, action_id) vector; the
# observations depend on the format chosen at world creation.
# ---------------------------------------------------------------------------
ACTION = 'action'

OBS_S_RAW = 's_raw'
OBS_REWARD = 'reward'
OBS_INFO = 'info'
OBS_STATE = 'state'           # 0 running, 1 terminated, 2 truncated

OBS_MATRIX = 'observation'

OBS_GRAPH_NODES = 'graph_nodes'
OBS_GRAPH_EDGES = 'graph_edges'
OBS_GRAPH_EDGE_LINKS = 'graph_edge_links'
OBS_GRAPH_NODE_INDEX = 'node_index'
OBS_GRAPH_POS_INDEX = 'pos_index'

# Episode-state codes carried in the OBS_STATE tensor. The dm_env_rpc
# EnvironmentStateType also carries this (RUNNING / TERMINATED / INTERRUPTED),
# but a plain code keeps the gymnasium terminated/truncated split explicit.
STATE_RUNNING = 0
STATE_TERMINATED = 1
STATE_TRUNCATED = 2


def is_graph_format(observation_format: str) -> bool:
    """Return whether an observation format produces a graph observation."""
    return observation_format in ('graph', 'graph_v1', 'graph_v2')


def pack_setting(value: Any) -> dm_env_rpc_pb2.Tensor:
    """Pack a world or reset setting into a tensor.

    Strings become ``STRING`` tensors, everything else an ``INT64`` tensor, so
    that integers survive the round trip without a float detour.
    """
    if isinstance(value, str):
        return tensor_utils.pack_tensor(value, dtype=dm_env_rpc_pb2.STRING)
    return tensor_utils.pack_tensor(int(value), dtype=dm_env_rpc_pb2.INT64)


def unpack_scalar(tensor: dm_env_rpc_pb2.Tensor) -> Any:
    """Unpack a scalar tensor into a plain Python value."""
    value = tensor_utils.unpack_tensor(tensor)
    return np.asarray(value).reshape(-1)[0].item()


def dumps_info(info: dict[str, Any]) -> str:
    """Serialise a reset or step info dict to JSON."""
    return json.dumps(info)


def loads_info(text: str) -> dict[str, Any]:
    """Parse a reset or step info dict from JSON."""
    info: dict[str, Any] = json.loads(text)
    return info
