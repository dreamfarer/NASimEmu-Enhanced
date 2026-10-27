"""Shared vocabulary of the dm_env_rpc transport.

The :mod:`nasimemu_enhanced.network` server and client exchange the same
environment over gRPC using Google DeepMind's ``dm_env_rpc`` protocol. This
module holds everything both sides have to agree on so that they cannot drift
apart: the names of the world and reset settings, the names, data types and
shapes of the action and observation tensors, the conversion between the
environment's ``step`` info and its typed tensors, and the property tree that
carries the ``reset`` metadata.

Design
------
**Typed observations, no JSON.** Every value an agent needs after a step is a
typed tensor: the formatted observation (``observation`` for the matrix format,
``graph.*`` for a graph format), the raw host matrix ``s_raw``, the ``reward``
and one tensor per field of the ``step`` info (``info.*``). A stock
``dm_env_rpc`` client such as :class:`~dm_env_rpc.v1.dm_env_adaptor.DmEnvAdaptor`
therefore receives the whole step as named, typed tensors and does not have to
know a private JSON schema. The dotted names (``graph.nodes``,
``info.services.names``) are what that adaptor turns back into nested dicts.

**The reward is named ``reward``.** ``DmEnvAdaptor`` treats an observation named
``reward`` as the step reward, so the name is kept.

**Episode state is the response state.** The dm_env_rpc ``StepResponse`` already
carries whether the episode is running, terminated or interrupted, so no extra
tensor encodes it.

**Metadata is a property tree.** The ``reset`` info (the host-row ``layout`` and
the ``actions`` catalogue) is static for a world, so it is served through the
dm_env_rpc properties extension rather than resent every step. The catalogue is
treated as generic records: the data type of every leaf is inferred from its
Python value, so no part of this module knows what an action is.

**Variable-length tensors.** Every tensor carries at most one variable
dimension (the number of hosts, nodes, edges or named values), which
``dm_env_rpc`` allows and marks with a negative size in the spec.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from dm_env_rpc.v1 import dm_env_rpc_pb2, tensor_utils
from numpy.typing import NDArray

SETTING_SCENARIO_NAME = 'scenario_name'  # base file name, e.g. 'uni.v2.yaml'
SETTING_SCENARIO_YAML = 'scenario_yaml'  # the scenario file content
SETTING_EMULATE = 'emulate'  # 0 / 1
SETTING_STEP_LIMIT = 'step_limit'  # -1 means no limit
SETTING_RANDOM_INIT = 'random_init'  # 0 / 1
SETTING_OBSERVATION_FORMAT = 'observation_format'
SETTING_FULLY_OBS = 'fully_obs'  # 0 / 1
SETTING_AUGMENT_WITH_ACTION = 'augment_with_action'  # 0 / 1

WORLD_SETTINGS = frozenset({
    SETTING_SCENARIO_NAME, SETTING_SCENARIO_YAML, SETTING_EMULATE, SETTING_STEP_LIMIT,
    SETTING_RANDOM_INIT, SETTING_OBSERVATION_FORMAT, SETTING_FULLY_OBS, SETTING_AUGMENT_WITH_ACTION,
})

SETTING_SEED = 'seed'  # -1 means no seed
RESET_SETTINGS = frozenset({SETTING_SEED})

ACTION = 'action'

OBS_REWARD = 'reward'
OBS_S_RAW = 's_raw'
OBS_MATRIX = 'observation'
OBS_GRAPH_NODES = 'graph.nodes'
OBS_GRAPH_EDGES = 'graph.edges'
OBS_GRAPH_EDGE_LINKS = 'graph.edge_links'
OBS_GRAPH_NODE_INDEX = 'node_index'
OBS_GRAPH_POS_INDEX = 'pos_index'

INFO_ACTION = 'info.action'
INFO_SUCCESS = 'info.success'
INFO_CONNECTION_ERROR = 'info.connection_error'
INFO_PERMISSION_ERROR = 'info.permission_error'
INFO_UNDEFINED_ERROR = 'info.undefined_error'
INFO_VALUE = 'info.value'
INFO_ACCESS = 'info.access'
INFO_STEP_IDX = 'info.step_idx'
INFO_R_TOT = 'info.r_tot'
INFO_CAPTURED = 'info.captured'
INFO_DISCOVERED = 'info.discovered'
INFO_NEWLY_DISCOVERED = 'info.newly_discovered'
INFO_SUBNET_GRAPH = 'info.subnet_graph'

INFO_MAPS = (
    ('services', 'info.services.names', 'info.services.values'),
    ('os', 'info.os.names', 'info.os.values'),
    ('processes', 'info.processes.names', 'info.processes.values'),
)

PROP_LAYOUT = 'layout'
PROP_ACTIONS = 'actions'
PROP_SEPARATOR = '.'

# A name (a service, OS or process label, or a column name) may be ``None`` in a
# scenario, which a STRING tensor cannot hold. It is carried as this sentinel,
# which starts with a null byte and so cannot collide with a scenario identifier.
_NONE = '\x00__none__'


def encode_name(name: Any) -> str:
    """Encode a name that may be ``None`` as a string for a STRING tensor."""
    return _NONE if name is None else str(name)


def decode_name(name: Any) -> Any:
    """Decode a name carried by a STRING tensor, back to ``None`` where needed."""
    return None if name == _NONE else str(name)


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


def observation_spec_descriptors(raw_cols: int, is_graph: bool, row_cols: int | None,
                                 node_cols: int | None, max_node_index: int
                                 ) -> list[tuple[str, int, list[int], tuple[Any, Any] | None]]:
    """Describe every observation tensor the server declares.

    Returns one ``(name, dtype, shape, bounds)`` per tensor, where ``bounds`` is
    ``None`` or a ``(minimum, maximum)`` pair. A negative shape entry marks the
    one variable dimension. The order and the graph bounds match the spaces that
    :class:`nasimemu_enhanced.env.NASimEmuEnv` declares locally, so the client's
    reconstructed spaces are identical.
    """
    double, int64, string, boolean = (
        dm_env_rpc_pb2.DOUBLE, dm_env_rpc_pb2.INT64, dm_env_rpc_pb2.STRING, dm_env_rpc_pb2.BOOL)

    specs: list[tuple[str, int, list[int], tuple[Any, Any] | None]] = [
        (OBS_REWARD, double, [], None),
        (OBS_S_RAW, double, [-1, raw_cols], None),
    ]
    if is_graph:
        assert node_cols is not None
        specs += [
            (OBS_GRAPH_NODES, double, [-1, node_cols], None),
            (OBS_GRAPH_EDGES, int64, [-1], None),
            (OBS_GRAPH_EDGE_LINKS, int64, [-1, 2], None),
            (OBS_GRAPH_NODE_INDEX, int64, [-1, 2], (-1, max_node_index)),
            (OBS_GRAPH_POS_INDEX, double, [-1], (0.0, np.inf)),
        ]
    else:
        assert row_cols is not None
        specs.append((OBS_MATRIX, double, [-1, row_cols], None))

    specs += [
        (INFO_ACTION, int64, [3], None),
        (INFO_SUCCESS, boolean, [], None),
        (INFO_CONNECTION_ERROR, boolean, [], None),
        (INFO_PERMISSION_ERROR, boolean, [], None),
        (INFO_UNDEFINED_ERROR, boolean, [], None),
        (INFO_VALUE, double, [], None),
        (INFO_ACCESS, int64, [], None),
        (INFO_STEP_IDX, int64, [], None),
        (INFO_R_TOT, double, [], None),
        (INFO_CAPTURED, int64, [], None),
        (INFO_DISCOVERED, int64, [-1, 2], None),
        (INFO_NEWLY_DISCOVERED, int64, [-1, 2], None),
        (INFO_SUBNET_GRAPH, int64, [-1, 2], None),
    ]
    for _key, names_name, values_name in INFO_MAPS:
        specs.append((names_name, string, [-1], None))
        specs.append((values_name, double, [-1], None))
    return specs


def _addr_array(pairs: Any) -> NDArray[Any]:
    """Turn a list of ``[a, b]`` pairs into an ``(n, 2)`` int64 array."""
    pairs = list(pairs)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def info_to_tensors(info: dict[str, Any]) -> dict[str, Any]:
    """Convert a ``step`` info dict into the named observation values.

    The inverse of :func:`tensors_to_info`. Every returned value is a NumPy
    scalar or array with the data type its spec declares, so it can be packed
    straight through a :class:`~dm_env_rpc.v1.spec_manager.SpecManager`.
    """
    tensors: dict[str, Any] = {
        INFO_ACTION: np.asarray(info['action'], dtype=np.int64).reshape(3),
        INFO_SUCCESS: np.bool_(info['success']),
        INFO_CONNECTION_ERROR: np.bool_(info['connection_error']),
        INFO_PERMISSION_ERROR: np.bool_(info['permission_error']),
        INFO_UNDEFINED_ERROR: np.bool_(info['undefined_error']),
        INFO_VALUE: np.float64(info['value']),
        INFO_ACCESS: np.int64(info['access']),
        INFO_STEP_IDX: np.int64(info['step_idx']),
        INFO_R_TOT: np.float64(info['r_tot']),
        INFO_CAPTURED: np.int64(info['captured']),
        INFO_DISCOVERED: _addr_array(info['discovered']),
        INFO_NEWLY_DISCOVERED: _addr_array(info['newly_discovered']),
        INFO_SUBNET_GRAPH: _addr_array(info['subnet_graph']),
    }
    for key, names_name, values_name in INFO_MAPS:
        mapping: dict[str, float] = info[key]
        tensors[names_name] = np.asarray([encode_name(k) for k in mapping], dtype=np.str_)
        tensors[values_name] = np.asarray(list(mapping.values()), dtype=np.float64)
    return tensors


def tensors_to_info(values: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a ``step`` info dict from the unpacked observation values.

    The inverse of :func:`info_to_tensors`; the result equals the dict the local
    :meth:`nasimemu_enhanced.env.NASimEmuEnv.step` returns.
    """
    info: dict[str, Any] = {
        'action': [int(x) for x in np.asarray(values[INFO_ACTION]).reshape(-1)],
        'success': bool(np.asarray(values[INFO_SUCCESS]).reshape(-1)[0]),
        'connection_error': bool(np.asarray(values[INFO_CONNECTION_ERROR]).reshape(-1)[0]),
        'permission_error': bool(np.asarray(values[INFO_PERMISSION_ERROR]).reshape(-1)[0]),
        'undefined_error': bool(np.asarray(values[INFO_UNDEFINED_ERROR]).reshape(-1)[0]),
        'value': float(np.asarray(values[INFO_VALUE]).reshape(-1)[0]),
        'access': int(np.asarray(values[INFO_ACCESS]).reshape(-1)[0]),
        'step_idx': int(np.asarray(values[INFO_STEP_IDX]).reshape(-1)[0]),
        'r_tot': float(np.asarray(values[INFO_R_TOT]).reshape(-1)[0]),
        'captured': int(np.asarray(values[INFO_CAPTURED]).reshape(-1)[0]),
        'discovered': [[int(a), int(b)] for a, b in _addr_array(np.asarray(values[INFO_DISCOVERED]).reshape(-1, 2))],
        'newly_discovered': [[int(a), int(b)] for a, b in
                             _addr_array(np.asarray(values[INFO_NEWLY_DISCOVERED]).reshape(-1, 2))],
        'subnet_graph': [[int(a), int(b)] for a, b in
                         _addr_array(np.asarray(values[INFO_SUBNET_GRAPH]).reshape(-1, 2))],
    }
    for key, names_name, values_name in INFO_MAPS:
        names = [decode_name(n) for n in np.asarray(values[names_name]).reshape(-1)]
        vals = [float(v) for v in np.asarray(values[values_name]).reshape(-1)]
        info[key] = dict(zip(names, vals))
    return info


def neutral_info() -> dict[str, Any]:
    """The step info of the initial observation, before any action is taken.

    The first observation of a sequence has no action result, so every field
    takes the value a failed, cost-free action would have. The client's
    :meth:`reset` returns the metadata instead of this, and ``dm_env_rpc`` marks
    the first step of a sequence as ``FIRST`` with no reward, so these values are
    never read as a real result; they only satisfy the tensor specs.
    """
    return {
        'action': [0, 0, 0], 'success': False, 'connection_error': False, 'permission_error': False,
        'undefined_error': False, 'value': 0.0, 'access': 0, 'step_idx': 0, 'r_tot': 0.0, 'captured': 0,
        'discovered': [], 'newly_discovered': [], 'subnet_graph': [], 'services': {}, 'os': {}, 'processes': {},
    }


def build_property_tree(metadata: dict[str, Any]) -> dict[str, Any]:
    """Turn the reset metadata into the nested dict the property store serves.

    ``layout`` is kept as is; the ``actions`` list becomes a dict keyed by the
    string of each ``action_id`` so that the catalogue is a listable node whose
    entries the client can read back in order.
    """
    return {
        PROP_LAYOUT: metadata['layout'],
        PROP_ACTIONS: {str(entry['id']): entry for entry in metadata['actions']},
    }


def reconstruct_metadata(tree: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the reset metadata from the nested dict the client walked.

    The inverse of :func:`build_property_tree`; the ``actions`` dict is returned
    to a list ordered by ``action_id``.
    """
    actions = tree[PROP_ACTIONS]
    return {
        'layout': tree[PROP_LAYOUT],
        'actions': [actions[str(i)] for i in range(len(actions))],
    }


def pack_property_value(value: Any) -> dm_env_rpc_pb2.Tensor:
    """Pack a property leaf, inferring the data type from the Python value.

    Scalars and lists of ``bool``, ``int``, ``float`` or ``str`` are supported;
    ``bool`` is checked before ``int`` because it is a subclass of it. Nothing
    here knows the meaning of the value, so the action catalogue stays a generic
    record.
    """
    array, dtype = _property_array(value)
    return tensor_utils.pack_tensor(array, dtype=dtype)


def property_spec(name: str, value: Any) -> dm_env_rpc_pb2.TensorSpec:
    """Build the tensor spec a property leaf is listed with."""
    array, dtype = _property_array(value)
    return dm_env_rpc_pb2.TensorSpec(name=name, shape=list(array.shape), dtype=dtype)


def decode_leaf(value: Any) -> Any:
    """Convert an unpacked property leaf back into a scalar or a list.

    String leaves may carry the ``None`` sentinel, which is decoded back to
    ``None`` (see :func:`decode_name`).
    """
    array = np.asarray(value)
    if array.dtype.kind in ('U', 'S', 'O'):  # a string leaf
        if array.ndim == 0:
            return decode_name(array.item())
        return [decode_name(x) for x in array.tolist()]
    return array.item() if array.ndim == 0 else array.tolist()


def _property_array(value: Any) -> tuple[NDArray[Any], int]:
    """Return the NumPy array and dm_env_rpc dtype a property value packs to.

    A ``None`` (on its own or inside a string list) is carried as the ``None``
    sentinel, since a STRING tensor cannot hold ``None``.
    """
    is_list = isinstance(value, (list, tuple))
    items = list(value) if is_list else [value]
    dtype = _infer_dtype(items)
    if dtype == int(dm_env_rpc_pb2.STRING):
        data: list[Any] = [encode_name(x) for x in items]
        np_dtype: Any = np.str_
    else:
        data = items
        np_dtype = {int(dm_env_rpc_pb2.BOOL): np.bool_, int(dm_env_rpc_pb2.INT64): np.int64,
                    int(dm_env_rpc_pb2.DOUBLE): np.float64}[dtype]
    array = np.asarray(data, dtype=np_dtype) if is_list else np.asarray(data[0], dtype=np_dtype)
    return array, dtype


def _infer_dtype(items: list[Any]) -> int:
    """Infer one dm_env_rpc dtype for a homogeneous list of scalars.

    An empty list and a list that mixes strings with ``None`` both map to
    ``STRING``; ``bool`` is checked before ``int`` because it is a subclass.
    """
    if items and all(isinstance(x, bool) for x in items):
        return int(dm_env_rpc_pb2.BOOL)
    if items and all(isinstance(x, bool) or isinstance(x, int) for x in items):
        return int(dm_env_rpc_pb2.INT64)
    if items and all(isinstance(x, (int, float)) for x in items):
        return int(dm_env_rpc_pb2.DOUBLE)
    if all(x is None or isinstance(x, str) for x in items):
        return int(dm_env_rpc_pb2.STRING)
    raise ValueError(f'cannot represent property value as a tensor: {items!r}')
