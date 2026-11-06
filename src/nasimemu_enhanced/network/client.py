"""A gymnasium client for the NASimEmu dm_env_rpc server.

:class:`RemoteNASimEmuEnv` connects to a :func:`nasimemu_enhanced.network.server`
and behaves like a local :class:`NASimEmuEnv`: the same constructor arguments,
the same :meth:`reset` and :meth:`step` return values, the same declared spaces
and the same ``s_raw``, ``a_raw``, ``subnet_graph`` and ``action_list``
attributes the agents read. Agent code therefore runs unchanged whether the
environment is in-process or across the network.

The environment lives in the server process; this class only sends actions,
unpacks the typed observation tensors and rebuilds the ``step`` info dict. The
reset metadata (the host-row layout and the action catalogue) is read once at
join from the server's properties and returned from every :meth:`reset`. A new
scenario instance is generated on every :meth:`reset`, exactly as locally.

By default the client uses gRPC local credentials, matching the server; pass
``credentials`` for TLS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from dm_env_rpc.v1 import connection as dm_connection
from dm_env_rpc.v1 import dm_env_rpc_pb2, spec_manager, tensor_spec_utils
from dm_env_rpc.v1.extensions import properties as dm_properties
from gymnasium import spaces

from nasimemu_enhanced.network import protocol as p

try:
    import grpc
except ModuleNotFoundError as error:
    raise ModuleNotFoundError("the network client needs grpc; install nasimemu-enhanced[network]") from error


class RemoteNASimEmuEnv(gym.Env[Any, Any]):
    """A :class:`NASimEmuEnv` driven over dm_env_rpc.

    Parameters
    ----------
    scenario_name : str
        path to a single ``.v2.yaml`` scenario file; its content is read and
        sent to the server, so the server needs no shared filesystem. Unlike
        the local environment, a colon-joined list of scenarios is not
        supported over the network.
    address : str, optional
        host:port, or ``unix:`` socket, of the server (default='localhost:50051')
    credentials : grpc.ChannelCredentials, optional
        channel credentials; by default gRPC local credentials, matching the
        server (default=None)

    The remaining arguments match :class:`NASimEmuEnv`: ``emulate``,
    ``step_limit``, ``random_init``, ``observation_format``, ``fully_obs`` and
    ``augment_with_action``. ``verbose`` is accepted and ignored.

    Attributes
    ----------
    s_raw, a_raw, subnet_graph, action_list, action_space, observation_space
        mirror the local environment; see :class:`NASimEmuEnv`. ``a_raw`` holds
        the last action as an array rather than the action object, and the
        ``action_list`` classes are stand-ins, since the action objects are not
        sent over the wire.
    env : None
        there is no local wrapped environment; present so that code guarding on
        ``hasattr(env.env, 'scenario')`` falls through.
    """

    def __init__(self, scenario_name: str, emulate: bool = False, step_limit: int | None = None,
                 random_init: bool = False, observation_format: str = 'matrix', fully_obs: bool = False,
                 augment_with_action: bool = False, verbose: bool = False,
                 address: str = 'localhost:50051',
                 credentials: grpc.ChannelCredentials | None = None) -> None:
        if ':' in Path(scenario_name).name:
            raise NotImplementedError('the network client supports a single scenario file, not a colon-joined list')

        self.observation_format = observation_format
        self.graph_version = {'graph': 1, 'graph_v1': 1, 'graph_v2': 2}.get(observation_format)
        self.env = None  # no local wrapped environment

        self.s_raw: Any = None
        self.a_raw: Any = None
        self.subnet_graph: set[tuple[int, int]] = set()
        self.action_list: list[tuple[type, dict[str, Any]]] = []

        if credentials is None:
            local = grpc.LocalConnectionType.UDS if address.startswith('unix:') else grpc.LocalConnectionType.LOCAL_TCP
            credentials = grpc.local_channel_credentials(local)
        options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
        self._channel = grpc.secure_channel(address, credentials, options=options)
        self._connection = dm_connection.Connection(self._channel)
        self._closed = False

        settings = {
            p.SETTING_SCENARIO_NAME: p.pack_setting(Path(scenario_name).name),
            p.SETTING_SCENARIO_YAML: p.pack_setting(Path(scenario_name).read_text()),
            p.SETTING_EMULATE: p.pack_setting(int(emulate)),
            p.SETTING_STEP_LIMIT: p.pack_setting(-1 if step_limit is None else int(step_limit)),
            p.SETTING_RANDOM_INIT: p.pack_setting(int(random_init)),
            p.SETTING_OBSERVATION_FORMAT: p.pack_setting(observation_format),
            p.SETTING_FULLY_OBS: p.pack_setting(int(fully_obs)),
            p.SETTING_AUGMENT_WITH_ACTION: p.pack_setting(int(augment_with_action)),
        }
        created = self._connection.send(dm_env_rpc_pb2.CreateWorldRequest(settings=settings))
        self._world_name = created.world_name

        joined = self._connection.send(dm_env_rpc_pb2.JoinWorldRequest(world_name=self._world_name))
        self._bind_specs(joined.specs)
        self._define_spaces()
        self._read_metadata()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Generate a new scenario instance and start an episode."""
        super().reset(seed=seed)
        settings = {p.SETTING_SEED: p.pack_setting(-1 if seed is None else int(seed))}
        self._connection.send(dm_env_rpc_pb2.ResetRequest(settings=settings))

        obs, _reward, _info, _state = self._advance(actions={})
        self.subnet_graph = set()
        self.a_raw = None
        return obs, dict(self._metadata)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Perform one action given as ``(subnet, host, action_id)``."""
        value = np.asarray(action, dtype=np.int64).reshape(-1)
        packed = self._action_manager.pack({p.ACTION: value})
        obs, reward, info, state = self._advance(actions=packed)

        self.a_raw = value
        self.subnet_graph = {(int(a), int(b)) for a, b in info['subnet_graph']}
        terminated = state == dm_env_rpc_pb2.EnvironmentStateType.TERMINATED
        truncated = state == dm_env_rpc_pb2.EnvironmentStateType.INTERRUPTED
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Leave and destroy the world, then close the connection."""
        if self._closed:
            return
        self._closed = True
        try:
            self._connection.send(dm_env_rpc_pb2.LeaveWorldRequest())
            self._connection.send(dm_env_rpc_pb2.DestroyWorldRequest(world_name=self._world_name))
        finally:
            self._connection.close()
            self._channel.close()

    def _bind_specs(self, specs: dm_env_rpc_pb2.ActionObservationSpecs) -> None:
        self._action_specs = dict(specs.actions)
        self._obs_specs = dict(specs.observations)
        self._action_manager = spec_manager.SpecManager(self._action_specs)
        self._obs_manager = spec_manager.SpecManager(self._obs_specs)
        self._obs_uids = sorted(self._obs_specs.keys())

    def _advance(self, actions: dict[int, dm_env_rpc_pb2.Tensor]) -> tuple[Any, float, dict[str, Any], int]:
        response = self._connection.send(
            dm_env_rpc_pb2.StepRequest(actions=actions, requested_observations=self._obs_uids))
        values = self._obs_manager.unpack(dict(response.observations))

        self.s_raw = np.asarray(values[p.OBS_S_RAW], dtype=np.float32)
        reward = float(np.asarray(values[p.OBS_REWARD]).reshape(-1)[0])
        info = p.tensors_to_info(values)

        if self.graph_version is None:
            obs: Any = np.asarray(values[p.OBS_MATRIX], dtype=np.float32)
        else:
            obs = {
                'graph': spaces.GraphInstance(
                    nodes=np.asarray(values[p.OBS_GRAPH_NODES], dtype=np.float64),
                    edges=np.asarray(values[p.OBS_GRAPH_EDGES], dtype=np.int64).reshape(-1),
                    edge_links=np.asarray(values[p.OBS_GRAPH_EDGE_LINKS], dtype=np.int64).reshape(-1, 2)),
                'node_index': np.asarray(values[p.OBS_GRAPH_NODE_INDEX], dtype=np.int64).reshape(-1, 2),
                'pos_index': np.asarray(values[p.OBS_GRAPH_POS_INDEX], dtype=np.float64).reshape(-1),
            }
        return obs, reward, info, int(response.state)

    def _read_metadata(self) -> None:
        """Read the layout and action catalogue from the server's properties."""
        props = dm_properties.PropertiesExtension(self._connection)
        self._metadata = p.reconstruct_metadata(self._walk_properties(props, ''))
        # action_list mirrors the local env: the catalogue without the terminal
        # entry, as (stand-in class, parameters). The action objects themselves
        # are not sent, so the classes are stand-ins named after the action type.
        self.action_list = [
            (type(str(entry['type']), (), {}), {k: v for k, v in entry.items() if k not in ('id', 'type')})
            for entry in self._metadata['actions'][:-1]
        ]

    def _walk_properties(self, props: dm_properties.PropertiesExtension, key: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for spec in props.list(key):
            name = spec.key.rsplit(p.PROP_SEPARATOR, 1)[-1]
            if spec.listable:
                result[name] = self._walk_properties(props, spec.key)
            else:
                result[name] = p.decode_leaf(props.read(spec.key))
        return result

    def _define_spaces(self) -> None:
        by_name = {spec.name: spec for spec in self._obs_specs.values()}

        action_max = np.asarray(self._action_specs[next(iter(self._action_specs))].max.int64s.array)
        self.action_space = spaces.MultiDiscrete((action_max + 1).tolist())

        if self.graph_version is None:
            cols = int(by_name[p.OBS_MATRIX].shape[1])
            self.observation_space = spaces.Sequence(
                spaces.Box(-np.inf, np.inf, (cols,), dtype=np.float32), stack=True)
        else:
            node_cols = int(by_name[p.OBS_GRAPH_NODES].shape[1])
            node_bounds = tensor_spec_utils.bounds(by_name[p.OBS_GRAPH_NODE_INDEX])
            self.observation_space = spaces.Dict({
                'graph': spaces.Graph(node_space=spaces.Box(-np.inf, np.inf, (node_cols,), dtype=np.float64),
                                      edge_space=spaces.Discrete(1)),
                'node_index': spaces.Sequence(
                    spaces.Box(int(np.asarray(node_bounds.min).reshape(-1)[0]),
                               int(np.asarray(node_bounds.max).reshape(-1)[0]), (2,), dtype=np.int64), stack=True),
                'pos_index': spaces.Sequence(spaces.Box(0, np.inf, (), dtype=np.float64), stack=True),
            })
