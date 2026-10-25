"""A dm_env_rpc server wrapping :class:`NASimEmuEnv`.

The server lets an agent in another process, language or machine drive the
environment over gRPC, using Google DeepMind's ``dm_env_rpc`` protocol. One
client connection owns one environment: the client creates a world (choosing
the scenario and the observation format), joins it to learn the action and
observation specs, and then steps and resets it exactly as it would a local
:class:`NASimEmuEnv`.

The transport is unencrypted and unauthenticated, so the server binds to
localhost by default and refuses emulation unless it is started with
``allow_emulation=True``: a remote caller could otherwise drive real exploits
against the emulated network.

See :mod:`nasimemu_enhanced.network.protocol` for the shared names and types and
:class:`nasimemu_enhanced.network.client.RemoteNASimEmuEnv` for the client.
"""
from __future__ import annotations

import logging
import os
import tempfile
from concurrent import futures
from typing import Any, Iterable, Iterator, cast

import grpc
import numpy as np
from dm_env_rpc.v1 import dm_env_rpc_pb2, dm_env_rpc_pb2_grpc, spec_manager, tensor_spec_utils, tensor_utils
from google.rpc import code_pb2, status_pb2

from nasimemu_enhanced.env import NASimEmuEnv
from nasimemu_enhanced.network import protocol as p

logger = logging.getLogger(__name__)

_ACTION_UID = 1
_OBS_UID_START = 101


def _spec(name: str, shape: list[int], dtype: Any) -> dm_env_rpc_pb2.TensorSpec:
    """Build a tensor spec; a negative size marks the variable dimension."""
    return dm_env_rpc_pb2.TensorSpec(name=name, shape=shape, dtype=dtype)


class _Session:
    """The per-connection state: one world, one environment.

    A connection moves through create → join → (reset / step)* → leave/destroy.
    After a reset the first :meth:`step` returns the initial observation and
    ignores its action, matching the dm_env_rpc convention that a sequence
    starts with an action-free step.
    """

    def __init__(self, allow_emulation: bool) -> None:
        self._allow_emulation = allow_emulation
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None

        self.env: NASimEmuEnv | None = None
        self.world_name: str | None = None
        self.joined = False

        self._action_specs: dict[int, dm_env_rpc_pb2.TensorSpec] = {}
        self._obs_specs: dict[int, dm_env_rpc_pb2.TensorSpec] = {}
        self._action_manager: spec_manager.SpecManager | None = None
        self._obs_manager: spec_manager.SpecManager | None = None

        self._pending: dict[str, Any] | None = None  # initial observation after a reset
        self._pending_state = p.STATE_RUNNING

    # -- world lifecycle ---------------------------------------------------
    def create_world(self, request: dm_env_rpc_pb2.CreateWorldRequest) -> dm_env_rpc_pb2.CreateWorldResponse:
        if self.env is not None:
            raise ValueError('a world already exists on this connection')

        settings = {name: p.unpack_scalar(tensor) for name, tensor in request.settings.items()}

        scenario_name = settings.get(p.SETTING_SCENARIO_NAME)
        if not scenario_name:
            raise ValueError(f'missing required setting {p.SETTING_SCENARIO_NAME!r}')

        scenario_path = self._resolve_scenario(str(scenario_name), settings.get(p.SETTING_SCENARIO_YAML))

        emulate = bool(settings.get(p.SETTING_EMULATE, 0))
        if emulate and not self._allow_emulation:
            raise ValueError('emulation is disabled on this server (start it with allow_emulation=True)')

        step_limit_setting = int(settings.get(p.SETTING_STEP_LIMIT, -1))
        step_limit = None if step_limit_setting < 0 else step_limit_setting

        self.env = NASimEmuEnv(
            scenario_name=scenario_path,
            emulate=emulate,
            step_limit=step_limit,
            random_init=bool(settings.get(p.SETTING_RANDOM_INIT, 0)),
            observation_format=str(settings.get(p.SETTING_OBSERVATION_FORMAT, 'matrix')),
            fully_obs=bool(settings.get(p.SETTING_FULLY_OBS, 0)),
            augment_with_action=bool(settings.get(p.SETTING_AUGMENT_WITH_ACTION, 0)),
        )

        # A probe reset tells us the fixed tensor widths (host-row size, graph
        # node-feature width); the variable dimension stays open in the spec.
        obs, _ = self.env.reset()
        self._build_specs(obs)
        self._stash(obs, reward=0.0, info={}, state=p.STATE_RUNNING)

        self.world_name = f'nasimemu-{id(self):x}'
        return dm_env_rpc_pb2.CreateWorldResponse(world_name=self.world_name)

    def join_world(self, request: dm_env_rpc_pb2.JoinWorldRequest) -> dm_env_rpc_pb2.JoinWorldResponse:
        if self.env is None or request.world_name != self.world_name:
            raise ValueError(f'unknown world {request.world_name!r}')
        if self.joined:
            raise ValueError('already joined a world on this connection')
        self.joined = True
        return dm_env_rpc_pb2.JoinWorldResponse(specs=self._specs())

    def reset(self, request: dm_env_rpc_pb2.ResetRequest) -> dm_env_rpc_pb2.ResetResponse:
        if self.env is None or not self.joined:
            raise ValueError('cannot reset before joining a world')
        settings = {name: p.unpack_scalar(tensor) for name, tensor in request.settings.items()}
        seed_setting = int(settings.get(p.SETTING_SEED, -1))
        seed = None if seed_setting < 0 else seed_setting

        obs, info = self.env.reset(seed=seed)
        self._stash(obs, reward=0.0, info=info, state=p.STATE_RUNNING)
        return dm_env_rpc_pb2.ResetResponse(specs=self._specs())

    def leave_world(self) -> dm_env_rpc_pb2.LeaveWorldResponse:
        self.joined = False
        return dm_env_rpc_pb2.LeaveWorldResponse()

    def destroy_world(self, request: dm_env_rpc_pb2.DestroyWorldRequest) -> dm_env_rpc_pb2.DestroyWorldResponse:
        if request.world_name != self.world_name:
            raise ValueError(f'unknown world {request.world_name!r}')
        if self.joined:
            raise ValueError('cannot destroy a world while joined to it')
        self._cleanup()
        return dm_env_rpc_pb2.DestroyWorldResponse()

    # -- stepping ----------------------------------------------------------
    def step(self, request: dm_env_rpc_pb2.StepRequest) -> dm_env_rpc_pb2.StepResponse:
        if self.env is None or not self.joined:
            raise ValueError('cannot step before joining a world')

        if self._pending is not None:
            # First step of a sequence: return the initial observation and
            # ignore the action, as dm_env_rpc expects.
            tensors, state = self._pending, self._pending_state
            self._pending = None
        else:
            action = self._read_action(request.actions)
            obs, reward, terminated, truncated, info = self.env.step(action)
            state = p.STATE_TERMINATED if terminated else p.STATE_TRUNCATED if truncated else p.STATE_RUNNING
            tensors = self._observation_tensors(obs, reward=reward, info=info, state=state)

        packed = self._obs_manager.pack(tensors)  # type: ignore[union-attr]
        requested = request.requested_observations or list(packed.keys())
        observations = {uid: packed[uid] for uid in requested if uid in packed}

        return dm_env_rpc_pb2.StepResponse(state=_state_type(state), observations=observations)

    # -- helpers -----------------------------------------------------------
    def _resolve_scenario(self, scenario_name: str, scenario_yaml: Any) -> str:
        if not scenario_yaml:
            return scenario_name  # a path on the server's own filesystem
        if '.v2' not in scenario_name:
            raise ValueError(f'scenario name must be a .v2.yaml file: {scenario_name!r}')
        self._tempdir = tempfile.TemporaryDirectory(prefix='nasimemu-scenario-')
        path = os.path.join(self._tempdir.name, os.path.basename(scenario_name))
        with open(path, 'w') as handle:
            handle.write(str(scenario_yaml))
        return path

    def _build_specs(self, obs: Any) -> None:
        assert self.env is not None
        nvec = cast(Any, self.env.action_space).nvec
        max_subnets, max_hosts, n_action_ids = (int(x) for x in nvec)

        action_spec = _spec(p.ACTION, [3], dm_env_rpc_pb2.INT64)
        tensor_spec_utils.set_bounds(
            action_spec, minimum=np.array([0, 0, 0]),
            maximum=np.array([max_subnets - 1, max_hosts - 1, n_action_ids - 1]))
        self._action_specs = {_ACTION_UID: action_spec}

        raw_cols = int(np.asarray(self.env.s_raw).shape[1])
        specs = [
            _spec(p.OBS_S_RAW, [-1, raw_cols], dm_env_rpc_pb2.DOUBLE),
            _spec(p.OBS_REWARD, [], dm_env_rpc_pb2.DOUBLE),
            _spec(p.OBS_INFO, [], dm_env_rpc_pb2.STRING),
            _spec(p.OBS_STATE, [], dm_env_rpc_pb2.INT64),
        ]
        if isinstance(obs, dict):  # graph observation
            node_cols = int(np.asarray(obs['graph'].nodes).shape[1])
            specs += [
                _spec(p.OBS_GRAPH_NODES, [-1, node_cols], dm_env_rpc_pb2.DOUBLE),
                _spec(p.OBS_GRAPH_EDGES, [-1], dm_env_rpc_pb2.INT64),
                _spec(p.OBS_GRAPH_EDGE_LINKS, [-1, 2], dm_env_rpc_pb2.INT64),
                _spec(p.OBS_GRAPH_NODE_INDEX, [-1, 2], dm_env_rpc_pb2.INT64),
                _spec(p.OBS_GRAPH_POS_INDEX, [-1], dm_env_rpc_pb2.DOUBLE),
            ]
        else:  # matrix observation
            specs.append(_spec(p.OBS_MATRIX, [-1, int(np.asarray(obs).shape[1])], dm_env_rpc_pb2.DOUBLE))

        self._obs_specs = {_OBS_UID_START + i: s for i, s in enumerate(specs)}
        self._action_manager = spec_manager.SpecManager(self._action_specs)
        self._obs_manager = spec_manager.SpecManager(self._obs_specs)

    def _specs(self) -> dm_env_rpc_pb2.ActionObservationSpecs:
        return dm_env_rpc_pb2.ActionObservationSpecs(actions=self._action_specs, observations=self._obs_specs)

    def _read_action(self, actions: Any) -> list[int]:
        assert self._action_manager is not None
        unpacked = self._action_manager.unpack(dict(actions))
        value = np.asarray(unpacked[p.ACTION]).reshape(-1)
        if value.size != 3:
            raise ValueError(f'action must have three elements, got {value.size}')
        spec = self._action_specs[_ACTION_UID]
        bounds = tensor_spec_utils.bounds(spec)
        if np.any(value < np.asarray(bounds.min)) or np.any(value > np.asarray(bounds.max)):
            raise ValueError(f'action {value.tolist()} is out of bounds [{bounds.min}, {bounds.max}]')
        return [int(x) for x in value]

    def _observation_tensors(self, obs: Any, reward: float, info: dict[str, Any], state: int) -> dict[str, Any]:
        assert self.env is not None
        tensors: dict[str, Any] = {
            p.OBS_S_RAW: np.asarray(self.env.s_raw, dtype=np.float64),
            p.OBS_REWARD: np.float64(reward),
            p.OBS_INFO: p.dumps_info(info),
            p.OBS_STATE: np.int64(state),
        }
        if isinstance(obs, dict):
            graph = obs['graph']
            tensors[p.OBS_GRAPH_NODES] = np.asarray(graph.nodes, dtype=np.float64)
            tensors[p.OBS_GRAPH_EDGES] = np.asarray(graph.edges, dtype=np.int64)
            tensors[p.OBS_GRAPH_EDGE_LINKS] = np.asarray(graph.edge_links, dtype=np.int64).reshape(-1, 2)
            tensors[p.OBS_GRAPH_NODE_INDEX] = np.asarray(obs['node_index'], dtype=np.int64).reshape(-1, 2)
            tensors[p.OBS_GRAPH_POS_INDEX] = np.asarray(obs['pos_index'], dtype=np.float64).reshape(-1)
        else:
            tensors[p.OBS_MATRIX] = np.asarray(obs, dtype=np.float64)
        return tensors

    def _stash(self, obs: Any, reward: float, info: dict[str, Any], state: int) -> None:
        self._pending = self._observation_tensors(obs, reward=reward, info=info, state=state)
        self._pending_state = state

    def _cleanup(self) -> None:
        self.env = None
        self.world_name = None
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None


def _state_type(state: int) -> Any:
    if state == p.STATE_TERMINATED:
        return dm_env_rpc_pb2.EnvironmentStateType.TERMINATED
    if state == p.STATE_TRUNCATED:
        return dm_env_rpc_pb2.EnvironmentStateType.INTERRUPTED
    return dm_env_rpc_pb2.EnvironmentStateType.RUNNING


class NASimEmuServicer(dm_env_rpc_pb2_grpc.EnvironmentServicer):  # type: ignore[misc]
    """Routes each dm_env_rpc request of a connection to its :class:`_Session`."""

    def __init__(self, allow_emulation: bool = False) -> None:
        self._allow_emulation = allow_emulation

    def Process(self, request_iterator: Iterable[dm_env_rpc_pb2.EnvironmentRequest],
                context: grpc.ServicerContext) -> Iterator[dm_env_rpc_pb2.EnvironmentResponse]:
        session = _Session(self._allow_emulation)
        try:
            for request in request_iterator:
                response = dm_env_rpc_pb2.EnvironmentResponse()
                field = request.WhichOneof('payload')
                try:
                    self._dispatch(session, field, request, response)
                except Exception as error:  # a bad request fails this call, not the connection
                    logger.warning('request %r failed: %s', field, error)
                    response.error.CopyFrom(status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message=str(error)))
                yield response
        finally:
            session._cleanup()

    @staticmethod
    def _dispatch(session: _Session, field: str | None, request: dm_env_rpc_pb2.EnvironmentRequest,
                  response: dm_env_rpc_pb2.EnvironmentResponse) -> None:
        if field == 'create_world':
            response.create_world.CopyFrom(session.create_world(request.create_world))
        elif field == 'join_world':
            response.join_world.CopyFrom(session.join_world(request.join_world))
        elif field == 'step':
            response.step.CopyFrom(session.step(request.step))
        elif field == 'reset':
            response.reset.CopyFrom(session.reset(request.reset))
        elif field == 'leave_world':
            response.leave_world.CopyFrom(session.leave_world())
        elif field == 'destroy_world':
            response.destroy_world.CopyFrom(session.destroy_world(request.destroy_world))
        else:
            raise ValueError(f'unsupported request {field!r}')


def serve(address: str = 'localhost:50051', *, allow_emulation: bool = False,
          max_workers: int = 10) -> grpc.Server:
    """Start a NASimEmu dm_env_rpc server and return it (not yet waiting).

    Parameters
    ----------
    address : str, optional
        host:port to bind to; ``localhost`` keeps the server off the network
        (default='localhost:50051')
    allow_emulation : bool, optional
        allow clients to request emulation, which drives real exploits against
        the emulated network; off by default (default=False)
    max_workers : int, optional
        number of connections served in parallel (default=10)

    Returns
    -------
    grpc.Server
        the started server; call ``wait_for_termination()`` to block
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    dm_env_rpc_pb2_grpc.add_EnvironmentServicer_to_server(NASimEmuServicer(allow_emulation=allow_emulation), server)
    server.add_insecure_port(address)
    server.start()
    logger.info('NASimEmu dm_env_rpc server listening on %s (emulation %s)',
                address, 'allowed' if allow_emulation else 'disabled')
    return server


def main() -> None:
    """Run a server from the command line until interrupted."""
    import argparse

    parser = argparse.ArgumentParser(description='NASimEmu dm_env_rpc server')
    parser.add_argument('--address', default='localhost:50051', help='host:port to bind to')
    parser.add_argument('--allow-emulation', action='store_true', help='allow clients to request emulation')
    parser.add_argument('--verbose', action='store_true', help='log each connection')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    server = serve(args.address, allow_emulation=args.allow_emulation)
    server.wait_for_termination()


if __name__ == '__main__':
    main()
