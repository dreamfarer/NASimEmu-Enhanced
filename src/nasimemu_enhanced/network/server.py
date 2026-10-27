"""A dm_env_rpc server wrapping :class:`NASimEmuEnv`.

The server lets an agent in another process, language or machine drive the
environment over gRPC, using Google DeepMind's ``dm_env_rpc`` protocol. A
connection creates one or more worlds (choosing the scenario and the observation
format), joins one to learn the action and observation specs, then steps and
resets it exactly as it would a local :class:`NASimEmuEnv`. Each world runs in
its own process (see :mod:`nasimemu_enhanced.network.worker`), so worlds do not
share a random stream and cannot block one another.

A world belongs to the connection that created it: it is destroyed when that
connection destroys it, leaves and the stream closes, or the server stops. The
reset metadata (the host-row layout and the action catalogue) is served through
the dm_env_rpc properties extension; everything an agent needs after a step is a
typed observation tensor.

The transport uses gRPC local credentials by default, so the server only accepts
loopback connections whatever address it binds to, and it refuses emulation
unless started with ``allow_emulation=True``: a remote caller could otherwise
drive real exploits against the emulated network. Pass ``credentials`` to
:func:`serve` to use TLS instead.

See :mod:`nasimemu_enhanced.network.protocol` for the shared names and types and
:class:`nasimemu_enhanced.network.client.RemoteNASimEmuEnv` for the client.
"""
from __future__ import annotations

import logging
import uuid
from concurrent import futures
from typing import Any, Iterable, Iterator

import grpc
import numpy as np
from dm_env_rpc.v1 import dm_env_rpc_pb2, dm_env_rpc_pb2_grpc, spec_manager, tensor_spec_utils, tensor_utils
from dm_env_rpc.v1.extensions import properties_pb2
from google.rpc import code_pb2, status_pb2

from nasimemu_enhanced.network import protocol as p
from nasimemu_enhanced.network.worker import WorldError, WorldProcess

logger = logging.getLogger(__name__)

_ACTION_UID = 1
_OBS_UID_START = 101


class _RpcError(Exception):
    """An error carrying the ``google.rpc`` code to report to the client."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.rpc_code = code


class _PropertyStore:
    """Serves the reset metadata as a read-only dm_env_rpc property tree.

    A dict value is a listable node; anything else is a readable leaf whose data
    type is inferred from its Python value, so the action catalogue stays a set
    of generic records.
    """

    def __init__(self, tree: dict[str, Any]) -> None:
        self._tree = tree

    def handle(self, request: properties_pb2.PropertyRequest) -> properties_pb2.PropertyResponse:
        if request.HasField('read_property'):
            tensor = self._read(request.read_property.key)
            return properties_pb2.PropertyResponse(
                read_property=properties_pb2.ReadPropertyResponse(value=tensor))
        if request.HasField('list_property'):
            specs = self._list(request.list_property.key)
            return properties_pb2.PropertyResponse(
                list_property=properties_pb2.ListPropertyResponse(values=specs))
        if request.HasField('write_property'):
            raise _RpcError(code_pb2.PERMISSION_DENIED, 'the reset metadata is read-only')
        raise _RpcError(code_pb2.INVALID_ARGUMENT, 'empty property request')

    def _read(self, key: str) -> dm_env_rpc_pb2.Tensor:
        node = self._resolve(key)
        if isinstance(node, dict):
            raise _RpcError(code_pb2.INVALID_ARGUMENT, f'property {key!r} is a node, not a value')
        return p.pack_property_value(node)

    def _list(self, key: str) -> list[properties_pb2.PropertySpec]:
        node = self._resolve(key) if key else self._tree
        if not isinstance(node, dict):
            raise _RpcError(code_pb2.INVALID_ARGUMENT, f'property {key!r} is a value, not a node')
        specs: list[properties_pb2.PropertySpec] = []
        for name, value in node.items():
            full = f'{key}{p.PROP_SEPARATOR}{name}' if key else name
            if isinstance(value, dict):
                specs.append(properties_pb2.PropertySpec(
                    spec=dm_env_rpc_pb2.TensorSpec(name=full), is_listable=True))
            else:
                specs.append(properties_pb2.PropertySpec(
                    spec=p.property_spec(full, value), is_readable=True))
        return specs

    def _resolve(self, key: str) -> Any:
        node: Any = self._tree
        for part in key.split(p.PROP_SEPARATOR):
            if not isinstance(node, dict) or part not in node:
                raise _RpcError(code_pb2.NOT_FOUND, f'unknown property {key!r}')
            node = node[part]
        return node


class _World:
    """One world of a connection: its worker process, specs and sequence state.

    A world moves through create -> join -> (reset / step)*. After a reset the
    first :meth:`step` returns the initial observation and ignores its action, as
    dm_env_rpc expects; once an episode ends, the next step begins a new one and
    again ignores its action, so a stock client that keeps stepping is served a
    fresh episode instead of a state-transition error.
    """

    def __init__(self, name: str, settings: dict[str, Any], allow_emulation: bool) -> None:
        self.name = name
        self._process = WorldProcess(settings, allow_emulation)
        descriptor = self._process.descriptor

        self._action_spec = _action_spec(descriptor['action_bounds'])
        obs_specs = _observation_specs(descriptor)
        self._specs_msg = dm_env_rpc_pb2.ActionObservationSpecs(
            actions={_ACTION_UID: self._action_spec}, observations=obs_specs)
        self._obs_manager = spec_manager.SpecManager(obs_specs)

        self.properties = _PropertyStore(p.build_property_tree(descriptor['metadata']))

        self._pending: dict[str, Any] | None = descriptor['frame']
        self._ended = False

    def specs(self) -> dm_env_rpc_pb2.ActionObservationSpecs:
        return self._specs_msg

    def reset(self, seed: int | None) -> None:
        self._pending = self._process.reset(seed)
        self._ended = False

    def next_frame(self, action: list[int] | None) -> tuple[dict[str, Any], int]:
        """Return the next frame and its episode state, applying the dm_env_rpc
        rule that the first step of a sequence ignores its action."""
        if self._pending is not None:
            frame, self._pending, self._ended = self._pending, None, False
            return frame, dm_env_rpc_pb2.EnvironmentStateType.RUNNING
        if self._ended:
            self._ended = False
            return self._process.reset(None), dm_env_rpc_pb2.EnvironmentStateType.RUNNING
        assert action is not None
        frame = self._process.step(action)
        if frame['terminated']:
            self._ended = True
            return frame, dm_env_rpc_pb2.EnvironmentStateType.TERMINATED
        if frame['truncated']:
            self._ended = True
            return frame, dm_env_rpc_pb2.EnvironmentStateType.INTERRUPTED
        return frame, dm_env_rpc_pb2.EnvironmentStateType.RUNNING

    def expects_action(self) -> bool:
        """Whether the next step performs an action rather than opening a sequence."""
        return self._pending is None and not self._ended

    def pack_observations(self, frame: dict[str, Any], requested: Iterable[int]) -> dict[int, Any]:
        values: dict[str, Any] = dict(frame['obs_tensors'])
        values[p.OBS_S_RAW] = frame['s_raw']
        values[p.OBS_REWARD] = np.float64(frame['reward'])
        values.update(p.info_to_tensors(frame['info']))
        packed = self._obs_manager.pack(values)

        requested = list(requested)
        for uid in requested:
            if uid not in packed:
                raise _RpcError(code_pb2.INVALID_ARGUMENT, f'unknown observation uid {uid}')
        return {uid: packed[uid] for uid in requested}

    def read_action(self, actions: dict[int, dm_env_rpc_pb2.Tensor]) -> list[int]:
        for uid in actions:
            if uid != _ACTION_UID:
                raise _RpcError(code_pb2.INVALID_ARGUMENT, f'unknown action uid {uid}')
        if _ACTION_UID not in actions:
            # dm_env_rpc lets a step omit an action; the default scans the empty
            # address (0, 0), which fails with a connection error and only costs.
            return [0, 0, 0]
        tensor = actions[_ACTION_UID]

        expected = tensor_utils.data_type_to_np_type(self._action_spec.dtype)
        if tensor_utils.get_tensor_type(tensor) != expected:
            raise _RpcError(code_pb2.INVALID_ARGUMENT,
                            f'action must be {np.dtype(expected).name}, got a different type')
        if len(tensor.shape) > 1:
            raise _RpcError(code_pb2.INVALID_ARGUMENT, f'action must be a vector, got shape {list(tensor.shape)}')

        value: Any = np.asarray(tensor_utils.unpack_tensor(tensor)).reshape(-1)
        if value.size == 1:  # a scalar broadcasts to the whole action
            value = np.repeat(value, 3)
        if value.size != 3:
            raise _RpcError(code_pb2.INVALID_ARGUMENT, f'action must have three elements, got {value.size}')

        bounds = tensor_spec_utils.bounds(self._action_spec)
        if np.any(value < np.asarray(bounds.min)) or np.any(value > np.asarray(bounds.max)):
            raise _RpcError(code_pb2.INVALID_ARGUMENT,
                            f'action {value.tolist()} is out of bounds [{bounds.min}, {bounds.max}]')
        return [int(x) for x in value]

    def close(self) -> None:
        self._process.close()


def _action_spec(action_bounds: list[int]) -> dm_env_rpc_pb2.TensorSpec:
    spec = dm_env_rpc_pb2.TensorSpec(name=p.ACTION, shape=[3], dtype=dm_env_rpc_pb2.INT64)
    tensor_spec_utils.set_bounds(spec, minimum=np.array([0, 0, 0]), maximum=np.asarray(action_bounds))
    return spec


def _observation_specs(descriptor: dict[str, Any]) -> dict[int, dm_env_rpc_pb2.TensorSpec]:
    descriptors = p.observation_spec_descriptors(
        raw_cols=descriptor['raw_cols'], is_graph=descriptor['is_graph'],
        row_cols=descriptor['row_cols'], node_cols=descriptor['node_cols'],
        max_node_index=descriptor['max_node_index'])
    specs: dict[int, dm_env_rpc_pb2.TensorSpec] = {}
    for i, (name, dtype, shape, bounds) in enumerate(descriptors):
        spec = dm_env_rpc_pb2.TensorSpec(name=name, shape=shape, dtype=dtype)
        if bounds is not None:
            tensor_spec_utils.set_bounds(spec, minimum=bounds[0], maximum=bounds[1])
        specs[_OBS_UID_START + i] = spec
    return specs


class _Session:
    """The per-connection state: the worlds this connection created and which one
    it is joined to. Worlds live and die with the connection."""

    def __init__(self, allow_emulation: bool) -> None:
        self._allow_emulation = allow_emulation
        self.worlds: dict[str, _World] = {}
        self.joined: str | None = None

    def create_world(self, request: dm_env_rpc_pb2.CreateWorldRequest) -> dm_env_rpc_pb2.CreateWorldResponse:
        unknown = set(request.settings) - p.WORLD_SETTINGS
        if unknown:
            raise _RpcError(code_pb2.INVALID_ARGUMENT, f'unknown world settings: {sorted(unknown)}')
        settings = {name: p.unpack_scalar(tensor) for name, tensor in request.settings.items()}

        if not settings.get(p.SETTING_SCENARIO_NAME):
            raise _RpcError(code_pb2.INVALID_ARGUMENT, f'missing required setting {p.SETTING_SCENARIO_NAME!r}')
        if bool(settings.get(p.SETTING_EMULATE, 0)) and not self._allow_emulation:
            raise _RpcError(code_pb2.PERMISSION_DENIED,
                            'emulation is disabled on this server (start it with allow_emulation=True)')

        name = f'nasimemu-{uuid.uuid4().hex}'
        try:
            self.worlds[name] = _World(name, settings, self._allow_emulation)
        except WorldError as error:
            raise _RpcError(code_pb2.INVALID_ARGUMENT, str(error)) from error
        return dm_env_rpc_pb2.CreateWorldResponse(world_name=name)

    def join_world(self, request: dm_env_rpc_pb2.JoinWorldRequest) -> dm_env_rpc_pb2.JoinWorldResponse:
        world = self._world(request.world_name)
        if request.settings:
            raise _RpcError(code_pb2.INVALID_ARGUMENT, f'unknown join settings: {sorted(request.settings)}')
        if self.joined is not None:
            raise _RpcError(code_pb2.FAILED_PRECONDITION, 'already joined a world on this connection')
        self.joined = world.name
        return dm_env_rpc_pb2.JoinWorldResponse(specs=world.specs())

    def reset(self, request: dm_env_rpc_pb2.ResetRequest) -> dm_env_rpc_pb2.ResetResponse:
        world = self._joined_world()
        world.reset(self._seed(request.settings))
        return dm_env_rpc_pb2.ResetResponse(specs=world.specs())

    def reset_world(self, request: dm_env_rpc_pb2.ResetWorldRequest) -> dm_env_rpc_pb2.ResetWorldResponse:
        world = self._world(request.world_name)
        world.reset(self._seed(request.settings))
        return dm_env_rpc_pb2.ResetWorldResponse()

    def leave_world(self) -> dm_env_rpc_pb2.LeaveWorldResponse:
        self.joined = None
        return dm_env_rpc_pb2.LeaveWorldResponse()

    def destroy_world(self, request: dm_env_rpc_pb2.DestroyWorldRequest) -> dm_env_rpc_pb2.DestroyWorldResponse:
        world = self._world(request.world_name)
        if self.joined == world.name:
            raise _RpcError(code_pb2.FAILED_PRECONDITION, 'cannot destroy a world while joined to it')
        self.worlds.pop(world.name).close()
        return dm_env_rpc_pb2.DestroyWorldResponse()

    def step(self, request: dm_env_rpc_pb2.StepRequest) -> dm_env_rpc_pb2.StepResponse:
        world = self._joined_world()
        action = world.read_action(dict(request.actions)) if world.expects_action() else None
        frame, state = world.next_frame(action)
        observations = world.pack_observations(frame, request.requested_observations)
        return dm_env_rpc_pb2.StepResponse(state=state, observations=observations)

    def extension(self, request: Any) -> Any:
        world = self._joined_world()
        prop_request = properties_pb2.PropertyRequest()
        if not request.Unpack(prop_request):
            raise _RpcError(code_pb2.UNIMPLEMENTED, 'unsupported extension request')
        return world.properties.handle(prop_request)

    def _world(self, world_name: str) -> _World:
        world = self.worlds.get(world_name)
        if world is None:
            raise _RpcError(code_pb2.NOT_FOUND, f'unknown world {world_name!r}')
        return world

    def _joined_world(self) -> _World:
        if self.joined is None:
            raise _RpcError(code_pb2.FAILED_PRECONDITION, 'not joined to a world')
        return self.worlds[self.joined]

    def _seed(self, settings: Any) -> int | None:
        unknown = set(settings) - p.RESET_SETTINGS
        if unknown:
            raise _RpcError(code_pb2.INVALID_ARGUMENT, f'unknown reset settings: {sorted(unknown)}')
        if p.SETTING_SEED not in settings:
            return None
        seed_setting = int(p.unpack_scalar(settings[p.SETTING_SEED]))
        return None if seed_setting < 0 else seed_setting

    def close(self) -> None:
        for world in self.worlds.values():
            world.close()
        self.worlds.clear()
        self.joined = None


def _error_code(error: Exception) -> int:
    code = getattr(error, 'rpc_code', None)
    if code is not None:
        return int(code)
    if isinstance(error, KeyError):
        return int(code_pb2.NOT_FOUND)
    if isinstance(error, PermissionError):
        return int(code_pb2.PERMISSION_DENIED)
    if isinstance(error, NotImplementedError):
        return int(code_pb2.UNIMPLEMENTED)
    if isinstance(error, WorldError):
        return int(code_pb2.INTERNAL)
    return int(code_pb2.INVALID_ARGUMENT)


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
                    code = _error_code(error)
                    if code == code_pb2.INTERNAL:
                        logger.warning('request %r failed: %s', field, error)
                    response.error.CopyFrom(status_pb2.Status(code=code, message=str(error)))
                yield response
        finally:
            session.close()

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
        elif field == 'reset_world':
            response.reset_world.CopyFrom(session.reset_world(request.reset_world))
        elif field == 'leave_world':
            response.leave_world.CopyFrom(session.leave_world())
        elif field == 'destroy_world':
            response.destroy_world.CopyFrom(session.destroy_world(request.destroy_world))
        elif field == 'extension':
            response.extension.Pack(session.extension(request.extension))
        else:
            raise _RpcError(code_pb2.UNIMPLEMENTED, f'unsupported request {field!r}')


def serve(address: str = 'localhost:50051', *, allow_emulation: bool = False, max_workers: int = 10,
          credentials: grpc.ServerCredentials | None = None) -> grpc.Server:
    """Start a NASimEmu dm_env_rpc server and return it (not yet waiting).

    Parameters
    ----------
    address : str, optional
        host:port, or ``unix:`` socket, to bind to (default='localhost:50051')
    allow_emulation : bool, optional
        allow clients to request emulation, which drives real exploits against
        the emulated network; off by default (default=False)
    max_workers : int, optional
        number of connections served in parallel (default=10)
    credentials : grpc.ServerCredentials, optional
        server credentials; by default gRPC local credentials, which accept only
        loopback connections. Pass TLS credentials to serve over the network
        (default=None)

    Returns
    -------
    grpc.Server
        the started server; call ``wait_for_termination()`` to block
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    dm_env_rpc_pb2_grpc.add_EnvironmentServicer_to_server(NASimEmuServicer(allow_emulation=allow_emulation), server)

    if credentials is None:
        local = grpc.LocalConnectionType.UDS if address.startswith('unix:') else grpc.LocalConnectionType.LOCAL_TCP
        credentials = grpc.local_server_credentials(local)
    server.add_secure_port(address, credentials)
    server.start()
    logger.info('NASimEmu dm_env_rpc server listening on %s (emulation %s)',
                address, 'allowed' if allow_emulation else 'disabled')
    return server


def main() -> None:
    """Run a server from the command line until interrupted."""
    import argparse

    parser = argparse.ArgumentParser(description='NASimEmu dm_env_rpc server')
    parser.add_argument('--address', default='localhost:50051', help='host:port or unix: socket to bind to')
    parser.add_argument('--allow-emulation', action='store_true', help='allow clients to request emulation')
    parser.add_argument('--verbose', action='store_true', help='log each connection')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    server = serve(args.address, allow_emulation=args.allow_emulation)
    server.wait_for_termination()


if __name__ == '__main__':
    main()
