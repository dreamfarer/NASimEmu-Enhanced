"""The per-world worker process.

Each world the server creates owns one :class:`~nasimemu_enhanced.env.NASimEmuEnv`
inside a separate process, started here by :class:`WorldProcess`. Isolating a
world in its own process gives it its own global ``random`` and ``numpy.random``
state, so two seeded worlds stepped by different connections no longer share one
random stream and cannot diverge; it also keeps a crash or a slow emulation step
in one world from blocking the others.

The process is created with the ``spawn`` start method, because the server runs
gRPC threads and forking a process with live threads is unsafe. The server and
the worker exchange plain Python objects and NumPy arrays over a pipe: commands
in one direction, a result frame back. Nothing here depends on gRPC, so the heavy
environment never imports the transport.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

import numpy as np

from nasimemu_enhanced.env import NASimEmuEnv
from nasimemu_enhanced.network import protocol as p

_CMD_RESET = 'reset'
_CMD_STEP = 'step'
_CMD_CLOSE = 'close'


class WorldError(Exception):
    """Raised in the server when a worker reports or hits an error."""


class WorldProcess:
    """A handle, held by the server, to one world running in its own process.

    Construction starts the process, builds the environment and blocks until the
    worker has produced its world descriptor (the fixed tensor widths, the action
    bounds, the reset metadata and the initial observation). Afterwards
    :meth:`reset` and :meth:`step` drive the environment one call at a time.
    """

    def __init__(self, settings: dict[str, Any], allow_emulation: bool) -> None:
        context = mp.get_context('spawn')
        self._parent_conn, child_conn = context.Pipe()
        self._process: BaseProcess | None = context.Process(
            target=_worker_main, args=(child_conn, settings, allow_emulation), daemon=True)
        self._process.start()
        child_conn.close()  # only the worker keeps its end

        tag, payload = self._recv()
        if tag != 'ready':
            self.close()
            raise WorldError(str(payload))
        self.descriptor: dict[str, Any] = payload

    def reset(self, seed: int | None) -> dict[str, Any]:
        """Start a new episode and return its initial frame."""
        return self._call((_CMD_RESET, seed))

    def step(self, action: list[int]) -> dict[str, Any]:
        """Perform one action and return its frame."""
        return self._call((_CMD_STEP, action))

    def close(self) -> None:
        """Stop the worker process and release the pipe."""
        if self._process is None:
            return
        try:
            if self._process.is_alive():
                self._parent_conn.send((_CMD_CLOSE, None))
                self._process.join(timeout=5)
        except (OSError, BrokenPipeError, EOFError, ValueError):
            pass
        finally:
            if self._process.is_alive():
                self._process.terminate()
            self._parent_conn.close()
            self._process = None

    def _call(self, command: tuple[str, Any]) -> dict[str, Any]:
        self._parent_conn.send(command)
        tag, payload = self._recv()
        if tag == 'error':
            raise WorldError(str(payload))
        frame: dict[str, Any] = payload
        return frame

    def _recv(self) -> tuple[str, Any]:
        try:
            tag, payload = self._parent_conn.recv()
        except EOFError as error:  # the worker died
            raise WorldError('the world process stopped unexpectedly') from error
        return tag, payload


def _worker_main(conn: Connection, settings: dict[str, Any], allow_emulation: bool) -> None:
    """Entry point of the worker process: build the env, then serve commands."""
    try:
        worker = _Worker(settings, allow_emulation)
        conn.send(('ready', worker.descriptor))
    except Exception as error:  # report a failed build, then stop
        conn.send(('error', f'{type(error).__name__}: {error}'))
        return

    try:
        while True:
            command, argument = conn.recv()
            if command == _CMD_CLOSE:
                break
            try:
                if command == _CMD_RESET:
                    frame = worker.reset(argument)
                elif command == _CMD_STEP:
                    frame = worker.step(argument)
                else:
                    raise WorldError(f'unknown worker command {command!r}')
                conn.send(('frame', frame))
            except Exception as error:  # a bad step fails this call, not the world
                conn.send(('error', f'{type(error).__name__}: {error}'))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        worker.close()
        conn.close()


class _Worker:
    """Owns the environment inside the worker process."""

    def __init__(self, settings: dict[str, Any], allow_emulation: bool) -> None:
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None

        scenario_name = settings.get(p.SETTING_SCENARIO_NAME)
        if not scenario_name:
            raise WorldError(f'missing required setting {p.SETTING_SCENARIO_NAME!r}')
        scenario_path = self._resolve_scenario(str(scenario_name), settings.get(p.SETTING_SCENARIO_YAML))

        emulate = bool(settings.get(p.SETTING_EMULATE, 0))
        if emulate and not allow_emulation:
            raise WorldError('emulation is disabled on this server (start it with allow_emulation=True)')

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

        obs, metadata = self.env.reset()  # a probe reset fixes the tensor widths
        max_subnets, max_hosts, n_action_ids = (int(x) for x in self.env.action_space.nvec)  # type: ignore[attr-defined]
        is_graph = self.env.graph_version is not None

        self.descriptor = {
            'action_bounds': [max_subnets - 1, max_hosts - 1, n_action_ids - 1],
            'raw_cols': int(np.asarray(self.env.s_raw).shape[1]),
            'is_graph': is_graph,
            'row_cols': None if is_graph else int(np.asarray(obs).shape[1]),
            'node_cols': int(np.asarray(obs['graph'].nodes).shape[1]) if is_graph else None,
            'max_node_index': max(max_subnets, max_hosts),
            'metadata': metadata,
            'frame': self._frame(obs, reward=0.0, info=p.neutral_info(), terminated=False, truncated=False),
        }

    def reset(self, seed: int | None) -> dict[str, Any]:
        obs, _metadata = self.env.reset(seed=seed)
        return self._frame(obs, reward=0.0, info=p.neutral_info(), terminated=False, truncated=False)

    def step(self, action: list[int]) -> dict[str, Any]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._frame(obs, reward=reward, info=info, terminated=terminated, truncated=truncated)

    def close(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def _resolve_scenario(self, scenario_name: str, scenario_yaml: Any) -> str:
        if not scenario_yaml:
            return scenario_name  # a path on the server's own filesystem
        if '.v2' not in scenario_name:
            raise WorldError(f'scenario name must be a .v2.yaml file: {scenario_name!r}')
        self._tempdir = tempfile.TemporaryDirectory(prefix='nasimemu-scenario-')
        path = os.path.join(self._tempdir.name, os.path.basename(scenario_name))
        with open(path, 'w') as handle:
            handle.write(str(scenario_yaml))
        return path

    def _frame(self, obs: Any, reward: float, info: dict[str, Any],
               terminated: bool, truncated: bool) -> dict[str, Any]:
        return {
            'obs_tensors': self._split_obs(obs),
            's_raw': np.asarray(self.env.s_raw, dtype=np.float64),
            'reward': float(reward),
            'info': info,
            'terminated': bool(terminated),
            'truncated': bool(truncated),
        }

    def _split_obs(self, obs: Any) -> dict[str, Any]:
        if isinstance(obs, dict):  # graph observation
            graph = obs['graph']
            return {
                p.OBS_GRAPH_NODES: np.asarray(graph.nodes, dtype=np.float64),
                p.OBS_GRAPH_EDGES: np.asarray(graph.edges, dtype=np.int64).reshape(-1),
                p.OBS_GRAPH_EDGE_LINKS: np.asarray(graph.edge_links, dtype=np.int64).reshape(-1, 2),
                p.OBS_GRAPH_NODE_INDEX: np.asarray(obs['node_index'], dtype=np.int64).reshape(-1, 2),
                p.OBS_GRAPH_POS_INDEX: np.asarray(obs['pos_index'], dtype=np.float64).reshape(-1),
            }
        return {p.OBS_MATRIX: np.asarray(obs, dtype=np.float64)}
