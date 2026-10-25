# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [2.0.0] - Unreleased

NASimEmu-Enhanced 2.0.0 moves from gym 0.21 to [Gymnasium](https://gymnasium.farama.org/)
and gives the environment a declared, self-describing interface. It is not
backwards compatible; see [Migrating from 1.x](#migrating-from-1x).

### Added

- Python 3.10 and newer are supported; 1.x was limited to Python 3.10.
- `NASimEmuEnv` declares standard Gymnasium spaces and passes Gymnasium's
  `check_env`:
  - `action_space` is `MultiDiscrete([subnets, hosts, actions + 1])`.
  - `observation_space` is a `Sequence` of host rows in the matrix format and a
    `Dict` around a `Graph` in the graph formats.
- `reset()` takes `seed`, which seeds the global `random` and `numpy.random`
  generators the simulation draws from.
- The info returned by `reset()` describes the observations and actions:
  - `info['layout']` names the columns of a host row: `row_size`, `fields`
    (field name to `[start, stop)` columns) and the names behind the `os`,
    `services` and `processes` columns.
  - `info['actions']` has one dict per `action_id` with `id`, `type` and the
    parameters of the action; the last entry is the terminal action.
- The info returned by `step()` holds plain data only, so it can be serialized,
  e.g. as JSON, and has the same keys in simulation and emulation:
  - `info['action']` is the performed action as `[subnet, host, action_id]`.
  - The result of the action: `info['success']`, `info['connection_error']`,
    `info['permission_error']`, `info['undefined_error']`, `info['value']`,
    `info['access']`, the revealed `info['services']`, `info['os']` and
    `info['processes']`, and the `info['discovered']` and
    `info['newly_discovered']` hosts.
  - The episode counters `info['step_idx']`, `info['r_tot']`,
    `info['captured']` and `info['subnet_graph']`.
- `env.a_raw` holds the last performed action object, next to `env.s_raw` and
  `env.subnet_graph`.
- Scenario `xs_entry_dmz_one_subnet.v2.yaml`, small enough to emulate on a
  laptop.

### Changed

- **Breaking:** the package is imported as `nasimemu_enhanced` instead of
  `nasimemu`, and its modules are regrouped by responsibility (see
  [Migrating from 1.x](#migrating-from-1x)).
- **Breaking:** gym 0.21 is replaced by Gymnasium 1.3:
  - `reset()` returns `(obs, info)`.
  - `step()` returns `(obs, reward, terminated, truncated, info)`.
- **Breaking:** the environment no longer resets itself at the end of an
  episode; call `reset()`.
- **Breaking:** the episode end is split into two flags:
  - `terminated` is set only when the agent chooses the terminal action.
  - `truncated` is set when `step_limit` is reached.
- **Breaking:** an action is `(subnet, host, action_id)` instead of
  `((subnet, host), action_id)`. The terminal action is
  `action_id = len(env.action_list)` instead of `-1`.
  `graph.get_possible_actions()` returns actions in the new format.
- **Breaking:** graph observations are a dict instead of the tuple
  `(node_feats, edge_index, node_index, pos_index)`:
  - `graph` is a `GraphInstance`. Its `nodes` are the old `node_feats`, and its
    `edge_links` are the old `edge_index` transposed.
  - `node_index` and `pos_index` are unchanged.
- **Breaking:** the `step()` info no longer contains `s_raw`, `s_true`,
  `a_raw`, `done` and `d_true`. Its result fields are plain values (see Added)
  instead of numpy values, tuple-keyed dicts and sets.
- An action against an address without a host fails with a connection error
  and costs, instead of raising an `AssertionError`.
- The Vagrant generator is run as
  `python3 -m nasimemu_enhanced.emulation.vagrant_gen`; `setup_vagrant.sh` is
  updated.
- Dependencies: `gym` is replaced by `gymnasium==1.3`, `numpy==2.2.6` is
  pinned, and `matplotlib` and `pillow` are no longer required.

### Removed

- **Breaking:** V1 scenarios and NASim's static benchmark scenarios; only
  `.v2.yaml` scenarios load.
- **Breaking:** generated scenarios (names such as `huge-gen-rgoal-stoch`
  instead of a file path) and the NASim scenario generator. Use upstream
  [NASimEmu](https://github.com/jaromiru/NASimEmu) for these.
- NASim's flat action and observation mode, example agents, scripts, demo,
  graph viewer and its own gym registrations.

### Fixed

- Emulation: an action waits for a Metasploit console left busy by the previous
  action instead of failing.
- Emulation: `ssh-rsa` is re-enabled so that `rsync` to the Linux target boxes
  works.
- `undefined_error` is reported in the `step()` info.

### Migrating from 1.x

**Install and import.** The PyPI name stays `nasimemu-enhanced`; the import
changes.

```python
# 1.x
import gym, nasimemu
import nasimemu.env_utils as env_utils

# 2.0
import gymnasium as gym, nasimemu_enhanced
import nasimemu_enhanced.graph as graph
```

**Module paths.**

| 1.x | 2.0 |
| --- | --- |
| `nasimemu.env` | `nasimemu_enhanced.env` |
| `nasimemu.env_utils` | `nasimemu_enhanced.graph` |
| `nasimemu.env_emu` | `nasimemu_enhanced.emulation.network` |
| `nasimemu.msf_interface` | `nasimemu_enhanced.emulation.msf_client` |
| `nasimemu.vagrant_gen` | `nasimemu_enhanced.emulation.vagrant_gen` |
| `nasimemu.nasim.envs.action` | `nasimemu_enhanced.actions` |
| `nasimemu.nasim.envs.environment` | `nasimemu_enhanced.simulation.environment` |
| `nasimemu.nasim.envs.host_vector` | `nasimemu_enhanced.simulation.host_vector` |
| `nasimemu.nasim.envs.network`, `.observation`, `.state` | `nasimemu_enhanced.simulation.network`, `.observation`, `.state` |
| `nasimemu.nasim.envs.utils` | `nasimemu_enhanced.utils` |
| `nasimemu.nasim.scenarios` | `nasimemu_enhanced.scenarios` |

**Environment attributes.** Gymnasium wrappers no longer forward attributes of
the wrapped environment. Reach `action_list`, `s_raw`, `a_raw`,
`subnet_graph`, `render(s)` and `render_state()` through `env.unwrapped`.

**The interaction loop.**

```python
# 1.x
s = env.reset()
s, r, done, info = env.step(((subnet, host), action_id))   # -1 ends the episode
# on done, s is already the first observation of the next episode

# 2.0
s, info = env.reset(seed=0)
s, r, terminated, truncated, info = env.step((subnet, host, action_id))   # len(env.unwrapped.action_list) ends the episode
if terminated or truncated:
    s, info = env.reset()
```

**Graph observations.**

```python
# 1.x
node_feats, edge_index, node_index, pos_index = s

# 2.0
node_feats, edge_index = s['graph'].nodes, s['graph'].edge_links.T
node_index, pos_index = s['node_index'], s['pos_index']
```

**`step()` info.**

| 1.x key | 2.0 replacement |
| --- | --- |
| `info['s_true']` | the returned observation; the environment no longer resets itself, so it is always the true one |
| `info['done']`, `info['d_true']` | `terminated or truncated` |
| `info['s_raw']` | `env.unwrapped.s_raw` |
| `info['a_raw']` | `info['action']` as `[subnet, host, action_id]`, or the action object `env.unwrapped.a_raw` |
| `info['subnet_graph']` (set of tuples) | `info['subnet_graph']` (sorted list of `[from, to]`), or the set `env.unwrapped.subnet_graph` |
| `info['discovered']`, `info['newly_discovered']` (`{(subnet, host): bool}` over all addresses) | sorted lists of the `[subnet, host]` addresses that are `True` |
| `info['access']` (`AccessLevel`, float or `{}`) | `int`: 0 = none, 1 = user, 2 = root |

`env.unwrapped.s_raw` and `env.unwrapped.a_raw` hold the latest step only; read
them right after `step()`.

**Plotting.**

```python
# 1.x
env_utils.plot_network(info['s_raw'], info['subnet_graph'], info['a_raw'])

# 2.0
e = env.unwrapped
graph.plot_network(e.s_raw, e.subnet_graph, e.a_raw)
```

**Scenarios.** Pass `.v2.yaml` files only. Generated scenario names and V1
scenario files raise an error.

## [1.0.1] - 2026-10-04

### Changed

- The package version is derived from the git tag (`hatch-vcs`).
- The vendored `pymetasploit3` fork is replaced by the `pymetasploit3` package
  from PyPI.
- Type checking with mypy in strict mode runs in CI.

## [1.0.0] - 2026-09-22

### Added

- First release on PyPI as `nasimemu-enhanced`, packaging
  [NASimEmu](https://github.com/jaromiru/NASimEmu) for gym 0.21 and Python 3.10.

[2.0.0]: https://github.com/dreamfarer/NASimEmu-Enhanced/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/dreamfarer/NASimEmu-Enhanced/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/dreamfarer/NASimEmu-Enhanced/releases/tag/v1.0.0
