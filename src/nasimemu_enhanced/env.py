import gymnasium as gym, random
from gymnasium import spaces
from typing import Any
import numpy as np
from numpy.typing import NDArray
from nasimemu_enhanced import graph

from nasimemu_enhanced.actions import Action, ActionResult, Exploit, PrivilegeEscalation, ServiceScan, OSScan, SubnetScan, ProcessScan, NoOp
from nasimemu_enhanced.scenarios import load_scenario

from nasimemu_enhanced.simulation.environment import SimulationEnvironment
from nasimemu_enhanced.simulation.host_vector import HostVector
from nasimemu_enhanced.emulation.environment import EmulationEnvironment

class TerminalAction():
    pass

class PartiallyObservableWrapper():
    def reset(self, s: NDArray[Any]) -> NDArray[Any]:
        self.__obs: dict[Any, Any] = dict()
        obs = self.__update_obs(s)
        return obs

    def step(self, s: NDArray[Any]) -> NDArray[Any]:
        obs = self.__update_obs(s)
        return obs

    def __update_obs(self, s: NDArray[Any]) -> NDArray[Any]:
        for host_data in s[:-1]:
            address = HostVector(host_data).address

            if address == (0, 0): # skip invalid entries
                continue

            if address in self.__obs:
                new_fields = host_data != 0
                self.__obs[address][new_fields] = host_data[new_fields]
            else:
                self.__obs[address] = host_data

        # construct and return new observation
        action_result = s[-1]
        obs = np.vstack([list(self.__obs.values()), action_result])

        return obs

# observation_format in ['matrix', 'graph']
class NASimEmuEnv(gym.Env[Any, Any]):
    def __init__(self, scenario_name: str, emulate: bool = False, step_limit: int | None = None, random_init: bool = False, observation_format: str = 'matrix', fully_obs: bool = False, augment_with_action: bool = False, verbose: bool = False) -> None:
        # different processes need different seeds
        random.seed()
        np.random.seed()

        self.emulate = emulate
        self.step_limit = step_limit
        self.verbose = verbose
        self.fully_obs = fully_obs
        self.augment_with_action = augment_with_action
        self.observation_format = observation_format
        self.graph_version = {'graph': 1, 'graph_v1': 1, 'graph_v2': 2}.get(observation_format) # any other format is the matrix

        self.scenario_name = scenario_name
        self.random_init = random_init

        self._define_spaces()

    # the spaces have to be known before the first reset(), so every listed scenario is instantiated once and has to agree on them
    def _define_spaces(self) -> None:
        specs = []

        for scenario_name in self.scenario_name.split(':'):
            scenario = load_scenario(scenario_name)
            self.env: SimulationEnvironment = SimulationEnvironment(scenario, fully_obs=self.fully_obs)
            _, _, self.action_list = self._create_action_lists()

            specs.append( (scenario.address_space_bounds, self._get_layout(), self._get_action_catalogue()) )

        if any(spec != specs[0] for spec in specs):
            raise ValueError(f"Scenarios with different address space bounds, host vectors or actions cannot share one environment: {self.scenario_name}")

        (max_subnets, max_hosts), layout, action_catalogue = specs[0]
        row_size = layout['row_size']

        # action = (subnet, host, action_id); the last action_id is the terminal action
        self.action_shape = (max_subnets, max_hosts, len(action_catalogue))
        self.action_space = spaces.MultiDiscrete(list(self.action_shape))

        if self.graph_version is None:
            # one row per observed host and a last row with the action result; the number of rows grows as hosts are discovered
            self.observation_space = spaces.Sequence(spaces.Box(-np.inf, np.inf, (row_size,), dtype=np.float32), stack=True)
        else:
            # hosts and subnets are nodes; a node has its type (0 = host, 1 = subnet) prepended to its row
            # the edges carry no information, but a Graph space only accepts edge links together with edge features
            self.observation_space = spaces.Dict({
                'graph': spaces.Graph(node_space=spaces.Box(-np.inf, np.inf, (row_size + 1,), dtype=np.float64), edge_space=spaces.Discrete(1)),
                'node_index': spaces.Sequence(spaces.Box(-1, max(max_subnets, max_hosts), (2,), dtype=np.int64), stack=True), # (subnet, host), host = -1 for subnets
                'pos_index': spaces.Sequence(spaces.Box(0, np.inf, (), dtype=np.float64), stack=True),
            })

    # where the fields are in a host row, as [start, stop) columns
    def _get_layout(self) -> dict[str, Any]:
        hv = HostVector
        fields = {
            'subnet': [hv._subnet_address_idx, hv._host_address_idx],
            'host': [hv._host_address_idx, hv._compromised_idx],
            'compromised': [hv._compromised_idx, hv._compromised_idx + 1],
            'reachable': [hv._reachable_idx, hv._reachable_idx + 1],
            'discovered': [hv._discovered_idx, hv._discovered_idx + 1],
            'value': [hv._value_idx, hv._value_idx + 1],
            'discovery_value': [hv._discovery_value_idx, hv._discovery_value_idx + 1],
            'access': [hv._access_idx, hv._access_idx + 1],
            'os': [hv._os_start_idx, hv._service_start_idx],
            'services': [hv._service_start_idx, hv._process_start_idx],
            'processes': [hv._process_start_idx, hv.state_size],
        }
        row_size = hv.state_size

        if self.augment_with_action:
            fields['last_action'] = [row_size, row_size + len(self.action_list)]
            row_size += len(self.action_list)

        return {
            'row_size': row_size,
            'fields': fields,
            'os': sorted(hv.os_idx_map, key=lambda x: hv.os_idx_map[x]),
            'services': sorted(hv.service_idx_map, key=lambda x: hv.service_idx_map[x]),
            'processes': sorted(hv.process_idx_map, key=lambda x: hv.process_idx_map[x]),
        }

    # the actions an agent can choose from, by action_id
    def _get_action_catalogue(self) -> list[dict[str, Any]]:
        catalogue: list[dict[str, Any]] = []

        for action_id, (a_class, a_params) in enumerate(self.action_list):
            params = {k: (v if isinstance(v, str) else int(v) if isinstance(v, int) else float(v)) for k, v in a_params.items() if v is not None}
            catalogue.append({'id': action_id, 'type': a_class.__name__, **params})

        catalogue.append({'id': len(self.action_list), 'type': TerminalAction.__name__})
        return catalogue

    def _generate_env(self) -> None:
        if ':' in self.scenario_name: # there are multiple possible scenarios
            scenarios = self.scenario_name.split(':')
            scenario_name = random.choice(scenarios)
        else:
            scenario_name = self.scenario_name

        scenario = load_scenario(scenario_name)

        if self.emulate:
            self.env = EmulationEnvironment(scenario=scenario)
        else:
            self.env = SimulationEnvironment(scenario, fully_obs=self.fully_obs)

        if not self.fully_obs:
            self.env_po_wrapper = PartiallyObservableWrapper()

        # self.edge_index = self._gen_edge_index()
        self.exploit_list, self.privesc_list, self.action_list = self._create_action_lists()
        self.action_cls = [x[0] for x  in self.action_list]

        host_num_map = self.env.scenario.host_num_map
        self.host_index = np.array( sorted(host_num_map, key=lambda h: host_num_map[h]) )# fixed order node index
        self.subnet_index = np.array( [(x, -1) for x in range(len(self.env.scenario.subnets))] )

        self.discovered_subnets: set[int] = set()
        self.subnet_graph: set[tuple[int, int]] = set() # (from, to)

    def _create_action_lists(self) -> tuple[list[Any], list[Any], list[tuple[type[Action], dict[str, Any]]]]:
        exploit_list = sorted(self.env.scenario.exploits.items())
        privesc_list = sorted(self.env.scenario.privescs.items())

        action_list: list[tuple[type[Action], dict[str, Any]]] = [
                (ServiceScan, {'cost': self.env.scenario.service_scan_cost}),
                (OSScan, {'cost': self.env.scenario.os_scan_cost}),
                (SubnetScan, {'cost': self.env.scenario.subnet_scan_cost}),
                (ProcessScan, {'cost': self.env.scenario.process_scan_cost}),
        ]

        for exploit_name, exploit_params in exploit_list:
            action_list.append( (Exploit, {'name': exploit_name, **exploit_params}))

        for privesc_name, privesc_params in privesc_list:
            action_list.append( (PrivilegeEscalation, {'name': privesc_name, **privesc_params}))

        return exploit_list, privesc_list, action_list

    def _translate_action(self, action: Any) -> Any:
        subnet, host, action_id = (int(x) for x in action)

        if action_id == len(self.action_list): # terminal action
            return TerminalAction()

        a_class, a_params = self.action_list[action_id]
        # print(a_class, a_params, subnet, host, action_id)
        return a_class(target=(subnet, host), **a_params)

    # the action space covers the whole address space, but only some of the addresses are hosts
    def _is_valid_target(self, a: Action) -> bool:
        if self.emulate: # in emulation, we don't know the exact scenario configuration
            return True

        return a.target in self.env.scenario.hosts

    # an action against an address without a host fails with a connection error and still costs
    def _step_invalid_target(self, a: Action) -> tuple[NDArray[Any], float, dict[str, Any]]:
        assert self.env.current_state is not None
        result = ActionResult(False, 0.0, connection_error=True)
        obs = self.env.current_state.get_observation(a, result, self.fully_obs)
        return obs.numpy(), -a.cost, result.info()

    # marks the actions that target an observed host, in the shape of the action space; it repeats what the observation shows and says nothing about whether an action will succeed
    def _get_action_mask(self, s: NDArray[Any]) -> NDArray[np.bool_]:
        action_mask = np.zeros(self.action_shape, dtype=np.bool_)

        for host_data in s[:-1]:
            subnet, host = HostVector(host_data).address

            if (subnet, host) != (0, 0): # skip invalid entries
                action_mask[subnet, host] = True

        return action_mask

    def _to_observation(self, s: NDArray[Any]) -> Any:
        if self.graph_version is None:
            return s

        node_feats, edge_index, node_index, pos_index = graph.convert_to_graph(s, self.subnet_graph, version=self.graph_version)

        return {
            'graph': spaces.GraphInstance(nodes=node_feats, edges=np.zeros(edge_index.shape[1], dtype=np.int64), edge_links=edge_index.T),
            'node_index': node_index,
            'pos_index': pos_index,
        }

    def _get_subnets(self, s: NDArray[Any]) -> set[int]:
        return {HostVector(x).address[0] for x in s[:-1]}

    def _augment_with_action(self, s: NDArray[Any], action: Any) -> NDArray[Any]:
        action_matrix = np.zeros((len(s), len(self.action_list)), dtype=np.float32)

        if action is not None:
            subnet, host, action_id = (int(x) for x in action)

            host_addresses = [HostVector(host_data).address for host_data in s[:-1]]

            if (subnet, host) in host_addresses: # the target does not have to be an observed host
                action_column = -1 if action_id == len(self.action_list) else action_id # the terminal action shares the last column
                action_matrix[host_addresses.index((subnet, host)), action_column] = 1.

        s = np.concatenate((s, action_matrix), axis=1)

        return s

    # action = (subnet, host, action_id)
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        s: Any
        if isinstance(action, (Action, TerminalAction)):
            a = action
        else:
            a = self._translate_action(action)

        # ignore the flags from the environment, the agent has to choose to terminate
        terminated = isinstance(a, TerminalAction)

        if terminated:
            s, r, _, _, i = self.env.step(NoOp())
            r = 0.

        elif isinstance(a, Action) and not self._is_valid_target(a):
            s, r, i = self._step_invalid_target(a)

        else:
            s, r, _, _, i = self.env.step(a)

        r /= 10. # reward scaling
        self.r_tot += r

        if r > 0:
            self.captured += 1

        if not self.fully_obs:
            s = self.env_po_wrapper.step(s)

        # track newly discovered subnets and remember the origin
        if isinstance(a, SubnetScan):
            s_subnets = self._get_subnets(s)
            new_subnets = s_subnets - self.discovered_subnets
            origin_subnet = a.target[0]

            for new_subnet in new_subnets:
                self.subnet_graph.add( (origin_subnet, new_subnet) )

            self.discovered_subnets = s_subnets

        self.step_idx += 1

        if self.verbose:
            print("===========================================")
            print(f"Step: {self.step_idx}")
            # print(f"Raw state: \n {s}")
            print(f"Action: {a}")
            print(f"R: {r} D: {terminated}")
            print(f"Info: {i}")

            self.env.render(obs=s)
            # self.env.render() # show observation

        self.s_raw = s # before any augmentation / transformation (e.g., into graph)

        i['s_raw'] = s
        i['a_raw'] = a
        i['action_mask'] = self._get_action_mask(s)

        # optionally, include the last performed action
        if self.augment_with_action:
            s = self._augment_with_action(s, action)

        # optionally, convert the observation into a graph
        s = self._to_observation(s)

        i['s_true'] = s
        i['step_idx'] = self.step_idx
        i['subnet_graph'] = self.subnet_graph
        i['r_tot'] = self.r_tot
        i['captured'] = self.captured

        truncated = (not terminated) and (self.step_limit is not None) and (self.step_idx >= self.step_limit)

        i['done'] = terminated or truncated
        i['d_true'] = terminated or truncated # fix: this will disable the difference between true termination and step_limit exceedance - both are treated the same

        return s, r, terminated, truncated, i

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None: # the simulation draws from the global generators
            random.seed(seed)
            np.random.seed(seed)

        self.step_idx = 0
        self.r_tot = 0.
        self.captured = 0

        self._generate_env() # generate new env

        s: Any
        s, _ = self.env.reset()

        if not self.fully_obs:
            s = self.env_po_wrapper.reset(s)

        self.s_raw = s
        action_mask = self._get_action_mask(s)

        self.discovered_subnets = self._get_subnets(s)
        self.subnet_graph = set()

        # optionally, include the last performed action (zeros in reset())
        if self.augment_with_action:
            s = self._augment_with_action(s, None)

        s = self._to_observation(s)

        # break the tie with random offset
        if self.random_init:
            assert self.step_limit is not None
            self.step_idx = np.random.randint(self.step_limit)
            self.random_init = False

        if self.verbose:
            print()
            print("-------")
            print("reset()")
            print("-------")

            self.env.render_state()
            self.env.render(obs=s)

        return s, {'layout': self._get_layout(), 'actions': self._get_action_catalogue(), 'action_mask': action_mask}

    def render(self, s: NDArray[Any]) -> None:  # type: ignore[override]  # renders the given observation
        self.env.render(obs=s)

    def render_state(self) -> None:
        self.env.render_state()