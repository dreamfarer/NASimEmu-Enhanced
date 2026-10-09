"""The public Gymnasium environment of NASimEmu.

:class:`NASimEmuEnv` is the single entry point for agents. It wraps either the
simulator or the emulator, declares standard Gymnasium spaces and describes
its observations and actions as plain data, so that an agent does not need
any other part of the package.
"""
import gymnasium as gym, random
import numpy as np

from gymnasium import spaces
from typing import Any
from numpy.typing import NDArray

from nasimemu_enhanced.graph import convert_to_graph
from nasimemu_enhanced.actions import Action, ActionResult, Exploit, PrivilegeEscalation, ServiceScan, OSScan, \
    SubnetScan, ProcessScan, NoOp
from nasimemu_enhanced.scenarios import load_scenario
from nasimemu_enhanced.simulation.environment import SimulationEnvironment
from nasimemu_enhanced.simulation.host_vector import HostVector
from nasimemu_enhanced.emulation.environment import EmulationEnvironment


class TerminalAction:
    """The action with which the agent ends the episode.

    It is selected with the last ``action_id`` of the action space and is not
    forwarded to the network.
    """


class PartiallyObservableWrapper:
    """Accumulate what the agent has observed over an episode.

    The simulator only reports what the last action revealed. This wrapper
    remembers every host seen so far and merges new information into it, so
    the observation has one row per observed host and grows as hosts are
    discovered.
    """

    def reset(self, s: NDArray[Any]) -> NDArray[Any]:
        """Forget everything and start from the initial observation.

        Parameters
        ----------
        s : numpy.ndarray
            initial observation of the wrapped environment

        Returns
        -------
        numpy.ndarray
            the accumulated observation
        """
        self.__obs: dict[Any, Any] = dict()
        obs = self.__update_obs(s)
        return obs

    def step(self, s: NDArray[Any]) -> NDArray[Any]:
        """Merge the observation of the last action into what is known.

        Parameters
        ----------
        s : numpy.ndarray
            observation of the wrapped environment after an action

        Returns
        -------
        numpy.ndarray
            the accumulated observation
        """
        obs = self.__update_obs(s)
        return obs

    def __update_obs(self, s: NDArray[Any]) -> NDArray[Any]:
        """Update the known hosts and build the observation from them.

        Non-zero fields of a host row overwrite what was known about that
        host. The last row (the result of the action) is passed through.
        """
        for host_data in s[:-1]:
            address = HostVector(host_data).address

            if address == (0, 0):  # skip invalid entries
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


class NASimEmuEnv(gym.Env[Any, Any]):
    """A penetration-testing environment on a simulated or emulated network.

    Implements the Gymnasium interface and is registered as ``NASimEmu-v0``.
    A new scenario instance is generated on every :meth:`reset`.

    Parameters
    ----------
    scenario_name : str
        path to a ``.v2.yaml`` scenario file. Several paths can be joined with
        ``:``; one of them is then chosen at random on every reset. All of
        them have to share address space bounds, host row layout and actions.
    emulate : bool, optional
        run against the emulated network instead of the simulator
        (default=False)
    step_limit : int, optional
        number of steps after which an episode is truncated; no limit if None
        (default=None)
    random_init : bool, optional
        start the step counter of the first episode at a random offset below
        ``step_limit``, so that parallel environments do not end their
        episodes in the same step. Requires ``step_limit`` (default=False)
    observation_format : str, optional
        ``'graph'`` (or ``'graph_v1'``) and ``'graph_v2'`` return the
        observation as a graph; any other value returns the matrix
        (default='matrix')
    fully_obs : bool, optional
        observe the whole network from the start instead of only what the
        agent has discovered (default=False)
    augment_with_action : bool, optional
        append the last action to the observation as additional columns
        (default=False)
    verbose : bool, optional
        print every step and reset (default=False)

    Attributes
    ----------
    action_space : gymnasium.spaces.MultiDiscrete
        an action is ``(subnet, host, action_id)``. The last ``action_id`` is
        the terminal action, which ends the episode.
    observation_space : gymnasium.spaces.Space
        in the matrix format a ``Sequence`` of host rows, in the graph formats
        a ``Dict`` with a ``Graph`` and two index arrays (see Notes)
    action_list : list
        the actions an agent can choose, as ``(action class, parameters)``,
        indexed by ``action_id``
    s_raw : numpy.ndarray
        the last observation as a matrix, before the last action is appended
        and before the conversion into a graph
    subnet_graph : set
        ``(from, to)`` pairs of subnets, recording from which subnet another
        one was discovered

    Raises
    ------
    ValueError
        if the listed scenarios do not share the same spaces

    Notes
    -----
    **Matrix observation.** One row per observed host, followed by one row
    with the result of the last action. The number of rows grows as hosts are
    discovered; the number of columns is fixed. ``info['layout']`` of
    :meth:`reset` names the columns.

    **Graph observation.** A dict with

    - ``graph``: a ``GraphInstance``. Hosts and subnets are nodes; a node is
      its row with the node type prepended (0 = host, 1 = subnet). The edges
      carry a constant feature, only ``edge_links`` is meaningful.
    - ``node_index``: ``(subnet, host)`` of every node, with host = -1 for
      subnet nodes
    - ``pos_index``: position of every host node, 0 for subnet nodes

    In ``'graph'`` hosts of a subnet are connected to each other, in
    ``'graph_v2'`` they are connected to their subnet node.

    **Episode end.** ``terminated`` is only set by the terminal action; the
    environment never ends an episode on its own when all sensitive hosts are
    compromised. ``truncated`` is set when ``step_limit`` is reached.

    **Randomness.** The simulation draws from the global ``random`` and
    ``numpy.random`` generators, which :meth:`reset` seeds when given a seed.
    Environments in the same process therefore share one random stream.

    Examples
    --------
    >>> import gymnasium as gym
    >>> import nasimemu_enhanced
    >>> env = gym.make('NASimEmu-v0', scenario_name='scenarios/uni.v2.yaml')
    >>> obs, info = env.reset(seed=0)
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    def __init__(self, scenario_name: str, emulate: bool = False, step_limit: int | None = None,
                 random_init: bool = False, observation_format: str = 'matrix', fully_obs: bool = False,
                 augment_with_action: bool = False, verbose: bool = False) -> None:
        random.seed()
        np.random.seed()

        self.emulate = emulate
        self.step_limit = step_limit
        self.verbose = verbose
        self.fully_obs = fully_obs
        self.augment_with_action = augment_with_action
        self.observation_format = observation_format
        self.graph_version = {'graph': 1, 'graph_v1': 1, 'graph_v2': 2}.get(
            observation_format)  # any other format is the matrix

        self.scenario_name = scenario_name
        self.random_init = random_init

        self._define_spaces()

    def _define_spaces(self) -> None:
        """Set the action and observation space.

        The spaces have to be known before the first reset(), so every listed
        scenario is instantiated once and has to agree on them.

        Raises
        ------
        ValueError
            if the scenarios differ in address space bounds, host row layout
            or actions
        """
        specs = []

        for scenario_name in self.scenario_name.split(':'):
            scenario = load_scenario(scenario_name)
            self.env: SimulationEnvironment = SimulationEnvironment(scenario, fully_obs=self.fully_obs)
            _, _, self.action_list = self._create_action_lists()

            specs.append((scenario.address_space_bounds, self._get_layout(), self._get_action_catalogue()))

        if any(spec != specs[0] for spec in specs):
            raise ValueError(
                f"Scenarios with different address space bounds, host vectors or actions cannot share one environment: {self.scenario_name}")

        (max_subnets, max_hosts), layout, action_catalogue = specs[0]
        row_size = layout['row_size']

        # action = (subnet, host, action_id); the last action_id is the terminal action
        self.action_shape = (max_subnets, max_hosts, len(action_catalogue))
        self.action_space = spaces.MultiDiscrete(list(self.action_shape))

        if self.graph_version is None:
            # one row per observed host and a last row with the action result; the number of rows grows as hosts are discovered
            self.observation_space = spaces.Sequence(spaces.Box(-np.inf, np.inf, (row_size,), dtype=np.float32),
                                                     stack=True)
        else:
            # hosts and subnets are nodes; a node has its type (0 = host, 1 = subnet) prepended to its row
            # the edges carry no information, but a Graph space only accepts edge links together with edge features
            self.observation_space = spaces.Dict({
                'graph': spaces.Graph(node_space=spaces.Box(-np.inf, np.inf, (row_size + 1,), dtype=np.float64),
                                      edge_space=spaces.Discrete(1)),
                'node_index': spaces.Sequence(spaces.Box(-1, max(max_subnets, max_hosts), (2,), dtype=np.int64),
                                              stack=True),  # (subnet, host), host = -1 for subnets
                'pos_index': spaces.Sequence(spaces.Box(0, np.inf, (), dtype=np.float64), stack=True),
            })

    def _get_layout(self) -> dict[str, Any]:
        """Describe the columns of a host row.

        Returns
        -------
        dict
            ``row_size`` is the number of columns. ``fields`` maps every field
            (``subnet``, ``host``, ``compromised``, ``reachable``,
            ``discovered``, ``value``, ``discovery_value``, ``access``,
            ``os``, ``services``, ``processes`` and, with
            ``augment_with_action``, ``last_action``) to its ``[start, stop)``
            columns. ``os``, ``services`` and ``processes`` list the names
            behind the columns of these fields, in column order.
        """
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

    def _get_action_catalogue(self) -> list[dict[str, Any]]:
        """Describe the actions an agent can choose from.

        Returns
        -------
        list of dict
            one entry per ``action_id`` with ``id``, ``type`` (the name of the
            action class) and the parameters of the action, such as ``name``,
            ``cost``, ``prob``, ``os``, ``service`` or ``process``. The last
            entry is the terminal action.
        """
        catalogue: list[dict[str, Any]] = []

        for action_id, (a_class, a_params) in enumerate(self.action_list):
            params = {k: (v if isinstance(v, str) else int(v) if isinstance(v, int) else float(v)) for k, v in
                      a_params.items() if v is not None}
            catalogue.append({'id': action_id, 'type': a_class.__name__, **params})

        catalogue.append({'id': len(self.action_list), 'type': TerminalAction.__name__})
        return catalogue

    def _generate_env(self) -> None:
        """Generate a new scenario instance and the environment running it.

        With several scenarios, one is chosen at random.
        """
        if ':' in self.scenario_name:  # there are multiple possible scenarios
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
        self.action_cls = [x[0] for x in self.action_list]

        host_num_map = self.env.scenario.host_num_map
        self.host_index = np.array(sorted(host_num_map, key=lambda h: host_num_map[h]))  # fixed order node index
        self.subnet_index = np.array([(x, -1) for x in range(len(self.env.scenario.subnets))])

        self.discovered_subnets: set[int] = set()
        self.subnet_graph: set[tuple[int, int]] = set()  # (from, to)

    def _create_action_lists(self) -> tuple[list[Any], list[Any], list[tuple[type[Action], dict[str, Any]]]]:
        """List the exploits, privilege escalations and all actions of the scenario.

        Returns
        -------
        list
            ``(name, parameters)`` of every exploit, sorted by name
        list
            ``(name, parameters)`` of every privilege escalation, sorted by
            name
        list
            ``(action class, parameters)`` of every action, indexed by
            ``action_id``: the four scans, then the exploits, then the
            privilege escalations
        """
        exploit_list = sorted(self.env.scenario.exploits.items())
        privesc_list = sorted(self.env.scenario.privescs.items())

        action_list: list[tuple[type[Action], dict[str, Any]]] = [
            (ServiceScan, {'cost': self.env.scenario.service_scan_cost}),
            (OSScan, {'cost': self.env.scenario.os_scan_cost}),
            (SubnetScan, {'cost': self.env.scenario.subnet_scan_cost}),
            (ProcessScan, {'cost': self.env.scenario.process_scan_cost}),
        ]

        for exploit_name, exploit_params in exploit_list:
            action_list.append((Exploit, {'name': exploit_name, **exploit_params}))

        for privesc_name, privesc_params in privesc_list:
            action_list.append((PrivilegeEscalation, {'name': privesc_name, **privesc_params}))

        return exploit_list, privesc_list, action_list

    def _translate_action(self, action: Any) -> Any:
        """Turn an action of the action space into an action object.

        Parameters
        ----------
        action : sequence of int
            ``(subnet, host, action_id)``

        Returns
        -------
        Action or TerminalAction
            the action to perform
        """
        subnet, host, action_id = (int(x) for x in action)

        if action_id == len(self.action_list):  # terminal action
            return TerminalAction()

        a_class, a_params = self.action_list[action_id]
        # print(a_class, a_params, subnet, host, action_id)
        return a_class(target=(subnet, host), **a_params)

    def _is_valid_target(self, a: Action) -> bool:
        """Check whether the action targets an address with a host.

        The action space covers the whole address space, but only some of the
        addresses are hosts. In emulation the hosts are not known, so every
        target counts as valid.
        """
        if self.emulate:  # in emulation, we don't know the exact scenario configuration
            return True

        return a.target in self.env.scenario.hosts

    def _step_invalid_target(self, a: Action) -> tuple[NDArray[Any], float, dict[str, Any]]:
        """Perform an action against an address without a host.

        The action fails with a connection error, leaves the network unchanged
        and still costs.

        Returns
        -------
        numpy.ndarray
            observation of the failed action
        float
            the negative cost of the action
        dict
            the result of the action
        """
        assert self.env.current_state is not None
        result = ActionResult(False, 0.0, connection_error=True)
        obs = self.env.current_state.get_observation(a, result, self.fully_obs)
        return obs.numpy(), -a.cost, result.info()

    def _get_action_mask(self, s: NDArray[Any]) -> NDArray[np.bool_]:
        """Mark the actions that target an observed host.

        The mask repeats what the observation shows and says nothing about
        whether an action will succeed. The environment does not enforce it.

        Parameters
        ----------
        s : numpy.ndarray
            observation as a matrix

        Returns
        -------
        numpy.ndarray
            boolean array in the shape of the action space, True for every
            ``(subnet, host, action_id)`` whose host is in the observation
        """
        action_mask = np.zeros(self.action_shape, dtype=np.bool_)

        for host_data in s[:-1]:
            subnet, host = HostVector(host_data).address

            if (subnet, host) != (0, 0):  # skip invalid entries
                action_mask[subnet, host] = True

        return action_mask

    def _to_observation(self, s: NDArray[Any]) -> Any:
        """Convert the matrix into the configured observation format.

        Returns
        -------
        numpy.ndarray or dict
            the matrix itself, or the graph observation described in the
            Notes of :class:`NASimEmuEnv`
        """
        if self.graph_version is None:
            return s

        node_feats, edge_index, node_index, pos_index = convert_to_graph(s, self.subnet_graph,
                                                                         version=self.graph_version)

        return {
            'graph': spaces.GraphInstance(nodes=node_feats, edges=np.zeros(edge_index.shape[1], dtype=np.int64),
                                          edge_links=edge_index.T),
            'node_index': node_index,
            'pos_index': pos_index,
        }

    def _get_subnets(self, s: NDArray[Any]) -> set[int]:
        """Return the subnets of the hosts in the observation."""
        return {HostVector(x).address[0] for x in s[:-1]}

    def _augment_with_action(self, s: NDArray[Any], action: Any) -> NDArray[Any]:
        """Append the last action to the observation.

        One column per action is added to every row. The column of the last
        action is set to 1 in the row of its target host, if that host is in
        the observation; the terminal action uses the last column.

        Parameters
        ----------
        s : numpy.ndarray
            observation as a matrix
        action : sequence of int or None
            the last action as ``(subnet, host, action_id)``, None after a
            reset

        Returns
        -------
        numpy.ndarray
            the observation with the additional columns
        """
        action_matrix = np.zeros((len(s), len(self.action_list)), dtype=np.float32)

        if action is not None:
            subnet, host, action_id = (int(x) for x in action)

            host_addresses = [HostVector(host_data).address for host_data in s[:-1]]

            if (subnet, host) in host_addresses:  # the target does not have to be an observed host
                action_column = -1 if action_id == len(
                    self.action_list) else action_id  # the terminal action shares the last column
                action_matrix[host_addresses.index((subnet, host)), action_column] = 1.

        s = np.concatenate((s, action_matrix), axis=1)

        return s

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Perform one action.

        Implements gymnasium.Env.step().

        Parameters
        ----------
        action : sequence of int
            ``(subnet, host, action_id)``, an element of the action space.
            The last ``action_id`` is the terminal action. An action against
            an address without a host fails with a connection error and still
            costs.

        Returns
        -------
        numpy.ndarray or dict
            the observation, an element of the observation space
        float
            the value gained minus the cost of the action, divided by 10;
            0 for the terminal action
        bool
            terminated, True if the agent chose the terminal action
        bool
            truncated, True if ``step_limit`` was reached
        dict
            auxiliary information. Besides the result of the action (see
            :meth:`nasimemu_enhanced.actions.ActionResult.info`) it holds

            - ``action_mask``: boolean array in the shape of the action space,
              True for the actions that target an observed host
            - ``s_raw``: the observation as a matrix, before the last action
              is appended and before the conversion into a graph
            - ``s_true``: the returned observation
            - ``a_raw``: the performed action object
            - ``subnet_graph``: ``(from, to)`` pairs of discovered subnets
            - ``step_idx``: number of steps in this episode
            - ``r_tot``: sum of the rewards in this episode
            - ``captured``: number of steps with a positive reward in this
              episode
            - ``done`` and ``d_true``: terminated or truncated

        Notes
        -----
        The environment does not reset itself at the end of an episode; call
        :meth:`reset`.
        """
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

        r /= 10.  # reward scaling
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
                self.subnet_graph.add((origin_subnet, new_subnet))

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

        self.s_raw = s  # before any augmentation / transformation (e.g., into graph)

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
        i[
            'd_true'] = terminated or truncated  # fix: this will disable the difference between true termination and step_limit exceedance - both are treated the same

        return s, r, terminated, truncated, i

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Generate a new scenario instance and start an episode.

        Implements gymnasium.Env.reset().

        Parameters
        ----------
        seed : int, optional
            seed for the global ``random`` and ``numpy.random`` generators the
            simulation draws from; they are left as they are if None
            (default=None)
        options : dict, optional
            not used (default=None)

        Returns
        -------
        numpy.ndarray or dict
            the initial observation, an element of the observation space
        dict
            auxiliary information:

            - ``layout``: the columns of a host row, as ``row_size``,
              ``fields`` (field name to ``[start, stop)`` columns) and the
              names behind the ``os``, ``services`` and ``processes`` columns
            - ``actions``: one dict per ``action_id`` with ``id``, ``type``
              and the parameters of the action
            - ``action_mask``: boolean array in the shape of the action space,
              True for the actions that target an observed host
        """
        super().reset(seed=seed)

        if seed is not None:  # the simulation draws from the global generators
            random.seed(seed)
            np.random.seed(seed)

        self.step_idx = 0
        self.r_tot = 0.
        self.captured = 0

        self._generate_env()  # generate new env

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
        """Print an observation as a table.

        Parameters
        ----------
        s : numpy.ndarray
            observation as a matrix, e.g. ``s_raw``
        """
        self.env.render(obs=s)

    def render_state(self) -> None:
        """Print the full state of the network, including undiscovered hosts."""
        self.env.render_state()
