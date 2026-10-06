import numpy as np
from collections.abc import Collection
from typing import Any, cast

import nasimemu.nasim.scenarios.utils as u
from nasimemu.nasim.scenarios.host import Host

# (subnet, host) address of a host
Address = tuple[int, int]


class Scenario:

    def __init__(self, scenario_dict: dict[str, Any], name: str | None = None, permute_subnets: bool = True) -> None:

        self.scenario_dict = scenario_dict
        self.name = name
        self._e_map: dict[str, dict[str | None, dict[str, Any]]] | None = None
        self._pe_map: dict[str, dict[str | None, dict[str, Any]]] | None = None

        if permute_subnets:
            self._permute_subnets()

        # this is used for consistent positioning of
        # host state and obs in state and obs matrices
        self.host_num_map: dict[Address, int] = {}
        for host_num, host_addr in enumerate(self.hosts):
            self.host_num_map[host_addr] = host_num

    def _permute_subnets(self) -> None:
        # create the permutation sequence
        perm = np.concatenate([[0], np.random.permutation(np.arange(1, len(self.subnets)))])
        orig = np.arange(len(self.subnets))
        # print(f'perm={list(perm)}')
        # print(f'orig={list(orig)}')

        # permute the subnets' sizes
        subnets = np.array(self.scenario_dict[u.SUBNETS])
        subnets[perm] = subnets[orig]
        self.scenario_dict[u.SUBNETS] = subnets

        # permute topology
        topology = np.array(self.scenario_dict[u.TOPOLOGY])

        topology[perm, :] = topology[orig, :] # rows
        topology[:, perm] = topology[:, orig] # cols

        self.scenario_dict[u.TOPOLOGY] = topology

        # permute sensitive_hosts
        sensitive_hosts: dict[Address, float] = {}
        for ((subnet, host_addr), value) in self.scenario_dict[u.SENSITIVE_HOSTS].items():
            sensitive_hosts[(perm[subnet], host_addr)] = value

        self.scenario_dict[u.SENSITIVE_HOSTS] = sensitive_hosts

        # permute host_configs
        hosts: dict[Address, Host] = {}
        for ((subnet, host_addr), host_config) in self.scenario_dict[u.HOSTS].items():
            # alter the host's address
            host_config.address = (perm[subnet], host_addr)

            # alter the host firewall
            fw_dict: dict[Any, list[str]] = {}
            for ((fw_subnet, fw_host_addr), fw_config) in host_config.firewall.items():
                fw_dict[(perm[fw_subnet], fw_host_addr)] = fw_config
            host_config.firewall = fw_dict

            # add to the correct spot
            hosts[(perm[subnet], host_addr)] = host_config

        self.scenario_dict[u.HOSTS] = hosts

        # permute firewall
        firewall: dict[tuple[int, int], Collection[str]] = {}
        for ((fw_from, fw_to), fw_config) in self.scenario_dict[u.FIREWALL].items():
            firewall[(perm[fw_from], perm[fw_to])] = fw_config

        self.scenario_dict[u.FIREWALL] = firewall

    @property
    def step_limit(self) -> int | None:
        return self.scenario_dict.get(u.STEP_LIMIT, None)

    @property
    def services(self) -> list[str]:
        return cast(list[str], self.scenario_dict[u.SERVICES])

    @property
    def num_services(self) -> int:
        return len(self.services)

    @property
    def os(self) -> list[str]:
        return cast(list[str], self.scenario_dict[u.OS])

    @property
    def num_os(self) -> int:
        return len(self.os)

    @property
    def processes(self) -> list[str]:
        return cast(list[str], self.scenario_dict[u.PROCESSES])

    @property
    def num_processes(self) -> int:
        return len(self.processes)

    @property
    def exploits(self) -> dict[str, dict[str, Any]]:
        return cast(dict[str, dict[str, Any]], self.scenario_dict[u.EXPLOITS])

    @property
    def privescs(self) -> dict[str, dict[str, Any]]:
        return cast(dict[str, dict[str, Any]], self.scenario_dict[u.PRIVESCS])

    @property
    def exploit_map(self) -> dict[str, dict[str | None, dict[str, Any]]]:
        """A nested dictionary for all exploits in scenario.

        I.e. {service_name: {
                 os_name: {
                     name: e_name,
                     cost: e_cost,
                     prob: e_prob,
                     access: e_access
                 }
             }
        """
        if self._e_map is None:
            e_map: dict[str, dict[str | None, dict[str, Any]]] = {}
            for e_name, e_def in self.exploits.items():
                srv_name = e_def[u.EXPLOIT_SERVICE]
                if srv_name not in e_map:
                    e_map[srv_name] = {}
                srv_map = e_map[srv_name]

                os = e_def[u.EXPLOIT_OS]
                if os not in srv_map:
                    srv_map[os] = {
                        "name": e_name,
                        u.EXPLOIT_SERVICE: srv_name,
                        u.EXPLOIT_OS: os,
                        u.EXPLOIT_COST: e_def[u.EXPLOIT_COST],
                        u.EXPLOIT_PROB: e_def[u.EXPLOIT_PROB],
                        u.EXPLOIT_ACCESS: e_def[u.EXPLOIT_ACCESS]
                    }
            self._e_map = e_map
        return self._e_map

    @property
    def privesc_map(self) -> dict[str, dict[str | None, dict[str, Any]]]:
        """A nested dictionary for all privilege escalation actions in scenario.

        I.e. {process_name: {
                 os_name: {
                     name: pe_name,
                     cost: pe_cost,
                     prob: pe_prob,
                     access: pe_access
                 }
             }
        """
        if self._pe_map is None:
            pe_map: dict[str, dict[str | None, dict[str, Any]]] = {}
            for pe_name, pe_def in self.privescs.items():
                proc_name = pe_def[u.PRIVESC_PROCESS]
                if proc_name not in pe_map:
                    pe_map[proc_name] = {}
                proc_map = pe_map[proc_name]

                os = pe_def[u.PRIVESC_OS]
                if os not in proc_map:
                    proc_map[os] = {
                        "name": pe_name,
                        u.PRIVESC_PROCESS: proc_name,
                        u.PRIVESC_OS: os,
                        u.PRIVESC_COST: pe_def[u.PRIVESC_COST],
                        u.PRIVESC_PROB: pe_def[u.PRIVESC_PROB],
                        u.PRIVESC_ACCESS: pe_def[u.PRIVESC_ACCESS]
                    }
            self._pe_map = pe_map
        return self._pe_map

    @property
    def subnets(self) -> Any:
        return self.scenario_dict[u.SUBNETS]

    @property
    def topology(self) -> Any:
        return self.scenario_dict[u.TOPOLOGY]

    @property
    def sensitive_hosts(self) -> dict[Address, float]:
        return cast(dict[Address, float], self.scenario_dict[u.SENSITIVE_HOSTS])

    @property
    def sensitive_addresses(self) -> list[Address]:
        return list(self.sensitive_hosts.keys())

    @property
    def firewall(self) -> dict[tuple[int, int], Collection[str]]:
        return cast(dict[tuple[int, int], Collection[str]], self.scenario_dict[u.FIREWALL])

    @property
    def hosts(self) -> dict[Address, Host]:
        return cast(dict[Address, Host], self.scenario_dict[u.HOSTS])

    @property
    def address_space(self) -> list[Address]:
        return list(self.hosts.keys())

    @property
    def service_scan_cost(self) -> float:
        return cast(float, self.scenario_dict[u.SERVICE_SCAN_COST])

    @property
    def os_scan_cost(self) -> float:
        return cast(float, self.scenario_dict[u.OS_SCAN_COST])

    @property
    def subnet_scan_cost(self) -> float:
        return cast(float, self.scenario_dict[u.SUBNET_SCAN_COST])

    @property
    def process_scan_cost(self) -> float:
        return cast(float, self.scenario_dict[u.PROCESS_SCAN_COST])

    @property
    def address_space_bounds(self) -> tuple[int, int]:
        if "address_space_bounds" in self.scenario_dict:
            return cast(tuple[int, int], self.scenario_dict["address_space_bounds"])

        else:
            return len(self.subnets), max(self.subnets)
